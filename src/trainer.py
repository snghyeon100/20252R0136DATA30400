import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from . import config, utils


class EarlyStopping:
    """
    F1 Score(Max) 기준으로 작동하는 Early Stopping
    """
    def __init__(self, patience=3, delta=0, path=config.MODEL_SAVE_PATH, mode='max'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.val_score_best = -np.inf if mode == 'max' else np.inf

    def __call__(self, score, model):
        if self.mode == 'max':
            improvement = (score > self.best_score + self.delta) if self.best_score is not None else True
        else:
            improvement = (score < self.best_score - self.delta) if self.best_score is not None else True

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif improvement:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'[EarlyStopping] Count: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        print(f'[EarlyStopping] Score improved ({self.val_score_best:.6f} --> {score:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_score_best = score


class Trainer:
    """
    Supervised-only 모델 학습 및 평가 (Self-Training 제거)
    """
    def __init__(self, model, taxonomy, train_loader, val_loader=None, device=config.DEVICE):
        self.model = model.to(device)
        self.taxonomy = taxonomy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # GNN용 데이터 준비
        self.adj_matrix = self.taxonomy.get_adjacency_matrix(device)
        self.class_features = self._prepare_class_features()

        self.early_stopping = EarlyStopping(patience=3, path=config.MODEL_SAVE_PATH, mode='max')

        pos_weight = torch.ones([config.NUM_CLASSES]) * 20.0
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(device))

    def _prepare_class_features(self):
        print("[Trainer] Preparing Class Features from LLM data...")
        if os.path.exists(config.EXPANDED_KEYWORDS_PATH):
            with open(config.EXPANDED_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
        else:
            llm_data = {}

        texts = []
        for cid in range(config.NUM_CLASSES):
            cname = self.taxonomy.id2name[cid]
            info = llm_data.get(str(cid), {})
            keywords = info.get("keywords", [])
            desc = info.get("description", "")
            if not keywords:
                keywords = self.taxonomy.raw_keywords.get(cid, [])
            text = f"{cname}: {', '.join(keywords)}. {desc}"
            texts.append(text)

        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        encoder = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(self.device)
        encoder.eval()

        features = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                out = encoder(**encoded)
                token_embeddings = out.last_hidden_state
                input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                features.append(emb)
        return torch.cat(features, dim=0).detach()

    def _get_optimizer(self):
        bert_params = list(map(id, self.model.bert.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params, self.model.parameters())

        lr_bert = config.LR_BERT_P1
        lr_base = config.LR_BASE_P1

        optimizer = AdamW([
            {'params': self.model.bert.parameters(), 'lr': lr_bert},
            {'params': base_params, 'lr': lr_base}
        ], weight_decay=0.01)
        return optimizer

    def train(self):
        print(f"[Trainer] Start Supervised Training for {config.NUM_EPOCHS} epochs.")
        self.optimizer = self._get_optimizer()

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)

            val_loss, val_f1 = self.evaluate()

            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

            self.early_stopping(val_f1, self.model)
            if self.early_stopping.early_stop:
                print("[Trainer] Early stopping triggered.")
                break

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            silver_labels = batch['labels'].to(self.device)

            logits, proj_feat = self.model(input_ids, mask, self.class_features, self.adj_matrix)

            loss_cls = self._compute_taxonomy_aware_loss(logits, silver_labels)
            loss_con = self._compute_contrastive_loss(proj_feat, silver_labels)
            loss = loss_cls + (0.1 * loss_con)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        """
        검증 데이터셋 평가 (Micro F1 Score 반환)
        - Self-Training 제거 -> 항상 supervised 기준
        - 예측은 (sigmoid>0.5) 대신, 너 predict()에서 쓰는 규칙처럼
          "min 2, max 3"로 뽑아 F1을 계산 (실제 제출과 정렬)
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits, proj_feat = self.model(input_ids, mask, self.class_features, self.adj_matrix)

            # loss
            loss = self._compute_taxonomy_aware_loss(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            # (중요) min2/max3 디코딩으로 multi-hot preds 만들기
            preds = self._decode_min2_max3(probs, threshold=0.5)

            # 디버그 출력
            print(f"Pred Prob Mean: {probs.mean().item():.4f}")
            print(f"Label Sum: {labels.sum().item()}")
            avg_pred_count = preds.sum(dim=1).mean().item()
            print(f"Avg Labels per Data: {avg_pred_count:.2f}")

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / len(self.val_loader)

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        f1 = f1_score(all_labels, all_preds, average='micro')
        return avg_loss, f1

    def _decode_min2_max3(self, probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        probs: (B, C)
        return: multi-hot (B, C) where each row has 2~3 labels
        """
        B, C = probs.shape
        preds = torch.zeros_like(probs)

        top_vals, top_inds = torch.topk(probs, k=3, dim=1)  # (B, 3)

        for i in range(B):
            chosen = []
            for val, idx in zip(top_vals[i], top_inds[i]):
                if val.item() > threshold or len(chosen) < 2:
                    chosen.append(int(idx.item()))
            preds[i, chosen] = 1.0

        return preds

    def _compute_taxonomy_aware_loss(self, logits, silver_labels):
        bce_loss = self.bce_loss(logits, silver_labels)
        mask = torch.ones_like(silver_labels, device=self.device)

        for i in range(silver_labels.shape[0]):
            core_classes = torch.where(silver_labels[i] == 1)[0].tolist()
            for c in core_classes:
                children = self.taxonomy.get_children(c)
                if children:
                    children_tensor = torch.tensor(children, device=self.device)
                    mask[i].index_fill_(0, children_tensor, 0.0)

        masked_loss_sum = (bce_loss * mask).sum()
        valid_elements_count = mask.sum()

        if valid_elements_count.item() == 0:
            return torch.tensor(0.0, device=self.device)

        return masked_loss_sum / valid_elements_count

    def _compute_contrastive_loss(self, features, labels, temperature=0.07):
        labels_float = labels.float()
        label_dot = torch.matmul(labels_float, labels_float.T)
        pos_mask = (label_dot > 0).float()

        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(pos_mask.shape[0]).view(-1, 1).to(self.device),
            0
        )
        pos_mask = pos_mask * logits_mask

        sim_matrix = torch.matmul(features, features.T) / temperature
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos
        loss = loss[pos_mask.sum(1) > 0].mean()

        if torch.isnan(loss):
            return torch.tensor(0.0, device=self.device)
        return loss

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        all_preds = []
        all_pids = []

        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            pids = batch['pid']

            logits, _ = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            probs = torch.sigmoid(logits)

            # [Kaggle Rule] Min 2, Max 3 Labels
            for i in range(len(probs)):
                top_vals, top_inds = torch.topk(probs[i], k=3)
                valid_inds = []
                for val, idx in zip(top_vals, top_inds):
                    if val > 0.5 or len(valid_inds) < 2:
                        valid_inds.append(idx.item())
                all_preds.append(valid_inds)

            all_pids.extend(pids)

        return all_pids, all_preds
