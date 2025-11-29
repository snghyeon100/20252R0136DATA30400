import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from . import config, utils

class EarlyStopping:
    """조기 종료를 위한 헬퍼 클래스"""
    def __init__(self, patience=3, delta=0, path=config.MODEL_SAVE_PATH):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'[EarlyStopping] Count: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Validation loss가 감소하면 모델을 저장한다.'''
        print(f'[EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Trainer:
    """
    [Phase 2 & 3] 모델 학습 및 평가
    """
    def __init__(self, model, taxonomy, train_loader, val_loader=None, device=config.DEVICE):
        self.model = model.to(device)
        self.taxonomy = taxonomy
        self.train_loader = train_loader
        self.val_loader = val_loader # 검증 데이터 로더 추가
        self.device = device
        
        # GNN용 데이터 준비
        self.adj_matrix = self.taxonomy.get_adjacency_matrix(device)
        self.class_features = self._prepare_class_features()
        
        # Early Stopping 설정 (Patience=3: 3번 참아줌)
        self.early_stopping = EarlyStopping(patience=3, path=config.MODEL_SAVE_PATH)
        
        # Loss Function
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _prepare_class_features(self):
        # (이전과 동일: LLM 설명문 임베딩 생성 코드)
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
                batch_texts = texts[i : i+batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                
                # SBERT Mean Pooling 권장
                out = encoder(**encoded)
                token_embeddings = out.last_hidden_state
                input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                features.append(emb)
                
        return torch.cat(features, dim=0).detach()

    def _get_optimizer(self, phase=1):
        # (이전과 동일: Optimizer 설정 코드)
        bert_params = list(map(id, self.model.bert.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params, self.model.parameters())
        
        if phase == 1:
            lr_bert = config.LR_BERT_P1
            lr_base = config.LR_BASE_P1
        else:
            lr_bert = config.LR_BERT_P2
            lr_base = config.LR_BASE_P2
            
        optimizer = AdamW([
            {'params': self.model.bert.parameters(), 'lr': lr_bert},
            {'params': base_params, 'lr': lr_base}
        ], weight_decay=0.01)
        return optimizer

    def train(self):
        """전체 학습 파이프라인 (Early Stopping 적용)"""
        print(f"[Trainer] Start Training for {config.NUM_EPOCHS} epochs.")
        
        # 초기 Optimizer 설정
        self.optimizer = self._get_optimizer(phase=1)
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            # Phase 2 전환 체크
            if epoch > 3: 
                print(f"[Trainer] >>> Switching to Phase 2 (Self-Training) <<<")
                self.optimizer = self._get_optimizer(phase=2)
                mode = "self_train"
            else:
                mode = "supervised"
                
            # 1. 학습
            train_loss = self.train_epoch(epoch, mode)
            
            # 2. 검증 (Validation)
            val_loss = self.evaluate(mode)
            
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mode: {mode}")
            
            # 3. Early Stopping 체크
            self.early_stopping(val_loss, self.model)
            
            if self.early_stopping.early_stop:
                print("[Trainer] Early stopping triggered.")
                break

    def train_epoch(self, epoch, mode):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            silver_labels = batch['labels'].to(self.device) 
            
            logits, proj_feat = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            
            if mode == "supervised":
                loss_cls = self._compute_taxonomy_aware_loss(logits, silver_labels)
                loss_con = self._compute_contrastive_loss(proj_feat, silver_labels)
                loss = loss_cls + (0.1 * loss_con)
            else: 
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    target_q = self._compute_target_q(probs)
                loss = F.binary_cross_entropy_with_logits(logits, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, mode):
        """검증 데이터셋 평가"""
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        
        # 검증은 빠르게 진행 (tqdm 없이 하거나 간단히)
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits, proj_feat = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            
            # 검증 Loss도 학습 모드와 동일하게 계산하여 비교
            if mode == "supervised":
                loss_cls = self._compute_taxonomy_aware_loss(logits, labels)
                loss_con = self._compute_contrastive_loss(proj_feat, labels)
                loss = loss_cls + (0.1 * loss_con)
            else:
                probs = torch.sigmoid(logits)
                target_q = self._compute_target_q(probs)
                loss = F.binary_cross_entropy_with_logits(logits, target_q)
                
            total_loss += loss.item()
            
        return total_loss / len(self.val_loader)

    # ... (나머지 _compute_taxonomy_aware_loss, _compute_contrastive_loss, _compute_target_q, predict 함수는 이전과 동일) ...
    # 코드 길이상 생략된 부분은 이전에 드린 코드의 함수들을 그대로 붙여넣으시면 됩니다.
    def _compute_taxonomy_aware_loss(self, logits, silver_labels):
        # (이전과 동일)
        bce_loss = self.bce_loss(logits, silver_labels)
        mask = torch.ones_like(silver_labels)
        pred_indices = silver_labels.nonzero()
        silver_labels_cpu = silver_labels.cpu()
        for i in range(silver_labels.shape[0]):
            core_classes = torch.where(silver_labels[i] == 1)[0].cpu().tolist()
            for c in core_classes:
                children = self.taxonomy.get_children(c)
                if children:
                    mask[i, children] = 0.0
        masked_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-9)
        return masked_loss

    def _compute_contrastive_loss(self, features, labels, temperature=0.07):
        # (이전과 동일)
        labels_float = labels.float()
        label_dot = torch.matmul(labels_float, labels_float.T)
        mask = (label_dot > 0).float()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask
        sim_matrix = torch.matmul(features, features.T) / temperature
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = - mean_log_prob_pos
        loss = loss[mask.sum(1) > 0].mean()
        if torch.isnan(loss): return torch.tensor(0.0, device=self.device)
        return loss

    def _compute_target_q(self, p):
        # (이전과 동일)
        weight = p ** 2 / p.sum(0)
        return (weight.t() / weight.sum(1)).t()

    @torch.no_grad()
    def predict(self, loader):
        # (이전과 동일)
        self.model.eval()
        all_preds = []
        all_pids = []
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            pids = batch['pid']
            logits, _ = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            probs = torch.sigmoid(logits)
            preds = []
            for i in range(len(probs)):
                top_vals, top_inds = torch.topk(probs[i], k=3)
                valid_inds = []
                for val, idx in zip(top_vals, top_inds):
                    if val > 0.5 or len(valid_inds) < 2:
                        valid_inds.append(idx.item())
                all_preds.append(valid_inds)
            all_pids.extend(pids)
        return all_pids, all_preds