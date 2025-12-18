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
    + (중요) evaluate/predict 디코딩을 "path 디코딩"으로 변경
      - leaf topK -> best path -> sibling softmax prob로 leaf 포함/제외
      - 결과는 항상 root-parent 또는 root-parent-leaf (2개 or 3개)
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

        # ---------------------------
        # Path 디코딩 준비 (캐시/하이퍼파라미터)
        # ---------------------------
        self.num_classes = config.NUM_CLASSES

        self.leaf_ids = self._collect_leaf_nodes()
        self.leaf_ids_tensor = torch.tensor(self.leaf_ids, device=self.device, dtype=torch.long)

        self._is_single_parent_graph = self._check_single_parent_graph()
        self._fixed_chain_cache = self._precompute_fixed_leaf_chains() if self._is_single_parent_graph else None

        # path 선택 점수 가중치 (silver_label.py와 동일한 디폴트)
        self.w_root = float(getattr(config, "PATH_W_ROOT", 0.4))
        self.w_parent = float(getattr(config, "PATH_W_PARENT", 0.7))
        self.w_leaf = float(getattr(config, "PATH_W_LEAF", 1.0))

        # leaf 후보 top-k
        self.top_k_leaf = int(getattr(config, "SBERT_LEAF_TOP_K", 30))

        # leaf 포함 여부 결정: sibling softmax 확률 임계값
        self.tau_prob = float(getattr(config, "LEAF_PROB_THRESHOLD", 0.62))
        self.temperature = float(getattr(config, "LEAF_SOFTMAX_TEMPERATURE", 1.0))
        self.temperature = max(self.temperature, 1e-6)

        # eval 디버그 프린트(너무 많이 찍히면 느려짐)
        self.eval_debug = bool(getattr(config, "EVAL_DEBUG", False))

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

        (중요 변경)
        - 기존: sigmoid>0.5 + min2/max3 topk
        - 변경: "path 디코딩"으로 2~3개 라벨을 항상 한 path로 만들고 F1 계산
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for bidx, batch in enumerate(self.val_loader):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits, proj_feat = self.model(input_ids, mask, self.class_features, self.adj_matrix)

            # loss
            loss = self._compute_taxonomy_aware_loss(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            # (핵심) path 디코딩으로 preds 생성 (multi-hot)
            preds = self._decode_path(probs)

            if self.eval_debug and bidx == 0:
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

    # ---------------------------
    # (NEW) Path Decoding
    # ---------------------------
    def _decode_path(self, probs: torch.Tensor) -> torch.Tensor:
        """
        probs: (B, C)
        return: multi-hot (B, C) where each row has 2 or 3 labels, and single-path constraint
        """
        B, C = probs.shape
        preds = torch.zeros_like(probs)

        for i in range(B):
            doc_probs = probs[i]  # (C,)

            # 1) leaf top-K 후보
            leaf_scores = doc_probs.index_select(0, self.leaf_ids_tensor)  # (num_leaf,)
            k = min(self.top_k_leaf, leaf_scores.numel())
            top_vals, top_pos = torch.topk(leaf_scores, k=k)

            cand_leaf_ids = [self.leaf_ids[idx] for idx in top_pos.tolist()]

            # 2) 후보 leaf들 중 best path 1개 선택
            best_chain = None
            best_chain_score = -1e9
            best_leaf_id = None

            for leaf_id in cand_leaf_ids:
                chain = self._get_chain_root_to_leaf(leaf_id, doc_probs)  # [root,...,leaf]
                chain_score = self._score_chain(chain, doc_probs)

                if chain_score > best_chain_score:
                    best_chain_score = chain_score
                    best_chain = chain
                    best_leaf_id = leaf_id

            if not best_chain:
                continue

            # 3) root/parent/leaf 결정
            if len(best_chain) == 1:
                # root만 있는 이상 케이스: 자식 중 하나 붙여 2개 맞추기
                root_id = best_chain[0]
                children = self.taxonomy.get_children(root_id) or []
                if children:
                    child_scores = [(c, float(doc_probs[c].item())) for c in children]
                    child_scores.sort(key=lambda x: x[1], reverse=True)
                    parent_id = child_scores[0][0]
                    preds[i, root_id] = 1.0
                    preds[i, parent_id] = 1.0
                else:
                    preds[i, root_id] = 1.0
                continue

            root_id = best_chain[0]
            parent_id = best_chain[-2]
            leaf_id = best_chain[-1]

            # 4) sibling softmax prob로 leaf 포함 여부 결정
            include_leaf = self._include_leaf_by_softmax_prob(
                parent_id=parent_id,
                chosen_leaf_id=leaf_id,
                doc_probs=doc_probs
            )

            preds[i, root_id] = 1.0
            preds[i, parent_id] = 1.0
            if include_leaf and leaf_id != parent_id:
                preds[i, leaf_id] = 1.0

        return preds

    def _include_leaf_by_softmax_prob(self, parent_id: int, chosen_leaf_id: int, doc_probs: torch.Tensor) -> bool:
        """
        parent 아래 sibling leaf들 점수에 softmax를 적용하고,
        chosen_leaf의 softmax 확률이 tau_prob 이상이면 leaf 포함.

        - sibling set = taxonomy.get_children(parent_id) 중 leaf만
        - sibling leaf가 0/1개면 비교가 무의미하므로 leaf 포함(True)
        """
        children = self.taxonomy.get_children(parent_id) or []
        if not children:
            return True

        sibling_leaf_ids = [c for c in children if len(self.taxonomy.get_children(c) or []) == 0]

        if chosen_leaf_id not in sibling_leaf_ids:
            return True

        if len(sibling_leaf_ids) <= 1:
            return True

        sib_tensor = torch.tensor(sibling_leaf_ids, device=doc_probs.device, dtype=torch.long)
        sib_scores = doc_probs.index_select(0, sib_tensor)
        probs_sib = torch.softmax(sib_scores / self.temperature, dim=0)

        idx = sibling_leaf_ids.index(chosen_leaf_id)
        p = float(probs_sib[idx].item())
        return p >= self.tau_prob

    def _score_chain(self, chain, doc_probs: torch.Tensor) -> float:
        if not chain:
            return -1e9
        if len(chain) == 1:
            return float(doc_probs[chain[0]].item())
        if len(chain) == 2:
            r, l = chain[0], chain[1]
            return float((self.w_root * doc_probs[r] + self.w_leaf * doc_probs[l]).item())

        root_id = chain[0]
        parent_id = chain[-2]
        leaf_id = chain[-1]
        score = self.w_root * doc_probs[root_id] + self.w_parent * doc_probs[parent_id] + self.w_leaf * doc_probs[leaf_id]
        return float(score.item())

    def _get_chain_root_to_leaf(self, leaf_id: int, doc_probs: torch.Tensor):
        """
        leaf_id의 경로를 [root,...,leaf]로 반환
        - 단일부모 트리면 캐시 사용
        - 다중부모(DAG)면 doc_probs 기준으로 가장 큰 부모를 선택해 경로 1개로 만듦
        """
        if self._fixed_chain_cache is not None and leaf_id in self._fixed_chain_cache:
            return self._fixed_chain_cache[leaf_id]

        chain = [leaf_id]
        cur = leaf_id
        visited = {cur}

        while True:
            parents = self.taxonomy.get_parents(cur) or []
            if not parents:
                break

            if len(parents) == 1:
                next_p = parents[0]
            else:
                next_p = max(parents, key=lambda p: float(doc_probs[p].item()))

            if next_p in visited:
                break

            chain.append(next_p)
            visited.add(next_p)
            cur = next_p

        chain.reverse()
        return chain

    def _collect_leaf_nodes(self):
        leaf_ids = []
        for cid in range(self.num_classes):
            children = self.taxonomy.get_children(cid) or []
            if len(children) == 0:
                leaf_ids.append(cid)

        if not leaf_ids:
            leaf_ids = list(range(self.num_classes))

        print(f"[Trainer] #Leaf nodes = {len(leaf_ids)}")
        return leaf_ids

    def _check_single_parent_graph(self) -> bool:
        for cid in range(self.num_classes):
            parents = self.taxonomy.get_parents(cid) or []
            if len(parents) > 1:
                print("[Trainer] Detected multi-parent nodes (DAG). Will choose best-parent per sample.")
                return False
        print("[Trainer] Single-parent taxonomy detected. Using fixed path cache.")
        return True

    def _precompute_fixed_leaf_chains(self):
        cache = {}
        for leaf_id in self.leaf_ids:
            chain = [leaf_id]
            cur = leaf_id
            visited = {cur}

            while True:
                parents = self.taxonomy.get_parents(cur) or []
                if not parents:
                    break
                p = parents[0]
                if p in visited:
                    break
                chain.append(p)
                visited.add(p)
                cur = p

            chain.reverse()
            cache[leaf_id] = chain
        return cache

    # ---------------------------
    # Loss (unchanged)
    # ---------------------------
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

    # ---------------------------
    # Predict (중요 변경)
    # ---------------------------
    @torch.no_grad()
    def predict(self, loader):
        """
        (중요 변경)
        - 기존: 각 샘플에서 probs top3를 threshold로 min2/max3
        - 변경: evaluate와 동일한 path 디코딩으로 2~3개를 뽑음
          => submission의 path invalid가 거의 0%로 떨어져야 정상
        """
        self.model.eval()
        all_preds = []
        all_pids = []

        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            pids = batch['pid']

            logits, _ = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            probs = torch.sigmoid(logits)

            # path 디코딩 -> multi-hot
            mh = self._decode_path(probs)  # (B,C) 0/1

            # 제출 형식용: 각 row의 label id list
            for i in range(mh.size(0)):
                idx = torch.where(mh[i] > 0.5)[0].tolist()
                idx = sorted(map(int, idx))
                all_preds.append(idx)

            all_pids.extend(pids)

        return all_pids, all_preds
