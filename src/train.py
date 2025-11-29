import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from . import config, utils

class Trainer:
    """
    [Phase 2 & 3] 모델 학습 및 평가를 담당하는 클래스
    """
    def __init__(self, model, taxonomy, train_loader, test_loader=None, device=config.DEVICE):
        self.model = model.to(device)
        self.taxonomy = taxonomy
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 1. GNN용 족보 행렬 (Self-loop & Norm 처리됨)
        # data_loader.py에 추가한 함수 호출
        self.adj_matrix = self.taxonomy.get_adjacency_matrix(device)
        
        # 2. GNN용 클래스 피처 (LLM 설명문 임베딩) 준비
        self.class_features = self._prepare_class_features()
        
        # 3. Optimizer 설정 (Differential Learning Rate)
        self.optimizer = self._get_optimizer(phase=1)
        
        # 4. Loss Function
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') # 마스킹을 위해 none 설정

    def _prepare_class_features(self):
        """
        LLM이 생성한 '클래스 설명(keywords+description)'을 로드하여
        고정된 텐서 (531, 768)로 변환합니다. (GNN 입력용)
        """
        print("[Trainer] Preparing Class Features from LLM data...")
        
        # 파일 로드
        if os.path.exists(config.EXPANDED_KEYWORDS_PATH):
            with open(config.EXPANDED_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
        else:
            print("[Warning] LLM data not found. Using raw keywords only.")
            llm_data = {}

        # 텍스트 리스트 생성 (ID 순서대로)
        texts = []
        for cid in range(config.NUM_CLASSES):
            cname = self.taxonomy.id2name[cid]
            # LLM 데이터가 있으면 쓰고, 없으면 기본 정보 사용
            info = llm_data.get(str(cid), {})
            keywords = info.get("keywords", [])
            desc = info.get("description", "")
            
            if not keywords: # LLM 데이터 없을 때 fallback
                keywords = self.taxonomy.raw_keywords.get(cid, [])
                
            text = f"{cname}: {', '.join(keywords)}. {desc}"
            texts.append(text)
            
        # BERT/SBERT를 이용해 임베딩 (학습 없이 Inference만)
        # 메모리 절약을 위해 별도의 tokenizer 사용 혹은 model 내부 모듈 활용
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        encoder = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(self.device)
        encoder.eval()
        
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i+batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                
                # [CLS] 토큰 또는 Mean Pooling 사용
                out = encoder(**encoded)
                # 여기선 간단히 CLS 사용 (SBERT면 Mean Pooling 권장하지만 CLS도 무방)
                emb = out.last_hidden_state[:, 0, :] 
                features.append(emb)
                
        return torch.cat(features, dim=0).detach() # Gradient 끊음 (초기값으로만 사용)

    def _get_optimizer(self, phase=1):
        """
        학습 단계에 따라 다른 Learning Rate 적용
        Phase 1: 초기 학습 (BERT 5e-5)
        Phase 2: Self-Training (BERT 1e-6)
        """
        # 파라미터 그룹 분리
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
        """전체 학습 파이프라인"""
        print(f"[Trainer] Start Training for {config.NUM_EPOCHS} epochs.")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            # Phase Switch: 절반 지나면 Self-Training 모드 & LR 감소
            if epoch > 3: # 예: 4에폭부터 Self-Training
                print(f"[Trainer] >>> Switching to Phase 2 (Self-Training) <<<")
                self.optimizer = self._get_optimizer(phase=2)
                mode = "self_train"
            else:
                mode = "supervised"
                
            loss = self.train_epoch(epoch, mode)
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Loss: {loss:.4f} | Mode: {mode}")
            
            # 모델 저장 (매 에폭 혹은 Best)
            torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)

    def train_epoch(self, epoch, mode):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # 데이터 GPU 이동
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            # Silver Labels (Phase 1 결과물)
            silver_labels = batch['labels'].to(self.device) 
            
            # 1. Forward Pass
            # class_features와 adj_matrix는 매번 넣어줌
            logits, proj_feat = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            
            # 2. Loss Calculation
            if mode == "supervised":
                # Phase 1: Silver Label 기준 학습 + Contrastive Loss
                loss_cls = self._compute_taxonomy_aware_loss(logits, silver_labels)
                loss_con = self._compute_contrastive_loss(proj_feat, silver_labels)
                loss = loss_cls + (0.1 * loss_con)
                
            else: # mode == "self_train"
                # Phase 2: Self-Training (KL Div)
                # 25 Step마다 Target Q 업데이트 (논문 구현) -> 여기선 배치 단위 근사(Approximation) 사용
                # 현재 배치의 예측값(P)을 강화해서 정답(Q)으로 삼음
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    target_q = self._compute_target_q(probs)
                
                # KL Divergence Loss
                log_probs = torch.log_softmax(logits, dim=1) # BCE 대신 KL 사용 시
                # Multi-label에서는 각 클래스별로 KL을 계산해야 함.
                # 여기선 간단히 강화된 Q와 P 사이의 BCE로 대체하거나 MSE 사용 가능
                # 논문 수식 (9) 구현: Q * log(Q/P)
                # 여기서는 구현 편의상 'Soft Label에 대한 BCE'로 근사
                loss = F.binary_cross_entropy_with_logits(logits, target_q)

            # 3. Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def _compute_taxonomy_aware_loss(self, logits, silver_labels):
        """
        [핵심 전략] 족보 기반 마스킹 BCE Loss
        - Positive(1): Core Class + Parents
        - Ignore(Mask): Core Class의 Children -> Loss 계산에서 제외
        - Negative(0): 나머지
        """
        # 1. 기본 BCE Loss (Reduction None -> 픽셀별 Loss 계산)
        bce_loss = self.bce_loss(logits, silver_labels)
        
        # 2. 마스크 생성 (기본 1.0)
        mask = torch.ones_like(silver_labels)
        
        # 배치의 각 샘플마다 Ignore 할 자식 노드 찾기
        # (벡터화가 어렵기 때문에 배치 내 루프가 불가피함 - Batch 64라 빠름)
        pred_indices = silver_labels.nonzero() # (N, 2) -> [batch_idx, class_idx]
        
        # Core Class(1로 설정된 애들)의 자식들을 찾아서 마스크 0으로 설정
        # CPU로 잠시 내려서 처리 (Graph 접근 때문)
        silver_labels_cpu = silver_labels.cpu()
        
        for i in range(silver_labels.shape[0]):
            # 이 문서의 정답 클래스들
            core_classes = torch.where(silver_labels[i] == 1)[0].cpu().tolist()
            
            for c in core_classes:
                children = self.taxonomy.get_children(c)
                if children:
                    # 자식 인덱스들의 마스크를 0으로 (Loss 제외)
                    mask[i, children] = 0.0
                    
        # 3. 마스킹 적용 후 평균
        masked_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-9)
        return masked_loss

    def _compute_contrastive_loss(self, features, labels, temperature=0.07):
        """
        Supervised Contrastive Loss
        같은 Core Class를 가진 리뷰끼리 당기기
        """
        # 라벨이 겹치는 것이 하나라도 있으면 Positive Pair로 간주
        # (Batch x Batch) 유사도 행렬
        labels_float = labels.float()
        # 공통된 라벨 개수 계산 (Batch, Batch)
        # A와 B가 같은 라벨을 공유하면 > 0
        label_dot = torch.matmul(labels_float, labels_float.T)
        mask = (label_dot > 0).float()
        
        # 자기 자신 제외
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device), 
            0
        )
        mask = mask * logits_mask
        
        # Feature 유사도
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        # Log-Softmax
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Positive Pair에 대한 Loss
        # 나랑 같은 팀이 하나도 없는 경우(분모 0) 방지
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        # mask.sum(1)이 0인 경우(나랑 같은 팀 없음) Loss 0 처리
        loss = - mean_log_prob_pos
        loss = loss[mask.sum(1) > 0].mean() # 유효한 것만 평균
        
        if torch.isnan(loss): return torch.tensor(0.0, device=self.device)
        return loss

    def _compute_target_q(self, p):
        """
        Self-Training용 Target Q 생성 (Sharpening)
        공식: q_ij = p_ij^2 / sum(p) ... (정규화)
        확률분포를 뾰족하게 만들어서(0.7 -> 0.9, 0.3 -> 0.1) 확신을 강화함
        """
        # p: (Batch, Num_Classes)
        weight = p ** 2 / p.sum(0)
        return (weight.t() / weight.sum(1)).t()

    @torch.no_grad()
    def predict(self, loader):
        """제출용 예측 함수"""
        self.model.eval()
        all_preds = []
        all_pids = []
        
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            pids = batch['pid']
            
            # 예측 (GNN 피처와 족보는 그대로 사용)
            logits, _ = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            probs = torch.sigmoid(logits)
            
            # Thresholding (0.5 이상이면 정답)
            # 혹은 Top-K (최소 2개, 최대 3개)
            preds = []
            for i in range(len(probs)):
                # 상위 3개 뽑기 (Kaggle Rule: 2~3 labels)
                top_vals, top_inds = torch.topk(probs[i], k=3)
                
                # 확률이 너무 낮은 건 자르되, 최소 2개는 보장
                valid_inds = []
                for val, idx in zip(top_vals, top_inds):
                    if val > 0.5 or len(valid_inds) < 2:
                        valid_inds.append(idx.item())
                
                all_preds.append(valid_inds)
            
            all_pids.extend(pids)
            
        return all_pids, all_preds