import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from . import config

class GraphEncoder(nn.Module):
    """
    [우뇌] 클래스(Taxonomy) 정보를 처리하는 GNN 모듈
    
    역할:
    - LLM이 생성한 '클래스 설명' 임베딩을 초기값으로 사용
    - '족보(Adjacency Matrix)'를 통해 부모-자식 간 정보를 교환(Propagation)
    - 최종적으로 '구조적 의미가 담긴 클래스 벡터'를 생성
    """
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        # GCN Layer: Feature Transformation (W)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, feature_matrix, adj_matrix):
        """
        Args:
            feature_matrix: (Num_Classes, 768) - 초기 클래스 임베딩 (LLM 설명문)
            adj_matrix: (Num_Classes, Num_Classes) - 정규화된 인접 행렬
        Returns:
            (Num_Classes, 768) - 족보 정보가 반영된 클래스 임베딩
        """
        # 1. 선형 변환 (Transformation): X * W
        h = self.linear(feature_matrix)
        
        # 2. 정보 전파 (Propagation): A * H
        # 이웃 노드(부모/자식)의 정보가 나에게 흘러들어옴
        h = torch.matmul(adj_matrix, h)
        
        # 3. 비선형 활성화 & 드롭아웃
        h = self.activation(h)
        h = self.dropout(h)
        
        # (옵션) Residual Connection: 원래 내 정보도 잊지 않도록 더해줌
        h = h + feature_matrix 
        
        return h

class DualEncoder(nn.Module):
    """
    [Main Brain] BERT + GNN 결합 모델 (TaxoClass Architecture)
    
    구조:
    1. Document Encoder (BERT): 리뷰 텍스트 -> 벡터
    2. Class Encoder (GNN): 텍스트 + 그래프 -> 벡터
    3. Interaction: 두 벡터의 내적(Dot Product)으로 분류 점수 계산
    4. Projection Head: 대조 학습(Contrastive Learning)을 위한 벡터 압축
    """
    def __init__(self, num_classes=config.NUM_CLASSES, hidden_dim=768):
        super(DualEncoder, self).__init__()
        
        # 1. Document Encoder (BERT)
        print(f"[Model] Loading Pretrained BERT: {config.BERT_MODEL_NAME}...")
        self.bert = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        
        # 2. Class Encoder (GNN)
        # BERT의 출력 차원(768)을 그대로 유지하며 그래프 연산 수행
        self.gnn = GraphEncoder(hidden_dim, hidden_dim)
        
        # 3. Interaction Bias
        # 내적 계산 시 각 클래스별 고유한 편향(Bias)을 학습
        self.final_bias = nn.Parameter(torch.zeros(num_classes))

        # 4. Contrastive Learning Projection Head
        # 768차원 -> 128차원으로 압축하여 벡터 공간상에서 군집화 유도
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128) # 128차원으로 축소
        )
        
        # BERT 파라미터 일부 고정 (선택 사항)
        # 초기 학습 안정화를 위해 하위 레이어를 얼릴 수도 있음 (여기선 다 풂)
        # for param in self.bert.parameters():
        #     param.requires_grad = True

    def forward(self, input_ids, attention_mask, class_features, adj_matrix):
        """
        Args:
            input_ids, attention_mask: 리뷰 데이터 배치 (Batch, Seq_Len)
            class_features: 모든 클래스의 초기 텍스트 임베딩 (531, 768)
            adj_matrix: 족보 그래프 (531, 531)
        
        Returns:
            logits: 분류 점수 (Batch, 531)
            proj_feat: 대조 학습용 임베딩 (Batch, 128)
        """
        
        # --- A. Document Encoding (좌뇌) ---
        # BERT를 통과시켜 [CLS] 토큰 벡터 추출
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        doc_emb = outputs.last_hidden_state[:, 0, :] # (Batch, 768)
        
        # --- B. Class Encoding (우뇌) ---
        # GNN을 통과시켜 구조적 정보가 담긴 클래스 벡터 생성
        # 이 연산은 배치 크기와 무관하게 한 번만 수행하면 되지만, 
        # 학습 중에는 매번 수행하여 업데이트된 가중치를 반영함
        class_emb = self.gnn(class_features, adj_matrix) # (531, 768)
        
        # --- C. Prediction (유사도 계산) ---
        # 문서 벡터와 클래스 벡터 간의 내적 (Dot Product)
        # (Batch, 768) x (768, 531) -> (Batch, 531)
        logits = torch.matmul(doc_emb, class_emb.t()) + self.final_bias
        
        # --- D. Contrastive Learning Output ---
        # 대조 학습을 위해 벡터 투영 및 정규화 (L2 Normalization)
        proj_feat = self.projection_head(doc_emb)
        proj_feat = F.normalize(proj_feat, p=2, dim=1)
        
        return logits, proj_emb