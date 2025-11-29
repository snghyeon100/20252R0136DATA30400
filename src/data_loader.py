import os
import networkx as nx
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from . import config

class Taxonomy:
    """
    계층 구조(Taxonomy)와 클래스 정보를 관리하는 클래스입니다.
    데이터 로딩 시 가장 먼저 초기화해야 합니다.
    """
    def __init__(self):
        self.id2name = {}
        self.name2id = {}
        self.graph = nx.DiGraph() # 방향 그래프 (Parent -> Child)
        self.raw_keywords = {}    # 제공된 기본 키워드

        # 데이터 로드 실행
        self._load_classes()
        self._load_hierarchy()
        self._load_keywords()

    def _load_classes(self):
        """classes.txt 로드: ID <-> Name 매핑 생성"""
        print(f"[Data] Loading classes from {config.CLASS_PATH}")
        with open(config.CLASS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    cid, cname = parts
                    cid = int(cid)
                    self.id2name[cid] = cname
                    self.name2id[cname] = cid
        print(f"       - Total classes: {len(self.id2name)}")

    def _load_hierarchy(self):
        """class_hierarchy.txt 로드: 부모-자식 그래프 생성"""
        print(f"[Data] Loading hierarchy from {config.TAXONOMY_PATH}")
        with open(config.TAXONOMY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    parent, child = map(int, parts)
                    self.graph.add_edge(parent, child)
        print(f"       - Total edges: {self.graph.number_of_edges()}")

    def _load_keywords(self):
        """class_related_keywords.txt 로드"""
        # 형식: grocery_gourmet_food:snacks,condiments,...
        if not os.path.exists(config.KEYWORDS_PATH):
            print("[Warning] Keywords file not found.")
            return

        print(f"[Data] Loading keywords from {config.KEYWORDS_PATH}")
        with open(config.KEYWORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    cname, keywords_str = line.split(":", 1)
                    if cname in self.name2id: # 유효한 클래스인지 확인
                        self.raw_keywords[self.name2id[cname]] = keywords_str.split(",")

    # --- 유틸리티 함수들 ---
    def get_children(self, class_id):
        """특정 클래스의 직계 자식 ID 반환"""
        if self.graph.has_node(class_id):
            return list(self.graph.successors(class_id))
        return []

    def get_parents(self, class_id):
        """특정 클래스의 직계 부모 ID 반환"""
        if self.graph.has_node(class_id):
            return list(self.graph.predecessors(class_id))
        return []
        
    def get_ancestors(self, class_id):
        """모든 조상 ID 반환 (재귀적)"""
        return list(nx.ancestors(self.graph, class_id))
    
    def get_adjacency_matrix(self, device):
        """
        GNN 학습용 인접 행렬 생성 (Self-loop + Normalization)
        """
        num_classes = len(self.id2name)
        
        # 1. 단위 행렬(Identity Matrix)로 초기화
        # 대각선이 모두 1이므로 '나 자신 포함(Self-loop)' 문제가 자동으로 해결됩니다.
        adj = torch.eye(num_classes, device=device)
        
        # 2. 그래프 연결 정보 채우기 (양방향)
        # GNN에서는 정보가 부모<->자식 양방향으로 흘러야 하므로 둘 다 1로 설정합니다.
        for u, v in self.graph.edges():
            adj[u, v] = 1 
            adj[v, u] = 1 
            
        # 3. 정규화 (Normalization)
        # 친구가 많은 노드는 값이 너무 커지는 것을 방지하기 위해, 연결된 수만큼 나눠줍니다.
        # (Row-normalization: D^-1 * A)
        row_sum = adj.sum(dim=1, keepdim=True)
        
        # 0으로 나누는 에러 방지 (혹시 고립된 노드가 있다면)
        row_sum[row_sum == 0] = 1 
        
        norm_adj = adj / row_sum
        
        return norm_adj

class ReviewDataset(Dataset):
    """
    BERT 학습을 위한 PyTorch 데이터셋
    """
    def __init__(self, corpus_path, tokenizer, max_len=128, silver_labels=None):
        self.data = []
        self.pids = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.silver_labels = silver_labels # {pid: [0, 1, 0...]} 형태의 텐서 또는 딕셔너리

        # 파일 로드 (pid \t text)
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1) # 탭 1번만 분리
                if len(parts) == 2:
                    pid, text = parts
                    self.data.append(text)
                    self.pids.append(pid)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        pid = self.pids[idx]

        # BERT 토크나이징
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "pid": pid
        }

        # 학습 시 Silver Label이 있으면 같이 반환
        if self.silver_labels is not None:
            # pid에 해당하는 라벨 가져오기 (만약 딕셔너리면)
            if isinstance(self.silver_labels, dict):
                 if pid in self.silver_labels:
                     item["labels"] = torch.tensor(self.silver_labels[pid], dtype=torch.float)
                 else:
                     # 라벨 없는 경우 (드문 케이스)
                     item["labels"] = torch.zeros(config.NUM_CLASSES, dtype=torch.float)
            # 텐서 리스트면 인덱스로 접근
            else:
                 item["labels"] = self.silver_labels[idx]

        return item

# 간단한 테스트 함수
if __name__ == "__main__":
    taxo = Taxonomy()
    print("Root nodes:", [n for n, d in taxo.graph.in_degree() if d == 0])