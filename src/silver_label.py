import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from . import config

class SilverLabeler:
    """
    [Phase 1] 가상 정답(Silver Label) 생성기
    Strategy: SBERT Retrieval (Top-K) -> Hierarchy Filtering -> Confidence Check
    """
    def __init__(self, taxonomy, data_loader, device=config.DEVICE):
        self.taxonomy = taxonomy
        self.data_loader = data_loader
        self.device = device
        
        # 고성능 임베딩 모델 로드 (Retrieval용)
        # all-mpnet-base-v2는 속도와 성능 균형이 매우 좋음
        print("[SilverLabeler] Loading SBERT Model (all-mpnet-base-v2)...")
        self.encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
        
        self.similarity_matrix = None
        self.silver_labels = {} 

    def run(self):
        """전체 파이프라인 실행"""
        # 1. 유사도 행렬 계산 (없으면 계산, 있으면 로드)
        if os.path.exists(config.SIMILARITY_MATRIX_PATH):
            print(f"[SilverLabeler] Loading similarity matrix from {config.SIMILARITY_MATRIX_PATH}")
            self.similarity_matrix = torch.load(config.SIMILARITY_MATRIX_PATH, map_location=self.device)
        else:
            self.similarity_matrix = self._calculate_similarity()
            torch.save(self.similarity_matrix, config.SIMILARITY_MATRIX_PATH)
            print("[SilverLabeler] Similarity matrix saved.")

        # 2. Core Class Mining (SOTA Strategy 적용)
        self._mine_core_classes_sota()

        # 3. 결과 저장
        torch.save(self.silver_labels, config.SILVER_LABELS_PATH)
        print(f"[SilverLabeler] Silver labels saved to {config.SILVER_LABELS_PATH}")

    def _calculate_similarity(self):
        """
        [Retrieval Step] 모든 리뷰와 클래스의 유사도를 계산 (Top-K 후보 선정용)
        """
        print("[SilverLabeler] Step 1: Calculating Embedding Similarity...")
        
        # A. 클래스 텍스트 생성 (Enrichment)
        class_texts = []
        for cid in range(config.NUM_CLASSES):
            cname = self.taxonomy.id2name[cid]
            keywords = self.taxonomy.raw_keywords.get(cid, [])
            # 포맷: "Class Name: keyword1, keyword2..."
            text = f"{cname}: {', '.join(keywords)}"
            class_texts.append(text)

        # B. 임베딩 생성
        print("   - Encoding Classes...")
        class_embeddings = self.encoder.encode(class_texts, convert_to_tensor=True, show_progress_bar=False)
        
        print("   - Encoding Reviews...")
        # data_loader에서 텍스트 리스트만 가져옴
        review_texts = self.data_loader.data 
        doc_embeddings = self.encoder.encode(review_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64)

        # C. 코사인 유사도 계산
        print("   - Computing Cosine Matrix...")
        # 결과 크기: (N_docs, 531)
        similarity_matrix = util.cos_sim(doc_embeddings, class_embeddings)
        
        return similarity_matrix

    def _mine_core_classes_sota(self):
        """
        [Filtering Step] Top-K Retrieval + Hierarchy Check 전략 적용
        """
        print("[SilverLabeler] Step 2: Mining Core Classes (Retrieval -> Filter)...")
        
        num_docs = self.similarity_matrix.shape[0]
        pids = self.data_loader.pids
        final_labels = {}
        
        # 검색할 후보 개수 (SBERT가 뽑을 1차 후보군)
        TOP_K = 20
        # 최소 유사도 임계값 (이 점수보다 낮으면 후보로도 안 침)
        MIN_SCORE_THRESHOLD = 0.3
        
        for i in tqdm(range(num_docs), desc="Mining"):
            doc_sims = self.similarity_matrix[i] # 현재 문서의 전체 점수
            
            # --- 1. Retrieval: 점수 높은 순으로 상위 K개 후보만 일단 가져옴 ---
            # values: 점수들, indices: 클래스 ID들
            top_k_values, top_k_indices = torch.topk(doc_sims, k=TOP_K)
            top_k_indices = top_k_indices.tolist()
            
            # --- 2. Hierarchy Filtering: "족보 있는 집안인가?" ---
            # 후보군(Top-K) 내에서 부모-자식이 연결되어 있는지 확인
            valid_candidates = []
            for cid in top_k_indices:
                # 점수 너무 낮으면 제외
                if doc_sims[cid] < MIN_SCORE_THRESHOLD:
                    continue

                parents = self.taxonomy.get_parents(cid)
                
                # 규칙: (내 부모 중 하나라도 Top-K 안에 있거나) OR (내가 루트 노드라면) 합격
                # 즉, 고립된(Isolated) 노드는 노이즈로 간주하고 버림
                is_connected = any(p in top_k_indices for p in parents)
                is_root = (len(parents) == 0)
                
                if is_connected or is_root:
                    valid_candidates.append(cid)
            
            if not valid_candidates:
                # 필터링 후 남은 게 없으면 라벨 없음 (학습 제외됨)
                final_labels[pids[i]] = torch.zeros(config.NUM_CLASSES)
                continue

            # --- 3. Confidence Check: "부모보다 확실히 더 구체적인가?" (TaxoClass Logic) ---
            core_classes = []
            for c in valid_candidates:
                # 부모/형제 점수 비교
                if self._check_local_confidence(c, doc_sims):
                    core_classes.append(c)

            # --- 4. Label Expansion: Core Class + 조상님들 ---
            label_vec = torch.zeros(config.NUM_CLASSES)
            if core_classes:
                # Core Class 마킹
                label_vec[core_classes] = 1.0
                
                # 조상 클래스들도 자동으로 1.0 처리 (Taxonomy Constraint)
                for core in core_classes:
                    ancestors = self.taxonomy.get_ancestors(core)
                    label_vec[list(ancestors)] = 1.0
            
            final_labels[pids[i]] = label_vec

        self.silver_labels = final_labels
        print(f"[SilverLabeler] Finished. Generated labels for {len(final_labels)} docs.")

    def _check_local_confidence(self, class_id, doc_sims):
        """
        본인 점수 - Max(가족 점수) > Threshold 확인
        """
        my_score = doc_sims[class_id].item()
        
        parents = self.taxonomy.get_parents(class_id)
        # 형제 찾기: 부모의 자식들 중 나를 뺀 것
        siblings = []
        for p in parents:
            sibs = self.taxonomy.get_children(p)
            siblings.extend([s for s in sibs if s != class_id])
            
        competitors = parents + siblings
        if not competitors:
            return True # 경쟁자 없으면 통과
            
        # 경쟁자 중 1등 점수
        comp_scores = [doc_sims[comp].item() for comp in competitors]
        max_comp_score = max(comp_scores) if comp_scores else 0
        
        # 내 점수가 경쟁자보다 0.05 이상 높아야 인정 (수치 조정 가능)
        return (my_score - max_comp_score) > 0.05