import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from . import config

# OpenAI 라이브러리 안전 로드
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("[Warning] 'openai' library not found. LLM expansion will be skipped.")

class SilverLabeler:
    """
    [Phase 1] 가상 정답(Silver Label) 생성기
    
    기능:
    1. LLM Enrichment: 키워드 확장 + 클래스 설명 생성 (GNN Feature용)
    2. Retrieval: SBERT를 이용한 Top-K 후보군 선정
    3. Filtering: 계층 구조 및 Confidence 기반 정제
    """
    def __init__(self, taxonomy, data_loader, device=config.DEVICE):
        self.taxonomy = taxonomy
        self.data_loader = data_loader
        self.device = device
        
        # API 호출 횟수 카운터
        self.api_call_count = 0
        
        # 모델 로드
        print("[SilverLabeler] Loading SBERT Model (all-mpnet-base-v2)...")
        self.encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
        
        self.similarity_matrix = None
        self.silver_labels = {} 
        self.llm_data = {} # {cid: {'keywords': [], 'description': ''}}

    def run(self):
        """전체 파이프라인 실행"""
        
        # 1. LLM 데이터 생성 (키워드 + 설명)
        self.llm_data = self._load_or_generate_llm_data()
        print(f"[API Stats] Total OpenAI API Calls used: {self.api_call_count}")

        # 2. 유사도 행렬 계산
        if os.path.exists(config.SIMILARITY_MATRIX_PATH):
            print(f"[SilverLabeler] Loading similarity matrix from {config.SIMILARITY_MATRIX_PATH}")
            self.similarity_matrix = torch.load(config.SIMILARITY_MATRIX_PATH, map_location=self.device)
        else:
            self.similarity_matrix = self._calculate_similarity()
            torch.save(self.similarity_matrix, config.SIMILARITY_MATRIX_PATH)
            print("[SilverLabeler] Similarity matrix saved.")

        # 3. Core Class Mining (SOTA Strategy)
        self._mine_core_classes_sota()

        # 4. 결과 저장
        torch.save(self.silver_labels, config.SILVER_LABELS_PATH)
        print(f"[SilverLabeler] Silver labels saved to {config.SILVER_LABELS_PATH}")

    def _load_or_generate_llm_data(self):
        """
        [Enrichment] LLM을 사용하여 클래스 정보를 확장합니다.
        - 배치 처리로 API 호출 최소화
        - 결과는 keywords_expanded.json에 저장
        """
        # 1. 파일이 있으면 로드 (비용 절약)
        if os.path.exists(config.EXPANDED_KEYWORDS_PATH):
            print(f"[SilverLabeler] Loading LLM data from {config.EXPANDED_KEYWORDS_PATH}")
            with open(config.EXPANDED_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)

        print("[SilverLabeler] Generating Data using LLM API (Keywords + Description)...")
        if OpenAI is None:
            return {}

        # API Key 확인
        api_key = os.getenv("OPENAI_API_KEY") # 혹은 직접 입력: "sk-..."
        if not api_key:
            print("[Error] OPENAI_API_KEY not found. Skipping LLM.")
            return {}
            
        client = OpenAI(api_key=api_key)
        
        generated_data = {}
        all_classes = list(self.taxonomy.id2name.items())
        
        # 배치 사이즈 20 (531 / 20 ≈ 27회 호출)
        BATCH_SIZE = 20
        
        for i in tqdm(range(0, len(all_classes), BATCH_SIZE), desc="LLM Querying"):
            batch = all_classes[i : i + BATCH_SIZE]
            
            # 프롬프트 구성: 키워드와 설명을 동시에 요청
            classes_str = "\n".join([f"ID {cid}: {cname}" for cid, cname in batch])
            prompt = (
                f"I am building a hierarchical text classifier for Amazon products.\n"
                f"For each category below, provide two things:\n"
                f"1. 'keywords': A list of 10 representative keywords or synonyms.\n"
                f"2. 'description': A detailed, single-paragraph description (approx 30-50 words) explaining what this category covers.\n\n"
                f"Output strictly in JSON format: {{ID: {{'keywords': [...], 'description': '...'}}}}.\n\n"
                f"Categories:\n{classes_str}"
            )

            try:
                self.api_call_count += 1 # 카운트 증가
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert taxonomist."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                batch_result = json.loads(content)
                
                # 결과 병합
                for str_cid, data in batch_result.items():
                    generated_data[str(str_cid)] = data
                    
            except Exception as e:
                print(f"[Warning] API call failed for batch {i}: {e}")
                continue

        # 결과 저장
        with open(config.EXPANDED_KEYWORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4)
            
        return generated_data

    def _calculate_similarity(self):
        """
        [Retrieval Step] 기본정보 + LLM 정보를 모두 합쳐서 임베딩 유사도 계산
        """
        print("[SilverLabeler] Step 1: Calculating Embedding Similarity...")
        
        class_texts = []
        for cid in range(config.NUM_CLASSES):
            cname = self.taxonomy.id2name[cid]
            raw_kwd = self.taxonomy.raw_keywords.get(cid, [])
            
            # LLM 데이터 가져오기
            llm_info = self.llm_data.get(str(cid), {})
            llm_kwd = llm_info.get("keywords", [])
            llm_desc = llm_info.get("description", "")
            
            # 키워드 합치기
            all_keywords = list(set(raw_kwd + llm_kwd))
            
            # 텍스트 구성: "이름: 키워드들. 설명문"
            # 이렇게 하면 SBERT가 문맥을 아주 잘 파악함
            text = f"{cname}: {', '.join(all_keywords)}. {llm_desc}"
            class_texts.append(text)

        # 임베딩 생성
        print("   - Encoding Classes (Enriched)...")
        class_embeddings = self.encoder.encode(class_texts, convert_to_tensor=True, show_progress_bar=False)
        
        print("   - Encoding Reviews...")
        review_texts = self.data_loader.data 
        doc_embeddings = self.encoder.encode(review_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64)

        # 코사인 유사도
        print("   - Computing Cosine Matrix...")
        similarity_matrix = util.cos_sim(doc_embeddings, class_embeddings)
        
        return similarity_matrix

    def _mine_core_classes_sota(self):
        """
        [Filtering Step] Top-K Retrieval + Hierarchy Check + Confidence
        """
        print("[SilverLabeler] Step 2: Mining Core Classes...")
        
        num_docs = self.similarity_matrix.shape[0]
        pids = self.data_loader.pids
        final_labels = {}
        
        TOP_K = 20
        MIN_SCORE_THRESHOLD = 0.3
        
        for i in tqdm(range(num_docs), desc="Mining"):
            doc_sims = self.similarity_matrix[i]
            
            # 1. Retrieval (Top-K)
            top_k_values, top_k_indices = torch.topk(doc_sims, k=TOP_K)
            top_k_indices = top_k_indices.tolist()
            
            # 2. Hierarchy Filter
            valid_candidates = []
            for cid in top_k_indices:
                if doc_sims[cid] < MIN_SCORE_THRESHOLD:
                    continue

                parents = self.taxonomy.get_parents(cid)
                is_connected = any(p in top_k_indices for p in parents)
                is_root = (len(parents) == 0)
                
                if is_connected or is_root:
                    valid_candidates.append(cid)
            
            if not valid_candidates:
                final_labels[pids[i]] = torch.zeros(config.NUM_CLASSES)
                continue

            # 3. Confidence Check
            core_classes = []
            for c in valid_candidates:
                if self._check_local_confidence(c, doc_sims):
                    core_classes.append(c)

            # 4. Label Expansion
            label_vec = torch.zeros(config.NUM_CLASSES)
            if core_classes:
                label_vec[core_classes] = 1.0
                for core in core_classes:
                    ancestors = self.taxonomy.get_ancestors(core)
                    label_vec[list(ancestors)] = 1.0
            
            final_labels[pids[i]] = label_vec

        self.silver_labels = final_labels
        print(f"[SilverLabeler] Finished. Labels generated for {len(final_labels)} docs.")

    def _check_local_confidence(self, class_id, doc_sims):
        """본인 점수 - Max(가족 점수) > Threshold 확인"""
        my_score = doc_sims[class_id].item()
        
        parents = self.taxonomy.get_parents(class_id)
        siblings = []
        for p in parents:
            sibs = self.taxonomy.get_children(p)
            siblings.extend([s for s in sibs if s != class_id])
            
        competitors = parents + siblings
        if not competitors:
            return True
            
        comp_scores = [doc_sims[comp].item() for comp in competitors]
        max_comp_score = max(comp_scores) if comp_scores else 0
        
        return (my_score - max_comp_score) > 0.05