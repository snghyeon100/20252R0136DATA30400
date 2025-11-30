import os
import json
import torch
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("[Warning] 'google-generativeai' library not installed.")

class SilverLabeler:
    """
    [Phase 1] 가상 정답(Silver Label) 생성기 (Ultimate Edition)
    Pipeline: LLM Enrichment -> Hybrid Retrieval -> BGE Reranking -> Hierarchy Filtering
    """
    def __init__(self, taxonomy, data_loader, device=config.DEVICE):
        self.taxonomy = taxonomy
        self.data_loader = data_loader
        self.device = device
        self.api_call_count = 0
        
        # 1. Retrieval용 SBERT 모델
        print("[SilverLabeler] Loading Retriever (SBERT: all-mpnet-base-v2)...")
        self.retriever = SentenceTransformer('all-mpnet-base-v2', device=device)
        
        # 2. Reranking용 BGE 모델
        print("[SilverLabeler] Loading Reranker (BGE: bge-reranker-v2-m3)...")
        #self.rerank_model_name = 'BAAI/bge-reranker-v2-m3'
        self.rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.rerank_model_name)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(self.rerank_model_name).to(device)
        self.rerank_model.eval()
        
        self.similarity_matrix = None
        self.silver_labels = {} 
        self.llm_data = {}
        self.class_texts_for_rerank = [] # 리랭킹용 텍스트 저장소

    def run(self):
        # 1. LLM 데이터 생성
        self.llm_data = self._load_or_generate_llm_data()
        
        # [수정] Reranking에 쓸 클래스 텍스트는 언제나 미리 준비되어야 함
        self._prepare_class_texts()
        
        # 2. Hybrid 유사도 행렬 계산 (1차 Retrieval용)
        if os.path.exists(config.SIMILARITY_MATRIX_PATH):
            print(f"[SilverLabeler] Loading similarity matrix...")
            self.similarity_matrix = torch.load(config.SIMILARITY_MATRIX_PATH, map_location=self.device)
        else:
            self.similarity_matrix = self._calculate_hybrid_similarity()
            torch.save(self.similarity_matrix, config.SIMILARITY_MATRIX_PATH)

        # 3. Reranking & Mining (핵심 파트)
        self._mine_core_classes_ultimate()

        # 4. 결과 저장
        torch.save(self.silver_labels, config.SILVER_LABELS_PATH)
        print(f"[SilverLabeler] Silver labels saved to {config.SILVER_LABELS_PATH}")

    def _prepare_class_texts(self):
        """
        Reranking 및 Hybrid 계산에 사용할 클래스 텍스트를 미리 생성하여 저장
        """
        self.class_texts_for_rerank = []
        for cid in range(config.NUM_CLASSES):
            cname = self.taxonomy.id2name[cid]
            raw_kwd = self.taxonomy.raw_keywords.get(cid, [])
            
            # LLM 데이터가 로드된 상태여야 함
            llm_info = self.llm_data.get(str(cid), {})
            llm_kwd = llm_info.get("keywords", [])
            llm_desc = llm_info.get("description", "")
            
            all_keywords = list(set(raw_kwd + llm_kwd))
            text = f"{cname}: {', '.join(all_keywords)}. {llm_desc}"
            self.class_texts_for_rerank.append(text)

    def _load_or_generate_llm_data(self):
        # (기존과 동일)
        if os.path.exists(config.EXPANDED_KEYWORDS_PATH):
            print(f"[SilverLabeler] Loading existing LLM data...")
            with open(config.EXPANDED_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)

        api_key = os.getenv("GOOGLE_API_KEY") 
        if not HAS_GEMINI or not api_key:
            return self._generate_dummy_data()

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        generated_data = {}
        all_classes = list(self.taxonomy.id2name.items())
        BATCH_SIZE = 5
        
        for i in tqdm(range(0, len(all_classes), BATCH_SIZE), desc="Gemini Querying"):
            batch = all_classes[i : i + BATCH_SIZE]
            batch_context = []
            for cid, cname in batch:
                existing_kwds = self.taxonomy.raw_keywords.get(cid, [])
                existing_str = ", ".join(existing_kwds) if existing_kwds else "None"
                batch_context.append(f"ID {cid} Name: '{cname}' (Existing: {existing_str})")
            
            classes_str = "\n".join(batch_context)
            prompt = (
                f"I need to classify products. For each category below, provide:\n"
                f"1. 'keywords': List of 5 NEW keywords (synonyms/slang).\n"
                f"2. 'description': A one-sentence description.\n"
                f"Output ONLY valid JSON: {{ID: {{'keywords': [], 'description': ''}}}}.\n\n"
                f"{classes_str}"
            )

            try:
                self.api_call_count += 1
                response = model.generate_content(prompt)
                text = response.text.replace("```json", "").replace("```", "").strip()
                if text.startswith('"') and text.endswith('"'): text = text[1:-1].replace('\\"', '"')
                batch_result = json.loads(text)
                
                for str_cid, data in batch_result.items():
                    generated_data[str(str_cid)] = {
                        "keywords": data.get("keywords", []),
                        "description": data.get("description", "")
                    }
                time.sleep(8) 
            except Exception as e:
                print(f"[Warning] Gemini Error: {e}. Fallback to dummy.")
                for cid, cname in batch:
                    if str(cid) not in generated_data:
                        generated_data[str(cid)] = {"keywords": self.taxonomy.raw_keywords.get(cid, []), "description": f"Category {cname}."}

        with open(config.EXPANDED_KEYWORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4)
        return generated_data

    def _generate_dummy_data(self):
        dummy_data = {}
        for cid, cname in self.taxonomy.id2name.items():
            raw_kwds = self.taxonomy.raw_keywords.get(cid, [])
            dummy_data[str(cid)] = {"keywords": [], "description": f"Category {cname}."}
        with open(config.EXPANDED_KEYWORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=4)
        return dummy_data

    def _calculate_hybrid_similarity(self):
        """[Hybrid Retrieval] SBERT + Lexical"""
        print("[SilverLabeler] Step 1: Calculating Hybrid Similarity...")
        
        # 1. 텍스트 준비 (self.class_texts_for_rerank 사용)
        class_texts = self.class_texts_for_rerank
        review_texts = self.data_loader.data

        # 2. Semantic (SBERT)
        print("   - [Semantic] Encoding with SBERT...")
        class_emb = self.retriever.encode(class_texts, convert_to_tensor=True, show_progress_bar=False)
        doc_emb = self.retriever.encode(review_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
        semantic_sim = util.cos_sim(doc_emb, class_emb).cpu().numpy()

        # 3. Lexical (TF-IDF)
        print("   - [Lexical] Calculating TF-IDF...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        all_corpus = class_texts + review_texts[:5000]
        vectorizer.fit(all_corpus)
        tfidf_docs = vectorizer.transform(review_texts)
        tfidf_classes = vectorizer.transform(class_texts)
        lexical_sim = (tfidf_docs * tfidf_classes.T).toarray()
        lexical_sim = (lexical_sim - lexical_sim.min()) / (lexical_sim.max() - lexical_sim.min() + 1e-9)

        # 4. Combine
        print("   - [Hybrid] Combining Scores...")
        alpha = 0.3
        hybrid_sim = (1 - alpha) * semantic_sim + alpha * lexical_sim
        
        return torch.tensor(hybrid_sim, device=self.device)

    def _mine_core_classes_ultimate(self):
        """[Reranking & Filtering] (Batch Processing)"""
        print("[SilverLabeler] Step 2: Mining Core Classes (Rerank & Filter)...")
        
        num_docs = self.similarity_matrix.shape[0]
        pids = self.data_loader.pids
        review_texts = self.data_loader.data
        final_labels = {}
        
        RETRIEVAL_TOP_K = 50 # (속도 위해 25개로 조정)
        RERANK_TOP_K = 20    # (속도 위해 10개로 조정)
        MIN_SCORE_THRESHOLD = 0.01
        BATCH_SIZE = 128      # 배치 사이즈 설정 (메모리에 따라 조절)

        # 배치 단위로 처리
        for start_idx in tqdm(range(0, num_docs, BATCH_SIZE), desc="Reranking"):
            end_idx = min(start_idx + BATCH_SIZE, num_docs)
            batch_indices = range(start_idx, end_idx)
            
            # --- 1. Retrieval (Hybrid) - 배치 ---
            # (Batch, 531)
            batch_doc_sims = self.similarity_matrix[start_idx:end_idx] 
            
            # (Batch, K)
            batch_top_k_vals, batch_top_k_inds = torch.topk(batch_doc_sims, k=RETRIEVAL_TOP_K)
            
            # --- 2. Reranking (BGE) - 배치 ---
            # Reranker 입력 쌍 만들기 (모든 배치의 후보들을 한 리스트에 담음)
            all_pairs = []
            pair_map = [] # (batch_relative_idx, candidate_idx) 매핑 정보
            
            for i, doc_idx in enumerate(batch_indices):
                current_review = review_texts[doc_idx]
                candidate_indices = batch_top_k_inds[i].tolist()
                
                for cid in candidate_indices:
                    all_pairs.append([current_review, self.class_texts_for_rerank[cid]])
                    pair_map.append((i, cid)) # i번째 문서의 cid 후보
            
            # BGE 모델 추론 (한방에 처리)
            # 쌍이 너무 많으면(32 * 25 = 800개) 여기서도 미니 배치로 나눠야 함
            # 안전하게 128개씩 끊어서 처리
            rerank_scores_list = []
            MINI_BATCH = 1024
            
            with torch.no_grad():
                for j in range(0, len(all_pairs), MINI_BATCH):
                    mini_pairs = all_pairs[j : j + MINI_BATCH]
                    inputs = self.rerank_tokenizer(mini_pairs, padding=True, truncation=True, return_tensors='pt', max_length=256).to(self.device)
                    scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1).float()
                    rerank_scores_list.append(torch.sigmoid(scores))
            
            all_rerank_scores = torch.cat(rerank_scores_list) # (Batch * K, )
            
            # 점수 재배치 (각 문서별로 다시 묶기)
            # 문서별 점수 저장소 초기화
            doc_candidate_scores = [{} for _ in range(len(batch_indices))]
            
            for k, score in enumerate(all_rerank_scores):
                doc_rel_idx, cid = pair_map[k]
                doc_candidate_scores[doc_rel_idx][cid] = score.item()
            
            # --- 3. Filter & Confidence & Expansion (문서별 처리) ---
            for i, doc_idx in enumerate(batch_indices):
                scores_map = doc_candidate_scores[i]
                
                # 점수 높은 순 정렬
                sorted_candidates = sorted(scores_map.keys(), key=lambda x: scores_map[x], reverse=True)
                
                # [핵심 변경] 복잡한 필터 다 버리고, 무조건 상위 3개를 챙깁니다.
                # 단, 점수가 너무 터무니없이 낮은(0.001 미만) 건 제외
                best_candidates = []
                TARGET_COUNT = 1
                MIN_SCORE = 0.01
                for cid in sorted_candidates:
                    if len(best_candidates) >= TARGET_COUNT: 
                        break # 3개 차면 즉시 종료
                        
                    if scores_map[cid] > MIN_SCORE:         
                        best_candidates.append(cid)
                
                # 만약 2개도 안 모였으면, 점수 낮아도 그냥 상위 2개는 무조건 채웁니다. (강제 할당)
                if len(best_candidates) < 2:
                    best_candidates = sorted_candidates[:2]

                # Expansion (선택된 애들의 조상 노드도 정답으로 인정)
                label_vec = torch.zeros(config.NUM_CLASSES)
                if best_candidates:
                    label_vec[best_candidates] = 1.0
                    for core in best_candidates:
                        ancestors = self.taxonomy.get_ancestors(core)
                        label_vec[list(ancestors)] = 1.0
                
                final_labels[pids[doc_idx]] = label_vec

        self.silver_labels = final_labels
        print(f"[SilverLabeler] Finished. Force Top-2/3 Strategy applied.")

    def _check_local_confidence_reranked(self, class_id, scores_map):
        my_score = scores_map.get(class_id, 0.0)
        parents = self.taxonomy.get_parents(class_id)
        siblings = []
        for p in parents:
            sibs = self.taxonomy.get_children(p)
            siblings.extend([s for s in sibs if s != class_id])
        competitors = parents + siblings
        if not competitors: return True
        comp_scores = [scores_map.get(comp, 0.0) for comp in competitors]
        max_comp_score = max(comp_scores) if comp_scores else 0
        return (my_score - max_comp_score) > 0.03