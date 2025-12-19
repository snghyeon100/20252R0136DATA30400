# silver_label.py
import os
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from . import config


# ---------------------------
# BM25 (class texts as "docs", review as "query")
# ---------------------------
class BM25Indexer:
    """
    BM25 indexer over class_texts (documents).
    Each review is treated as a query; returns score over classes.
    """

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)

        self.docs_tokens: List[List[str]] = [self.tokenize(t) for t in docs]
        self.N = len(self.docs_tokens)

        self.doc_len = [len(toks) for toks in self.docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N > 0 else 0.0

        # df & postings
        self.df = defaultdict(int)  # term -> df
        self.postings = defaultdict(list)  # term -> list[(doc_idx, tf)]

        for di, toks in enumerate(self.docs_tokens):
            tf = Counter(toks)
            for term, freq in tf.items():
                self.postings[term].append((di, freq))
            for term in tf.keys():
                self.df[term] += 1

        # idf
        self.idf = {}
        for term, df in self.df.items():
            # Okapi idf (+1 안정화)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        간단 토크나이저(영문/숫자/한글 포함).
        """
        if not text:
            return []
        text = text.lower()
        return re.findall(r"[0-9a-z가-힣]+", text)

    def get_scores(self, query: str) -> List[float]:
        """
        query(str) -> scores(List[float]) length = #classes
        """
        if self.N == 0:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return [0.0] * self.N

        qtf = Counter(q_tokens)
        scores = [0.0] * self.N

        for term, qcnt in qtf.items():
            if term not in self.postings:
                continue
            idf = self.idf.get(term, 0.0)
            for di, tf in self.postings[term]:
                dl = self.doc_len[di]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-12)))
                gain = idf * (tf * (self.k1 + 1.0) / (denom + 1e-12))
                scores[di] += qcnt * gain

        return scores


class SilverLabeler:
    """
    SBERT + BM25 Hybrid Silver Labeler
    - Leaf Top-K -> Path Restore -> Best Path 1개 선택
    - sibling softmax 확률로 leaf 포함 여부 결정 (uniform-baseline + delta)

    * 기존 SBERT-only에서:
      doc_sims = alpha * sbert_sim + (1-alpha) * bm25_norm
      로 변경
    """

    def __init__(self, taxonomy, data_loader, device: str = config.DEVICE):
        self.taxonomy = taxonomy
        self.data_loader = data_loader
        self.device = device  # SBERT encoder device (cuda 가능)

        self.leaf_prob_delta = float(getattr(config, "LEAF_PROB_DELTA", 0.20))

        # SBERT bi-encoder
        model_name = getattr(config, "SBERT_MODEL_NAME", "all-mpnet-base-v2")
        print(f"[SilverLabeler] Loading SBERT Model ({model_name})...")
        self.encoder = SentenceTransformer(model_name, device=device)

        # data
        self.review_texts: List[str] = self.data_loader.data
        self.pids = self.data_loader.pids

        self.num_classes: int = getattr(config, "NUM_CLASSES", len(self.taxonomy.id2name))
        self.silver_labels: Dict[str, torch.Tensor] = {}

        # leaf nodes
        self.leaf_ids: List[int] = self._collect_leaf_nodes()
        # silver labeling은 CPU에서 처리(=BM25 결합/전송 오버헤드 최소화)
        self.tensor_device = "cpu"
        self.leaf_ids_tensor = torch.tensor(self.leaf_ids, device=self.tensor_device, dtype=torch.long)

        # tree vs DAG
        self._is_single_parent_graph = self._check_single_parent_graph()
        self._fixed_chain_cache: Optional[Dict[int, List[int]]] = None
        if self._is_single_parent_graph:
            self._fixed_chain_cache = self._precompute_fixed_leaf_chains()

        # SBERT-only similarity cache path
        base_sim_path = getattr(config, "SIMILARITY_MATRIX_PATH", "similarity_matrix.pt")
        root, ext = os.path.splitext(base_sim_path)
        self.sbert_sim_path = getattr(
            config,
            "SBERT_ONLY_SIM_MATRIX_PATH",
            f"{root}_sbert_only{ext or '.pt'}"
        )

        self.sbert_sim_matrix: Optional[torch.Tensor] = None

        # =========================
        # BM25 + Hybrid 설정
        # =========================
        self.use_bm25 = bool(getattr(config, "USE_BM25", True))
        self.hybrid_alpha = float(getattr(config, "HYBRID_ALPHA", 0.7))  # SBERT weight
        self.hybrid_alpha = min(max(self.hybrid_alpha, 0.0), 1.0)

        self.bm25_k1 = float(getattr(config, "BM25_K1", 1.5))
        self.bm25_b = float(getattr(config, "BM25_B", 0.75))
        self.bm25_norm = getattr(config, "BM25_NORM", "minmax")  # 현재 minmax만 구현

        self._bm25_indexer: Optional[BM25Indexer] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(self):
        """
        1) SBERT similarity matrix 로드/계산
        2) (선택) BM25 index 빌드
        3) Leaf Top-K -> best path 1개 선택
        4) uniform-baseline softmax sibling prob로 leaf 포함 여부 결정
        5) silver label 저장
        """
        self.sbert_sim_matrix = self._load_or_build_sbert_similarity()

        if self.use_bm25:
            print("[SilverLabeler] Building BM25 index over class texts...")
            class_texts = self._build_class_texts()
            self._bm25_indexer = BM25Indexer(class_texts, k1=self.bm25_k1, b=self.bm25_b)
            print("[SilverLabeler] BM25 index ready.")

        print("[SilverLabeler] Mining silver labels (Leaf Top-K -> best path -> softmax leaf-include)...")
        self._mine_labels(self.sbert_sim_matrix)

        torch.save(self.silver_labels, config.SILVER_LABELS_PATH)
        print(f"[SilverLabeler] Silver labels saved to {config.SILVER_LABELS_PATH}")

    # ------------------------------------------------------------------
    # Text builders
    # ------------------------------------------------------------------
    def _build_class_texts(self) -> List[str]:
        """
        클래스 텍스트 구성: class name + raw keywords
        """
        class_texts = []
        for cid in range(self.num_classes):
            cname = self.taxonomy.id2name[cid]
            raw_kwd = self.taxonomy.raw_keywords.get(cid, [])
            if raw_kwd:
                text = f"{cname}: {', '.join(raw_kwd)}"
            else:
                text = cname
            class_texts.append(text)
        return class_texts

    # ------------------------------------------------------------------
    # SBERT Similarity
    # ------------------------------------------------------------------
    def _load_or_build_sbert_similarity(self) -> torch.Tensor:
        if os.path.exists(self.sbert_sim_path):
            print(f"[SilverLabeler] Loading SBERT similarity matrix from {self.sbert_sim_path}")
            sim = torch.load(self.sbert_sim_path, map_location="cpu")
            return sim

        print("[SilverLabeler] Building SBERT similarity matrix (docs x classes)...")
        class_texts = self._build_class_texts()

        print("   - Encoding classes with SBERT...")
        class_emb = self.encoder.encode(
            class_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        print("   - Encoding reviews with SBERT...")
        doc_emb = self.encoder.encode(
            self.review_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=getattr(config, "SBERT_DOC_BATCH_SIZE", 64)
        )

        print("   - Computing cosine similarity...")
        sim = util.cos_sim(doc_emb, class_emb)  # [-1, 1]
        sim = (sim + 1.0) / 2.0                 # [0, 1]
        sim = sim.detach().cpu()

        torch.save(sim, self.sbert_sim_path)
        print(f"[SilverLabeler] SBERT similarity matrix saved to {self.sbert_sim_path}")
        return sim

    # ------------------------------------------------------------------
    # Hybrid: SBERT + BM25
    # ------------------------------------------------------------------
    def _bm25_to_01(self, scores: List[float]) -> torch.Tensor:
        """
        BM25 scores(List[float]) -> torch.Tensor([0,1]) shape=(C,)
        """
        if not scores:
            return torch.zeros(self.num_classes, dtype=torch.float32, device=self.tensor_device)

        x = torch.tensor(scores, dtype=torch.float32, device=self.tensor_device)

        # min-max normalization per doc
        if self.bm25_norm == "minmax":
            mn = torch.min(x)
            mx = torch.max(x)
            if (mx - mn) < 1e-12:
                return torch.zeros_like(x)
            return (x - mn) / (mx - mn + 1e-12)

        # fallback
        mn = torch.min(x)
        mx = torch.max(x)
        if (mx - mn) < 1e-12:
            return torch.zeros_like(x)
        return (x - mn) / (mx - mn + 1e-12)

    def _get_doc_sims_hybrid(self, doc_idx: int, sbert_row: torch.Tensor) -> torch.Tensor:
        """
        sbert_row: (C,) on CPU
        return: (C,) hybrid score on CPU
        """
        if (not self.use_bm25) or (self._bm25_indexer is None):
            return sbert_row

        review = self.review_texts[doc_idx] if doc_idx < len(self.review_texts) else ""
        bm25_scores = self._bm25_indexer.get_scores(review)  # length C
        bm25_01 = self._bm25_to_01(bm25_scores)

        a = self.hybrid_alpha
        return a * sbert_row + (1.0 - a) * bm25_01

    # ------------------------------------------------------------------
    # Leaf Top-K -> Best Path -> Decide include leaf via softmax prob
    # ------------------------------------------------------------------
    def _mine_labels(self, sbert_sim_matrix: torch.Tensor):
        # leaf 후보 top-k
        top_k_leaf = int(getattr(config, "SBERT_LEAF_TOP_K", 30))

        # path score 가중치 (best leaf path 고를 때 사용)
        w_root = float(getattr(config, "PATH_W_ROOT", 0.4))
        w_parent = float(getattr(config, "PATH_W_PARENT", 0.7))
        w_leaf = float(getattr(config, "PATH_W_LEAF", 1.0))

        # leaf 후보 자체 최소 유사도(너무 낮은 leaf는 후보에서도 제외)
        min_leaf_sim = float(getattr(config, "MIN_LEAF_SIM", -1.0))

        # leaf 포함 여부 결정: sibling softmax 확률 임계값 (인자 호환용)
        tau_prob = float(getattr(config, "LEAF_PROB_THRESHOLD", 0.5))
        temp = float(getattr(config, "LEAF_SOFTMAX_TEMPERATURE", 1.0))
        temp = max(temp, 1e-6)

        num_docs = sbert_sim_matrix.shape[0]
        for i in tqdm(range(num_docs), desc="SilverLabeling"):
            sbert_row = sbert_sim_matrix[i].to(self.tensor_device)  # (C,)
            doc_sims = self._get_doc_sims_hybrid(i, sbert_row)      # (C,) hybrid

            # 1) leaf 대상으로만 top-k 후보 뽑기
            leaf_sims = doc_sims.index_select(0, self.leaf_ids_tensor)  # (num_leaf,)
            k = min(top_k_leaf, leaf_sims.numel())
            top_vals, top_pos = torch.topk(leaf_sims, k=k)
            cand_leaf_ids = [self.leaf_ids[idx] for idx in top_pos.tolist()]

            # 2) 후보 leaf들 중 "경로 점수"가 가장 큰 leaf의 경로 1개 선택
            best_chain: List[int] = []
            best_leaf_id: Optional[int] = None
            best_chain_score: float = -1e9

            for leaf_id, leaf_score in zip(cand_leaf_ids, top_vals.tolist()):
                if leaf_score < min_leaf_sim:
                    continue

                chain = self._get_chain_root_to_leaf(leaf_id, doc_sims)
                chain_score = self._score_chain(chain, doc_sims, w_root, w_parent, w_leaf)

                if chain_score > best_chain_score:
                    best_chain_score = chain_score
                    best_chain = chain
                    best_leaf_id = leaf_id

            # 3) 라벨 벡터 생성
            label_vec = torch.zeros(self.num_classes, device="cpu")

            if not best_chain or best_leaf_id is None:
                self.silver_labels[self.pids[i]] = label_vec
                continue

            # chain에서 root/parent/leaf 추출
            if len(best_chain) == 1:
                root_id = best_chain[0]
                children = self.taxonomy.get_children(root_id) or []
                if children:
                    child_scores = [(c, float(doc_sims[c].item())) for c in children]
                    child_scores.sort(key=lambda x: x[1], reverse=True)
                    parent_id = child_scores[0][0]
                    chosen = [root_id, parent_id]
                else:
                    chosen = [root_id]
                label_vec[chosen] = 1.0
                self.silver_labels[self.pids[i]] = label_vec
                continue

            root_id = best_chain[0]
            parent_id = best_chain[-2]
            leaf_id = best_chain[-1]

            # 4) leaf 포함 여부 결정 (uniform-baseline softmax prob)
            include_leaf = self._include_leaf_by_softmax_prob(
                parent_id=parent_id,
                chosen_leaf_id=leaf_id,
                doc_sims=doc_sims,
                tau_prob=tau_prob,          # 호환용
                temperature=temp
            )

            chosen_nodes = [root_id, parent_id, leaf_id] if include_leaf else [root_id, parent_id]
            label_vec[chosen_nodes] = 1.0
            self.silver_labels[self.pids[i]] = label_vec

    def _include_leaf_by_softmax_prob(
        self,
        parent_id: int,
        chosen_leaf_id: int,
        doc_sims: torch.Tensor,
        tau_prob: float,      # <- 호환용(현재는 baseline+delta 사용)
        temperature: float
    ) -> bool:
        """
        uniform-baseline 정규화:
          p(leaf | siblings) >= 1/n + delta  이면 leaf 포함
        """
        children = self.taxonomy.get_children(parent_id) or []
        if not children:
            return True

        sibling_leaf_ids = [c for c in children if len(self.taxonomy.get_children(c) or []) == 0]
        if chosen_leaf_id not in sibling_leaf_ids:
            return True

        n = len(sibling_leaf_ids)
        if n <= 1:
            return True

        sib_tensor = torch.tensor(sibling_leaf_ids, device=doc_sims.device, dtype=torch.long)
        sib_scores = doc_sims.index_select(0, sib_tensor)  # (n,)
        probs = torch.softmax(sib_scores / temperature, dim=0)

        idx = sibling_leaf_ids.index(chosen_leaf_id)
        p = float(probs[idx].item())

        baseline = 1.0 / n
        thr = baseline + float(getattr(config, "LEAF_PROB_DELTA", 0.20))
        if thr > 0.999:
            thr = 0.999

        return p >= thr

    def _score_chain(
        self,
        chain: List[int],
        doc_sims: torch.Tensor,
        w_root: float,
        w_parent: float,
        w_leaf: float
    ) -> float:
        if not chain:
            return -1e9
        if len(chain) == 1:
            return float(doc_sims[chain[0]].item())
        if len(chain) == 2:
            r, l = chain[0], chain[1]
            return float((w_root * doc_sims[r] + w_leaf * doc_sims[l]).item())

        root_id = chain[0]
        parent_id = chain[-2]
        leaf_id = chain[-1]
        score = w_root * doc_sims[root_id] + w_parent * doc_sims[parent_id] + w_leaf * doc_sims[leaf_id]
        return float(score.item())

    # ------------------------------------------------------------------
    # Chain Restore
    # ------------------------------------------------------------------
    def _get_chain_root_to_leaf(self, leaf_id: int, doc_sims: torch.Tensor) -> List[int]:
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
                next_p = max(parents, key=lambda p: float(doc_sims[p].item()))

            if next_p in visited:
                break

            chain.append(next_p)
            visited.add(next_p)
            cur = next_p

        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # Taxonomy helpers
    # ------------------------------------------------------------------
    def _collect_leaf_nodes(self) -> List[int]:
        leaf_ids = []
        for cid in range(self.num_classes):
            children = self.taxonomy.get_children(cid)
            if children is None or len(children) == 0:
                leaf_ids.append(cid)

        if not leaf_ids:
            leaf_ids = list(range(self.num_classes))

        print(f"[SilverLabeler] #Leaf nodes = {len(leaf_ids)}")
        return leaf_ids

    def _check_single_parent_graph(self) -> bool:
        for cid in range(self.num_classes):
            parents = self.taxonomy.get_parents(cid) or []
            if len(parents) > 1:
                print("[SilverLabeler] Detected multi-parent nodes (DAG). Will choose best-parent per doc.")
                return False
        print("[SilverLabeler] Single-parent taxonomy detected. Using fixed path cache.")
        return True

    def _precompute_fixed_leaf_chains(self) -> Dict[int, List[int]]:
        cache: Dict[int, List[int]] = {}
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
