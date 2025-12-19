# src/trainer.py
import os
import json
import torch
import torch.nn as nn
import numpy as np
import re
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from . import config, utils

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False


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

    [LLM Selective 변경]
    - 기존: top-3 path 중 선택 + use_leaf 판단
    - 변경: "best path의 root/parent"는 고정으로 주고,
            "parent의 leaf children 중 1개 선택 or NONE"만 LLM이 결정
      => 출력: [{"pid":"...","leaf_choice":0..K}]
         (0이면 leaf 미포함)
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
        self.tau_prob = float(getattr(config, "LEAF_PROB_THRESHOLD", 0.5))
        self.temperature = float(getattr(config, "LEAF_SOFTMAX_TEMPERATURE", 1.0))
        self.temperature = max(self.temperature, 1e-6)

        # eval 디버그 프린트(너무 많이 찍히면 느려짐)
        self.eval_debug = bool(getattr(config, "EVAL_DEBUG", False))

        # =========================
        # LLM Selective Predict 설정
        # =========================
        self.use_llm_selective = bool(getattr(config, "USE_LLM_SELECTIVE", True))

        self.llm_max_calls = int(getattr(config, "LLM_MAX_CALLS", 1000))
        self.llm_batch_size = int(getattr(config, "LLM_BATCH_SIZE", 5))
        self.llm_model_name = getattr(config, "LLM_MODEL_NAME", "gemini-2.5-flash")
        self.llm_review_max_chars = int(getattr(config, "LLM_REVIEW_MAX_CHARS", 1200))
        self.llm_timeout_fallback_auto = True  # LLM 실패 시 auto로 fallback

        # 불확실도 결합 가중치(랭크 퍼센타일 기반)
        self.uncert_w_margin = float(getattr(config, "UNCERT_W_MARGIN", 1.0))
        self.uncert_w_leafgap = float(getattr(config, "UNCERT_W_LEAFGAP", 1.0))

        # 카운터
        self.llm_calls_used = 0
        self.llm_items_sent = 0
        self.llm_items_parsed = 0

        self._freeze_bert()
        self.leaf_prob_delta = float(getattr(config, "LEAF_PROB_DELTA", 0.20))
        # =========================
        # LLM Prompt Logging
        # =========================
        self.llm_log_enabled = bool(getattr(config, "LLM_LOG_ENABLED", True))
        self.llm_log_save_response = bool(getattr(config, "LLM_LOG_SAVE_RESPONSE", True))
        self.llm_log_save_parsed = bool(getattr(config, "LLM_LOG_SAVE_PARSED", True))

        base_dir = getattr(config, "LLM_LOG_DIR", "logs/llm_prompts")
        self.llm_log_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llm_log_dir = Path(base_dir) / self.llm_log_run_id

        if self.llm_log_enabled:
            self.llm_log_dir.mkdir(parents=True, exist_ok=True)
            self.llm_log_jsonl = self.llm_log_dir / "llm_calls.jsonl"
            print(f"[LLM LOG] enabled. dir={self.llm_log_dir}")
        else:
            self.llm_log_jsonl = None

        # =========================
        # Gemini init (선택) - _gemini_model 하나로 통일
        # =========================
        self._gemini_model = None
        if self.use_llm_selective:
            if not _HAS_GEMINI:
                print("[LLM] google-generativeai not installed -> LLM disabled.")
                self.use_llm_selective = False
            else:
                # ✅ 환경변수만 사용 (없으면 비활성화)
                api_key = os.getenv("GOOGLE_API_KEY", "").strip()
                if not api_key:
                    print("[LLM] GOOGLE_API_KEY not set -> LLM disabled.")
                    self.use_llm_selective = False
                else:
                    genai.configure(api_key=api_key)
                    self._gemini_model = genai.GenerativeModel(self.llm_model_name)

    def _freeze_bert(self):
        """BERT 파라미터를 전부 고정 + dropout까지 끄기 위해 eval 고정"""
        for p in self.model.bert.parameters():
            p.requires_grad = False
        self.model.bert.eval()

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Freeze BERT: trainable params = {trainable:,} / total params = {total:,}")
    def _llm_log_write_jsonl(self, obj: Dict[str, Any]):
        if not self.llm_log_enabled or self.llm_log_jsonl is None:
            return
        obj = dict(obj)
        obj["ts"] = datetime.datetime.now().isoformat()
        with open(self.llm_log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _llm_log_write_text(self, filename: str, text: str):
        if not self.llm_log_enabled:
            return
        path = self.llm_log_dir / filename
        path.write_text(text or "", encoding="utf-8")

    # ---------------------------
    # pid -> raw text 매핑 (dataset 수정 없이도 동작)
    # ---------------------------
    def _build_pid2text_from_loader(self, loader) -> Dict[str, str]:
        ds = getattr(loader, "dataset", None)
        if ds is None:
            return {}
        pids = getattr(ds, "pids", None)
        data = getattr(ds, "data", None)
        if pids is None or data is None:
            return {}
        if len(pids) != len(data):
            return {}
        return {str(pid): str(txt) for pid, txt in zip(pids, data)}

    # ---------------------------
    # 퍼센타일 랭크 (작을수록 애매 => -value로 넣어서 큰게 애매가 되게 만들기)
    # 반환: [0,1] (큰 값일수록 "상위")
    # ---------------------------
    def _percentile_rank(self, x: torch.Tensor) -> torch.Tensor:
        N = x.numel()
        if N <= 1:
            return torch.zeros_like(x)
        order = torch.argsort(x)  # 오름차순
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(N, device=x.device, dtype=torch.float)
        return ranks / float(N - 1)

    # ---------------------------
    # (핵심) 한 샘플 분석: top3 후보 + margin + leaf_gap + auto_nodes
    # ---------------------------
    def _analyze_one_sample(
        self,
        doc_probs: torch.Tensor,
        top_n_paths: int = 3,
    ) -> Dict[str, Any]:
        leaf_scores = doc_probs.index_select(0, self.leaf_ids_tensor)
        k = min(self.top_k_leaf, leaf_scores.numel())
        top_vals, top_pos = torch.topk(leaf_scores, k=k)
        cand_leaf_ids = [self.leaf_ids[idx] for idx in top_pos.tolist()]

        scored: List[Tuple[float, List[int]]] = []
        for leaf_id in cand_leaf_ids:
            chain = self._get_chain_root_to_leaf(leaf_id, doc_probs)
            score = self._score_chain(chain, doc_probs)
            scored.append((score, chain))

        if not scored:
            return {
                "top_paths": [],
                "best_chain": None,
                "best_score": -1e9,
                "second_score": -1e9,
                "margin": 0.0,
                "leaf_p": 1.0,
                "leaf_gap": 1.0,
                "auto_nodes": [],
            }

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_chain = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else (best_score - 999.0)
        margin = float(best_score - second_score)

        leaf_p, leaf_gap, auto_nodes = self._auto_nodes_and_leaf_stats(best_chain, doc_probs)
        top_paths = [chain for (_, chain) in scored[:top_n_paths]]

        return {
            "top_paths": top_paths,
            "best_chain": best_chain,
            "best_score": float(best_score),
            "second_score": float(second_score),
            "margin": float(margin),
            "leaf_p": float(leaf_p),
            "leaf_gap": float(leaf_gap),
            "auto_nodes": auto_nodes,
        }

    def _auto_nodes_and_leaf_stats(self, best_chain: List[int], doc_probs: torch.Tensor) -> Tuple[float, float, List[int]]:
        if not best_chain:
            return 1.0, 1.0, []

        if len(best_chain) == 1:
            root_id = best_chain[0]
            children = self.taxonomy.get_children(root_id) or []
            if children:
                child_scores = [(c, float(doc_probs[c].item())) for c in children]
                child_scores.sort(key=lambda x: x[1], reverse=True)
                parent_id = child_scores[0][0]
                return 1.0, 1.0, [root_id, parent_id]
            return 1.0, 1.0, [root_id]

        root_id = best_chain[0]
        parent_id = best_chain[-2]
        leaf_id = best_chain[-1]

        children = self.taxonomy.get_children(parent_id) or []
        sibling_leaf_ids = [c for c in children if len(self.taxonomy.get_children(c) or []) == 0]

        p = 1.0
        if (leaf_id in sibling_leaf_ids) and (len(sibling_leaf_ids) > 1):
            sib_tensor = torch.tensor(sibling_leaf_ids, device=doc_probs.device, dtype=torch.long)
            sib_scores = doc_probs.index_select(0, sib_tensor)
            probs_sib = torch.softmax(sib_scores / self.temperature, dim=0)
            idx = sibling_leaf_ids.index(leaf_id)
            p = float(probs_sib[idx].item())

        leaf_gap = abs(p - float(self.tau_prob))
        include_leaf = (p >= float(self.tau_prob))

        if include_leaf and (leaf_id != parent_id):
            return p, leaf_gap, [root_id, parent_id, leaf_id]
        return p, leaf_gap, [root_id, parent_id]

    def _chain_to_str(self, chain: List[int]) -> str:
        return " > ".join([self.taxonomy.id2name[cid] for cid in chain])

    # ---------------------------
    # ✅ (변경) parent의 leaf children 후보 수집
    # ---------------------------
    def _get_leaf_children_under_parent(self, parent_id: int) -> List[int]:
        children = self.taxonomy.get_children(parent_id) or []
        if not children:
            return []
        leafs = [c for c in children if len(self.taxonomy.get_children(c) or []) == 0]
        return leafs if leafs else children

    # ---------------------------
    # ✅ (변경) LLM item 구성: root/parent 고정 + leaf 후보들(이름만) + NONE
    # ---------------------------
    def _build_llm_prompt_items(self, pid: str, review: str, root_id: int, parent_id: int, leaf_ids: List[int]) -> Dict[str, Any]:
        review = (review or "")[: self.llm_review_max_chars]

        leaf_names = []
        for lid in leaf_ids:
            leaf_names.append(str(self.taxonomy.id2name.get(lid, str(lid))))

        item = {
            "pid": pid,
            "review": review,
            "root_id": int(root_id),
            "parent_id": int(parent_id),
            "root_name": str(self.taxonomy.id2name.get(root_id, str(root_id))),
            "parent_name": str(self.taxonomy.id2name.get(parent_id, str(parent_id))),
            "leaf_ids": [int(x) for x in leaf_ids],     # 매핑용(LLM에는 id 안 줌)
            "leaf_names": leaf_names,                  # LLM 입력용(이름만)
        }
        return item

    # ---------------------------
    # ✅ (변경) LLM prompt 구성 (배치): leaf_choice만 반환
    # ---------------------------
    def _build_llm_prompt_from_batch(self, batch_payload: List[Dict[str, Any]]) -> str:
        lines = []
        lines.append("You are a product taxonomy leaf selector.")
        lines.append("Given a review and a fixed root/parent category, choose ONE leaf under the parent, or NONE.")
        lines.append("Return ONLY a JSON array. No extra text, no markdown.")
        lines.append("")
        lines.append('Output format: [{"pid":"...","leaf_choice":0}]')
        lines.append("")
        lines.append("Rules:")
        lines.append("- leaf_choice is an integer.")
        lines.append("- 0 means NONE (do NOT include a leaf).")
        lines.append("- 1..K selects the corresponding leaf candidate.")
        lines.append("- Do not output anything except the JSON array.")
        lines.append("")
        lines.append("Items:")

        for it in batch_payload:
            pid = it.get("pid", "")
            review = it.get("review", "") or ""
            root_name = it.get("root_name", "")
            parent_name = it.get("parent_name", "")
            leaf_names = it.get("leaf_names", []) or []

            lines.append(f"pid: {pid}")
            lines.append(f"root: {root_name}")
            lines.append(f"parent: {parent_name}")
            lines.append("review:")
            lines.append(review)
            lines.append("leaf_candidates:")
            lines.append("0) NONE")
            for j, nm in enumerate(leaf_names, start=1):
                lines.append(f"{j}) {nm}")
            lines.append("")

        return "\n".join(lines)

    # ---------------------------
    # ✅ (변경) LLM 응답 파싱: leaf_choice
    # ---------------------------
    def _parse_llm_output(self, raw_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        기대 포맷:
          [{"pid":"...","leaf_choice":0}, ...]
        (호환) choice 키가 오면 leaf_choice로 간주
        """
        if not raw_text:
            return None

        m = re.search(r"\[[\s\S]*\]", raw_text)
        if not m:
            return None

        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except Exception:
            return None

        if not isinstance(obj, list):
            return None

        out = []
        for x in obj:
            if not isinstance(x, dict):
                continue
            if "pid" not in x:
                continue
            if ("leaf_choice" not in x) and ("choice" not in x):
                continue
            out.append(x)

        return out if out else None

    # ---------------------------
    # LLM 호출 (배치 1콜)
    # ---------------------------
    def _llm_choose_batch(self, batch_payload, debug=False):
        if self._gemini_model is None:
            if debug:
                print("[LLM DEBUG] _gemini_model is None -> fallback to auto")
            return None

        # ✅ 이번 호출 번호(파일명용)
        call_id = self.llm_calls_used + 1

        prompt = self._build_llm_prompt_from_batch(batch_payload)
        prompt_chars = len(prompt)

        # --- (A) prompt 저장 ---
        if self.llm_log_enabled:
            pids = [str(x.get("pid", "")) for x in batch_payload]
            self._llm_log_write_text(f"call_{call_id:06d}_prompt.txt", prompt)
            self._llm_log_write_jsonl({
                "call_id": call_id,
                "event": "prompt_saved",
                "prompt_chars": prompt_chars,
                "batch_size": len(batch_payload),
                "pids": pids,
            })

        if debug:
            print("\n" + "="*60)
            print(f"[LLM DEBUG] prompt_chars={prompt_chars} | batch_items={len(batch_payload)}")
            print("[LLM DEBUG] pids:", [x.get("pid") for x in batch_payload])
            print("="*60)

        # ✅ 실제 호출 시도할 때만 카운트 증가
        self.llm_calls_used += 1

        try:
            resp = self._gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
            )
            raw_text = getattr(resp, "text", "") or ""

            # --- (B) response 저장 ---
            if self.llm_log_enabled and self.llm_log_save_response:
                self._llm_log_write_text(f"call_{call_id:06d}_response.txt", raw_text)
                self._llm_log_write_jsonl({
                    "call_id": call_id,
                    "event": "response_saved",
                    "response_chars": len(raw_text),
                })

        except Exception as e:
            if debug:
                print(f"[LLM DEBUG] exception={e}")

            # --- (C) 에러 로그 ---
            if self.llm_log_enabled:
                self._llm_log_write_jsonl({
                    "call_id": call_id,
                    "event": "exception",
                    "error": repr(e),
                })
            return None

        out = self._parse_llm_output(raw_text)
        if out is None:
            if debug:
                print("[LLM DEBUG] raw_response_head:")
                print(raw_text[:800])
                print("[LLM DEBUG] parse failed")

            if self.llm_log_enabled:
                self._llm_log_write_jsonl({
                    "call_id": call_id,
                    "event": "parse_failed",
                })
            return None

        # --- (D) parsed 저장 ---
        if self.llm_log_enabled and self.llm_log_save_parsed:
            self._llm_log_write_text(
                f"call_{call_id:06d}_parsed.json",
                json.dumps(out, ensure_ascii=False, indent=2)
            )
            self._llm_log_write_jsonl({
                "call_id": call_id,
                "event": "parse_ok",
                "parsed_items": len(out),
            })

        if debug:
            print("[LLM DEBUG] raw_response_head:")
            print(raw_text[:800])
            print("[LLM DEBUG] parsed_first2:")
            print(out[:2])
            print("="*60 + "\n")

        return out


    # ---------------------------
    # Class feature 준비
    # ---------------------------
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
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                out = encoder(**encoded)
                token_embeddings = out.last_hidden_state
                input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                features.append(emb)

        return torch.cat(features, dim=0).detach()

    def _get_optimizer(self):
        lr_base = config.LR_BASE_P1
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters! (All parameters are frozen)")
        optimizer = AdamW(trainable_params, lr=lr_base, weight_decay=0.01)
        return optimizer

    def train(self):
        print(f"[Trainer] Start Supervised Training for {config.NUM_EPOCHS} epochs.")
        self.optimizer = self._get_optimizer()

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.evaluate()

            print(
                f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
            )

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

            loss = self._compute_taxonomy_aware_loss(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
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
    # Path Decoding
    # ---------------------------
    def _decode_path(self, probs: torch.Tensor) -> torch.Tensor:
        B, C = probs.shape
        preds = torch.zeros_like(probs)

        for i in range(B):
            doc_probs = probs[i]

            leaf_scores = doc_probs.index_select(0, self.leaf_ids_tensor)
            k = min(self.top_k_leaf, leaf_scores.numel())
            top_vals, top_pos = torch.topk(leaf_scores, k=k)
            cand_leaf_ids = [self.leaf_ids[idx] for idx in top_pos.tolist()]

            best_chain = None
            best_chain_score = -1e9

            for leaf_id in cand_leaf_ids:
                chain = self._get_chain_root_to_leaf(leaf_id, doc_probs)
                chain_score = self._score_chain(chain, doc_probs)
                if chain_score > best_chain_score:
                    best_chain_score = chain_score
                    best_chain = chain

            if not best_chain:
                continue

            if len(best_chain) == 1:
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
        children = self.taxonomy.get_children(parent_id) or []
        if not children:
            return True

        sibling_leaf_ids = [c for c in children if len(self.taxonomy.get_children(c) or []) == 0]
        if chosen_leaf_id not in sibling_leaf_ids:
            return True

        n = len(sibling_leaf_ids)
        if n <= 1:
            return True

        sib_tensor = torch.tensor(sibling_leaf_ids, device=doc_probs.device, dtype=torch.long)
        sib_scores = doc_probs.index_select(0, sib_tensor)
        probs_sib = torch.softmax(sib_scores / self.temperature, dim=0)

        idx = sibling_leaf_ids.index(chosen_leaf_id)
        p = float(probs_sib[idx].item())

        baseline = 1.0 / n
        thr = baseline + self.leaf_prob_delta
        if thr > 0.999:
            thr = 0.999

        return p >= thr

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
    # Loss
    # ---------------------------
    def _compute_taxonomy_aware_loss(self, logits, silver_labels):
        bce_loss = self.bce_loss(logits, silver_labels)
        mask = torch.ones_like(silver_labels, device=self.device)

        for i in range(silver_labels.shape[0]):
            positives = torch.where(silver_labels[i] == 1)[0].tolist()
            for c in positives:
                children = self.taxonomy.get_children(c) or []
                if not children:
                    continue

                children_tensor = torch.tensor(children, device=self.device)
                child_is_positive = silver_labels[i].index_select(0, children_tensor)
                to_mask = children_tensor[child_is_positive == 0]

                if to_mask.numel() > 0:
                    mask[i].index_fill_(0, to_mask, 0.0)

        masked_loss_sum = (bce_loss * mask).sum()
        valid_elements_count = mask.sum().clamp_min(1.0)
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
    # ✅ (변경) predict: 애매샘플만 LLM, leaf 선택(or NONE)
    # ---------------------------
    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        # reset counters
        self.llm_calls_used = 0
        self.llm_items_sent = 0
        self.llm_items_parsed = 0

        pid2text = self._build_pid2text_from_loader(loader)

        pids_all: List[str] = []
        auto_nodes_all: List[List[int]] = []
        best_chain_all: List[Optional[List[int]]] = []
        margins: List[float] = []
        leaf_gaps: List[float] = []

        # 1) 전체 샘플 best_chain + margin/leaf_gap 수집
        for batch in tqdm(loader, desc="Predicting(collect ambiguity)"):
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            pids = [str(x) for x in batch["pid"]]

            logits, _ = self.model(input_ids, mask, self.class_features, self.adj_matrix)
            probs = torch.sigmoid(logits)

            B = probs.size(0)
            for i in range(B):
                pid = pids[i]
                doc_probs = probs[i]
                info = self._analyze_one_sample(doc_probs, top_n_paths=3)

                pids_all.append(pid)
                auto_nodes_all.append(info["auto_nodes"])
                best_chain_all.append(info["best_chain"])
                margins.append(info["margin"])
                leaf_gaps.append(info["leaf_gap"])

        N = len(pids_all)
        if N == 0:
            return [], []

        # 2) 불확실도 계산(퍼센타일 랭크)
        device = self.device if torch.cuda.is_available() else "cpu"
        m_t = torch.tensor(margins, dtype=torch.float, device=device)
        g_t = torch.tensor(leaf_gaps, dtype=torch.float, device=device)

        u_margin = self._percentile_rank(-m_t)
        u_leaf = self._percentile_rank(-g_t)
        uncert = self.uncert_w_margin * u_margin + self.uncert_w_leafgap * u_leaf

        # 3) 예산에 맞는 top-N 애매 샘플 선택
        budget_items = self.llm_max_calls * max(self.llm_batch_size, 1)
        n_select = min(int(budget_items), N)
        if not self.use_llm_selective:
            n_select = 0

        if n_select > 0:
            _, sel_idx = torch.topk(uncert, k=n_select, largest=True)
            sel_idx = sel_idx.tolist()
        else:
            sel_idx = []

        # 4) 기본(auto) 예측을 최종 공간에 복사
        final_nodes_all = [nodes[:] for nodes in auto_nodes_all]

        # 5) LLM 호출(선택된 샘플만): root/parent 고정 + leaf 선택(or NONE)
        debug_printed = False
        if n_select > 0:
            llm_items: List[Tuple[int, Dict[str, Any]]] = []

            for idx in sel_idx:
                pid = pids_all[idx]
                review = pid2text.get(pid, "")

                chain = best_chain_all[idx]
                if not chain or len(chain) == 0:
                    continue

                # root / parent 결정
                if len(chain) == 1:
                    root_id = int(chain[0])
                    root_children = self.taxonomy.get_children(root_id) or []
                    if not root_children:
                        continue
                    # parent는 root의 child 중 확률 가장 높은 걸로 (doc_probs가 여기 없음 -> best_chain이 root만인 케이스는 거의 없음)
                    # 안전하게: 첫 child를 parent로 둔다
                    parent_id = int(root_children[0])
                else:
                    root_id = int(chain[0])
                    parent_id = int(chain[-2])

                leaf_ids = self._get_leaf_children_under_parent(parent_id)
                if not leaf_ids:
                    continue

                item = self._build_llm_prompt_items(pid, review, root_id, parent_id, leaf_ids)
                llm_items.append((idx, item))

            bs = max(self.llm_batch_size, 1)
            for s in tqdm(range(0, len(llm_items), bs), desc="LLM selective batches(leaf pick)"):
                chunk = llm_items[s:s+bs]
                batch_payload = [it for (_, it) in chunk]
                self.llm_items_sent += len(batch_payload)

                debug = (not debug_printed)  # 첫 배치만
                out = self._llm_choose_batch(batch_payload, debug=debug)
                if debug:
                    debug_printed = True

                if out is None:
                    continue

                # pid -> leaf_choice
                pred_map: Dict[str, int] = {}
                for obj in out:
                    if not isinstance(obj, dict):
                        continue
                    pid = str(obj.get("pid", "")).strip()
                    choice = obj.get("leaf_choice", obj.get("choice", None))
                    if not pid or choice is None:
                        continue
                    try:
                        choice = int(choice)
                    except Exception:
                        continue
                    pred_map[pid] = choice

                # 적용
                for idx, item in chunk:
                    pid = item["pid"]
                    if pid not in pred_map:
                        continue

                    leaf_choice = pred_map[pid]
                    root_id = int(item["root_id"])
                    parent_id = int(item["parent_id"])
                    leaf_ids = item.get("leaf_ids", []) or []

                    base_nodes = [root_id, parent_id]

                    if leaf_choice == 0:
                        final_nodes_all[idx] = base_nodes
                        self.llm_items_parsed += 1
                        continue

                    if 1 <= leaf_choice <= len(leaf_ids):
                        chosen_leaf_id = int(leaf_ids[leaf_choice - 1])
                        final_nodes_all[idx] = base_nodes + [chosen_leaf_id]
                        self.llm_items_parsed += 1
                        continue

                    # 범위 밖이면 무시 (auto 유지)

        # 6) summary
        calls_budget = self.llm_max_calls
        print("\n==============================")
        print("[LLM Selective Predict Summary]")
        print(f"total_samples={N}")
        print(f"use_llm_selective={self.use_llm_selective}")
        print(f"llm_batch_size={self.llm_batch_size}")
        print(f"budget_calls={calls_budget}  => budget_items={calls_budget * max(self.llm_batch_size,1)}")
        print(f"selected_items={n_select} ({(n_select/N)*100:.2f}%)")
        print(f"calls_used={self.llm_calls_used}")
        print(f"items_sent={self.llm_items_sent}")
        print(f"items_parsed={self.llm_items_parsed}")
        print("==============================\n")

        # 7) 제출 형식: (pids, list_of_label_ids)
        all_pids = pids_all
        all_preds = []
        for nodes in final_nodes_all:
            uniq = sorted(set(map(int, nodes)))
            all_preds.append(uniq)

        return all_pids, all_preds
