#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_outputs.py

1) submission.csv 분석
   - 행 내부 중복 라벨 체크 (예: "3,3,35")
   - 라벨 개수 분포 (min/max/mean, count by k)
   - 라벨 빈도 top-K
   - 라벨 조합(combo) 빈도 top-K
   - (옵션) taxonomy edge 기반 path 제약 체크

2) silver_labels.pt 분석 (torch.save된 dict[pid] = multi-hot tensor)
   - 라벨 개수 분포 (min/max/mean, count by k)
   - 라벨 빈도 top-K
   - 라벨 조합(combo) 빈도 top-K
   - (옵션) taxonomy edge 기반 path 제약 체크

Usage 예시:
  # submission만
  python analyze_outputs.py --submission submission.csv

  # silver만
  python analyze_outputs.py --silver silver_labels.pt

  # 둘 다 + taxonomy path 체크
  python analyze_outputs.py --submission submission.csv --silver silver_labels.pt --taxonomy taxonomy.txt

taxonomy.txt 형식:
  각 줄에 "parent child" (공백/탭/콤마 구분 모두 허용)
"""

import argparse
import csv
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional

import torch


# -----------------------------
# Parsing: submission.csv
# -----------------------------
def parse_labels(label_str: str) -> List[int]:
    if label_str is None:
        return []
    s = str(label_str).strip().strip('"').strip("'")
    if not s or s.lower() in {"nan", "none"}:
        return []

    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    labels = []
    for p in parts:
        if not p:
            continue
        try:
            labels.append(int(p))
        except ValueError:
            continue
    return labels


def load_submission(path: str) -> Tuple[List[str], List[List[int]], List[List[int]]]:
    """
    return:
      pids: List[str]
      labels_raw: List[List[int]]  (중복 포함 원본)
      labels_uniq: List[List[int]] (행 내부 중복 제거 + 정렬)
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        lower_cols = {c.lower(): c for c in fieldnames}

        pid_col = None
        for cand in ["pid", "id"]:
            if cand in lower_cols:
                pid_col = lower_cols[cand]
                break

        labels_col = None
        for cand in ["labels", "label"]:
            if cand in lower_cols:
                labels_col = lower_cols[cand]
                break

        if labels_col is None:
            raise ValueError(f"[ERROR] label/labels 컬럼을 못 찾음. 컬럼들={fieldnames}")

        pids, labels_raw, labels_uniq = [], [], []
        for idx, row in enumerate(reader):
            pid = str(row[pid_col]) if pid_col is not None else str(idx)
            raw = parse_labels(row.get(labels_col, ""))
            uniq = sorted(set(raw))
            pids.append(pid)
            labels_raw.append(raw)
            labels_uniq.append(uniq)

    return pids, labels_raw, labels_uniq


# -----------------------------
# Loading: silver_labels.pt
# -----------------------------
def load_silver_labels(path: str) -> Tuple[List[str], List[List[int]]]:
    """
    silver_labels: torch.save(dict[pid] = multi-hot tensor(C))
    return:
      pids: List[str]
      labels_uniq: List[List[int]]  (각 pid의 1인 클래스 인덱스 리스트, 정렬)
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"[ERROR] silver file is not a dict. type={type(obj)}")

    pids = []
    labels_uniq = []

    for pid, vec in obj.items():
        if torch.is_tensor(vec):
            # vec: (C,) multi-hot (0/1 float)
            idx = torch.where(vec > 0.5)[0].tolist()
        else:
            # 혹시 list/np 등으로 저장된 경우 대비
            try:
                t = torch.tensor(vec)
                idx = torch.where(t > 0.5)[0].tolist()
            except Exception:
                idx = []

        idx = sorted(set(map(int, idx)))
        pids.append(str(pid))
        labels_uniq.append(idx)

    return pids, labels_uniq


# -----------------------------
# Taxonomy loading (edge list)
# -----------------------------
def load_taxonomy_edges(edge_path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Set[int]]:
    parents: Dict[int, List[int]] = defaultdict(list)
    children: Dict[int, List[int]] = defaultdict(list)
    nodes: Set[int] = set()

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()

            if len(parts) < 2:
                continue

            try:
                p = int(parts[0])
                c = int(parts[1])
            except ValueError:
                continue

            parents[c].append(p)
            children[p].append(c)
            nodes.add(p)
            nodes.add(c)

    for k in list(parents.keys()):
        parents[k] = sorted(set(parents[k]))
    for k in list(children.keys()):
        children[k] = sorted(set(children[k]))

    return parents, children, nodes


def build_ancestor_utils(parents: Dict[int, List[int]]):
    @lru_cache(maxsize=None)
    def ancestors(node: int) -> frozenset:
        res: Set[int] = set()
        for p in parents.get(node, []):
            res.add(p)
            res |= set(ancestors(p))
        return frozenset(res)

    def is_ancestor(a: int, b: int) -> bool:
        return (a == b) or (a in ancestors(b))

    @lru_cache(maxsize=None)
    def depth_min(node: int) -> int:
        ps = parents.get(node, [])
        if not ps:
            return 0
        return 1 + min(depth_min(p) for p in ps)

    def count_roots_in_set(labels: List[int]) -> int:
        return sum(1 for x in labels if len(parents.get(x, [])) == 0)

    return is_ancestor, depth_min, count_roots_in_set


def check_path_constraint(
    labels: List[int],
    is_ancestor_fn,
    depth_fn,
    strict_levels: bool = True
) -> Tuple[bool, str]:
    """
    - 2개: 둘 중 하나가 다른 하나의 조상이어야 함
      strict면 depth 차이=1 요구(보수적)
    - 3개: root->parent->leaf 한 체인
      strict면 depth가 연속(0,1,2 같은)이어야 함
    """
    n = len(labels)
    if n < 2 or n > 3:
        return False, f"label_count={n} (expected 2 or 3)"

    if n == 2:
        a, b = labels
        ok = is_ancestor_fn(a, b) or is_ancestor_fn(b, a)
        if not ok:
            return False, "2-label not ancestor-descendant"
        if strict_levels:
            da, db = depth_fn(a), depth_fn(b)
            if abs(da - db) != 1:
                return False, f"2-label ancestor ok, but depth_diff={abs(da-db)} != 1 (strict)"
        return True, "OK"

    # n == 3
    s = labels
    root_cands = [x for x in s if all(is_ancestor_fn(x, y) for y in s)]
    leaf_cands = [x for x in s if all(is_ancestor_fn(y, x) for y in s)]
    if not root_cands or not leaf_cands:
        return False, "3-label not a single chain (no root/leaf in set)"

    root = min(root_cands, key=lambda x: depth_fn(x))
    leaf = max(leaf_cands, key=lambda x: depth_fn(x))
    middle = [x for x in s if x != root and x != leaf]
    if len(middle) != 1:
        return False, "3-label chain ambiguous (no unique middle)"
    parent = middle[0]

    if not (is_ancestor_fn(root, parent) and is_ancestor_fn(parent, leaf)):
        return False, "3-label order fails: root !-> parent !-> leaf"

    if strict_levels:
        dr, dp, dl = depth_fn(root), depth_fn(parent), depth_fn(leaf)
        if not (dr + 1 == dp and dp + 1 == dl):
            return False, f"3-label ancestor ok, but depths={dr,dp,dl} not consecutive (strict)"

    return True, "OK"


# -----------------------------
# Common analysis
# -----------------------------
def summarize_labels(pids: List[str], labels_raw: Optional[List[List[int]]], labels_uniq: List[List[int]], name: str, topk: int):
    N = len(pids)
    k_list = [len(x) for x in labels_uniq]
    k_counter = Counter(k_list)

    print(f"\n==============================")
    print(f"[{name}] rows={N}")
    print(f"min/max/mean: {min(k_list) if k_list else None} {max(k_list) if k_list else None} {sum(k_list)/N if N else None:.6f}")
    print("count by k:", dict(sorted(k_counter.items())))

    # 내부 중복(행 내부) 체크: submission에만 의미 있음
    if labels_raw is not None:
        dup_rows = 0
        dup_tokens = 0
        examples = []
        for pid, raw, uniq in zip(pids, labels_raw, labels_uniq):
            d = len(raw) - len(uniq)
            if d > 0:
                dup_rows += 1
                dup_tokens += d
                if len(examples) < 10:
                    examples.append((pid, raw, uniq))
        print(f"[Row-Internal Duplicates] dup_rows={dup_rows} ({dup_rows/N:.2%}), total_dup_tokens={dup_tokens}")
        if examples:
            print("Examples (up to 10):")
            for pid, raw, uniq in examples:
                print(f"  pid={pid} raw={raw} -> uniq={uniq}")

    # 라벨 빈도 / combo 빈도
    label_freq = Counter()
    combo_freq = Counter()
    for uniq in labels_uniq:
        label_freq.update(uniq)
        combo_freq[tuple(uniq)] += 1

    print(f"\n== Label Frequency Top-{topk} ==")
    for lab, cnt in label_freq.most_common(topk):
        print(f"{lab}: {cnt}")

    print(f"\n== Combo Frequency Top-{topk} ==")
    for combo, cnt in combo_freq.most_common(topk):
        print(f"{combo}: {cnt}")

    topk_sum = sum(cnt for _, cnt in combo_freq.most_common(topk))
    print(f"\n[Combo Concentration] top{topk} combos cover {topk_sum}/{N} = {topk_sum/N:.2%} of rows")


def path_check(pids: List[str], labels_uniq: List[List[int]], taxonomy_path: str, strict_levels: bool, show_examples: int, name: str):
    parents, children, nodes = load_taxonomy_edges(taxonomy_path)
    is_ancestor_fn, depth_fn, count_roots_fn = build_ancestor_utils(parents)

    N = len(pids)
    invalid = 0
    invalid_ns = 0
    missing_root = 0
    multi_root = 0
    examples = []

    for pid, uniq in zip(pids, labels_uniq):
        rcount = count_roots_fn(uniq)
        if rcount == 0:
            missing_root += 1
        if rcount >= 2:
            multi_root += 1

        ok, reason = check_path_constraint(uniq, is_ancestor_fn, depth_fn, strict_levels=strict_levels)
        ok_ns, _ = check_path_constraint(uniq, is_ancestor_fn, depth_fn, strict_levels=False)

        if not ok:
            invalid += 1
            if len(examples) < show_examples:
                examples.append((pid, uniq, reason))
        if not ok_ns:
            invalid_ns += 1

    print(f"\n== Path Constraint Check: {name} ==")
    print(f"  taxonomy={taxonomy_path}")
    print(f"  strict_levels={strict_levels}")
    print(f"  invalid={invalid}/{N} = {invalid/N:.2%}")
    print(f"  invalid(non-strict reference)={invalid_ns}/{N} = {invalid_ns/N:.2%}")
    print(f"  missing_root_in_pred_set={missing_root}/{N} = {missing_root/N:.2%}")
    print(f"  multi_root_in_pred_set={multi_root}/{N} = {multi_root/N:.2%}")

    if examples:
        print(f"\nExamples (up to {show_examples}):")
        for pid, uniq, reason in examples:
            print(f"  pid={pid} labels={uniq} reason={reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, default=None, help="submission.csv 경로(옵션)")
    ap.add_argument("--silver", type=str, default=None, help="silver_labels.pt 경로(옵션)")
    ap.add_argument("--taxonomy", type=str, default=None, help="taxonomy edge 파일 경로(옵션)")
    ap.add_argument("--topk", type=int, default=20, help="top-k 출력")
    ap.add_argument("--show_examples", type=int, default=15, help="위반 예시 출력 개수")
    ap.add_argument("--strict_levels", action="store_true", help="depth 연속 레벨 엄격 체크")
    ap.add_argument("--no_strict_levels", action="store_true", help="strict 레벨 체크 끄기")
    args = ap.parse_args()

    strict = True
    if args.no_strict_levels:
        strict = False
    if args.strict_levels:
        strict = True

    did_any = False

    if args.submission:
        did_any = True
        pids, labels_raw, labels_uniq = load_submission(args.submission)
        summarize_labels(pids, labels_raw, labels_uniq, name=f"SUBMISSION({args.submission})", topk=args.topk)
        if args.taxonomy:
            path_check(pids, labels_uniq, args.taxonomy, strict_levels=strict, show_examples=args.show_examples, name="SUBMISSION")

    if args.silver:
        did_any = True
        spids, slabels_uniq = load_silver_labels(args.silver)
        summarize_labels(spids, labels_raw=None, labels_uniq=slabels_uniq, name=f"SILVER({args.silver})", topk=args.topk)
        if args.taxonomy:
            path_check(spids, slabels_uniq, args.taxonomy, strict_levels=strict, show_examples=args.show_examples, name="SILVER")

    if not did_any:
        print("[ERROR] --submission 또는 --silver 중 최소 하나는 넣어줘.")


if __name__ == "__main__":
    main()
