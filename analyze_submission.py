#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_submission.py

- submission.csv(또는 submissin.csv)에서 라벨 문자열을 파싱
- (A) 행 내부 중복 라벨 체크: 같은 라벨이 한 행에 여러 번 등장하는지
- (B) 전체 조합(라벨 set)이 지나치게 중복되는지: top-20 조합 빈도
- (C) taxonomy(부모-자식 edge 파일) 기반 path 제약 체크:
    * 라벨 2개: 둘 중 하나가 다른 하나의 조상(ancestor)이어야 함
    * 라벨 3개: root->parent->leaf 형태로 한 체인이어야 함
    * (옵션) strict level: depth 차이가 (0,1,2)로 딱 맞는지

Usage:
  python analyze_submission.py --submission submission.csv --taxonomy taxonomy.txt
  python analyze_submission.py --submission submissin.csv --taxonomy taxonomy.txt --write_dedup deduped.csv

taxonomy.txt 형식:
  각 줄에 "parent child" (공백/탭/콤마 구분 모두 허용), 예:
    0 10
    10 65
"""

import argparse
import csv
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional


# -----------------------------
# IO / Parsing
# -----------------------------
def parse_labels(label_str: str) -> List[int]:
    if label_str is None:
        return []
    s = str(label_str).strip().strip('"').strip("'")
    if not s or s.lower() in {"nan", "none"}:
        return []

    # 쉼표/공백 혼용 대응
    # ex) "10,65" / "10, 65" / "10 65"
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    labels = []
    for p in parts:
        if p == "":
            continue
        try:
            labels.append(int(p))
        except ValueError:
            # 혹시 이상 토큰 있으면 무시
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

        # pid 컬럼 탐색
        pid_col = None
        for cand in ["pid", "id"]:
            if cand in lower_cols:
                pid_col = lower_cols[cand]
                break

        # labels 컬럼 탐색
        labels_col = None
        for cand in ["labels", "label"]:
            if cand in lower_cols:
                labels_col = lower_cols[cand]
                break

        if labels_col is None:
            raise ValueError(
                f"[ERROR] labels 컬럼을 못 찾음. 컬럼들={fieldnames} "
                f"(labels/label 중 하나가 있어야 함)"
            )

        pids: List[str] = []
        labels_raw: List[List[int]] = []
        labels_uniq: List[List[int]] = []

        for idx, row in enumerate(reader):
            pid = str(row[pid_col]) if pid_col is not None else str(idx)
            lab = parse_labels(row.get(labels_col, ""))

            # uniq(순서-중복 제거는 “정렬된 set”으로 통일)
            uniq = sorted(set(lab))

            pids.append(pid)
            labels_raw.append(lab)
            labels_uniq.append(uniq)

    return pids, labels_raw, labels_uniq


# -----------------------------
# Taxonomy loading (edge list)
# -----------------------------
def load_taxonomy_edges(edge_path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Set[int]]:
    """
    edge_path: 각 줄에 parent child
    구분자: 공백/탭/콤마 모두 허용
    """
    parents: Dict[int, List[int]] = defaultdict(list)
    children: Dict[int, List[int]] = defaultdict(list)
    nodes: Set[int] = set()

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 콤마/공백 혼용 처리
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

    # 중복 제거
    for k in list(parents.keys()):
        parents[k] = sorted(set(parents[k]))
    for k in list(children.keys()):
        children[k] = sorted(set(children[k]))

    return parents, children, nodes


# -----------------------------
# Path constraint checks
# -----------------------------
def build_ancestor_utils(
    parents: Dict[int, List[int]]
):
    @lru_cache(maxsize=None)
    def ancestors(node: int) -> frozenset:
        res: Set[int] = set()
        for p in parents.get(node, []):
            res.add(p)
            res |= set(ancestors(p))
        return frozenset(res)

    def is_ancestor(a: int, b: int) -> bool:
        # a가 b의 조상(또는 동일)인가?
        return (a == b) or (a in ancestors(b))

    @lru_cache(maxsize=None)
    def depth_min(node: int) -> int:
        ps = parents.get(node, [])
        if not ps:
            return 0
        return 1 + min(depth_min(p) for p in ps)

    def has_root(labels: List[int]) -> bool:
        # parents가 없는 노드(루트)가 labels 안에 존재?
        for x in labels:
            if len(parents.get(x, [])) == 0:
                return True
        return False

    def count_roots_in_set(labels: List[int]) -> int:
        return sum(1 for x in labels if len(parents.get(x, [])) == 0)

    return ancestors, is_ancestor, depth_min, has_root, count_roots_in_set


def check_path_constraint(
    labels: List[int],
    is_ancestor_fn,
    depth_fn,
    strict_levels: bool = True
) -> Tuple[bool, str]:
    """
    labels: uniq+sorted 된 라벨 리스트(길이 0~3)
    return: (valid?, reason)
    """
    n = len(labels)
    if n < 2 or n > 3:
        return False, f"label_count={n} (expected 2 or 3)"

    if n == 2:
        a, b = labels
        if is_ancestor_fn(a, b) or is_ancestor_fn(b, a):
            if strict_levels:
                da = depth_fn(a)
                db = depth_fn(b)
                # 2개면 보통 (root,parent) or (parent,leaf)라서 depth 차이=1이면 가장 자연스러움
                if abs(da - db) != 1:
                    return False, f"2-label ancestor ok, but depth_diff={abs(da-db)} != 1 (strict)"
            return True, "OK"
        return False, "2-label not ancestor-descendant"

    # n == 3
    s = labels

    # root 후보: 다른 둘의 조상인 노드
    root_cands = [x for x in s if all(is_ancestor_fn(x, y) for y in s)]
    # leaf 후보: 다른 둘의 자손인 노드
    leaf_cands = [x for x in s if all(is_ancestor_fn(y, x) for y in s)]

    if not root_cands or not leaf_cands:
        return False, "3-label not a single chain (no common root/leaf within set)"

    # 여러 후보면 depth로 가장 위/아래를 선택
    root = min(root_cands, key=lambda x: depth_fn(x))
    leaf = max(leaf_cands, key=lambda x: depth_fn(x))

    middle = [x for x in s if x != root and x != leaf]
    if len(middle) != 1:
        return False, "3-label chain ambiguous (cannot identify unique middle)"
    parent = middle[0]

    if not (is_ancestor_fn(root, parent) and is_ancestor_fn(parent, leaf)):
        return False, "3-label order fails: root !-> parent !-> leaf"

    if strict_levels:
        dr, dp, dl = depth_fn(root), depth_fn(parent), depth_fn(leaf)
        if not (dr + 1 == dp and dp + 1 == dl):
            return False, f"3-label ancestor ok, but depths={dr,dp,dl} not consecutive (strict)"

    return True, "OK"


# -----------------------------
# Reporting
# -----------------------------
def print_top(counter: Counter, title: str, k: int = 20):
    print(f"\n== {title} (top {k}) ==")
    for item, cnt in counter.most_common(k):
        print(f"{item}: {cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, required=True, help="submission.csv 경로")
    ap.add_argument("--taxonomy", type=str, default=None, help="taxonomy edge 파일 경로(없으면 path 체크 스킵)")
    ap.add_argument("--topk", type=int, default=20, help="top-k 출력 개수")
    ap.add_argument("--show_examples", type=int, default=15, help="위반 예시 출력 개수")
    ap.add_argument("--strict_levels", action="store_true", help="depth가 (연속 레벨)인지 엄격 체크")
    ap.add_argument("--no_strict_levels", action="store_true", help="strict 레벨 체크 끄기")
    ap.add_argument("--write_dedup", type=str, default=None, help="행 내부 중복 제거한 csv 저장 경로(optional)")
    args = ap.parse_args()

    strict = True
    if args.no_strict_levels:
        strict = False
    if args.strict_levels:
        strict = True

    pids, labels_raw, labels_uniq = load_submission(args.submission)
    N = len(pids)

    # -------------------------
    # (A) 행 내부 중복 체크
    # -------------------------
    internal_dup_rows = 0
    internal_dup_amount = 0
    internal_dup_examples = []

    raw_len_dist = Counter()
    uniq_len_dist = Counter()

    for pid, raw, uniq in zip(pids, labels_raw, labels_uniq):
        raw_len_dist[len(raw)] += 1
        uniq_len_dist[len(uniq)] += 1

        dup = len(raw) - len(uniq)
        if dup > 0:
            internal_dup_rows += 1
            internal_dup_amount += dup
            if len(internal_dup_examples) < args.show_examples:
                internal_dup_examples.append((pid, raw, uniq))

    print(f"\n[Submission] rows={N}")
    print(f"[Row-Internal Duplicates] dup_rows={internal_dup_rows} ({internal_dup_rows/N:.2%}), "
          f"total_dup_tokens={internal_dup_amount}")

    print("\n== Raw label count distribution (len(raw)) ==")
    for k in sorted(raw_len_dist.keys()):
        print(f"  {k}: {raw_len_dist[k]}")
    print("\n== Unique label count distribution (len(unique)) ==")
    for k in sorted(uniq_len_dist.keys()):
        print(f"  {k}: {uniq_len_dist[k]}")

    if internal_dup_examples:
        print(f"\n== Examples: row-internal duplicates (up to {args.show_examples}) ==")
        for pid, raw, uniq in internal_dup_examples:
            print(f"  pid={pid} raw={raw} -> uniq={uniq}")

    # -------------------------
    # (B) 전체 빈도/조합 중복 체크
    # -------------------------
    label_freq_raw = Counter()
    label_freq_uniq = Counter()
    combo_freq = Counter()  # sorted(unique_labels) tuple

    for raw, uniq in zip(labels_raw, labels_uniq):
        for x in raw:
            label_freq_raw[x] += 1
        for x in uniq:
            label_freq_uniq[x] += 1
        combo_freq[tuple(uniq)] += 1

    print_top(label_freq_uniq, "Label Frequency (unique-per-row)", k=args.topk)
    print_top(combo_freq, "Label-Combo Frequency (unique labels per row, as tuple)", k=args.topk)

    # 조합 중복 “심함”을 빠르게 체감할 수 있게: top-20가 전체에서 차지하는 비율
    topk_sum = sum(cnt for _, cnt in combo_freq.most_common(args.topk))
    print(f"\n[Combo Concentration] top{args.topk} combos cover {topk_sum}/{N} = {topk_sum/N:.2%} of rows")

    # -------------------------
    # (C) Path 제약 체크
    # -------------------------
    if args.taxonomy is None:
        print("\n[Path Check] taxonomy 파일이 없어서 스킵함. --taxonomy taxonomy.txt로 실행해줘.")
    else:
        parents, children, nodes = load_taxonomy_edges(args.taxonomy)
        ancestors_fn, is_ancestor_fn, depth_fn, has_root_fn, count_roots_fn = build_ancestor_utils(parents)

        invalid = 0
        invalid_examples = []
        missing_root = 0
        multi_root = 0

        # strict/non-strict 둘 다 보고 싶을 때 대비: strict 결과 + (non-strict 참고)도 카운트
        invalid_nonstrict = 0

        for pid, uniq in zip(pids, labels_uniq):
            # 루트 관련 통계
            rcount = count_roots_fn(uniq)
            if rcount == 0:
                missing_root += 1
            if rcount >= 2:
                multi_root += 1

            ok, reason = check_path_constraint(uniq, is_ancestor_fn, depth_fn, strict_levels=strict)
            ok_ns, _ = check_path_constraint(uniq, is_ancestor_fn, depth_fn, strict_levels=False)

            if not ok:
                invalid += 1
                if len(invalid_examples) < args.show_examples:
                    invalid_examples.append((pid, uniq, reason))

            if not ok_ns:
                invalid_nonstrict += 1

        print("\n== Path Constraint Check ==")
        print(f"  strict_levels={strict}")
        print(f"  invalid={invalid}/{N} = {invalid/N:.2%}")
        print(f"  invalid(non-strict reference)={invalid_nonstrict}/{N} = {invalid_nonstrict/N:.2%}")
        print(f"  missing_root_in_pred_set={missing_root}/{N} = {missing_root/N:.2%}")
        print(f"  multi_root_in_pred_set={multi_root}/{N} = {multi_root/N:.2%}")

        if invalid_examples:
            print(f"\n== Examples: path-constraint violations (up to {args.show_examples}) ==")
            for pid, uniq, reason in invalid_examples:
                print(f"  pid={pid} labels={uniq} reason={reason}")

    # -------------------------
    # (Optional) write deduped file
    # -------------------------
    if args.write_dedup:
        # 주의: 여기서는 “행 내부 중복만 제거”하고 2~3 강제 padding은 안 함(확률이 없어서 임의 보정 위험).
        # 파일 포맷은 입력 헤더를 최대한 유지하려고 (pid/labels) 형태로 저장.
        out_path = args.write_dedup
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pid", "labels"])
            for pid, uniq in zip(pids, labels_uniq):
                w.writerow([pid, ",".join(map(str, uniq))])
        print(f"\n[Write] deduped submission saved: {out_path}")


if __name__ == "__main__":
    main()
