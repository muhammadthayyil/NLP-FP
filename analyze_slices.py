import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

NEG_RE = re.compile(
    r"\b(no|not|never|none|nothing|nobody|noone|can't|cannot|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|without)\b",
    re.IGNORECASE
)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

def has_negation(text: str) -> bool:
    return bool(NEG_RE.search(text or ""))

def length_bucket(text: str) -> str:
    n = len(tokenize(text))
    if n <= 5: return "short(<=5)"
    if n <= 15: return "medium(6-15)"
    return "long(>15)"

def jaccard_overlap(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def overlap_bucket(premise: str, hypothesis: str) -> str:
    j = jaccard_overlap(premise, hypothesis)
    if j >= 0.5: return "high(>=0.5)"
    if j >= 0.2: return "mid(0.2-0.5)"
    return "low(<0.2)"

def parse_label_map(s: str) -> Dict[int, str]:
    # default assumes SNLI labels are 0/1/2; edit if your mapping differs
    out: Dict[int, str] = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        out[int(k.strip())] = v.strip()
    return out

def fmt_label(x, id2name: Dict[int, str]) -> str:
    if isinstance(x, int) and x in id2name:
        return f"{x} ({id2name[x]})"
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--label_map",
                    default="0:entailment,1:neutral,2:contradiction",
                    help="Mapping for readability, e.g. '0:entailment,1:neutral,2:contradiction'")
    ap.add_argument("--show_examples", action="store_true")
    ap.add_argument("--examples_per_slice", type=int, default=3)
    ap.add_argument("--show_confusions", action="store_true")
    ap.add_argument("--top_confusions", type=int, default=10)
    args = ap.parse_args()

    id2name = parse_label_map(args.label_map)

    rows: List[Dict[str, Any]] = []
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = 0
    correct = 0

    slice_tot = Counter()
    slice_cor = Counter()
    slice_errs = defaultdict(list)

    # confusion counts across whole file
    conf = Counter()  # (gold, pred) -> count

    for r in rows:
        prem = r.get("premise", "")
        hyp = r.get("hypothesis", "")
        gold = r.get("label", None)
        pred = r.get("predicted_label", None)

        if gold is None or pred is None:
            continue

        is_correct = (gold == pred)
        total += 1
        correct += int(is_correct)
        conf[(gold, pred)] += 1

        neg = "negation" if has_negation(hyp) else "no_negation"
        lb = length_bucket(hyp)
        ob = overlap_bucket(prem, hyp)

        slices = [
            f"NEG::{neg}",
            f"LEN::{lb}",
            f"OVL::{ob}",
        ]

        for s in slices:
            slice_tot[s] += 1
            slice_cor[s] += int(is_correct)
            if (not is_correct) and args.show_examples and len(slice_errs[s]) < args.examples_per_slice:
                slice_errs[s].append(r)

    print(f"\nFile: {args.pred_file}")
    print(f"Scored: {total} examples")
    print(f"Overall accuracy: {correct/total:.4f}\n")

    def print_group(prefix: str, title: str):
        items = [(k, slice_tot[k], slice_cor[k] / slice_tot[k]) for k in slice_tot if k.startswith(prefix)]
        items.sort(key=lambda x: (-x[1], x[0]))  # by coverage desc
        print(title)
        print("-" * len(title))
        for k, n, acc in items:
            pct = 100.0 * n / total
            print(f"{k:<22} n={n:>6} ({pct:>5.1f}%) acc={acc:.4f}")
        print()

        if args.show_examples:
            # show examples from lowest-acc slice in this group (with at least some coverage)
            items_by_acc = sorted(items, key=lambda x: x[2])
            for k, n, acc in items_by_acc[:1]:
                if len(slice_errs[k]) == 0:
                    continue
                print(f"Example errors for {k} (acc={acc:.4f}):")
                for ex in slice_errs[k]:
                    g = ex["label"]; p = ex["predicted_label"]
                    print(f"- Premise: {ex.get('premise','')}")
                    print(f"  Hypothesis: {ex.get('hypothesis','')}")
                    print(f"  Gold: {fmt_label(g, id2name)} | Pred: {fmt_label(p, id2name)}\n")
                print()

    print_group("NEG::", "Negation slices (hypothesis)")
    print_group("LEN::", "Hypothesis length buckets")
    print_group("OVL::", "Lexical overlap buckets (Jaccard)")

    if args.show_confusions:
        print("Top confusions (gold -> pred), excluding correct")
        print("----------------------------------------------")
        bad: List[Tuple[Tuple[int, int], int]] = [((g, p), c) for (g, p), c in conf.items() if g != p]
        bad.sort(key=lambda x: -x[1])
        for (g, p), c in bad[: args.top_confusions]:
            print(f"{fmt_label(g, id2name):>18} -> {fmt_label(p, id2name):<18}  count={c}")
        print()

if __name__ == "__main__":
    main()
