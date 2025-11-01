#!/usr/bin/env python3
import argparse, pandas as pd, re
from typing import List, Tuple, Dict
from rapidfuzz import fuzz, process

RE_NON_ALNUM = re.compile(r"[^0-9a-z ]+")

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.lower().strip().replace("&"," and ").replace("/"," ").replace("-"," ")
    s = RE_NON_ALNUM.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokenize(s: str) -> List[str]:
    return [t for t in s.split() if t]

def load_synonyms(path: str) -> Dict[str, List[str]]:
    syn = {}
    if not path: return syn
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    for _, r in df.iterrows():
        raw = normalize_text(str(r.iloc[0]))
        expanded = str(r.iloc[1]).strip()
        if not expanded: continue
        exps = [normalize_text(e) for e in expanded.split("|") if e.strip()]
        if raw: syn[raw] = exps
    return syn

def build_nucc_index(nucc_df: pd.DataFrame) -> Tuple[Dict[str, List[int]], List[str], List[str]]:
    df = nucc_df.fillna("")
    fields = ["display_name","classification","specialization"]
    candidates, codes, index_by_norm = [], [], {}
    for _, row in df.iterrows():
        code = str(row.get("code") or row.get("Code") or row.get("NUCC_CODE") or "").strip()
        parts = []
        for f in fields:
            v = row.get(f) or row.get(f.capitalize()) or ""
            if str(v).strip(): parts.append(str(v))
        canon = normalize_text(" ".join(parts).strip())
        if not canon: continue
        candidates.append(canon)
        codes.append(code)
        index_by_norm.setdefault(canon, []).append(len(candidates)-1)
    return index_by_norm, candidates, codes

def expand_via_synonyms(text: str, synonyms: Dict[str, List[str]]) -> List[str]:
    norm = normalize_text(text)
    exps = [norm]
    if norm in synonyms:
        exps = list(dict.fromkeys(exps + synonyms[norm]))
    return [e for e in exps if e]

def classify_row(raw: str, synonyms: Dict[str, List[str]],
                 index_by_norm: Dict[str, List[int]],
                 candidates: List[str], codes: List[str],
                 fuzzy_cutoff=0.60) -> Tuple[str, float, str]:
    rn = normalize_text(raw)
    if rn == "":
        return "JUNK", 0.0, "blank input"
    exps = expand_via_synonyms(raw, synonyms)
    for e in exps:
        if e in index_by_norm:
            ids = index_by_norm[e]
            out = sorted({codes[i] for i in ids if codes[i]})
            return ("|".join(out) if out else "JUNK"), 1.0, f"exact match on '{e}'"
    mapping = {candidates[i]: i for i in range(len(candidates))}
    res = process.extract(exps[0], mapping.keys(), scorer=fuzz.token_set_ratio, limit=5)
    if not res:
        return "JUNK", 0.0, "no candidates"
    top_cand, top_score, _ = res[0]
    f = round(top_score/100.0, 4)
    if f < fuzzy_cutoff:
        return "JUNK", f, f"top fuzzy {f:.2f} below cutoff {fuzzy_cutoff:.2f}"
    near = [r for r in res if (r[1]/100.0) >= fuzzy_cutoff and (top_score - r[1]) <= 3]
    matched = sorted({codes[mapping[c]] for c,_,_ in near if codes[mapping[c]]})
    if not matched:
        return "JUNK", f, "fuzzy above cutoff but no codes"
    explain = "fuzzy match: " + "; ".join([f"'{c}' score={r/100:.2f}" for c,r,_ in near])
    return "|".join(matched), f, explain

def main():
    p = argparse.ArgumentParser(description="NUCC specialty standardizer with Label flag")
    p.add_argument("--nucc", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--synonyms", required=False, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    nucc_df = pd.read_csv(args.nucc, dtype=str).fillna("")
    in_df = pd.read_csv(args.input, dtype=str).fillna("")
    synonyms = load_synonyms(args.synonyms) if args.synonyms else {}

    index_by_norm, candidates, codes = build_nucc_index(nucc_df)

    if in_df.shape[1] == 1:
        col = in_df.columns[0]
    else:
        pref = [c for c in in_df.columns if c.lower() in ("raw_specialty","specialty","input","raw")]
        col = pref[0] if pref else in_df.columns[0]

    rows = []
    for raw in in_df[col].astype(str).tolist():
        nucc_codes, label, explain = classify_row(raw, synonyms, index_by_norm, candidates, codes, fuzzy_cutoff=0.60)
        rows.append({
            "raw_specialty": raw,
            "nucc_codes": nucc_codes,
            "Label": label,         # 1.0 for exact/synonym; fuzzy score (0–1) for fuzzy; 0.0 for blanks/no-candidate junk
            "explain": explain
        })

    out_df = pd.DataFrame(rows, columns=["raw_specialty","nucc_codes","Label","explain"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
