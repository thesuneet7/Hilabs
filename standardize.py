#!/usr/bin/env python3
import argparse, pandas as pd, re
from typing import List, Tuple, Dict
from rapidfuzz import fuzz, process
import json

RE_NON_ALNUM = re.compile(r"[^0-9a-z ]+")
NUCC_CODE_PATTERN = re.compile(r"\b[A-Z0-9]{10}\b", re.IGNORECASE)

def extract_nucc_codes(text: str):
    if not isinstance(text, str):
        return []
    matches = NUCC_CODE_PATTERN.findall(text.upper())
    return [m for m in matches if any(c.isdigit() for c in m) and m[-1].isalpha()]

def load_generic_words(path="generic_terms.json"):
    try:
        with open(path, "r") as f:
            words = json.load(f)
        return set(w.lower().strip() for w in words if str(w).strip())
    except Exception:
        return set()

GENERIC_WORDS = load_generic_words()

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("&"," and ").replace("/"," ").replace("-"," ")
    s = RE_NON_ALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in GENERIC_WORDS]
    return " ".join(tokens)

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
    direct_codes = extract_nucc_codes(raw)
    if direct_codes:
        codes_str = "|".join(sorted(set(c.upper() for c in direct_codes)))
        return codes_str, 1.0, "direct NUCC code detected in input"

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
    best_f = 0.0
    best_res = None
    best_exp = None
    for exp in exps:
        res = process.extract(exp, mapping.keys(), scorer=fuzz.token_set_ratio, limit=5)
        if res:
            top_cand, top_score, _ = res[0]
            f = top_score / 100.0
            if f > best_f:
                best_f, best_res, best_exp = f, res, exp

    if best_res is None:
        return "JUNK", 0.0, "no candidates"

    f = round(best_f, 4)
    res = best_res
    if f < fuzzy_cutoff:
        return "JUNK", f, f"best fuzzy {f:.2f} below cutoff {fuzzy_cutoff:.2f} (query='{best_exp}')"

    top_score = res[0][1]
    near = [r for r in res if (r[1]/100.0) >= fuzzy_cutoff and (top_score - r[1]) <= 3]
    matched = sorted({codes[mapping[c]] for c, _, _ in near if codes[mapping[c]]})
    if not matched:
        return "JUNK", f, "fuzzy above cutoff but no codes"

    explain = f"fuzzy match (query='{best_exp}'): " + "; ".join([f"'{c}' score={r/100:.2f}" for c, r, _ in near])
    return "|".join(matched), f, explain

# ---- NEW: detect if original raw contains 'clinic' or 'center' tokens (case-insensitive) ----
def has_clinic_or_center(raw: str) -> bool:
    if not isinstance(raw, str): return False
    s = raw.lower()
    # strict token check to avoid substrings like 'epicenter' or 'clinician'
    return bool(re.search(r"\bclinic\b", s) or re.search(r"\bcenter\b", s))

def append_code_preserve_order(code_str: str, extra_code: str) -> str:
    if not code_str or code_str.upper() == "JUNK":
        base = []
    else:
        base = [c for c in code_str.split("|") if c.strip()]
    base.append(extra_code)
    seen = set()
    out = []
    for c in base:
        cu = c.upper()
        if cu not in seen:
            seen.add(cu)
            out.append(cu)
    return ("|".join(out)) if out else "JUNK"

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

        # --- NEW: if raw contains 'clinic' or 'center', append 261Q00000X regardless of matches ---
        appended = False
        if has_clinic_or_center(raw):
            nucc_codes = append_code_preserve_order(nucc_codes, "261Q00000X")
            appended = True

        if appended:
            explain = (explain + " + appended 261Q00000X for 'clinic/center' rule") if explain else "appended 261Q00000X for 'clinic/center' rule"

        rows.append({
            "raw_specialty": raw,
            "nucc_codes": nucc_codes,
            "Label": label,        # unchanged: still reflects match confidence (exact=1.0, fuzzy in [0,1], or 0.0)
            "explain": explain
        })

    out_df = pd.DataFrame(rows, columns=["raw_specialty","nucc_codes","Label","explain"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
