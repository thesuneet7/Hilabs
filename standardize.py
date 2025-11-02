#!/usr/bin/env python3
import argparse, pandas as pd, re, json
from typing import List, Tuple, Dict
from rapidfuzz import fuzz, process

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

# --- dictionary-based segmentation (Code 2 feature) ---
def load_token_dict(path="nucc_dict.json"):
    try:
        with open(path, "r") as f:
            tokens = json.load(f)
        return set(t.lower().strip() for t in tokens if t.strip())
    except Exception:
        return set()

def segment_with_dictionary(text: str, token_dict: set) -> str:
    if not token_dict or not isinstance(text, str):
        return text
    if " " in text.strip():
        return text
    s = re.sub(r"[^A-Za-z]", "", text).lower()
    if len(s) < 6:
        return text
    n, i, tokens, matched_any = len(s), 0, [], False
    while i < n:
        match = None
        for j in range(n, i, -1):
            chunk = s[i:j]
            if chunk in token_dict:
                match = chunk
                tokens.append(chunk)
                matched_any = True
                i = j
                break
        if not match:
            tokens.append(s[i:])
            break
    if not matched_any:
        return text
    meaningful = [t for t in tokens if len(t) > 1]
    if len(meaningful) < 1:
        return text
    return " ".join(tokens).strip()

# --- clinic/center post-append (Code 1 feature) ---
def has_clinic_or_center(raw: str) -> bool:
    if not isinstance(raw, str): return False
    s = raw.lower()
    return bool(re.search(r"\bclinic\b", s) or re.search(r"\bcenter\b", s))

def append_code_preserve_order(code_str: str, extra_code: str) -> str:
    if not code_str or code_str.upper() == "JUNK":
        base = []
    else:
        base = [c for c in code_str.split("|") if c.strip()]
    base.append(extra_code)
    seen, out = set(), []
    for c in base:
        cu = c.upper()
        if cu not in seen:
            seen.add(cu)
            out.append(cu)
    return ("|".join(out)) if out else "JUNK"

# --- Function to check for mandatory codes based on "emergency", "medical", and "technician" ---
def has_emergency_medical_technician(raw: str) -> bool:
    raw = raw.lower()
    return all(word in raw for word in ["emergency", "medical", "technician"])

# --- classifier that composes both extras cleanly ---
def classify_row(raw: str, synonyms: Dict[str, List[str]],
                 index_by_norm: Dict[str, List[int]],
                 candidates: List[str], codes: List[str],
                 token_dict=None, fuzzy_cutoff=0.60) -> Tuple[str, float, str, bool]:
    # 1) direct NUCC codes
    direct_codes = extract_nucc_codes(raw)
    if direct_codes:
        codes_str = "|".join(sorted(set(c.upper() for c in direct_codes)))
        return codes_str, 1.0, "direct NUCC code detected in input", False

    # 2) spacing rescue (only prepares the query text; normalization happens later)
    query_text = raw
    used_spacing = False
    if token_dict:
        need_seg = (" " not in raw) and (len(re.sub(r"[^A-Za-z]", "", raw)) >= 6)
        if need_seg:
            seg = segment_with_dictionary(raw, token_dict)
            if seg != raw:
                query_text = seg
                used_spacing = True

    # 3) now run the normal pipeline on query_text
    rn = normalize_text(query_text)
    if rn == "":
        return "JUNK", 0.0, "blank input", used_spacing

    exps = expand_via_synonyms(query_text, synonyms)

    for e in exps:
        if e in index_by_norm:
            ids = index_by_norm[e]
            out = sorted({codes[i] for i in ids if codes[i]})
            expl = f"exact match on '{e}'"
            if used_spacing:
                expl += " + spacing rescue"
            return ("|".join(out) if out else "JUNK"), 1.0, expl, False

    mapping = {candidates[i]: i for i in range(len(candidates))}
    best_f, best_res, best_exp = 0.0, None, None
    for exp in exps:
        res = process.extract(exp, mapping.keys(), scorer=fuzz.token_set_ratio, limit=5)
        if res:
            top_cand, top_score, _ = res[0]
            f = top_score / 100.0
            if f > best_f:
                best_f, best_res, best_exp = f, res, exp

    if best_res is None:
        expl = "no candidates"
        if used_spacing: expl += " + spacing tried"
        return "JUNK", 0.0, expl, False

    f = round(best_f, 4)
    if f < fuzzy_cutoff:
        expl = f"best fuzzy {f:.2f} below cutoff {fuzzy_cutoff:.2f} (query='{best_exp}')"
        if used_spacing: expl += " + spacing tried"
        return "JUNK", f, expl, False

    top_score = best_res[0][1]
    near = [r for r in best_res if (r[1]/100.0) >= fuzzy_cutoff and (top_score - r[1]) <= 3]
    matched = sorted({codes[mapping[c]] for c, _, _ in near if codes[mapping[c]]})
    if not matched:
        expl = "fuzzy above cutoff but no codes"
        if used_spacing: expl += " + spacing tried"
        return "JUNK", f, expl, False

    explain = f"fuzzy match (query='{best_exp}'): " + "; ".join([f"'{c}' score={r/100:.2f}" for c, r, _ in near])
    if used_spacing:
        explain += " + spacing rescue"
    return "|".join(matched), f, explain, False

def main():
    p = argparse.ArgumentParser(description="NUCC standardizer with spacing rescue + clinic/center append")
    p.add_argument("--nucc", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--synonyms", required=False, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    nucc_df = pd.read_csv(args.nucc, dtype=str).fillna("")
    in_df = pd.read_csv(args.input, dtype=str).fillna("")
    synonyms = load_synonyms(args.synonyms) if args.synonyms else {}
    index_by_norm, candidates, codes = build_nucc_index(nucc_df)
    token_dict = load_token_dict("nucc_dict.json")

    if in_df.shape[1] == 1:
        col = in_df.columns[0]
    else:
        pref = [c for c in in_df.columns if c.lower() in ("raw_specialty","specialty","input","raw")]
        col = pref[0] if pref else in_df.columns[0]

    rows = []
    for raw in in_df[col].astype(str).tolist():
        nucc_codes, label, explain, _ = classify_row(
            raw, synonyms, index_by_norm, candidates, codes, token_dict=token_dict, fuzzy_cutoff=0.60
        )

        # 1. Append clinic/center code if applicable
        if has_clinic_or_center(raw):
            nucc_codes = append_code_preserve_order(nucc_codes, "261Q00000X")
            label = 1  # Overwrite label to 1 when manually adding clinic/center code
            explain = (explain + " + appended 261Q00000X for 'clinic/center' rule") if explain else "appended 261Q00000X for 'clinic/center' rule"

        # 2. Append emergency/medical/technician codes if all are present
        if has_emergency_medical_technician(raw):
            nucc_codes = append_code_preserve_order(nucc_codes, "146L00000X")
            nucc_codes = append_code_preserve_order(nucc_codes, "146M00000X")
            nucc_codes = append_code_preserve_order(nucc_codes, "146N00000X")
            explain += " + mandatory emergency/medical/technician codes appended"

            label=1

        rows.append({
            "raw_specialty": raw,
            "nucc_codes": nucc_codes,
            "Label": label,
            "explain": explain
        })

    out_df = pd.DataFrame(rows, columns=["raw_specialty","nucc_codes","Label","explain"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
