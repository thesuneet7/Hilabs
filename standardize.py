#!/usr/bin/env python3
"""
standardize.py
Baseline NUCC specialty mapper:
 - exact matching (normalized)
 - synonyms/abbrev expansion
 - fuzzy token-set matching (rapidfuzz)
Outputs: csv with columns raw_specialty, nucc_codes, confidence, explain
Deterministic (no randomness).
"""

import argparse
import csv
import math
from typing import List, Tuple, Dict, Any
import pandas as pd
import re
from rapidfuzz import fuzz, process

# -------------------------
# Utilities: normalization
# -------------------------
RE_NON_ALNUM = re.compile(r"[^0-9a-z ]+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = RE_NON_ALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [t for t in s.split() if t]

# -------------------------
# Load synonyms (CSV)
# -------------------------
def load_synonyms(path: str) -> Dict[str, List[str]]:
    # expected format: raw,expanded (you can include multiple expansions separated by '|')
    syn = {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    for _, row in df.iterrows():
        raw = normalize_text(str(row.iloc[0]))
        expanded = str(row.iloc[1]).strip()
        if expanded == "":
            continue
        exps = [normalize_text(e) for e in expanded.split("|") if e.strip()]
        if raw:
            syn[raw] = exps
    return syn

# -------------------------
# Build NUCC index
# -------------------------
def build_nucc_index(nucc_df: pd.DataFrame) -> Tuple[Dict[str, List[int]], List[str], List[str]]:
    """
    Returns:
      - index_by_norm: map normalized string -> list of row indices
      - candidates_texts: list of normalized candidate strings (for fuzzy matching)
      - candidate_codes: list of NUCC codes (same length)
    """
    df = nucc_df.fillna("")
    fields_to_concat = ["display_name", "classification", "specialization"]
    candidates = []
    codes = []
    index_by_norm = {}
    for i, row in df.iterrows():
        code = str(row.get("code") or row.get("Code") or row.get("NUCC_CODE") or "").strip()
        # Build a canonical text for fuzzy matching
        pieces = []
        for f in fields_to_concat:
            v = row.get(f) or row.get(f.capitalize()) or ""
            if pd.notna(v) and str(v).strip():
                pieces.append(str(v))
        canon = " ".join(pieces).strip()
        canon_norm = normalize_text(canon)
        if not canon_norm:
            continue
        candidates.append(canon_norm)
        codes.append(code)
        index_by_norm.setdefault(canon_norm, []).append(len(candidates)-1)
    return index_by_norm, candidates, codes

# -------------------------
# Matching logic
# -------------------------
def expand_via_synonyms(text: str, synonyms: Dict[str, List[str]]) -> List[str]:
    # produce a list of candidate expansions (including original normalized text)
    norm = normalize_text(text)
    expansions = [norm]
    # If the whole phrase matches a synonym key
    if norm in synonyms:
        expansions = list(dict.fromkeys(expansions + synonyms[norm]))
    # Also try token-level expansions (replace tokens that are keys)
    tokens = tokenize(norm)
    token_expanded = []
    replaced_any = False
    for t in tokens:
        if t in synonyms:
            token_expanded.append(" ".join([e if e != t else "|".join(synonyms[t]) for e in tokens]))
            replaced_any = True
    if replaced_any:
        # naive join, but we'll also return original
        expansions += [normalize_text(" ".join(tokens))]
    return list(dict.fromkeys([normalize_text(e) for e in expansions if e]))

def compute_confidence(is_exact: bool, fuzzy_score: float, synonym_hit: bool) -> float:
    # weights tuned for baseline
    wE = 0.5
    wF = 0.45
    wS = 0.05
    E = 1.0 if is_exact else 0.0
    F = max(0.0, min(1.0, fuzzy_score))
    S = 1.0 if synonym_hit else 0.0
    conf = wE*E + wF*F + wS*S
    return max(0.0, min(1.0, conf))

def match_single(raw: str, synonyms: Dict[str, List[str]], index_by_norm: Dict[str, List[int]],
                 candidates: List[str], codes: List[str], fuzzy_cutoff=0.60) -> Tuple[str, float, str]:
    raw_norm = normalize_text(raw)
    if raw_norm == "":
        return "JUNK", 0.0, "blank input"

    # Expand via synonyms
    expansions = expand_via_synonyms(raw, synonyms)
    synonym_hit = (len(expansions) > 1)

    # 1) exact match on NUCC normalized strings
    for e in expansions:
        if e in index_by_norm:
            ids = index_by_norm[e]
            out_codes = [codes[i] for i in ids if codes[i]]
            explain = f"exact match on '{e}'"
            confidence = compute_confidence(is_exact=True, fuzzy_score=1.0, synonym_hit=synonym_hit)
            return "|".join(sorted(set(out_codes))) if out_codes else "JUNK", round(confidence, 2), explain

    # 2) fuzzy match against candidates (token_set_ratio)
    # Use rapidfuzz.process.extract with token_set_ratio scorer
    # We'll score against the canonical list, get top candidate
    # Build mapping for process.extract (candidate->index)
    mapping = {candidates[i]: i for i in range(len(candidates))}
    # Flatten expansions into one query string (prefer first)
    query = expansions[0]
    # Use process.extract to get top matches
    results = process.extract(query, mapping.keys(), scorer=fuzz.token_set_ratio, limit=5)
    # results: list of (candidate_string, score, _) where score is 0-100
    if not results:
        return "JUNK", 0.0, "no candidates"

    top_candidate, top_score, _ = results[0]
    fuzzy_score = top_score / 100.0
    # handle borderline fuzzy matches
    if fuzzy_score < fuzzy_cutoff:
    # 👇 new condition: still accept if fuzzy_score is strong enough (>= 0.80)
        if fuzzy_score >= 0.80:
            idx = mapping[top_candidate]
            matched_codes = [codes[idx]]
            confidence = compute_confidence(False, fuzzy_score, synonym_hit)
            explain = f"high fuzzy match accepted (score={fuzzy_score:.2f}) to '{top_candidate}'"
            return "|".join(sorted(set([c for c in matched_codes if c]))) or "JUNK", round(confidence, 2), explain
        # otherwise treat as junk
        return "JUNK", round(compute_confidence(False, fuzzy_score, synonym_hit), 2), f"top fuzzy {fuzzy_score:.2f} below cutoff"


    # gather all candidates within 3 points (0.03) of top and above cutoff
    top_scores = [r for r in results if (r[1]/100.0) >= fuzzy_cutoff and (top_score - r[1]) <= 3]
    matched_codes = []
    expl_parts = []
    for cand_str, score, _ in top_scores:
        idx = mapping[cand_str]
        matched_codes.append(codes[idx])
        expl_parts.append(f"'{cand_str}' score={score/100.0:.2f}")

    confidence = compute_confidence(False, fuzzy_score, synonym_hit)
    explain = "fuzzy match: " + "; ".join(expl_parts)
    return "|".join(sorted(set([c for c in matched_codes if c])) ) or "JUNK", round(confidence, 2), explain

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="NUCC specialty standardizer (baseline)")
    parser.add_argument("--nucc", required=True, help="nucc_taxonomy_master.csv")
    parser.add_argument("--input", required=True, help="input_specialties.csv (single column of raw specialties)")
    parser.add_argument("--synonyms", required=False, default=None, help="synonyms.csv (raw,expanded)")
    parser.add_argument("--out", required=True, help="output csv path")
    args = parser.parse_args()

    nucc_df = pd.read_csv(args.nucc, dtype=str).fillna("")
    input_df = pd.read_csv(args.input, dtype=str).fillna("")

    synonyms = load_synonyms(args.synonyms) if args.synonyms else {}

    index_by_norm, candidates, codes = build_nucc_index(nucc_df)

    # identify input column (choose first column if header unknown)
    if input_df.shape[1] == 1:
        input_col = input_df.columns[0]
    else:
        # Prefer column named raw_specialty or specialty or input
        possible = [c for c in input_df.columns if c.lower() in ("raw_specialty", "specialty", "input", "raw")]
        input_col = possible[0] if possible else input_df.columns[0]

    outputs = []
    for raw in input_df[input_col].astype(str).tolist():
        nucc_codes, conf, explain = match_single(raw, synonyms, index_by_norm, candidates, codes)
        outputs.append({"raw_specialty": raw, "nucc_codes": nucc_codes, "confidence": f"{conf:.2f}", "explain": explain})

    out_df = pd.DataFrame(outputs, columns=["raw_specialty", "nucc_codes", "confidence", "explain"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
