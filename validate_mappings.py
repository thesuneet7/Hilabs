#!/usr/bin/env python3
# validate_mappings.py — single-file QA validator for NUCC fuzzy mappings

import argparse, re, json, pandas as pd, numpy as np
from typing import Dict, List, Tuple
from rapidfuzz import fuzz
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer   # install manually before running

# ---------- text utils ----------
RE_NON_ALNUM = re.compile(r"[^0-9a-z ]+")

def norm(s: str) -> str:
    if s is None: return ""
    s = s.lower().strip().replace("&"," and ").replace("/"," ").replace("-"," ")
    s = RE_NON_ALNUM.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def toks(s: str) -> List[str]:
    return [t for t in norm(s).split() if t and len(t) >= 3 and not t.isdigit()]

def bigrams(ts: List[str]) -> List[str]:
    return [" ".join(pair) for pair in zip(ts, ts[1:])]

# ---------- NUCC helpers ----------
def load_nucc_map(path: str) -> Dict[str, str]:
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        code = str(r.get("code") or r.get("Code") or r.get("NUCC_CODE") or "").strip()
        parts = []
        for f in ("display_name","classification","specialization"):
            v = r.get(f) or r.get(f.capitalize()) or ""
            if str(v).strip(): parts.append(str(v).strip())
        label = " ".join(parts).strip() or code
        if code: out[code] = label
    return out

def codes_to_best_label(code_str: str, code2label: Dict[str,str], raw: str) -> Tuple[str, List[str]]:
    codes = [c.strip() for c in str(code_str).split("|") if c.strip()]
    labels = [code2label.get(c, "") for c in codes if code2label.get(c, "")]
    if not labels: return "", []
    best = max(labels, key=lambda L: fuzz.token_set_ratio(norm(raw), norm(L)))
    return best, labels

# ---------- Method A (keyword overlap using JSON) ----------
def load_keyword_json(path:str)->Dict:
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def method_a_keyword_overlap_auto(raw:str, code:str,
                                  keyword_json:Dict,
                                  min_any_overlap:int=1)->Dict:
    """Return dict of metrics and flag."""
    code2tokens  = keyword_json.get("code2tokens",{})
    code2bigrams = keyword_json.get("code2bigrams",{})
    vocab_keep   = set(keyword_json.get("vocab_keep",[]))
    bigram_keep  = set(keyword_json.get("bigram_keep",[]))

    rt = toks(raw)
    raw_uni = {t for t in rt if t in vocab_keep}
    raw_bi  = {b for b in bigrams(rt) if b in bigram_keep}

    code_uni = set(code2tokens.get(code,[]))
    code_bi  = set(code2bigrams.get(code,[]))

    inter_uni = raw_uni & code_uni
    inter_bi  = raw_bi & code_bi

    overlap_score = len(inter_uni) + 2*len(inter_bi)
    flag = 0 if overlap_score >= min_any_overlap else 1
    uni_jacc = (len(inter_uni)/len(raw_uni|code_uni)) if (raw_uni|code_uni) else 0.0
    bi_jacc  = (len(inter_bi)/len(raw_bi|code_bi)) if (raw_bi|code_bi) else 0.0

    return {
        "methodA_flag": flag,
        "methodA_overlap_unigram": len(inter_uni),
        "methodA_overlap_bigram": len(inter_bi),
        "methodA_uni_jaccard": round(uni_jacc,4),
        "methodA_bi_jaccard": round(bi_jacc,4)
    }

# ---------- Method B (mutual fuzzy similarity) ----------
def method_b_mutual_similarity(raw: str, label: str, min_avg: float = 0.65) -> Tuple[int, float, float, float]:
    r = norm(raw); l = norm(label)
    f_fwd = fuzz.token_set_ratio(r, l) / 100.0
    f_rev = fuzz.token_set_ratio(l, r) / 100.0
    f_avg = 0.5*(f_fwd + f_rev)
    flag = 0 if f_avg >= min_avg else 1
    return flag, f_fwd, f_rev, f_avg

# ---------- Method C (semantic similarity) ----------
class SemanticChecker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def compute(self, raw:str, label:str, min_cos:float=0.60)->dict:
        emb_r = self.model.encode(raw, normalize_embeddings=True)
        emb_l = self.model.encode(label, normalize_embeddings=True)
        cos = float(np.dot(emb_r, emb_l))
        flag = 0 if cos >= min_cos else 1
        return {"methodC_flag": flag, "methodC_cosine": round(cos,4)}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Validate NUCC mappings (only for fuzzy rows).")
    ap.add_argument("--input", required=True, help="mapper output CSV: must have raw_specialty, nucc_codes, Label")
    ap.add_argument("--nucc", required=True, help="nucc_taxonomy_master.csv")
    ap.add_argument("--keywords", required=True, help="nucc_tokens.json from nucc_token_extractor.py")
    ap.add_argument("--out", required=True, help="output CSV with QA flags")
    ap.add_argument("--min_avg_sim", type=float, default=0.65, help="minimum average fuzzy sim (Method B)")
    ap.add_argument("--min_sem_cos", type=float, default=0.60, help="minimum cosine similarity (Method C)")
    ap.add_argument("--disable_semantic", action="store_true", help="skip Method C if desired")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str).fillna("")
    required_cols = {"raw_specialty","nucc_codes","Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {sorted(required_cols)}")

    df["Label_num"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0.0)
    code2label = load_nucc_map(args.nucc)
    nucc_keywords = load_keyword_json(args.keywords)
    sem = None if args.disable_semantic else SemanticChecker()

    # Initialize output columns
    df["validated_label_used"] = ""
    df["methodA_flag"] = ""
    df["methodA_overlap_unigram"] = ""
    df["methodA_overlap_bigram"] = ""
    df["methodA_uni_jaccard"] = ""
    df["methodA_bi_jaccard"] = ""
    df["methodB_flag"] = ""
    df["methodB_fwd"] = ""
    df["methodB_rev"] = ""
    df["methodB_avg"] = ""
    if not args.disable_semantic:
        df["methodC_flag"] = ""
        df["methodC_cosine"] = ""
    df["green_flag"] = 0

    for i in tqdm(df.index.tolist(), leave=False):
        raw = str(df.at[i, "raw_specialty"])
        codes = str(df.at[i, "nucc_codes"]).strip()
        L = float(df.at[i, "Label_num"])

        # Skip junk
        if codes == "JUNK" or codes == "":
            df.at[i, "green_flag"] = 0
            continue

        # Exact/synonym → auto green
        if L >= 1.0:
            df.at[i, "green_flag"] = 1
            continue

        # Fuzzy only
        label, _ = codes_to_best_label(codes, code2label, raw)
        df.at[i, "validated_label_used"] = label

        # --- Method A ---
        best_code = max(
            [c.strip() for c in str(codes).split("|") if c.strip()],
            key=lambda c: fuzz.token_set_ratio(norm(raw), norm(code2label.get(c,""))),
            default=""
        )
        a_info = method_a_keyword_overlap_auto(raw, best_code, nucc_keywords)
        df.at[i, "methodA_flag"] = int(a_info["methodA_flag"])
        df.at[i, "methodA_overlap_unigram"] = int(a_info["methodA_overlap_unigram"])
        df.at[i, "methodA_overlap_bigram"] = int(a_info["methodA_overlap_bigram"])
        df.at[i, "methodA_uni_jaccard"] = a_info["methodA_uni_jaccard"]
        df.at[i, "methodA_bi_jaccard"] = a_info["methodA_bi_jaccard"]

        # --- Method B ---
        b_flag, f_fwd, f_rev, f_avg = method_b_mutual_similarity(raw, label, min_avg=args.min_avg_sim)
        df.at[i, "methodB_flag"] = int(b_flag)
        df.at[i, "methodB_fwd"] = round(float(f_fwd),4)
        df.at[i, "methodB_rev"] = round(float(f_rev),4)
        df.at[i, "methodB_avg"] = round(float(f_avg),4)

        # --- Method C ---
        c_flag = 0
        if not args.disable_semantic:
            res = sem.compute(raw, label, min_cos=args.min_sem_cos)
            c_flag = res["methodC_flag"]
            df.at[i, "methodC_flag"] = int(c_flag)
            df.at[i, "methodC_cosine"] = res["methodC_cosine"]

        # --- green flag ---
        flags = [int(df.at[i, "methodA_flag"]), int(df.at[i, "methodB_flag"])]
        if not args.disable_semantic: flags.append(int(c_flag))
        df.at[i, "green_flag"] = 1 if sum(flags) == 0 else 0

    df.drop(columns=["Label_num"], inplace=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
