#!/usr/bin/env python3
# nucc_token_extractor.py
# Build informative NUCC keyword sets and export to JSON

import argparse, json, re, pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Set

RE_NON_ALNUM = re.compile(r"[^0-9a-z ]+")
GENERIC = {
  "a","an","and","or","of","the","for","to","from","with","by","in","on","at",
  "as","is","are","was","were","be","being","been",
  "clinic","clinical","center","centers","centre","centres","hospital","hospitals",
  "medical","medicine","health","healthcare","care","group","associates","services",
  "service","unit","units","department","institute","foundation","practice",
  "practices","office","offices"
}

def norm(s:str)->str:
    if s is None: return ""
    s = s.lower().strip().replace("&"," and ").replace("/"," ").replace("-"," ")
    s = RE_NON_ALNUM.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def toks(s:str)->List[str]:
    return [t for t in norm(s).split() if t and t not in GENERIC and len(t)>=3 and not t.isdigit()]

def bigrams(ts:List[str])->List[str]:
    return [" ".join(pair) for pair in zip(ts, ts[1:])]

def build_keywords(nucc_csv:str,
                   min_df_abs:int=2,
                   max_df_frac:float=0.6,
                   use_bigrams:bool=True)->Dict:
    df = pd.read_csv(nucc_csv, dtype=str).fillna("")
    rows = []
    code2label = {}
    for _, r in df.iterrows():
        code = str(r.get("code") or r.get("Code") or r.get("NUCC_CODE") or "").strip()
        if not code: continue
        parts = []
        for f in ("display_name","classification","specialization"):
            v = r.get(f) or r.get(f.capitalize()) or ""
            if str(v).strip(): parts.append(str(v))
        label = " ".join(parts).strip() or code
        code2label[code] = label
        rows.append((code, toks(label)))

    D = len(rows)
    df_uni, df_bi = Counter(), Counter()
    per_code_uni, per_code_bi = {}, {}

    for code, ts in rows:
        us = set(ts)
        bs = set(bigrams(ts)) if use_bigrams else set()
        for u in us: df_uni[u]+=1
        for b in bs: df_bi[b]+=1
        per_code_uni[code]=us
        per_code_bi[code]=bs

    vocab_keep = {u for u,dfu in df_uni.items() if dfu>=min_df_abs and dfu<=max_df_frac*D}
    bigram_keep = {b for b,dfb in df_bi.items() if dfb>=max(2,min_df_abs) and dfb<=0.4*D} if use_bigrams else set()

    code2tokens  = {c:list(per_code_uni[c]&vocab_keep) for c in per_code_uni}
    code2bigrams = {c:list(per_code_bi[c]&bigram_keep) for c in per_code_bi} if use_bigrams else {}

    return {
        "code2tokens": code2tokens,
        "code2bigrams": code2bigrams,
        "vocab_keep": list(vocab_keep),
        "bigram_keep": list(bigram_keep)
    }

def main():
    p = argparse.ArgumentParser(description="Extract NUCC keywords & bigrams to JSON")
    p.add_argument("--nucc", required=True, help="NUCC taxonomy CSV path")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--min_df_abs", type=int, default=2)
    p.add_argument("--max_df_frac", type=float, default=0.6)
    p.add_argument("--no_bigrams", action="store_true", help="Disable bigrams")
    args = p.parse_args()

    res = build_keywords(args.nucc,
                         min_df_abs=args.min_df_abs,
                         max_df_frac=args.max_df_frac,
                         use_bigrams=not args.no_bigrams)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
    print(f"✅ Saved NUCC tokens JSON to {args.out}")

if __name__=="__main__":
    main()
