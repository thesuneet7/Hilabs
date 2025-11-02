import pandas as pd
import json

def build_token_dictionary(nucc_df: pd.DataFrame):
    text_fields = ["display_name","classification","specialization"]
    all_tokens = set()
    for _, row in nucc_df.iterrows():
        for f in text_fields:
            v = str(row.get(f) or row.get(f.capitalize()) or "").strip().lower()
            all_tokens.update(v.split())
    # keep only alphabetic tokens with reasonable length
    all_tokens = {t for t in all_tokens if t.isalpha() and len(t) > 2}
    return sorted(all_tokens)

if __name__ == "__main__":
    nucc_path = "nucc_taxonomy_master.csv"   # adjust path if needed
    nucc_df = pd.read_csv(nucc_path, dtype=str).fillna("")
    tokens = build_token_dictionary(nucc_df)
    out_path = "nucc_dict.json"
    with open(out_path, "w") as f:
        json.dump(tokens, f, indent=2)
    print(f"Saved {len(tokens)} tokens to {out_path}")