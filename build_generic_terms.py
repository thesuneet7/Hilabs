import pandas as pd
import re
import json
from collections import Counter

RE_NON_ALNUM = re.compile(r"[^0-9a-z ]+")

def normalize_text(s):
    s = s.lower().strip()
    s = s.replace("&"," and ").replace("/"," ").replace("-"," ")
    s = RE_NON_ALNUM.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

# Load NUCC taxonomy
nucc = pd.read_csv("nucc_taxonomy_master.csv", dtype=str).fillna("")
fields = ["display_name","classification","specialization"]
texts = []

for _, row in nucc.iterrows():
    parts = []
    for f in fields:
        v = row.get(f) or row.get(f.capitalize()) or ""
        if str(v).strip():
            parts.append(str(v))
    texts.append(normalize_text(" ".join(parts)))

# Token frequency
all_tokens = []
for t in texts:
    all_tokens.extend(t.split())

freq = Counter(all_tokens)
total = sum(freq.values())

# Filter to frequent, non-specialty-like words
common = [w for w, c in freq.items() if c > 20]  # threshold adjustable
common = [w for w in common if len(w) > 2]       # skip tiny words

# Manually exclude obvious specialty prefixes
likely_generic = [w for w in common if w not in [
    "cardiology","medicine","surgery","pediatrics","family",
    "internal","radiology","psychiatry","anesthesiology","pathology"
]]

print("Top likely generic terms:", sorted(likely_generic)[:40])

# Save to JSON
with open("generic_terms.json","w") as f:
    json.dump(sorted(likely_generic), f, indent=2)
print("Saved generic_terms.json with", len(likely_generic), "entries.")
