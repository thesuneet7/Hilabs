# HiLabs Hackathon 2025: Specialty Standardization Challenge

## Problem Statement

Standardizing Provider Specialties to NUCC Taxonomy — creating a consistent mapping from messy provider specialty names to standardized NUCC taxonomy codes using fuzzy matching, token-based, and semantic methods.

---

## Overview

This project automates the standardization of provider specialties by:

* Cleaning and normalizing raw specialty names.
* Matching them to NUCC taxonomy entries using a combination of string similarity and semantic techniques.
* Validating the mappings through multi-method QA.

---

## Project Structure

```
Hilabs/
│
├── standardize.py            # Fuzzy matching and standardization
├── validate_mappings.py      # QA validation for fuzzy matches
├── nucc_taxonomy_master.csv  # NUCC taxonomy master reference
├── synonyms.csv              # Optional synonyms mapping
├── nucc_tokens.json          # Precomputed NUCC keyword tokens
├── output.csv                # Output after standardization
└── validated_output.csv      # Output after QA validation
```

---

## Step 1: Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-org>/hilabs-specialty-standardization.git
   cd hilabs-specialty-standardization
   ```
2. **Create a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Step 2: Running Standardization

The `standardize.py` script performs fuzzy matching between input specialties and NUCC taxonomy labels.

**Usage:**

```bash
python standardize.py --nucc nucc_taxonomy_master.csv --input input_specialties.csv --synonyms synonyms.csv --out output.csv
```

**Arguments:**

* `--nucc`: Path to NUCC taxonomy master file.
* `--input`: Input CSV containing raw specialty names.
* `--synonyms`: Optional synonyms file for enhanced matching.
* `--out`: Output CSV path.

---

## Step 3: Running Validation (QA)

The `validate_mappings.py` script checks the fuzzy matches using three complementary methods — **A**, **B**, and **C** — to ensure mapping accuracy.

**Usage:**

```bash
python validate_mappings.py --input output.csv --nucc nucc_taxonomy_master.csv --keywords nucc_tokens.json --out validated_output.csv
```

**Arguments:**

* `--input`: Output file from `standardize.py`.
* `--nucc`: NUCC taxonomy reference CSV.
* `--keywords`: Tokenized NUCC keyword JSON.
* `--out`: Output path for validated file.
* `--min_avg_sim`: (Optional) Threshold for fuzzy similarity (default 0.65).
* `--min_sem_cos`: (Optional) Threshold for semantic similarity (default 0.60).
* `--disable_semantic`: Add this flag to skip semantic validation.

---

## Validation Methods

### **Method A — Keyword Overlap (Rule-Based)**

Uses precomputed NUCC token and bigram dictionaries (`nucc_tokens.json`) to measure direct lexical overlap between raw specialties and NUCC labels.

* Calculates unigram and bigram overlaps.
* Assigns higher weight to bigram matches.
* Flags records where overlap is below a threshold.

### **Method B — Mutual Fuzzy Similarity**

Evaluates bidirectional fuzzy similarity (forward and reverse) between raw specialty and mapped NUCC label using `rapidfuzz`.

* Computes average fuzzy similarity score.
* Flags mappings below a defined minimum average similarity (default 0.65).

### **Method C — Semantic Similarity (Transformer-Based)**

Applies sentence embeddings from `sentence-transformers` (default: `all-MiniLM-L6-v2`) to assess contextual similarity between raw and NUCC label.

* Computes cosine similarity between normalized embeddings.
* Flags mappings below a cosine threshold (default 0.60).

Each row receives a **green_flag = 1** only if all active validation methods pass.

---

## Output Columns

| Column                 | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `validated_label_used` | Final NUCC label used for validation           |
| `methodA_flag`         | 0 if sufficient keyword overlap, else 1        |
| `methodB_flag`         | 0 if fuzzy match strong enough, else 1         |
| `methodC_flag`         | 0 if semantic similarity strong enough, else 1 |
| `green_flag`           | 1 if all active methods pass, else 0           |

---

## Step 4: How to Build and Run the Docker Container

1. **Create** `requirements.txt` and `Dockerfile` in your project directory.
2. **Build** the Docker image from the terminal in your project directory:

   ```bash
   docker build -t my-python-app .
   ```
3. **Run** the Docker container:

   ```bash
   docker run -it my-python-app
   ```
4. *(Optional)* To run a specific script inside the container, modify your Dockerfile’s `CMD` line:

   ```dockerfile
   CMD ["python", "your_script.py"]
   ```

---

## Example Workflow

1. Run `standardize.py` to generate fuzzy mappings.
2. Validate results using `validate_mappings.py`.
3. Review flags in the final CSV for QA.

---

## Credits

Developed by team **Mrittika_AI** for **HiLabs Hackathon 2025** — Specialty Standardization Challenge.
