# HiLabs Hackathon 2025: Specialty Standardization Challenge

## 🧩 Problem Statement

**Standardizing Provider Specialties to NUCC Taxonomy**

Healthcare provider data often contains free-text specialty names that are inconsistent, misspelled, or non-standardized. This project aims to **standardize provider specialties** by mapping them to the official NUCC Taxonomy using fuzzy matching, synonyms, and text normalization techniques.

---

## ⚙️ Features

* Cleans and normalizes free-text specialties
* Uses fuzzy matching (`rapidfuzz`) for closest taxonomy matches
* Handles synonyms and generic terms intelligently
* Flags ambiguous or low-confidence matches
* Outputs a standardized CSV with detailed match info

---

## 🧠 Methodology

1. **Inputs**

   * `nucc_taxonomy_master.csv` – official NUCC taxonomy data
   * `input_specialties.csv` – list of raw specialties to standardize
   * `synonyms.csv` – curated synonym list

2. **Processing**

   * Text normalization (lowercasing, removing punctuation and stopwords)
   * Fuzzy string matching using `rapidfuzz`
   * Tie-breaking for close similarity scores
   * Configurable cutoff thresholds

3. **Outputs**

   * `output.csv` containing:

     * Input specialty
     * Matched NUCC taxonomy
     * Confidence score
     * Match status (Exact / Fuzzy / Ambiguous / Unmatched)

---

## 💻 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/thesuneer7/Hilabs.git
cd Hilabs
```

### Step 2: Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # for Mac/Linux
# OR
venv\Scripts\activate        # for Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Basic command

```bash
python standardize.py \
  --nucc nucc_taxonomy_master.csv \
  --input input_specialties.csv \
  --synonyms synonyms.csv \
  --out output.csv
```

### Optional parameters

```bash
--fuzzy-cutoff <float>    # Minimum similarity threshold (default: 0.60)
--tie-window <float>      # Score range for ties (default: 3)
```

### Example

```bash
python standardize.py \
  --nucc nucc_taxonomy_master.csv \
  --input input_specialties.csv \
  --synonyms synonyms.csv \
  --out output.csv \
  --fuzzy-cutoff 0.7 \
  --tie-window 3
```

---

## 📁 File Descriptions

| File                       | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `standardize.py`           | Main standardization script                    |
| `synonyms.csv`             | List of known specialty synonyms               |
| `generic_terms.json`       | Generic terms to exclude from matching         |
| `nucc_taxonomy_master.csv` | NUCC taxonomy reference file                   |
| `input_specialties.csv`    | Raw input specialties                          |
| `output.csv`               | Output file with standardized results          |
| `problem_cases.csv`        | Logged ambiguous or unmatched cases (optional) |

---

## 🧪 Example Output

| Input Specialty     | Standardized NUCC Term                    | Confidence | Status    |
| ------------------- | ----------------------------------------- | ---------- | --------- |
| cardiologist        | Internal Medicine: Cardiovascular Disease | 0.99       | Exact     |
| skin clinic         | Dermatology                               | 0.87       | Fuzzy     |
| ortho surgeon       | Orthopedic Surgery                        | 0.92       | Fuzzy     |
| alternative healing | —                                         | —          | Unmatched |

---

## 👩‍💻 Contributors

**Vasudharaje Srivastava**
**Suneet Maharana**
**Aditya Mishra**
Team HiLabs Hackathon 2025

---

## 📜 License

This project was developed as part of the **HiLabs Hackathon 2025: Specialty Standardization Challenge**. All rights reserved by the organizing team and contributors.

---

## 🎯 Short repo description (one-line)

Standardize free-text provider specialties to NUCC taxonomy using normalization, synonyms, and fuzzy matching.

