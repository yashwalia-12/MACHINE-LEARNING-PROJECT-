# Project Transformation Report: Before vs. After

**Course:** UML501 – Machine Learning  
**Project:** Explainable Phishing URL Detection Framework  

---

## 1. High-Level Summary of Changes

| Component | BEFORE (Original State) | AFTER (Updated State) |
| :--- | :--- | :--- |
| **Preprocessing Architecture** | Only ad-hoc exploratory code scattered inside Jupyter Notebooks (`03_dataset_preparation.ipynb`). Hardcoded Windows paths (`C:\Users\elifo\...`). | Clean, modular Python preprocessing pipeline in `src/features/preprocess.py` that can be run from CLI or imported anywhere. |
| **Feature Extraction Engine** | Feature extraction functions were tightly coupled to notebooks and required scraping libraries. | Real-time 74-feature extraction engine (`src/features/extractor.py`) extracting 53 Lexical, 9 DNS, and 12 WHOIS parameters with safe timeouts and fallbacks. |
| **Dataset Availability** | Repo contained **no data files** (`final_dataset.csv` had to be manually searched on Kaggle). | Included curated, verified dataset (`data/sample_dataset.csv`) with 60 representative URLs and 74 extracted features for instant experimentation. |
| **Missing Value Handling** | Inconsistent null handling across notebooks. | Formalized two-tier strategy: `SimpleImputer(strategy='median')` for tree models, and median + `StandardScaler()` for linear/distance models. |
| **Data Leakage Safeguards** | Ad-hoc splits; risk of calculating feature statistics on the full dataset before splitting. | Guaranteed leakage-free pipeline: Stratified 80/20 train/test split performed **strictly before** fitting any transformer. |
| **Model Training & Inference** | Only ran in notebooks without an accessible programmatic API for predicting new URLs. | Created `src/models/train_and_predict.py` with one-line inference function `predict_url(url)` that returns verdict, risk percentage, and XAI triggers. |
| **Frontend / Dashboard** | **No frontend or user interface existed** in the repository. | Built **two** complete dashboards: <br>1. Interactive Streamlit app (`app.py`)<br>2. Zero-dependency standalone Web Dashboard (`app_web.py`). |
| **Documentation & Viva Prep** | Only research paper citation and basic setup instructions. | Comprehensive step-by-step user manual (`MANUAL_AND_VIVA_GUIDE.md`) explaining commands, architecture, and viva questions. |

---

## 2. File-by-File Breakdown of Additions & Modifications

### 1. `src/features/extractor.py` (New Core Engine)
- **What it does:** Extracts all 74 features on demand from any raw URL string.
- **Why it was needed:** The original repo had no way to analyze a new, unseen URL in real time.
- **Key Features Extracted:**
  - *Lexical (53)*: URL length, domain length, path segments count, Shannon Entropy (`entropy_url`, `entropy_hostname`), digit ratios, sensitive keywords (`login`, `verify`, `bank`, `update`), suspicious TLD detection, IP address detection.
  - *DNS (9)*: MX, TXT, NS records, TTL, IP count, private IP checks.
  - *WHOIS (12)*: Domain age, expiration days, creation year, registrar validity, whois privacy flags.

### 2. `src/features/preprocess.py` (New Preprocessing Module)
- **What it does:** Automates the complete data preparation workflow:
  - Deduplication and label validation.
  - Stratified 80/20 train-test splitting.
  - Median imputation for DNS/WHOIS timeout values.
  - StandardScaler normalization for linear models.
  - Exports clean CSV files (`train_tree_processed.csv`, `test_tree_processed.csv`, `train_scaled_processed.csv`) and serializes `artifacts/preprocessor_bundle.joblib`.
  - Produces an automated audit report in `results/preprocessing_report.json`.

### 3. `data/sample_dataset.csv` (New Built-In Dataset)
- Contains 60 diverse, representative URLs (30 Legitimate, 30 Phishing) across banks, tech giants, government portals, and active phishing vectors (`.xyz`, `.top`, IP-based domains).
- Enables immediate execution and verification without needing to download massive external archives.

### 4. `src/models/train_and_predict.py` (New Training & Inference Pipeline)
- Trains Random Forest and Logistic Regression on preprocessed data.
- Evaluates metrics (Accuracy, F1, ROC-AUC) matching research benchmarks.
- Saves model bundle to `artifacts/phishing_detector.joblib`.
- Exposes `predict_url(url)` with explainability reasoning (XAI factors).

### 5. `app.py` & `app_web.py` (New Frontends)
- **`app.py` (Streamlit)**: Live URL scanner, batch CSV processor, benchmark comparison table, and preprocessing architecture explorer.
- **`app_web.py` (Standalone Web UI)**: Runs on Python's built-in HTTP server with TailwindCSS. Zero extra libraries required.

---

## 3. What Remained Preserved
- All original research notebooks (`notebooks/data_collection/`, `notebooks/machine_learning/`, `notebooks/deep_learning/`) remain intact.
- The 74-feature schema and column naming conventions were preserved exactly so that models trained here remain 100% compatible with the published paper benchmarks.
