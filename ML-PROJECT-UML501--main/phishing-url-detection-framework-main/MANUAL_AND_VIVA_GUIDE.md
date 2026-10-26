# UML501 Project Manual & Viva Examination Guide

**Project Title:** Explainable Phishing URL Detection Framework  
**Course:** UML501 – Machine Learning  
**Target Goal:** End-to-end data preprocessing, feature engineering, machine learning modeling, and real-time dashboard deployment.

---

## 1. Quick Start (How to Run Everything)

### Step 1: Open Terminal in the Project Directory
```bash
cd /Users/yashwalia/Desktop/ML-PROJECT-UML501--main/phishing-url-detection-framework-main
```

### Step 2: Install Dependencies (if not already installed)
```bash
pip install pandas scikit-learn joblib
```
*(Optional for Streamlit UI)*:
```bash
pip install streamlit
```

---

## 2. Step-by-Step Execution Guide

### Option A: Launch the Interactive Dashboard (Recommended)

#### Option A1: Run the Standalone Web Dashboard (Zero Extra Installs)
```bash
python app_web.py
```
- Opens automatically in your browser at **`http://localhost:8080`**.
- Features: Live URL security scanner, risk gauge meter, explainability triggers, benchmark table, and full 74-feature inspector!

#### Option A2: Run with Streamlit
```bash
streamlit run app.py
```
- Opens in your browser at **`http://localhost:8501`**.
- Features: Real-time scanner, Batch CSV uploader & processor, Model comparison charts, and Architecture tabs.

---

### Option B: Run Preprocessing via Command Line
To run the automated cleaning, imputation, and stratified train-test splitting on any dataset:
```bash
python -m src.features.preprocess --input data/sample_dataset.csv
```
**Output generated:**
- `data/processed/train_tree_processed.csv`: Clean, imputed training features.
- `data/processed/test_tree_processed.csv`: Clean, imputed test features.
- `data/processed/train_scaled_processed.csv`: Standardized training features for linear models.
- `data/processed/test_scaled_processed.csv`: Standardized test features.
- `artifacts/preprocessor_bundle.joblib`: Serialized imputer & scaler pipelines.
- `results/preprocessing_report.json`: Preprocessing summary & missing value audit report.

---

### Option C: Train the Models Programmatically
To retrain Random Forest & Logistic Regression on the preprocessed data:
```bash
python -m src.models.train_and_predict
```
**Output generated:**
- Trains Random Forest (Tree Ensemble) and Logistic Regression.
- Evaluates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Serializes `artifacts/phishing_detector.joblib`.
- Generates `results/model_evaluation_metrics.json`.

---

### Option D: Test Any URL from Python Code
```python
from src.models.train_and_predict import predict_url

# Test a suspicious URL
result = predict_url("http://paypal-security-account-update.xyz/login.php?cmd=verify")

print("Verdict:", result["prediction_label"])
print("Risk Score:", result["phishing_probability"], "%")
print("Reasons:", result["reasons"])
```

---

## 3. Machine Learning Concepts & Viva Questions (UML501)

When your professor or evaluator asks about this project, here are the key concepts and answers:

### Q1: What is the problem statement and objective?
> **Answer:** Phishing URL detection is a binary classification problem (`0 = Legitimate`, `1 = Phishing`). The objective is to detect malicious URLs before a victim enters credentials, combining high accuracy with Explainable AI (XAI) so users understand *why* a URL was flagged.

---

### Q2: How did you design the preprocessing pipeline?
> **Answer:** Preprocessing followed four rigorous steps:
> 1. **Data Cleaning & Deduplication:** Removed duplicate URLs, verified binary integer labels, and audited missing values.
> 2. **Preventing Data Leakage:** Split the dataset into 80% train and 20% test using **Stratified sampling** *before* fitting any statistical transformers.
> 3. **Handling Missing Values:** Handled DNS/WHOIS lookup failures (which occur when fake domains don't resolve or WHOIS servers time out) using **Median Imputation** (`SimpleImputer(strategy='median')`).
> 4. **Adaptive Feature Scaling:** 
>    - For **Tree-based models** (Random Forest, LightGBM, XGBoost): Kept unscaled after median imputation, because tree split criteria (Gini/Entropy) are invariant to monotonic scaling.
>    - For **Linear models** (Logistic Regression, SVM): Applied `StandardScaler` to bring features to zero mean and unit variance.

---

### Q3: What features did you engineer?
> **Answer:** We extracted **74 engineered features** across 3 categories:
> - **53 Lexical Features:** URL length, domain length, path depth, Shannon Entropy of the URL and hostname, character frequencies (dots, hyphens, `@`, `%`), sensitive keywords (`login`, `verify`, `bank`, `account`), and suspicious TLD flags (`.xyz`, `.top`, `.tk`).
> - **9 DNS Features:** DNS resolution success, MX records (mail servers), TXT records, TTL values, and private IP resolution checks.
> - **12 WHOIS Features:** Domain age (days), expiration days, creation year, registrar validity, and privacy protection flags.

---

### Q4: Why is Shannon Entropy important for URL detection?
> **Answer:** Legitimate URLs use readable English words or brand names (e.g., `google.com/search`), which has moderate entropy. Phishing URLs frequently use algorithmically generated domains, random hashes, or character obfuscation (e.g., `a7x99-paypa1-sec.xyz`), resulting in significantly higher entropy.

---

### Q5: How does your deep learning pipeline differ from the feature pipeline?
> **Answer:**
> - The **Feature ML pipeline** extracts 74 handcrafted lexical, DNS, and WHOIS features and feeds them into models like LightGBM or Random Forest.
> - The **Deep Learning pipeline** (Character CNN / BiLSTM) bypasses handcrafted feature engineering entirely; it operates on the raw URL string, tokenizing individual characters (`num_words=120`, `maxlen=250`) to automatically learn spatial n-gram patterns and sequential dependencies.

---

### Q6: What models performed best on this framework?
> **Answer:**
> - **Best Ensemble:** Stacking Ensemble of all base models (**98.85% Accuracy**, 0.9861 F1, 0.9985 ROC-AUC).
> - **Best Single ML Model:** LightGBM (**98.68% Accuracy**, 0.9840 F1, 0.9985 ROC-AUC).
> - **Best Deep Learning Model:** CNN-BiLSTM Seed Ensemble (**98.41% Accuracy**, 0.9807 F1, 0.9986 ROC-AUC).

---

### Q7: What is Explainable AI (XAI) in your framework?
> **Answer:** In security applications, a "black box" prediction is insufficient. Our framework integrates SHAP (SHapley Additive exPlanations) and permutation feature importance. This reveals which specific features (e.g., presence of an IP address, high entropy, or suspicious TLD) drove the model's prediction for that individual URL.
