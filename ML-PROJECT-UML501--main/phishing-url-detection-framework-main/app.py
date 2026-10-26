"""Interactive Phishing URL Detection Dashboard (Streamlit Frontend).

Designed for UML501 Machine Learning Project Demonstration.
Provides:
- Real-time single URL scanning with risk gauge & explainable AI triggers
- Batch URL preprocessing & classification
- Model performance benchmarks (LightGBM, Random Forest, XGBoost, CNN-BiLSTM)
- Full 74-feature inspection & preprocessing pipeline walkthrough
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    st = None

from src.models.train_and_predict import predict_url, train_models
from src.features.extractor import extract_all_features

# Check if model bundle exists, else auto-train
BUNDLE_PATH = "artifacts/phishing_detector.joblib"
if not os.path.exists(BUNDLE_PATH):
    print("[*] Pre-training model bundle for dashboard...")
    train_models()


def main():
    if st is None:
        return

    st.set_page_config(
        page_title="Phishing URL Detector | UML501 Project",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .safe-box {
            background-color: #ecfdf5;
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 20px;
            color: #065f46;
        }
        .phish-box {
            background-color: #fef2f2;
            border: 2px solid #ef4444;
            border-radius: 12px;
            padding: 20px;
            color: #991b1b;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=70)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🔍 Real-Time URL Scanner", "📁 Batch Preprocessing & Audit", "📊 Model Benchmark Comparison", "🧠 Preprocessing & Architecture"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Course**: UML501 - Machine Learning")
    st.sidebar.markdown("**Engine**: 74 Engineered Features")
    st.sidebar.markdown("**Best Model**: Stacking AllBase (98.85% Acc)")

    if page == "🔍 Real-Time URL Scanner":
        render_scanner()
    elif page == "📁 Batch Preprocessing & Audit":
        render_batch()
    elif page == "📊 Model Benchmark Comparison":
        render_benchmarks()
    elif page == "🧠 Preprocessing & Architecture":
        render_architecture()


def render_scanner():
    st.markdown('<div class="main-header">🛡️ Explainable Phishing URL Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live URL feature extraction, preprocessing, and machine learning inference pipeline.</div>', unsafe_allow_html=True)

    # Preset Quick-Pick buttons
    st.markdown("**Try a sample URL:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    sample_url = ""
    if col1.button("✅ Google (Legit)"):
        sample_url = "https://www.google.com/search?q=machine+learning"
    if col2.button("✅ GitHub (Legit)"):
        sample_url = "https://github.com/torvalds/linux/commit/master"
    if col3.button("⚠️ Fake PayPal (Phish)"):
        sample_url = "http://paypal-security-account-update.xyz/login.php?cmd=verify"
    if col4.button("⚠️ Fake Banking (Phish)"):
        sample_url = "http://192.168.1.105/bank-of-america/secure-login.html"
    if col5.button("⚠️ Fake Apple ID (Phish)"):
        sample_url = "http://appleid-support-icloud-security.top/auth/verify?token=ab823"

    url_input = st.text_input("Enter URL to analyze:", value=sample_url, placeholder="e.g. http://secure-login.chase-bank.xyz/update")
    
    col_btn, col_opt = st.columns([1, 3])
    with col_btn:
        scan_pressed = st.button("🚀 Analyze URL", type="primary", use_container_width=True)
    with col_opt:
        enable_network = st.checkbox("Perform live DNS/WHOIS queries (requires internet & may be slower)", value=False)

    if scan_pressed and url_input:
        with st.spinner("Extracting 74 lexical, DNS, and WHOIS features..."):
            result = predict_url(url_input, enable_network=enable_network)

        st.markdown("---")

        # Result Display
        c1, c2 = st.columns([2, 1])
        with c1:
            if result["is_phishing"]:
                st.markdown(f"""
                <div class="phish-box">
                    <h2 style="margin:0; color:#dc2626;">🚨 WARNING: PHISHING URL DETECTED</h2>
                    <p style="font-size:1.1rem; margin-top:8px;">Risk Probability: <b>{result['phishing_probability']}%</b> | Severity: <b>{result['risk_level']}</b></p>
                    <p style="font-family:monospace; background:rgba(0,0,0,0.05); padding:8px; border-radius:6px;">{result['url']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box">
                    <h2 style="margin:0; color:#16a34a;">✅ VERIFIED LEGITIMATE URL</h2>
                    <p style="font-size:1.1rem; margin-top:8px;">Safety Confidence: <b>{result['legitimate_probability']}%</b> | Severity: <b>{result['risk_level']}</b></p>
                    <p style="font-family:monospace; background:rgba(0,0,0,0.05); padding:8px; border-radius:6px;">{result['url']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🔍 Explainable AI (XAI) Decision Factors:")
            for reason in result["reasons"]:
                st.write(f"- ⚠️ **{reason}**" if result["is_phishing"] else f"- 🛡️ **{reason}**")

        with c2:
            st.metric(label="Phishing Probability", value=f"{result['phishing_probability']}%")
            st.metric(label="Legitimate Probability", value=f"{result['legitimate_probability']}%")
            st.metric(label="Risk Rating", value=result["risk_level"])

        # Feature Breakdown
        st.markdown("---")
        with st.expander("🔬 View Complete 74 Extracted Feature Values", expanded=False):
            f_df = pd.DataFrame(list(result["features"].items()), columns=["Feature Name", "Extracted Value"])
            
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.markdown("**Lexical Structural Features (Length, Characters, Entropy):**")
                st.dataframe(f_df.iloc[:35], use_container_width=True)
            with c_f2:
                st.markdown("**Security Flags, DNS & WHOIS Features:**")
                st.dataframe(f_df.iloc[35:], use_container_width=True)


def render_batch():
    st.markdown('<div class="main-header">📁 Batch URL Preprocessing & Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a list of URLs to extract 74 features, preprocess missing values, and run batch predictions.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV containing a 'url' column:", type=["csv"])

    # Provide option to load the project sample dataset
    if st.button("📂 Load Built-In Sample Dataset (data/sample_dataset.csv)"):
        uploaded_file = "data/sample_dataset.csv"

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"**Loaded {len(df)} URLs from input dataset:**")
        st.dataframe(df.head(10), use_container_width=True)

        if "url" not in df.columns:
            st.error("Uploaded CSV must have a 'url' column!")
            return

        if st.button("⚡ Run Full Preprocessing & Prediction Pipeline", type="primary"):
            progress_bar = st.progress(0)
            predictions = []
            probs = []
            risk_levels = []

            urls = df["url"].tolist()
            total = len(urls)

            for i, u in enumerate(urls):
                res = predict_url(str(u), enable_network=False)
                predictions.append(res["prediction_label"])
                probs.append(res["phishing_probability"])
                risk_levels.append(res["risk_level"])
                progress_bar.progress((i + 1) / total)

            df["predicted_label"] = predictions
            df["phishing_risk_percent"] = probs
            df["risk_level"] = risk_levels

            st.success(f"Completed analysis on {total} URLs!")
            
            # Summary Metrics
            s1, s2, s3 = st.columns(3)
            phish_count = sum(df["predicted_label"] == "PHISHING")
            legit_count = total - phish_count
            s1.metric("Total Scanned", total)
            s2.metric("Flagged as Phishing", f"{phish_count} ({(phish_count/total)*100:.1f}%)")
            s3.metric("Verified Legitimate", f"{legit_count} ({(legit_count/total)*100:.1f}%)")

            st.dataframe(df[["url", "predicted_label", "phishing_risk_percent", "risk_level"]], use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Dataset CSV",
                data=csv_data,
                file_name="phishing_predictions_scored.csv",
                mime="text/csv",
            )


def render_benchmarks():
    st.markdown('<div class="main-header">📊 Model Benchmark & Research Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparison across 579,920 URLs evaluated in the UML501 framework.</div>', unsafe_allow_html=True)

    benchmark_data = {
        "Model Architecture": [
            "Stacking Ensemble (AllBase)",
            "LightGBM",
            "Random Forest",
            "CatBoost",
            "CNN-BiLSTM (Deep Learning)",
            "XGBoost",
            "Decision Tree",
            "Logistic Regression"
        ],
        "Category": ["Ensemble", "Base ML", "Base ML", "Base ML", "Raw URL Deep Learning", "Base ML", "Base ML", "Baseline"],
        "Accuracy": [0.9885, 0.9868, 0.9851, 0.9832, 0.9841, 0.9816, 0.9749, 0.9440],
        "F1-Score": [0.9861, 0.9840, 0.9820, 0.9796, 0.9807, 0.9777, 0.9697, 0.9317],
        "ROC-AUC": [0.9985, 0.9985, 0.9983, 0.9979, 0.9986, 0.9976, 0.9738, 0.9845],
        "Test Time (ms/sample)": [0.093, 0.031, 0.035, 0.004, 0.120, 0.009, 0.003, 0.004]
    }
    b_df = pd.DataFrame(benchmark_data)
    st.dataframe(b_df.style.highlight_max(subset=["Accuracy", "F1-Score", "ROC-AUC"], color="#bbf7d0"), use_container_width=True)

    st.markdown("### 🏆 Performance Comparison Chart")
    st.bar_chart(b_df.set_index("Model Architecture")[["Accuracy", "F1-Score", "ROC-AUC"]])

    st.markdown("### 🎯 Top 10 Most Important Features (Global SHAP / Tree Importance):")
    feature_importance_data = {
        "Feature": [
            "is_suspicious_tld", "entropy_url", "url_length", "uses_https",
            "domain_age_days", "path_length", "ratio_digits", "num_dots",
            "contains_login", "ratio_special_chars"
        ],
        "Importance Score": [0.185, 0.142, 0.128, 0.105, 0.094, 0.082, 0.076, 0.068, 0.061, 0.059]
    }
    fi_df = pd.DataFrame(feature_importance_data).set_index("Feature")
    st.bar_chart(fi_df)


def render_architecture():
    st.markdown('<div class="main-header">🧠 Preprocessing Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed walkthrough of how data is cleaned, extracted, imputed, and scaled for UML501.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Data Ingestion & Deduplication
    - **Raw Data**: 579,920 URLs (339,074 Legitimate ~58.5%, 240,846 Phishing ~41.5%).
    - **Cleaning**: Deduplication based on normalized URL, removing empty/invalid records, casting binary target `label` to `{0, 1}`.

    ### 2. Feature Extraction (74 Features)
    - **53 Lexical Features**: Lengths (URL, domain, hostname, path), character frequencies (dots, hyphens, `@`, `%`), Shannon Entropy of strings, sensitive keyword matching (`login`, `secure`, `bank`).
    - **9 DNS Features**: DNS resolution status, MX records, TXT records, NS records, TTL value, IP count, private IP resolution check.
    - **12 WHOIS Features**: Registration date, expiration days, domain age, registrar validity, name servers count, whois privacy protection.

    ### 3. Preventing Data Leakage (The Golden Rule)
    - **Stratified Train-Test Split (80% / 20%)**: Performed **before** any imputation or scaling.
    - Fitting transformers exclusively on `X_train`, then applying `.transform()` to `X_test`.

    ### 4. Dual Preprocessing Streams
    - **Tree Ensembles (Random Forest, LightGBM, XGBoost, CatBoost)**:
      - Uses `SimpleImputer(strategy='median')` to handle missing WHOIS/DNS timeout values.
      - **No scaling required**: Tree algorithms make split decisions based on feature order, making them invariant to monotonic scaling.
    - **Linear & Distance Models (Logistic Regression, SVM, KNN)**:
      - Uses `SimpleImputer(strategy='median')` followed by `StandardScaler()` (centering mean to 0, unit variance).
    - **Deep Learning Stream (Character CNN & BiLSTM)**:
      - Operates directly on raw URL strings with character-level tokenization (`MAX_LEN=250`, vocabulary of 120 characters, zero post-padding).
    """)


if __name__ == "__main__":
    main()
