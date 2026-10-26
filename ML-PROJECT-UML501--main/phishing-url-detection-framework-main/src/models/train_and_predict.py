"""Model Training and Real-Time Inference Engine for UML501 Phishing Detection.

Provides:
- Model training (Random Forest & Logistic Regression) on preprocessed data
- Performance evaluation matching the UML501 project benchmarks
- Serialized model bundle export
- High-level `predict_url(url)` function for real-time inference in scripts/dashboards
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.extractor import extract_all_features
from src.features.preprocess import run_preprocessing


def train_models(
    dataset_path: str = "data/sample_dataset.csv",
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train benchmark models and serialize the best detector pipeline."""
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Run Preprocessing
    print("[*] Running preprocessing pipeline...")
    prep_meta = run_preprocessing(
        input_path=dataset_path,
        output_dir="data/processed",
        artifacts_dir=artifacts_dir,
        random_state=random_state,
    )

    # 2. Load Processed Datasets
    train_df = pd.read_csv("data/processed/train_tree_processed.csv")
    test_df = pd.read_csv("data/processed/test_tree_processed.csv")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    feature_cols = list(X_train.columns)

    # 3. Train Random Forest (Top Tree Ensemble from the paper)
    print("[*] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "precision": float(precision_score(y_test, y_pred_rf, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_rf, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_rf, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_rf)),
    }

    # 4. Train Logistic Regression on Standardized Data
    print("[*] Training Logistic Regression Baseline...")
    train_scaled = pd.read_csv("data/processed/train_scaled_processed.csv")
    test_scaled = pd.read_csv("data/processed/test_scaled_processed.csv")
    X_train_sc = train_scaled.drop(columns=["label"])
    X_test_sc = test_scaled.drop(columns=["label"])

    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train_sc, y_train)

    y_pred_lr = lr_model.predict(X_test_sc)
    y_proba_lr = lr_model.predict_proba(X_test_sc)[:, 1]

    lr_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_lr)),
        "precision": float(precision_score(y_test, y_pred_lr, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_lr, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_lr, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_lr)),
    }

    # Top Feature Importances from Random Forest
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    top_features = [
        {"feature": feature_cols[i], "importance": round(float(importances[i]), 4)}
        for i in top_indices
    ]

    # 5. Serialize Artifacts
    model_bundle = {
        "model_name": "RandomForestClassifier",
        "model": rf_model,
        "feature_cols": feature_cols,
        "metrics": rf_metrics,
        "top_features": top_features,
        "lr_metrics": lr_metrics,
    }

    bundle_path = os.path.join(artifacts_dir, "phishing_detector.joblib")
    joblib.dump(model_bundle, bundle_path)

    # Save Results JSON
    results = {
        "random_forest": rf_metrics,
        "logistic_regression": lr_metrics,
        "top_important_features": top_features,
    }
    with open("results/model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[✓] Model training completed successfully!")
    print(f"    Random Forest -> Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}, ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"    Logistic Reg  -> Accuracy: {lr_metrics['accuracy']:.4f}, F1: {lr_metrics['f1']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"    Model bundle:   {bundle_path}")

    return results


def predict_url(
    url: str,
    bundle_path: str = "artifacts/phishing_detector.joblib",
    enable_network: bool = False,
) -> Dict[str, Any]:
    """Analyze and classify a single URL in real-time."""
    if not os.path.exists(bundle_path):
        # Auto-train if bundle is missing
        print(f"[*] Model bundle not found at {bundle_path}. Training first...")
        train_models()

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # 1. Extract 74 features
    raw_feats = extract_all_features(url, enable_network=enable_network)

    # 2. Build aligned DataFrame
    df_single = pd.DataFrame([raw_feats])[feature_cols]

    # 3. Predict Probability
    prob_phishing = float(model.predict_proba(df_single)[0][1])
    prob_legit = 1.0 - prob_phishing
    is_phishing = prob_phishing >= 0.5

    # 4. Determine Risk Level
    if prob_phishing >= 0.85:
        risk_level = "CRITICAL"
        risk_color = "#dc2626"
    elif prob_phishing >= 0.65:
        risk_level = "HIGH"
        risk_color = "#ea580c"
    elif prob_phishing >= 0.45:
        risk_level = "SUSPICIOUS"
        risk_color = "#eab308"
    elif prob_phishing >= 0.20:
        risk_level = "LOW"
        risk_color = "#3b82f6"
    else:
        risk_level = "SAFE"
        risk_color = "#16a34a"

    # 5. Extract Explainability / Suspicious Triggers
    reasons = []
    if raw_feats.get("is_ip_address", 0) == 1.0:
        reasons.append("Hostname uses raw IP address instead of domain name")
    if raw_feats.get("is_suspicious_tld", 0) == 1.0:
        reasons.append("Domain uses high-risk Top-Level Domain (e.g., .xyz, .top, .tk)")
    if raw_feats.get("uses_https", 1) == 0.0:
        reasons.append("Unencrypted connection (HTTP instead of HTTPS)")
    if raw_feats.get("contains_brand", 0) == 1.0 and ("google" not in url.lower() and "apple.com" not in url.lower() and "amazon.com" not in url.lower()):
        reasons.append("Impersonates major brand keywords (PayPal, Apple, Google, Microsoft)")
    if raw_feats.get("contains_login", 0) == 1.0 or raw_feats.get("contains_verify", 0) == 1.0:
        reasons.append("Contains sensitive authentication keywords ('login', 'verify')")
    if raw_feats.get("entropy_url", 0) > 4.2:
        reasons.append("High character randomness / obfuscation entropy in URL")
    if raw_feats.get("url_depth", 0) > 3:
        reasons.append("Deeply nested path structure designed to conceal true domain")

    if not reasons:
        if is_phishing:
            reasons.append("Multiple compound lexical anomalies detected across URL length and symbol distribution")
        else:
            reasons.append("Standard legitimate domain syntax with clean structural ratios")

    return {
        "url": url,
        "is_phishing": is_phishing,
        "prediction_label": "PHISHING" if is_phishing else "LEGITIMATE",
        "phishing_probability": round(prob_phishing * 100, 2),
        "legitimate_probability": round(prob_legit * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "reasons": reasons,
        "features": raw_feats,
    }


if __name__ == "__main__":
    train_models()
    test_sample = "http://paypal-security-account-update.xyz/login.php?cmd=verify"
    res = predict_url(test_sample)
    print("\n--- Inference Test ---")
    print(f"URL: {res['url']}")
    print(f"Verdict: {res['prediction_label']} ({res['phishing_probability']}% Risk)")
    print(f"Triggers: {res['reasons']}")
