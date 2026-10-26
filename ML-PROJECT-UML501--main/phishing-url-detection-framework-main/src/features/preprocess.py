"""End-to-End Preprocessing Pipeline for UML501 Phishing URL Detection.

Handles:
- Data validation, deduplication, and sanity checking
- Missing value imputation (median strategy for DNS/WHOIS lookups)
- Stratified 80/20 train/test splitting (Strictly avoiding data leakage)
- Feature scaling for linear/distance models vs unscaled median-imputed for tree ensembles
- Exporting clean processed datasets and serializing preprocessor artifacts
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_preprocessing(
    input_path: str,
    output_dir: str = "data/processed",
    artifacts_dir: str = "artifacts",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, any]:
    """Execute the full preprocessing pipeline on a dataset CSV."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"[*] Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    print(f"    Loaded shape: {initial_shape}")

    # 1. Cleaning & Deduplication
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    if "label" not in df.columns:
        raise ValueError("Missing mandatory target column: 'label'")

    # Drop rows with missing labels and ensure binary int
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 2. Separate Features, URL, and Target
    target_col = "label"
    url_col = "url" if "url" in df.columns else None

    cols_to_drop = [target_col] + ([url_col] if url_col else [])
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Convert non-numeric or boolean features to numeric
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype == bool:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 3. Missing Value Audit Before Imputation
    missing_counts = X.isnull().sum()
    total_missing = int(missing_counts.sum())
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()

    print(f"[*] Missing value audit: {total_missing} missing values across {len(cols_with_missing)} columns.")

    # 4. Train-Test Split (BEFORE imputing/scaling to prevent data leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[*] Split complete: Train={X_train_raw.shape[0]} rows, Test={X_test_raw.shape[0]} rows (stratified)")

    # 5. Fit Preprocessing Transformers
    # Pipeline for Tree models (RandomForest, LightGBM, XGBoost, CatBoost)
    tree_imputer = SimpleImputer(strategy="median")
    tree_imputer.fit(X_train_raw)

    X_train_tree = pd.DataFrame(
        tree_imputer.transform(X_train_raw),
        columns=feature_cols,
        index=X_train_raw.index
    )
    X_test_tree = pd.DataFrame(
        tree_imputer.transform(X_test_raw),
        columns=feature_cols,
        index=X_test_raw.index
    )

    # Pipeline for Linear models (Logistic Regression) & Neural Nets
    scaler = StandardScaler()
    scaler.fit(X_train_tree)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_tree),
        columns=feature_cols,
        index=X_train_raw.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_tree),
        columns=feature_cols,
        index=X_test_raw.index
    )

    # 6. Save Processed Datasets
    # Tree format (imputed numeric)
    train_tree_df = X_train_tree.copy()
    train_tree_df["label"] = y_train
    test_tree_df = X_test_tree.copy()
    test_tree_df["label"] = y_test

    train_out_path = os.path.join(output_dir, "train_tree_processed.csv")
    test_out_path = os.path.join(output_dir, "test_tree_processed.csv")
    train_tree_df.to_csv(train_out_path, index=False)
    test_tree_df.to_csv(test_out_path, index=False)

    # Scaled format (imputed + standardized)
    train_scaled_df = X_train_scaled.copy()
    train_scaled_df["label"] = y_train
    test_scaled_df = X_test_scaled.copy()
    test_scaled_df["label"] = y_test

    train_scaled_path = os.path.join(output_dir, "train_scaled_processed.csv")
    test_scaled_path = os.path.join(output_dir, "test_scaled_processed.csv")
    train_scaled_df.to_csv(train_scaled_path, index=False)
    test_scaled_df.to_csv(test_scaled_path, index=False)

    # 7. Serialize Artifacts
    preprocessor_bundle = {
        "feature_cols": feature_cols,
        "imputer": tree_imputer,
        "scaler": scaler,
        "random_state": random_state,
        "test_size": test_size,
    }
    artifact_path = os.path.join(artifacts_dir, "preprocessor_bundle.joblib")
    joblib.dump(preprocessor_bundle, artifact_path)

    # 8. Compile Preprocessing Summary Report
    summary = {
        "status": "success",
        "initial_rows": initial_shape[0],
        "initial_cols": initial_shape[1],
        "feature_count": len(feature_cols),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "class_balance_train": {
            "legitimate (0)": int((y_train == 0).sum()),
            "phishing (1)": int((y_train == 1).sum()),
            "phishing_ratio": float((y_train == 1).mean()),
        },
        "total_missing_imputed": total_missing,
        "columns_imputed_count": len(cols_with_missing),
        "saved_files": {
            "train_tree": train_out_path,
            "test_tree": test_out_path,
            "train_scaled": train_scaled_path,
            "test_scaled": test_scaled_path,
            "preprocessor_bundle": artifact_path,
        }
    }

    report_path = "results/preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"[✓] Preprocessing finished successfully!")
    print(f"    Train processed: {train_out_path} ({len(y_train)} rows)")
    print(f"    Test processed:  {test_out_path} ({len(y_test)} rows)")
    print(f"    Bundle saved:    {artifact_path}")
    print(f"    Report saved:    {report_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess phishing URL dataset")
    parser.add_argument("--input", default="data/sample_dataset.csv", help="Path to input CSV")
    args = parser.parse_args()

    run_preprocessing(args.input)
