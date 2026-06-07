"""
train.py  –  Phase 2
Trains Random Forest and XGBoost on features.csv, saves the best model.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES_CSV = os.path.join("dataset", "features.csv")
MODEL_DIR    = os.path.join("backend", "detection", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    # ── Load Data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(FEATURES_CSV)
    print(f"[INFO] Loaded {len(df)} samples from {FEATURES_CSV}")

    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    # ── Split ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Scale ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model 1: Random Forest ─────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_acc = round(rf.score(X_test_s, y_test) * 100, 2)
    print(f"[Random Forest] Accuracy: {rf_acc}%")

    # ── Model 2: XGBoost ───────────────────────────────────────────────────────
    xgb = XGBClassifier(n_estimators=100, random_state=42,
                        eval_metric="logloss", verbosity=0)
    xgb.fit(X_train_s, y_train)
    xgb_acc = round(xgb.score(X_test_s, y_test) * 100, 2)
    print(f"[XGBoost]       Accuracy: {xgb_acc}%")

    # ── Select Best ────────────────────────────────────────────────────────────
    if xgb_acc >= rf_acc:
        best_model = xgb
        best_name  = "XGBoost"
        best_acc   = xgb_acc
    else:
        best_model = rf
        best_name  = "RandomForest"
        best_acc   = rf_acc

    print(f"\n[BEST] {best_name} selected with accuracy {best_acc}%")

    # ── Save Model + Scaler ────────────────────────────────────────────────────
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"[SAVED] Model  → backend/detection/models/best_model.pkl")
    print(f"[SAVED] Scaler → backend/detection/models/scaler.pkl")


if __name__ == "__main__":
    train()