"""
evaluation.py  –  Phase 2
Evaluates the saved model and prints a full metrics report.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

FEATURES_CSV = os.path.join("dataset", "features.csv")
MODEL_DIR    = os.path.join("backend", "detection", "models")


def evaluate():
    df     = pd.read_csv(FEATURES_CSV)
    X      = df.drop(columns=["filename", "label"])
    y      = df["label"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

    X_test_s = scaler.transform(X_test)
    y_pred   = model.predict(X_test_s)

    print("\n" + "="*45)
    print("       MODEL EVALUATION REPORT")
    print("="*45)
    print(f"  Accuracy  : {round(accuracy_score(y_test, y_pred)*100, 2)}%")
    print(f"  Precision : {round(precision_score(y_test, y_pred, zero_division=0)*100, 2)}%")
    print(f"  Recall    : {round(recall_score(y_test, y_pred, zero_division=0)*100, 2)}%")
    print(f"  F1 Score  : {round(f1_score(y_test, y_pred, zero_division=0)*100, 2)}%")
    print("="*45)
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["REAL", "FAKE"],
                                zero_division=0))


if __name__ == "__main__":
    evaluate()