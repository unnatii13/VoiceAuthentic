"""
dataset_builder.py  –  Phase 2
Reads dataset/real and dataset/fake, extracts audio features, saves features.csv
"""

import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings("ignore")

REAL_DIR   = os.path.join("dataset", "real")
FAKE_DIR   = os.path.join("dataset", "fake")
OUTPUT_CSV = os.path.join("dataset", "features.csv")


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=5.0)
    except Exception as e:
        print(f"  [WARN] Could not load {file_path}: {e}")
        return None

    features = {}

    # MFCC – 13 coefficients (mean + std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, val in enumerate(np.mean(mfccs, axis=1)):
        features[f"mfcc_{i+1}"] = round(float(val), 6)
    for i, val in enumerate(np.std(mfccs, axis=1)):
        features[f"mfcc_std_{i+1}"] = round(float(val), 6)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = round(float(np.mean(zcr)), 6)
    features["zcr_std"]  = round(float(np.std(zcr)),  6)

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = round(float(np.mean(rms)), 6)
    features["rms_std"]  = round(float(np.std(rms)),  6)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = round(float(np.mean(contrast)), 6)
    features["spectral_contrast_std"]  = round(float(np.std(contrast)),  6)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = round(float(np.mean(centroid)), 6)
    features["spectral_centroid_std"]  = round(float(np.std(centroid)),  6)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = round(float(np.mean(rolloff)), 6)
    features["spectral_rolloff_std"]  = round(float(np.std(rolloff)),  6)

    # Pitch Variability
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    features["pitch_variability"] = round(float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0, 6)

    return features


def build_dataset():
    rows = []

    for label_name, label_val, folder in [("real", 0, REAL_DIR), ("fake", 1, FAKE_DIR)]:
        if not os.path.isdir(folder):
            print(f"[ERROR] Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        print(f"[INFO] Processing {len(files)} {label_name} files...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            feats = extract_features(fpath)
            if feats is not None:
                feats["filename"] = fname
                feats["label"]    = label_val
                rows.append(feats)

    if not rows:
        print("[ERROR] No features extracted. Check dataset paths.")
        return

    df   = pd.DataFrame(rows)
    cols = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    df   = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[DONE] Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"       Real: {(df.label==0).sum()}  |  Fake: {(df.label==1).sum()}")


if __name__ == "__main__":
    build_dataset()