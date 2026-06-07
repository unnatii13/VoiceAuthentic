"""
predict.py  –  Phase 2
Loads a .wav file, extracts features, runs the saved model, returns prediction.
"""

import os
import numpy as np
import librosa
import joblib
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join("backend", "detection", "models")

def load_model():
    model         = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True, duration=5.0)
    features = {}

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, val in enumerate(np.mean(mfccs, axis=1)):
        features[f"mfcc_{i+1}"] = float(val)
    for i, val in enumerate(np.std(mfccs, axis=1)):
        features[f"mfcc_std_{i+1}"] = float(val)

    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"]  = float(np.std(rms))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = float(np.mean(contrast))
    features["spectral_contrast_std"]  = float(np.std(contrast))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"]  = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"]  = float(np.std(rolloff))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    features["pitch_variability"] = float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    return features


def predict_voice(file_path):
    model, scaler, feature_names = load_model()
    features = extract_features(file_path)

    # Align feature order with training
    X = np.array([features[f] for f in feature_names]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred       = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)), 4)
    label      = "FAKE" if pred == 1 else "REAL"

    return {
        "prediction": label,
        "confidence": confidence
    }


if __name__ == "__main__":
    # Quick test on one file from the dataset
    test_file = os.path.join("dataset", "real", "LJ001-0001.wav")
    if os.path.exists(test_file):
        result = predict_voice(test_file)
        print(f"File     : {test_file}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}")
    else:
        print("[ERROR] Test file not found. Update test_file path.")