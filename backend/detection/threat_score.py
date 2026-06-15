"""
threat_score.py  –  Phase 2
Converts model confidence into a cybersecurity threat score + explainability.
"""

import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")


def calculate_threat_score(prediction: str, confidence: float) -> dict:
    """
    Converts prediction + confidence into a 0-100 threat score.
    Higher score = more dangerous / more likely synthetic.
    """
    if prediction == "FAKE":
        if confidence >= 0.90:
            score    = 95
            severity = "CRITICAL"
        elif confidence >= 0.75:
            score    = 80
            severity = "HIGH"
        elif confidence >= 0.60:
            score    = 60
            severity = "MEDIUM"
        else:
            score    = 40
            severity = "LOW"
    else:  # REAL
        if confidence >= 0.90:
            score    = 5
            severity = "SAFE"
        elif confidence >= 0.75:
            score    = 15
            severity = "SAFE"
        else:
            score    = 30
            severity = "LOW"

    return {
        "threat_score": score,
        "severity":     severity
    }


def generate_explainability(file_path: str, prediction: str) -> dict:
    """
    Analyses audio features and generates human-readable reasons
    explaining why the voice was flagged as real or fake.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=5.0)
    except Exception:
        return {"reasons": ["Could not analyse audio file."]}

    reasons = []

    # ── Pitch variability ─────────────────────────────────────────────────────
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_std  = float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    if prediction == "FAKE":
        if pitch_std < 200:
            reasons.append("Low pitch variability — synthetic voices lack natural pitch fluctuation")
    else:
        if pitch_std >= 200:
            reasons.append("Natural pitch variability detected — consistent with human speech")

    # ── Spectral contrast ─────────────────────────────────────────────────────
    contrast     = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_std = float(np.std(contrast))

    if prediction == "FAKE":
        if contrast_std < 10:
            reasons.append("High spectral smoothness — AI voices often lack harmonic irregularities")
    else:
        if contrast_std >= 10:
            reasons.append("Rich spectral contrast — matches natural harmonic structure of human voice")

    # ── Zero crossing rate ────────────────────────────────────────────────────
    zcr      = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    if prediction == "FAKE":
        if zcr_mean > 0.1:
            reasons.append("Elevated zero crossing rate — indicates unnatural high-frequency noise")
        else:
            reasons.append("Synthetic harmonic pattern detected in frequency domain")
    else:
        reasons.append("Zero crossing rate within normal human speech range")

    # ── RMS energy ────────────────────────────────────────────────────────────
    rms      = librosa.feature.rms(y=y)
    rms_std  = float(np.std(rms))

    if prediction == "FAKE" and rms_std < 0.01:
        reasons.append("Unusually uniform energy levels — human speech has natural volume variation")

    # Fallback
    if not reasons:
        if prediction == "FAKE":
            reasons.append("Multiple audio features deviate from natural human speech patterns")
        else:
            reasons.append("Audio features are consistent with authentic human speech")

    return {"reasons": reasons}


def analyse_voice(file_path: str, prediction: str, confidence: float) -> dict:
    """
    Full threat analysis: score + severity + explainability.
    """
    threat  = calculate_threat_score(prediction, confidence)
    explain = generate_explainability(file_path, prediction)
    if prediction == "FAKE":
    threat_type = "AI Voice Clone"
else:
    threat_type = "Authentic Human Voice"

    return {
        "prediction":   prediction,
        "confidence":   confidence,
        "threat_score": threat["threat_score"],
        "severity":     threat["severity"],
        "reasons":      explain["reasons"]
    }


if __name__ == "__main__":
    # Test on a real and a fake file
    import json
    from predict import predict_voice

    for fpath in [
        os.path.join("dataset", "real", "LJ001-0001.wav"),
        os.path.join("dataset", "fake", "LJ001-0004_gen.wav"),
    ]:
        if os.path.exists(fpath):
            result     = predict_voice(fpath)
            full_report = analyse_voice(fpath, result["prediction"], result["confidence"])
            print(f"\n{'='*55}")
            print(f"File     : {fpath}")
            print(json.dumps(full_report, indent=2))