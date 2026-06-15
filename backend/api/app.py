from fastapi import FastAPI, UploadFile, File
import shutil
import os
import json

from backend.detection.threat_score import analyse_voice
from backend.detection.predict import predict_voice
from backend.logs.logger import log_incident

app = FastAPI(
    title="VoiceAuthentic API",
    version="1.0"
)

UPLOAD_DIR = "backend/uploads"

@app.get("/")
def home():
    return {
        "message": "VoiceAuthentic Backend Running"
    }
@app.get("/incidents")
def get_incidents():

    with open("backend/logs/incidents.json", "r") as f:
        data = json.load(f)

    return data
@app.get("/dashboard")
def dashboard():

    with open("backend/logs/incidents.json", "r") as f:
        incidents = json.load(f)

    total = len(incidents)

    fake_count = len([
        i for i in incidents
        if i["prediction"] == "FAKE"
    ])

    real_count = len([
        i for i in incidents
        if i["prediction"] == "REAL"
    ])

    return {
        "total_scans": total,
        "deepfakes_detected": fake_count,
        "real_voices": real_count
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_voice(file_path)

    report = analyse_voice(
        file_path,
        result["prediction"],
        result["confidence"]
    )
    log_incident({
    "filename": file.filename,
    "prediction": report["prediction"],
    "confidence": report["confidence"],
    "threat_score": report["threat_score"],
    "severity": report["severity"],
    "threat_type": report["threat_type"]
})

    return {
        "filename": file.filename,
        **report
    }