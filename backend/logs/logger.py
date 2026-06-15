import json
import os
from datetime import datetime

LOG_FILE = "backend/logs/incidents.json"

def log_incident(data):

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    with open(LOG_FILE, "r") as f:
        incidents = json.load(f)

    incident = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data
    }

    incidents.append(incident)

    with open(LOG_FILE, "w") as f:
        json.dump(incidents, f, indent=4)

    return incident