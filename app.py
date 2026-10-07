import csv
import math
import os
import pickle

from flask import Flask, render_template, request

# Reuse the KNN prediction function from train_model.py
from train_model import predict_additional_demand

app = Flask(__name__)

# ---------- File paths (work no matter where you run the app from) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "centres.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "demand_model.pkl")

# Dropdown options: label shown on the page -> minutes used by the model
DURATIONS = {
    "30 minutes": 30,
    "1 hour": 60,
    "2 hours": 120,
    "4 hours": 240,
}

# Rough map positions (in km) for each zone. Used to estimate the distance
# between two centres, because centres.csv has no distance column.
ZONE_POSITIONS = {
    "Central": (0, 0),
    "North": (0, 10),
    "South": (0, -10),
    "East": (10, 0),
    "West": (-10, 0),
}


def load_centres():
    """Read the centres from the CSV file as a list of dictionaries."""
    centres = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            centres.append({
                "centre_id": row["centre_id"],
                "capacity": int(row["capacity"]),
                "current_demand": int(row["current_demand"]),
                "distance":float(row["distance"]),
                "zone": row["zone"],
            })
    return centres


def load_model():
    """Load the trained KNN model saved by train_model.py (None if missing)."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def zone_distance(zone_a, zone_b):
    """Estimate the distance in km between two zones."""
    if zone_a == zone_b:
        return 3.0  # centres in the same zone are close together
    x1, y1 = ZONE_POSITIONS.get(zone_a, (0, 0))
    x2, y2 = ZONE_POSITIONS.get(zone_b, (0, 0))
    return round(math.hypot(x1 - x2, y1 - y2), 1)


def classify_risk(utilization):
    """Safe below 80%, Warning 80% to 100%, Overload above 100%."""
    if utilization > 100:
        return "Overload"
    if utilization >= 80:
        return "Warning"
    return "Safe"


# ---------------------------------------------------------------
# NEW: simple recommendation system
# ---------------------------------------------------------------
def scale(value, low, high):
    """Turn a value into a number between 0 and 1 (0 = lowest, 1 = highest)."""
    if high == low:
        return 0.5
    return (value - low) / (high - low)


def join_reasons(parts):
    """Turn ['a', 'b', 'c'] into 'A, b and c.'"""
    if len(parts) == 1:
        text = parts[0]
    else:
        text = ", ".join(parts[:-1]) + " and " + parts[-1]
    return text[0].upper() + text[1:] + "."


def recommend_centres(active_rows, top_n=3):
    """
    Rank the available centres and return the best few.

    Each centre gets 3 points (each between 0 and 1):
      - lower utilization     -> better
      - more remaining space  -> better
      - shorter distance      -> better
    The final score is the average of the 3, shown out of 100.
    Centres with no spare capacity left are not recommended.
    """
    candidates = [r for r in active_rows if r["remaining_capacity"] > 0]
    if not candidates:
        return []

    utils = [r["utilization"] for r in candidates]
    spaces = [r["remaining_capacity"] for r in candidates]
    dists = [r["distance"] for r in candidates]

    # Averages, used to write the short reasons
    avg_util = sum(utils) / len(utils)
    avg_space = sum(spaces) / len(spaces)
    avg_dist = sum(dists) / len(dists)

    ranked = []
    for r in candidates:
        util_points = 1 - scale(r["utilization"], min(utils), max(utils))
        space_points = scale(r["remaining_capacity"], min(spaces), max(spaces))
        dist_points = 1 - scale(r["distance"], min(dists), max(dists))
        score = round((util_points + space_points + dist_points) / 3 * 100)

        # Build a short reason from whatever this centre does well
        reasons = []
        if r["remaining_capacity"] >= avg_space:
            reasons.append("high available capacity")
        if r["utilization"] <= avg_util:
            reasons.append("relatively low predicted utilization")
        if r["distance"] <= avg_dist:
            reasons.append("short distance from the disrupted centre")
        if not reasons:
            reasons.append("still has spare capacity after the shift, but ranks lower on the other factors")

        item = dict(r)
        item["score"] = score
        item["reason"] = join_reasons(reasons)
        ranked.append(item)

    # Highest score first, then return only the top few
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_n]


def run_simulation(centres, model, disrupted_id=None, duration_minutes=60):
    """
    Build the results for every centre.
    If disrupted_id is None (or the model is missing), nothing is disrupted.
    """
    # Find the disrupted centre (if any)
    disrupted = None
    for c in centres:
        if c["centre_id"] == disrupted_id:
            disrupted = c

    results = []
    for c in centres:
        row = dict(c)
        row["predicted_shift"] = 0.0
        row["distance"] = 0.0

        if disrupted is not None and c["centre_id"] == disrupted["centre_id"]:
            # 1. The selected centre is unavailable
            row["status"] = "Disrupted"
            row["expected_demand"] = 0.0
            row["utilization"] = 0.0
            row["remaining_capacity"] = 0.0
            row["risk"] = "Unavailable"
        else:
            row["status"] = "Active"

            # 2. Ask the KNN model how much extra demand this centre will get
            if disrupted is not None and model is not None:
                row["distance"] = round(abs(c["distance"] - disrupted["distance"]), 1)
                situation = {
                    "current_demand": c["current_demand"],
                    "capacity": c["capacity"],
                    "distance": row["distance"],
                    "disruption_duration": duration_minutes,
                    # We have no history column, so use current demand instead
                    "historical_demand": c["current_demand"],
                }
                row["predicted_shift"] = round(
                    predict_additional_demand(model, situation), 1
                )

            # 3. Expected demand, utilization, space left and risk
            row["expected_demand"] = round(c["current_demand"] + row["predicted_shift"], 1)
            row["utilization"] = round(row["expected_demand"] / c["capacity"] * 100, 1)
            row["remaining_capacity"] = round(c["capacity"] - row["expected_demand"], 1)
            row["risk"] = classify_risk(row["utilization"])

        results.append(row)

    # Numbers for the summary cards
    active_rows = [r for r in results if r["status"] == "Active"]
    overloaded = [r for r in active_rows if r["risk"] == "Overload"]
    summary = {
        "active": len(active_rows),
        "disrupted": len(results) - len(active_rows),
        "overload_risks": len(overloaded),
        "overloaded_ids": [r["centre_id"] for r in overloaded],
        "demand_to_redistribute": disrupted["current_demand"] if disrupted else 0,
        "predicted_total": round(sum(r["predicted_shift"] for r in active_rows), 1),
    }

    # The highest-risk centre = the active centre with the highest utilization
    highest = max(active_rows, key=lambda r: r["utilization"]) if disrupted else None

    # The best 3 centres to receive redistributed demand
    recommendations = recommend_centres(active_rows) if disrupted else []

    return summary, results, highest, recommendations


@app.route("/", methods=["GET", "POST"])
def home():
    centres = load_centres()
    centre_ids = [c["centre_id"] for c in centres]

    selected_centre = None
    selected_duration = "1 hour"

    # When the user clicks "Simulate Disruption", the form is sent as POST
    if request.method == "POST":
        chosen = request.form.get("centre")
        if chosen in centre_ids:
            selected_centre = chosen
        chosen_duration = request.form.get("duration")
        if chosen_duration in DURATIONS:
            selected_duration = chosen_duration

    model = load_model()
    error = None
    if selected_centre and model is None:
        error = "Model file not found. Run  python train_model.py  first, then try again."
        selected_centre = None

    summary, results, highest, recommendations = run_simulation(
        centres, model, selected_centre, DURATIONS[selected_duration]
    )

    return render_template(
        "index.html",
        summary=summary,
        centres=results,
        highest=highest,
        recommendations=recommendations,
        error=error,
        centre_ids=centre_ids,
        durations=list(DURATIONS.keys()),
        selected_centre=selected_centre,
        selected_duration=selected_duration,
    )


if __name__ == "__main__":
    app.run(debug=True)