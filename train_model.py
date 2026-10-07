import csv
import math
import os
import pickle
import random

# ---------- Paths (work no matter where you run the script from) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "demand_model.pkl")

# Inputs the model learns from, and the value it predicts
FEATURES = [
    "current_demand",
    "capacity",
    "distance",
    "disruption_duration",
    "historical_demand",
]
TARGET = "additional_demand"

# How many similar past situations (neighbours) to look at
K = 5


# ---------------------------------------------------------------
# STEP 1: Make a small fake "historical" dataset
# ---------------------------------------------------------------
def create_synthetic_data(rows=500):
    random.seed(42)  # same data every time you run it
    data = []

    for _ in range(rows):
        current_demand = random.randint(50, 200)          # people now
        capacity = random.randint(80, 220)                # max people
        distance = round(random.uniform(1, 30), 1)        # km to the disrupted centre
        duration = random.choice([30, 60, 120, 240])      # minutes
        historical = round(current_demand * random.uniform(0.8, 1.2))

        # Simple rule for how much extra demand was received in the past:
        #  - closer centres receive more
        #  - longer disruptions send more people
        #  - busier centres (higher historical demand) receive more
        closeness = 1 / (1 + distance / 10)
        duration_factor = 0.5 + duration / 240
        additional = historical * 0.25 * closeness * duration_factor
        additional = additional + random.gauss(0, 2)      # some noise
        additional = round(max(additional, 0), 1)

        data.append({
            "current_demand": current_demand,
            "capacity": capacity,
            "distance": distance,
            "disruption_duration": duration,
            "historical_demand": historical,
            "additional_demand": additional,
        })
    return data


# ---------------------------------------------------------------
# STEP 2: Helper functions for KNN
# ---------------------------------------------------------------
def scale_row(values, mins, maxs):
    """Squeeze every value into the range 0 to 1, so no feature dominates."""
    scaled = []
    for i in range(len(values)):
        spread = maxs[i] - mins[i]
        if spread == 0:
            scaled.append(0.0)
        else:
            scaled.append((values[i] - mins[i]) / spread)
    return scaled


def euclidean_distance(a, b):
    """Straight-line distance between two lists of numbers."""
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return math.sqrt(total)


def predict_additional_demand(model, situation):
    """
    KNN prediction.
    'situation' is a dictionary with the 5 feature values.
    """
    # 1. Turn the new situation into scaled numbers
    values = [situation[name] for name in model["features"]]
    scaled = scale_row(values, model["mins"], model["maxs"])

    # 2. Measure the distance to EVERY past example
    distances = []
    for row, target in zip(model["X"], model["y"]):
        distances.append((euclidean_distance(scaled, row), target))

    # 3. Sort so the most similar past examples come first
    distances.sort(key=lambda pair: pair[0])

    # 4. Take the K closest and average their additional demand
    nearest = distances[: model["k"]]
    return sum(target for _, target in nearest) / len(nearest)


# ---------------------------------------------------------------
# STEP 3: Train, test, and save
# ---------------------------------------------------------------
def main():
    # Create the data and save it as a CSV (you can open it in Excel)
    data = create_synthetic_data()
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES + [TARGET])
        writer.writeheader()
        writer.writerows(data)
    print(f"Created {len(data)} rows of training data -> {DATA_PATH}")

    # Split: 80% for training, 20% for testing
    split = int(len(data) * 0.8)
    train_rows = data[:split]
    test_rows = data[split:]

    # Find the smallest and largest value of each feature (for scaling)
    mins, maxs = [], []
    for name in FEATURES:
        column = [row[name] for row in train_rows]
        mins.append(min(column))
        maxs.append(max(column))

    # "Training" KNN = just remembering the scaled examples
    model = {
        "k": K,
        "features": FEATURES,
        "mins": mins,
        "maxs": maxs,
        "X": [scale_row([r[n] for n in FEATURES], mins, maxs) for r in train_rows],
        "y": [r[TARGET] for r in train_rows],
    }

    # Test the model on rows it has never seen
    errors = []
    squared_errors = []
    actual_values = []
    for row in test_rows:
        predicted = predict_additional_demand(model, row)
        actual = row[TARGET]
        errors.append(abs(predicted - actual))
        squared_errors.append((predicted - actual) ** 2)
        actual_values.append(actual)

    mae = sum(errors) / len(errors)
    mean_actual = sum(actual_values) / len(actual_values)
    ss_total = sum((a - mean_actual) ** 2 for a in actual_values)
    r2 = 1 - sum(squared_errors) / ss_total

    print(f"Average error (MAE): {mae:.2f}")
    print(f"R2 score: {r2:.2f}  (closer to 1 is better)")

    # Save the model with pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {MODEL_PATH}")

    # Quick test: load the saved model and make one prediction
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)

    sample = {
        "current_demand": 120,
        "capacity": 150,
        "distance": 5,
        "disruption_duration": 120,
        "historical_demand": 115,
    }
    result = predict_additional_demand(loaded_model, sample)
    print(f"\nTest prediction: about {result:.1f} extra people expected")


if __name__ == "__main__":
    main()