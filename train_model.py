"""Train a reproducible student PASS/FAIL demonstration classifier."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
MODEL_PATH = ROOT / "student_result_model.pkl"
METRICS_PATH = ROOT / "metrics.json"
DATA_PATH = ROOT / "student_results.csv"


def create_dataset(n=600, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "attendance": rng.integers(40, 101, n),
        "internal_marks": rng.integers(0, 101, n),
        "assignment_marks": rng.integers(0, 101, n),
        "previous_score": rng.integers(0, 101, n),
    })
    score = (
        0.20 * data["attendance"]
        + 0.35 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.25 * data["previous_score"]
    )
    # A deterministic teaching label, not a real institutional grading rule.
    labels = (score >= 55).astype(int).to_numpy()
    # Add reproducible label noise so the quality gate can demonstrate a
    # meaningful failure at an artificially strict 0.99 threshold.
    flip_mask = rng.random(n) < 0.05
    labels[flip_mask] = 1 - labels[flip_mask]
    data["result"] = labels
    return data


def train_model():
    data = create_dataset()
    data.to_csv(DATA_PATH, index=False)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["result"], test_size=0.20,
        random_state=42, stratify=data["result"]
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    metrics = {
        "accuracy": accuracy,
        "minimum_required_accuracy": 0.85,
        "training_records": int(len(x_train)),
        "testing_records": int(len(x_test)),
        "features": FEATURES,
        "confusion_matrix_labels": ["FAIL", "PASS"],
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["FAIL", "PASS"], output_dict=True, zero_division=0
        ),
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Dataset records: {len(data)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved: {MODEL_PATH.name}")
    print(f"Metrics saved: {METRICS_PATH.name}")
    return metrics


if __name__ == "__main__":
    train_model()
