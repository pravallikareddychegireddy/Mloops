"""Flask API for the trained student result classifier."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl is missing. Run python train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
@app.get("/health")
def health_check():
    return jsonify({"status": "ok", "service": "student-result-prediction"}), 200


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    missing = [name for name in FEATURES if name not in payload]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

    values = {}
    for name in FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            return jsonify({"error": f"{name} must be numeric"}), 400
        if not 0 <= value <= 100:
            return jsonify({"error": f"{name} must be between 0 and 100"}), 400
        values[name] = value

    try:
        prediction_code = int(load_model().predict(pd.DataFrame([values], columns=FEATURES))[0])
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({
        "prediction": "PASS" if prediction_code == 1 else "FAIL",
        "prediction_code": prediction_code,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
