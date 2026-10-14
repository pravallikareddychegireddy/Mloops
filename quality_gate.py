"""Fail CI when model metrics do not meet the configured quality threshold."""
import json
import os
import sys
from pathlib import Path

metrics_path = Path("metrics.json")
threshold = float(os.getenv("MINIMUM_ACCURACY", "0.85"))

if not metrics_path.exists():
    print("QUALITY GATE FAILED: metrics.json is missing.")
    sys.exit(1)

try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    accuracy = float(metrics["accuracy"])
except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
    print(f"QUALITY GATE FAILED: invalid metrics: {exc}")
    sys.exit(1)

print(f"Model accuracy: {accuracy:.4f}")
print(f"Required accuracy: {threshold:.4f}")
if not 0.0 <= accuracy <= 1.0 or accuracy < threshold:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
