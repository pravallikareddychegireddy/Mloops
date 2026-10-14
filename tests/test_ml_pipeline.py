import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

import train_model


class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_model.train_model()

    def test_dataset_exists_and_has_expected_columns(self):
        data = pd.read_csv("student_results.csv")
        self.assertEqual(len(data), 600)
        self.assertTrue(set(train_model.FEATURES + ["result"]).issubset(data.columns))

    def test_model_and_metrics_exist(self):
        self.assertTrue(Path("student_result_model.pkl").exists())
        self.assertTrue(Path("metrics.json").exists())

    def test_accuracy_is_valid(self):
        metrics = json.loads(Path("metrics.json").read_text())
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["training_records"], 480)
        self.assertEqual(metrics["testing_records"], 120)

    def test_model_predicts_binary_label(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([{
            "attendance": 95, "internal_marks": 90,
            "assignment_marks": 90, "previous_score": 90,
        }])
        self.assertIn(int(model.predict(sample)[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
