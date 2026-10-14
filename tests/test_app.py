import unittest

import train_model
from app import app


class TestPredictionApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_model.train_model()

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post("/predict", json={
            "attendance": 95, "internal_marks": 90,
            "assignment_marks": 90, "previous_score": 90,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")

    def test_low_performance_prediction(self):
        response = self.client.post("/predict", json={
            "attendance": 45, "internal_marks": 20,
            "assignment_marks": 25, "previous_score": 20,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "FAIL")

    def test_missing_fields_returns_400(self):
        response = self.client.post("/predict", json={"attendance": 90})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_out_of_range_value_returns_400(self):
        response = self.client.post("/predict", json={
            "attendance": 120, "internal_marks": 80,
            "assignment_marks": 80, "previous_score": 80,
        })
        self.assertEqual(response.status_code, 400)

    def test_non_json_returns_400(self):
        response = self.client.post("/predict", data="not json")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
