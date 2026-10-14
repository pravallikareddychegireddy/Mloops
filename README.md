# Student Result Prediction — MLOps Practicals 2–7

A single connected demonstration project covering model training, CI, quality gates, artifacts, a Flask prediction API, Docker, and automated Docker validation in GitHub Actions.

> **Educational demo only.** The generated dataset and target labels are synthetic. This is not a validated student assessment system and must not be used to make real academic decisions.

## Practicals covered

| Practical | Implementation |
|---|---|
| 2 — ML CI | `train_model.py`, automated tests, GitHub Actions |
| 3 — Quality gate | `quality_gate.py`, minimum accuracy 0.85 |
| 4 — Artifacts | `student-result-ml-artifacts` workflow artifact |
| 5 — Continuous Delivery | Flask API, tests, Docker-ready release artifact |
| 6 — Docker | `Dockerfile`, `.dockerignore`, local container execution |
| 7 — Docker in CI | Automated image build, health check, prediction request, logs, cleanup |

## Requirements

- Python 3.11
- Git
- Docker Desktop (Windows/macOS) or Docker Engine (Linux), for Practicals 6–7 locally
- A GitHub account for hosted CI/CD

## Run locally

### 1. Create an environment and install dependencies

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train, check quality, and run tests

```bash
python train_model.py
python quality_gate.py
python -m unittest discover -s tests -v
```

Training generates `student_results.csv`, `student_result_model.pkl`, and `metrics.json` in the project root. These are generated outputs, not source files; the model and metrics are recreated by the workflow.

The quality gate requires accuracy >= 0.85. To demonstrate a failure, run:

```bash
MINIMUM_ACCURACY=0.99 python quality_gate.py
```

In Windows PowerShell:

```powershell
$env:MINIMUM_ACCURACY = "0.99"
python quality_gate.py
Remove-Item Env:MINIMUM_ACCURACY
```

A failure exits with status 1, causing CI to stop. Restore the threshold to 0.85 for a passing run.

### 3. Run the Flask API

First ensure `student_result_model.pkl` exists by running `python train_model.py`, then:

```bash
python app.py
```

Open `http://127.0.0.1:5000/health`, or send a prediction:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance":95,"internal_marks":90,"assignment_marks":90,"previous_score":90}'
```

Expected response includes `"prediction": "PASS"`. A low-score request should return `FAIL`. Missing or invalid values return HTTP 400.

### 4. Build and run with Docker

Run these commands from the project root, after training has generated the model and metrics:

```bash
docker --version
docker info
docker build -t student-result-ml:local .
docker run --rm -p 5000:5000 --name student-result-ml student-result-ml:local
```

In another terminal:

```bash
curl --fail http://localhost:5000/health
curl --fail -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance":95,"internal_marks":90,"assignment_marks":90,"previous_score":90}'
```

Stop the container with `Ctrl+C` if running in the foreground. If you ran it detached, use `docker rm -f student-result-ml`.

## Set up GitHub Actions

1. Create a GitHub repository named `student-result-ml-ci`.
2. Add all project files and folders from this directory, preserving `.github/workflows/ml-ci.yml`.
3. Commit and push to the `main` branch.
4. Open **Actions** and select **Student Result ML CI-CD**.
5. Inspect both jobs: `train-test-package` and `docker-validation`.
6. On a successful run, download `student-result-ml-artifacts` or `student-result-ml-release` from the run's **Artifacts** section.

The workflow runs on pushes to `main`, pull requests targeting `main`, and manual `workflow_dispatch`. It does not deploy to a production server or publish the image to a registry.

## Repository structure

```text
student-result-ml-ci/
├── .github/
│   └── workflows/
│       └── ml-ci.yml
├── tests/
│   ├── test_ml_pipeline.py
│   └── test_app.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── quality_gate.py
├── requirements.txt
└── train_model.py
```

## Troubleshooting

- **`docker info` fails:** start Docker Desktop and wait for the engine to finish starting.
- **Model file missing:** run `python train_model.py` before starting the app or building the image.
- **Port 5000 already in use:** stop the other process or use a different host port, e.g. `-p 5001:5000`, then access port 5001.
- **Quality gate fails:** inspect `metrics.json`; confirm `MINIMUM_ACCURACY` is set to `0.85`.
- **GitHub Actions fails:** open the failed step's log. The Docker job prints container logs and removes the container even after a failure.
- **Windows script activation blocked:** use `python -m venv .venv`, then invoke `.venv\\Scripts\\python.exe` directly, or follow your organization's PowerShell policy.

## API contract

- `GET /` and `GET /health` → `200`, JSON service status.
- `POST /predict` with four numeric fields in the range 0–100 → `200`, prediction label and code.
- Missing fields, non-numeric values, out-of-range values, or invalid JSON → `400`.
