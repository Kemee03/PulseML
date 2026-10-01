# PulseML learner guide

We will build and learn PulseML one small milestone at a time. The aim is a working system you can explain in an interview, not a pile of tools installed at once.

## Step 1 — Run the first prediction service

1. Install Python if the `py` command is not recognized.
2. Open this project folder in File Explorer.
3. Click the address bar, type `powershell`, and press Enter.
4. Follow **Start PulseML on Windows** in `README.md`, one command at a time. Use Python 3.10–3.13 so the same environment can serve both `.joblib` and KeplerVision CNN models.
5. Visit `http://127.0.0.1:8000` and click **Run a sample prediction**.
6. Open `http://127.0.0.1:8000/docs` to see the API that the dashboard calls.

To try real batch inputs, use **Predict from a CSV** on the dashboard and choose KeplerVision's `exoTest.csv`. It runs up to 1,000 rows at once. The CSV must include all 3,197 `FLUX` columns. If it has a `LABEL` column, PulseML ignores it; predictions are not automatically evaluated for correctness.

What the parts mean:

- **FastAPI** receives web requests and validates their shape.
- **AstroLens** takes 3,197 measurements and returns a class and probability.
- **SQLite** saves a record of each request on your computer.
- **The dashboard** reads the API and displays model status and request metrics.

## Roadmap

1. **Serving and local setup** — included: load AstroLens and return a prediction.
2. **Prediction history and health metrics** — included: save requests and show latency, counts, and errors.
3. **Model registry lifecycle** — included: upload a trusted `.joblib` or a `.keras` CNN with its matching scaler as a staging version, promote it to production, then roll back to the previous production version.
4. **Data and prediction drift** — included: compare active-version feature distributions with a sampled training reference using PSI, show sample counts and top features, and open alerts for high PSI.
5. **Batch CSV inference** — included: run up to 1,000 real feature rows from a CSV and log predictions under the production version.
6. **Feedback and model quality** — included: submit labels separately for a matching prediction batch and calculate accuracy, precision, recall, F1, and confusion counts.
7. **Reliability and security** — included: high error-rate and high-latency alerts on recent inference requests, plus optional API-key protection for API routes. Set `PULSEML_API_KEY` before starting the server to enable it; leaving it unset keeps local development open.
8. **Production-shaped local stack** — included: PostgreSQL database support, local Docker Compose, and a merge script to transfer existing SQLite history. Docker startup still needs local verification.
9. **Quality and delivery** — included: SQLite behavior tests, a PostgreSQL round-trip check, and a GitHub Actions workflow for Python 3.10 and 3.13. GitHub will run the workflow after this project is pushed to a repository.

We will keep the commands you need for each step in the README, and explain new folders and concepts when they arrive. You can ask “explain this file” or paste an error message any time.

## Important limitation

The dashboard's sample button sends 3,197 zeros just to exercise the request flow. It is not real telescope data and its output should not be interpreted as an astronomical result. The included small classifier was already trained in KeplerVision; PulseML does not retrain it.

Model uploads must be trusted. A `.joblib` model must accept 3,197 features; the KeplerVision `.keras` model must be uploaded with its matching scaler. Registering a version does not automatically deploy it: it first appears in staging. Promotion changes the version used for new predictions, and rollback switches back to the previous production version.

The feature drift reference summarizes KeplerVision's training data. PSI appears once there are at least 1,000 saved predictions for the currently deployed version. A high PSI identifies a distribution change to investigate; it does not prove that predictions are wrong. Use real light curves for drift checks, not the dashboard's zero-filled demo request.
