"""PulseML's local model registry, inference API, and monitoring endpoints."""

from __future__ import annotations

import json
import io
import hashlib
import os
import re
import secrets
import shutil
import sqlite3
import time
import warnings
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
try:
    import psycopg
except ImportError:  # SQLite-only development can run without the PostgreSQL driver.
    psycopg = None
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.drift import MINIMUM_PRODUCTION_ROWS, ReferenceProfile

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("PULSEML_DB_PATH", ROOT / "pulseml.db"))
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_PATH = Path(os.getenv("PULSEML_MODEL_PATH", ROOT / "models" / "astrolens.joblib"))
REFERENCE_PATH = Path(os.getenv("PULSEML_REFERENCE_PATH", ROOT / "data" / "kepler_reference.npz"))
FEATURE_COUNT = 3197
FEATURE_COLUMNS = tuple(f"FLUX.{index}" for index in range(1, FEATURE_COUNT + 1))
MAX_BATCH_ROWS = 1000
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
RELIABILITY_WINDOW = 100
MIN_RELIABILITY_REQUESTS = 20
ERROR_RATE_ALERT_PERCENT = 5.0
LATENCY_ALERT_MS = 500.0
MODEL_ID = "astrolens"
MODEL_VERSION = "1.0.0"
LABELS = {0: "no_planet", 1: "planet"}
_artifacts: dict[tuple[str, str], object] = {}
_reference_profile: ReferenceProfile | None = None


class Record(dict):
    """Row mapping that supports both named and positional access."""

    def __getitem__(self, key):
        if isinstance(key, int):
            key = tuple(self.keys())[key]
        return super().__getitem__(key)


def hybrid_row(cursor):
    if cursor.description is None:
        return lambda values: values
    names = [column.name for column in cursor.description]
    def make_row(values):
        return Record(zip(names, values))
    return make_row


class DatabaseConnection:
    """Small adapter for the SQLite and PostgreSQL query styles used here."""

    def __init__(self, connection, dialect: str):
        self.connection = connection
        self.dialect = dialect

    def _sql(self, sql: str) -> str:
        if self.dialect == "postgresql":
            sql = sql.replace("?", "%s")
            if sql.lstrip().upper().startswith("INSERT OR IGNORE INTO"):
                sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO", 1).rstrip().rstrip(";")
                sql += " ON CONFLICT DO NOTHING"
        return sql

    def execute(self, sql: str, parameters=()):
        return self.connection.execute(self._sql(sql), parameters)

    def executemany(self, sql: str, rows):
        cursor = self.connection.cursor()
        cursor.executemany(self._sql(sql), rows)
        return cursor

    def executescript(self, sql: str):
        if self.dialect == "postgresql":
            sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
            sql = sql.replace("probability REAL", "probability DOUBLE PRECISION")
            sql = sql.replace("latency_ms REAL", "latency_ms DOUBLE PRECISION")
            for statement in sql.split(";"):
                if statement.strip():
                    self.execute(statement)
        else:
            self.connection.executescript(sql)

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


@contextmanager
def connect():
    """Open a configured database connection for one operation and close it."""
    if DATABASE_URL:
        if psycopg is None:
            raise RuntimeError("Install the 'psycopg[binary]' package to use PostgreSQL.")
        raw_connection = psycopg.connect(DATABASE_URL, row_factory=hybrid_row)
        connection = DatabaseConnection(raw_connection, "postgresql")
    else:
        raw_connection = sqlite3.connect(DB_PATH)
        raw_connection.row_factory = sqlite3.Row
        connection = DatabaseConnection(raw_connection, "sqlite")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS model_versions (
          model_id TEXT NOT NULL, version TEXT NOT NULL, name TEXT NOT NULL,
          algorithm TEXT NOT NULL, artifact_path TEXT NOT NULL,
          stage TEXT NOT NULL DEFAULT 'staging', created_at TEXT NOT NULL,
          PRIMARY KEY (model_id, version)
        );
        CREATE TABLE IF NOT EXISTS predictions (
          request_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, version TEXT NOT NULL,
          prediction INTEGER NOT NULL, probability REAL NOT NULL,
          latency_ms REAL NOT NULL, status TEXT NOT NULL, features TEXT NOT NULL,
          created_at TEXT NOT NULL, batch_id TEXT, batch_row INTEGER
        );
        CREATE TABLE IF NOT EXISTS prediction_batches (
          batch_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, version TEXT NOT NULL,
          row_count INTEGER NOT NULL, feature_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL,
          evaluation_json TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_feedback (
          request_id TEXT PRIMARY KEY, actual_label INTEGER NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY(request_id) REFERENCES predictions(request_id)
        );
        CREATE TABLE IF NOT EXISTS deployments (
          model_id TEXT PRIMARY KEY, version TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS deployment_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT NOT NULL,
          version TEXT NOT NULL, deployed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS alerts (
          id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL,
          model_id TEXT NOT NULL, version TEXT NOT NULL, severity TEXT NOT NULL,
          title TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL,
          resolved_at TEXT
        );
        CREATE TABLE IF NOT EXISTS drift_state (
          model_id TEXT NOT NULL, version TEXT NOT NULL, severity TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY (model_id, version)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_open_alert_per_fingerprint
          ON alerts(fingerprint) WHERE resolved_at IS NULL;
        """)
        if db.dialect == "sqlite":
            prediction_columns = {row["name"] for row in db.execute("PRAGMA table_info(predictions)")}
            batch_columns = {row["name"] for row in db.execute("PRAGMA table_info(prediction_batches)")}
        else:
            prediction_columns = {row["name"] for row in db.execute(
                "SELECT column_name AS name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=?",
                ("predictions",))}
            batch_columns = {row["name"] for row in db.execute(
                "SELECT column_name AS name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=?",
                ("prediction_batches",))}
        if "batch_id" not in prediction_columns:
            db.execute("ALTER TABLE predictions ADD COLUMN batch_id TEXT")
        if "batch_row" not in prediction_columns:
            db.execute("ALTER TABLE predictions ADD COLUMN batch_row INTEGER")
        if "evaluation_json" not in batch_columns:
            db.execute("ALTER TABLE prediction_batches ADD COLUMN evaluation_json TEXT")
        if db.dialect == "postgresql":
            # SQLite preserves tiny probabilities as 64-bit floats. PostgreSQL REAL is
            # only 32-bit and rejects such values as underflow during history migration.
            db.execute("ALTER TABLE predictions ALTER COLUMN probability TYPE DOUBLE PRECISION USING probability::DOUBLE PRECISION")
            db.execute("ALTER TABLE predictions ALTER COLUMN latency_ms TYPE DOUBLE PRECISION USING latency_ms::DOUBLE PRECISION")
        db.execute("""INSERT OR IGNORE INTO model_versions
          (model_id, version, name, algorithm, artifact_path, stage, created_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)""",
          (MODEL_ID, MODEL_VERSION, "AstroLens", "scikit-learn classifier",
           str(MODEL_PATH), "production", datetime.now(timezone.utc).isoformat()))
        db.execute("INSERT OR IGNORE INTO deployments VALUES (?, ?, ?)",
                   (MODEL_ID, MODEL_VERSION, datetime.now(timezone.utc).isoformat()))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _reference_profile
    initialize()
    with connect() as db:
        deployed = db.execute("SELECT model_id, version FROM deployments").fetchall()
    for deployment in deployed:
        load_registered_model(deployment["model_id"], deployment["version"])
    if REFERENCE_PATH.is_file():
        _reference_profile = ReferenceProfile(REFERENCE_PATH)
    yield


app = FastAPI(title="PulseML", description="Local model serving and reliability platform", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Protect API routes when an operator configures PULSEML_API_KEY."""
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        configured_key = os.environ.get("PULSEML_API_KEY")
        if configured_key:
            supplied_key = request.headers.get("x-api-key", "")
            if not secrets.compare_digest(supplied_key, configured_key):
                return JSONResponse(status_code=401, content={"detail": "Enter the PulseML API key to continue."})
    return await call_next(request)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)


@app.get("/api/health")
def health():
    with connect() as db:
        deployment = db.execute("SELECT model_id, version FROM deployments ORDER BY model_id LIMIT 1").fetchone()
    loaded = deployment is not None and (deployment["model_id"], deployment["version"]) in _artifacts
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded,
            "model": deployment["model_id"] if deployment else None,
            "version": deployment["version"] if deployment else None}


def _check_reliability_alerts(db: DatabaseConnection, model_id: str, version: str) -> None:
    """Open operational alerts from the most recent inference requests."""
    rows = db.execute("""SELECT status, latency_ms FROM predictions
                        WHERE model_id=? AND version=?
                        ORDER BY created_at DESC LIMIT ?""",
                      (model_id, version, RELIABILITY_WINDOW)).fetchall()
    if len(rows) < MIN_RELIABILITY_REQUESTS:
        return

    now = datetime.now(timezone.utc).isoformat()
    errors = sum(row["status"] != "success" for row in rows)
    error_rate = errors / len(rows) * 100
    if error_rate > ERROR_RATE_ALERT_PERCENT:
        db.execute("""INSERT OR IGNORE INTO alerts
                    (fingerprint, model_id, version, severity, title, detail, created_at)
                    VALUES (?, ?, ?, 'high', ?, ?, ?)""",
                   (f"error-rate:{model_id}:{version}", model_id, version,
                    "High prediction error rate",
                    f"{error_rate:.1f}% of the latest {len(rows)} inference requests failed (threshold: {ERROR_RATE_ALERT_PERCENT:.0f}%).",
                    now))

    successful_latencies = [row["latency_ms"] for row in rows if row["status"] == "success"]
    if len(successful_latencies) >= MIN_RELIABILITY_REQUESTS:
        average_latency = sum(successful_latencies) / len(successful_latencies)
        if average_latency > LATENCY_ALERT_MS:
            db.execute("""INSERT OR IGNORE INTO alerts
                        (fingerprint, model_id, version, severity, title, detail, created_at)
                        VALUES (?, ?, ?, 'high', ?, ?, ?)""",
                       (f"latency:{model_id}:{version}", model_id, version,
                        "High average inference latency",
                        f"Average inference latency is {average_latency:.1f} ms across the latest {len(successful_latencies)} successful requests (threshold: {LATENCY_ALERT_MS:.0f} ms).",
                        now))


@app.get("/api/models")
def models():
    with connect() as db:
        rows = db.execute("""SELECT m.*, d.version AS deployed_version,
                          EXISTS(SELECT 1 FROM deployment_history h WHERE h.model_id=m.model_id) AS has_previous
                          FROM model_versions m LEFT JOIN deployments d ON d.model_id=m.model_id
                          ORDER BY m.created_at DESC""").fetchall()
    return {"models": [dict(row) for row in rows]}


def _estimator(artifact):
    return artifact.get("model") if isinstance(artifact, dict) else artifact


class KerasModelAdapter:
    """Give a saved KeplerVision Keras model the predict_proba interface PulseML uses."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.feature_names_in_ = getattr(scaler, "feature_names_in_", None)

    def predict_proba(self, features):
        if self.feature_names_in_ is not None:
            features = pd.DataFrame(features, columns=self.feature_names_in_)
        scaled = self.scaler.transform(features).astype(np.float32, copy=False)
        planet_probability = self.model.predict(scaled[..., np.newaxis], verbose=0).reshape(-1)
        return np.column_stack((1 - planet_probability, planet_probability))


def _load_joblib(path: Path):
    """Load trusted sklearn artifacts and bridge the known LogisticRegression 1.9→1.7 rename."""
    from sklearn.exceptions import InconsistentVersionWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artifact = joblib.load(path)
    version_warnings = [warning for warning in caught
                        if issubclass(warning.category, InconsistentVersionWarning)]
    estimator = _estimator(artifact)
    if version_warnings:
        classifier = estimator.named_steps.get("classifier") if isinstance(estimator, Pipeline) else None
        if not isinstance(classifier, LogisticRegression) or hasattr(classifier, "multi_class"):
            raise ValueError("This joblib was saved with an incompatible scikit-learn version. Use a matching runtime.")
        # scikit-learn 1.9 removed this parameter; it was 'auto' in the included artifact.
        classifier.multi_class = "auto"
    return artifact


def _load_keras(path: Path, scaler_path: Path | None = None):
    scaler_path = scaler_path or path.with_suffix(".scaler.joblib")
    if not scaler_path.is_file():
        raise ValueError("The CNN scaler file is missing beside the .keras model.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("Keras CNN models require TensorFlow. Use the Python 3.10–3.13 runtime with TensorFlow 2.21.") from exc
    model = tf.keras.models.load_model(path, compile=False)
    return KerasModelAdapter(model, joblib.load(scaler_path))


def _load_artifact(path: Path):
    return _load_keras(path) if path.suffix.lower() == ".keras" else _load_joblib(path)


def _features_for_model(estimator, values):
    names = getattr(estimator, "feature_names_in_", None)
    return pd.DataFrame(values, columns=names) if names is not None else values


def load_registered_model(model_id: str, version: str):
    key = (model_id, version)
    if key in _artifacts:
        return _artifacts[key]
    with connect() as db:
        row = db.execute("SELECT artifact_path FROM model_versions WHERE model_id=? AND version=?",
                         key).fetchone()
    if row is None:
        return None
    path = Path(row["artifact_path"])
    if not path.is_file():
        return None
    artifact = _load_artifact(path)
    estimator = _estimator(artifact)
    if estimator is None or not hasattr(estimator, "predict_proba"):
        raise ValueError("The uploaded model must support predict_proba().")
    _artifacts[key] = artifact
    return artifact


@app.post("/api/models/register")
async def register_model(
    model_id: str = Form(...),
    name: str = Form(...),
    version: str = Form(...),
    artifact_file: UploadFile = File(...),
    scaler_file: UploadFile | None = File(None),
):
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", model_id):
        raise HTTPException(422, "Model ID must use lowercase letters, numbers, and hyphens.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise HTTPException(422, "Version may contain letters, numbers, dots, underscores, and hyphens.")
    if not name.strip() or len(name) > 100:
        raise HTTPException(422, "Model name must contain 1 to 100 characters.")
    suffix = Path(artifact_file.filename or "").suffix.lower()
    if suffix not in {".joblib", ".keras"}:
        raise HTTPException(422, "Upload a trusted .joblib artifact or a .keras model with its scaler .joblib file.")
    has_scaler = scaler_file is not None and bool(scaler_file.filename)
    if suffix == ".keras" and (not has_scaler or Path(scaler_file.filename).suffix.lower() != ".joblib"):
        raise HTTPException(422, "A KeplerVision .keras model also needs its matching .scaler.joblib file.")
    if suffix == ".joblib" and has_scaler:
        raise HTTPException(422, "Select a scaler file only when uploading a .keras model.")

    model_dir = ROOT / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / f"{version}{suffix}"
    scaler_destination = model_dir / f"{version}.scaler.joblib" if suffix == ".keras" else None
    if destination.exists() or (scaler_destination and scaler_destination.exists()):
        raise HTTPException(409, "That model version already has an artifact.")
    temporary = model_dir / f".{version}.uploading{suffix}"
    temporary_scaler = model_dir / f".{version}.uploading.scaler.joblib" if suffix == ".keras" else None
    artifact = None
    try:
        content = await artifact_file.read(200 * 1024 * 1024 + 1)
        if len(content) > 200 * 1024 * 1024:
            raise HTTPException(413, "Model artifacts must be 200 MB or smaller each.")
        temporary.write_bytes(content)
        if suffix == ".keras":
            scaler_content = await scaler_file.read(200 * 1024 * 1024 + 1)
            if len(scaler_content) > 200 * 1024 * 1024:
                raise HTTPException(413, "Scaler artifacts must be 200 MB or smaller.")
            temporary_scaler.write_bytes(scaler_content)
            artifact = _load_keras(temporary, temporary_scaler)
        else:
            artifact = _load_joblib(temporary)
        estimator = _estimator(artifact)
        if estimator is None or not hasattr(estimator, "predict_proba"):
            raise ValueError("The model must provide a predict_proba() method.")
        probabilities = estimator.predict_proba(_features_for_model(estimator, np.zeros((1, FEATURE_COUNT))))
        if np.asarray(probabilities).shape != (1, 2):
            raise ValueError("The model must return probabilities for exactly two classes.")
        temporary.replace(destination)
        if temporary_scaler:
            temporary_scaler.replace(scaler_destination)
    except HTTPException:
        temporary.unlink(missing_ok=True)
        if temporary_scaler:
            temporary_scaler.unlink(missing_ok=True)
        if destination.exists() and not _artifacts.get((model_id, version)):
            destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if temporary_scaler:
            temporary_scaler.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        if scaler_destination:
            scaler_destination.unlink(missing_ok=True)
        raise HTTPException(422, f"This model cannot serve PulseML's 3,197-feature input: {exc}") from exc
    finally:
        await artifact_file.close()
        if scaler_file:
            await scaler_file.close()

    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        try:
            db.execute("""INSERT INTO model_versions
                      (model_id, version, name, algorithm, artifact_path, stage, created_at)
                      VALUES (?, ?, ?, ?, ?, 'staging', ?)""",
                       (model_id, version, name.strip(),
                        "Keras multi-scale CNN" if suffix == ".keras" else type(estimator).__name__,
                        str(destination), now))
        except (sqlite3.IntegrityError, getattr(psycopg, "IntegrityError", sqlite3.IntegrityError)) as exc:
            destination.unlink(missing_ok=True)
            if scaler_destination:
                scaler_destination.unlink(missing_ok=True)
            raise HTTPException(409, "That model version is already registered.") from exc
    _artifacts[(model_id, version)] = artifact
    return {"model_id": model_id, "name": name.strip(), "version": version,
            "stage": "staging", "message": "Version registered. Promote it to serve predictions."}


@app.post("/api/models/{model_id}/{version}/promote")
def promote(model_id: str, version: str):
    with connect() as db:
        found = db.execute("SELECT 1 FROM model_versions WHERE model_id=? AND version=?", (model_id, version)).fetchone()
        if not found:
            raise HTTPException(404, "Model version not found")
        if not load_registered_model(model_id, version):
            raise HTTPException(409, "This model artifact is missing or invalid.")
        current = db.execute("SELECT version FROM deployments WHERE model_id=?", (model_id,)).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        if current and current["version"] != version:
            db.execute("INSERT INTO deployment_history (model_id, version, deployed_at) VALUES (?, ?, ?)",
                       (model_id, current["version"], now))
        db.execute("UPDATE deployments SET version=?, updated_at=? WHERE model_id=?",
                   (version, now, model_id))
        if not current:
            db.execute("INSERT INTO deployments VALUES (?, ?, ?)", (model_id, version, now))
        db.execute("UPDATE model_versions SET stage=CASE WHEN version=? THEN 'production' ELSE 'staging' END WHERE model_id=?",
                   (version, model_id))
    return {"model": model_id, "production_version": version}


@app.post("/api/models/{model_id}/rollback")
def rollback(model_id: str):
    with connect() as db:
        previous = db.execute("""SELECT id, version FROM deployment_history WHERE model_id=?
                                ORDER BY id DESC LIMIT 1""", (model_id,)).fetchone()
        current = db.execute("SELECT version FROM deployments WHERE model_id=?", (model_id,)).fetchone()
        if previous is None or current is None:
            raise HTTPException(409, "No previous production version is available to roll back to.")
        target = previous["version"]
        if not load_registered_model(model_id, target):
            raise HTTPException(409, "The previous model artifact is missing or invalid.")
        now = datetime.now(timezone.utc).isoformat()
        db.execute("DELETE FROM deployment_history WHERE id=?", (previous["id"],))
        db.execute("INSERT INTO deployment_history (model_id, version, deployed_at) VALUES (?, ?, ?)",
                   (model_id, current["version"], now))
        db.execute("UPDATE deployments SET version=?, updated_at=? WHERE model_id=?",
                   (target, now, model_id))
        db.execute("UPDATE model_versions SET stage=CASE WHEN version=? THEN 'production' ELSE 'staging' END WHERE model_id=?",
                   (target, model_id))
    return {"model": model_id, "production_version": target, "message": "Rollback complete."}


@app.post("/api/predict")
def predict(request: PredictRequest):
    with connect() as db:
        deployment = db.execute("SELECT model_id, version FROM deployments ORDER BY model_id LIMIT 1").fetchone()
    if deployment is None:
        raise HTTPException(503, "AstroLens artifact is missing. See the setup guide.")
    model_id, version = deployment["model_id"], deployment["version"]
    try:
        artifact = load_registered_model(model_id, version)
    except Exception as exc:
        raise HTTPException(503, f"The deployed model could not be loaded: {exc}") from exc
    if artifact is None:
        raise HTTPException(503, "The deployed model artifact is missing.")
    values = np.asarray(request.features, dtype=float).reshape(1, -1)
    if not np.isfinite(values).all():
        raise HTTPException(422, "Features must be finite numbers.")
    started = time.perf_counter()
    try:
        estimator = _estimator(artifact)
        probabilities = estimator.predict_proba(_features_for_model(estimator, values))[0]
        label = int(np.argmax(probabilities))
        probability = float(probabilities[-1])
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        now = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            db.execute("""INSERT INTO predictions
                        (request_id, model_id, version, prediction, probability, latency_ms, status, features, created_at)
                        VALUES (?, ?, ?, -1, 0, ?, 'error', ?, ?)""",
                       (str(uuid4()), model_id, version, latency, json.dumps(request.features), now))
            _check_reliability_alerts(db, model_id, version)
        raise HTTPException(503, f"Model prediction failed: {exc}") from exc
    latency = (time.perf_counter() - started) * 1000
    request_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        db.execute("""INSERT INTO predictions
                   (request_id, model_id, version, prediction, probability, latency_ms, status, features, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (request_id, model_id, version, label, probability, latency,
                    "success", json.dumps(request.features), now))
        _check_reliability_alerts(db, model_id, version)
    return {"request_id": request_id, "prediction": label, "class_name": LABELS.get(label, str(label)),
            "probability": probability, "model": model_id, "version": version,
            "latency_ms": round(latency, 2)}


@app.post("/api/predict/batch")
async def predict_batch(csv_file: UploadFile = File(...)):
    if not csv_file.filename or Path(csv_file.filename).suffix.lower() != ".csv":
        raise HTTPException(422, "Upload a CSV file containing KeplerVision flux columns.")
    try:
        content = await csv_file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "CSV files must be 200 MB or smaller.")
        frame = pd.read_csv(io.BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read the uploaded CSV: {exc}") from exc
    finally:
        await csv_file.close()
    if frame.empty:
        raise HTTPException(422, "The CSV must contain at least one observation.")
    if len(frame) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"Upload at most {MAX_BATCH_ROWS} observations at a time.")
    if not frame.columns.is_unique:
        raise HTTPException(422, "The CSV contains duplicate column names.")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise HTTPException(422, f"The CSV is missing {len(missing)} flux columns; expected FLUX.1 through FLUX.3197.")
    try:
        values = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Flux feature values must all be numeric.") from exc
    if not np.isfinite(values).all():
        raise HTTPException(422, "Flux features cannot contain missing or infinite values.")

    with connect() as db:
        deployment = db.execute("SELECT model_id, version FROM deployments ORDER BY model_id LIMIT 1").fetchone()
    if deployment is None:
        raise HTTPException(503, "No model version is deployed.")
    model_id, version = deployment["model_id"], deployment["version"]
    try:
        artifact = load_registered_model(model_id, version)
        if artifact is None:
            raise ValueError("The deployed model artifact is missing.")
        estimator = _estimator(artifact)
        started = time.perf_counter()
        probabilities = np.asarray(estimator.predict_proba(_features_for_model(estimator, values)), dtype=np.float64)
        total_latency = (time.perf_counter() - started) * 1000
        if probabilities.shape != (len(values), 2):
            raise ValueError("The deployed model must return two class probabilities for every row.")
    except Exception as exc:
        raise HTTPException(503, f"Batch prediction failed: {exc}") from exc

    labels = np.argmax(probabilities, axis=1).astype(int)
    now = datetime.now(timezone.utc).isoformat()
    per_row_latency = total_latency / len(values)
    batch_id = str(uuid4())
    records = [
        (str(uuid4()), model_id, version, int(label), float(probability[1]),
         per_row_latency, "success", json.dumps(row.tolist()), now, batch_id, index)
        for index, (row, label, probability) in enumerate(zip(values, labels, probabilities, strict=True))
    ]
    with connect() as db:
        db.execute("""INSERT INTO prediction_batches
                   (batch_id, model_id, version, row_count, feature_fingerprint, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                   (batch_id, model_id, version, len(records), hashlib.sha256(values.tobytes()).hexdigest(), now))
        db.executemany("""INSERT INTO predictions
                      (request_id, model_id, version, prediction, probability, latency_ms, status,
                       features, created_at, batch_id, batch_row)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", records)
        _check_reliability_alerts(db, model_id, version)
    return {
        "model": model_id,
        "version": version,
        "batch_id": batch_id,
        "count": len(records),
        "batch_latency_ms": round(total_latency, 2),
        "average_latency_per_row_ms": round(per_row_latency, 2),
        "note": "Any LABEL column in the uploaded CSV was ignored; predictions are not automatically treated as ground truth.",
        "sample_predictions": [
            {"request_id": records[index][0], "class_name": LABELS.get(int(labels[index]), str(labels[index])),
             "probability": round(float(probabilities[index, 1]), 6)}
            for index in range(min(5, len(records)))
        ],
    }


@app.post("/api/feedback/batch")
async def evaluate_batch(batch_id: str = Form(...), csv_file: UploadFile = File(...)):
    """Attach ground-truth LABELs to an earlier batch, then report binary quality metrics."""
    try:
        content = await csv_file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "CSV files must be 200 MB or smaller.")
        frame = pd.read_csv(io.BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read the feedback CSV: {exc}") from exc
    finally:
        await csv_file.close()
    if len(frame) > MAX_BATCH_ROWS or frame.empty:
        raise HTTPException(422, "The feedback CSV must contain between 1 and 1,000 rows.")
    if "LABEL" not in frame.columns:
        raise HTTPException(422, "The feedback CSV needs the KeplerVision LABEL column (1=no_planet, 2=planet).")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise HTTPException(422, "The feedback CSV must match the feature rows used for that prediction batch.")
    try:
        values = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        raw_labels = pd.to_numeric(frame["LABEL"], errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Flux features and LABEL must be numeric.") from exc
    if not np.isfinite(values).all() or not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [1, 2]).all():
        raise HTTPException(422, "Flux values must be finite and LABEL values must be 1 or 2.")
    with connect() as db:
        batch = db.execute("SELECT * FROM prediction_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch is None:
            raise HTTPException(404, "Prediction batch not found. Run batch predictions first and use its batch ID.")
        if len(frame) != batch["row_count"] or hashlib.sha256(values.tobytes()).hexdigest() != batch["feature_fingerprint"]:
            raise HTTPException(422, "This CSV's feature rows/order do not match that prediction batch.")
        predictions_for_batch = db.execute(
            "SELECT request_id, prediction FROM predictions WHERE batch_id=? ORDER BY batch_row", (batch_id,)
        ).fetchall()
        truths = (raw_labels.astype(int) - 1).tolist()
        now = datetime.now(timezone.utc).isoformat()
        existing_feedback = db.execute(
            "SELECT f.actual_label FROM prediction_feedback f JOIN predictions p USING(request_id) "
            "WHERE p.batch_id=? ORDER BY p.batch_row", (batch_id,)
        ).fetchall()
        if existing_feedback:
            if [row["actual_label"] for row in existing_feedback] != truths:
                raise HTTPException(409, "This batch already has different labels saved; feedback cannot be replaced.")
        else:
            db.executemany("INSERT INTO prediction_feedback VALUES (?, ?, ?)",
                           [(row["request_id"], truth, now) for row, truth in zip(predictions_for_batch, truths, strict=True)])
    predicted = [int(row["prediction"]) for row in predictions_for_batch]
    result = _evaluation_summary(batch_id, batch["model_id"], batch["version"], predicted, truths)
    with connect() as db:
        db.execute("UPDATE prediction_batches SET evaluation_json=? WHERE batch_id=?",
                   (json.dumps(result), batch_id))
    return result


def _evaluation_summary(batch_id: str, model_id: str, version: str,
                        predicted: list[int], truths: list[int]) -> dict:
    tp = sum(p == 1 and y == 1 for p, y in zip(predicted, truths, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, truths, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, truths, strict=True))
    tn = sum(p == 0 and y == 0 for p, y in zip(predicted, truths, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    planet_rows = tp + fn
    warning = (
        f"Only {planet_rows} planet examples are in this batch. Planet metrics are highly uncertain, "
        "and accuracy may look high because no-planet examples dominate."
        if planet_rows < 30 else None
    )
    return {"batch_id": batch_id, "model": model_id, "version": version,
            "count": len(truths), "planet_rows": planet_rows, "accuracy": (tp + tn) / len(truths),
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": {"true_negative": tn, "false_positive": fp,
                                 "false_negative": fn, "true_positive": tp},
            "warning": warning,
            "note": "Positive class is planet. Metrics describe this labeled batch and are not a guarantee of future performance."}


@app.get("/api/feedback/latest")
def latest_feedback():
    with connect() as db:
        batch = db.execute("SELECT batch_id, model_id, version, evaluation_json FROM prediction_batches "
                           "ORDER BY created_at DESC LIMIT 1").fetchone()
    if batch is None:
        return {"evaluated": False, "batch_id": None, "message": "No prediction batches yet."}
    with connect() as db:
        row_count = db.execute("SELECT COUNT(*) FROM predictions WHERE batch_id=?", (batch["batch_id"],)).fetchone()[0]
    if batch["evaluation_json"]:
        result = json.loads(batch["evaluation_json"])
    else:
        with connect() as db:
            saved = db.execute("SELECT p.prediction, f.actual_label FROM predictions p "
                               "JOIN prediction_feedback f USING(request_id) WHERE p.batch_id=? "
                               "ORDER BY p.batch_row", (batch["batch_id"],)).fetchall()
        if not saved:
            return {"evaluated": False, "batch_id": batch["batch_id"], "model": batch["model_id"],
                    "version": batch["version"], "count": row_count,
                    "message": "This batch has not been evaluated yet."}
        result = _evaluation_summary(batch["batch_id"], batch["model_id"], batch["version"],
                                     [int(row["prediction"]) for row in saved],
                                     [int(row["actual_label"]) for row in saved])
        with connect() as db:
            db.execute("UPDATE prediction_batches SET evaluation_json=? WHERE batch_id=?",
                       (json.dumps(result), batch["batch_id"]))
    result["evaluated"] = True
    result["count"] = result.get("count", row_count)
    return result


@app.get("/api/predictions")
def predictions(limit: int = Query(100, ge=1, le=500)):
    with connect() as db:
        rows = db.execute("SELECT request_id, model_id, version, prediction, probability, latency_ms, status, created_at FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return {"predictions": [dict(row) for row in rows]}


@app.get("/api/metrics")
def metrics():
    with connect() as db:
        row = db.execute("""SELECT COUNT(*) AS requests,
          COALESCE(AVG(latency_ms),0) AS avg_latency,
          COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END),0) AS errors,
          COALESCE(SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END),0) AS positive,
          COALESCE(SUM(CASE WHEN prediction=0 THEN 1 ELSE 0 END),0) AS negative
          FROM predictions""").fetchone()
    data = dict(row)
    data["error_rate"] = (data["errors"] / data["requests"] * 100) if data["requests"] else 0
    return data


@app.get("/api/monitoring")
def monitoring():
    summary = metrics()
    # Prediction distribution is a useful signal; a shift asks for investigation,
    # but is not proof that the model is wrong.
    total = summary["positive"] + summary["negative"]
    positive_rate = summary["positive"] / total if total else 0
    prediction_drift = "insufficient_data" if total < 20 else (
        "watch" if positive_rate < .05 or positive_rate > .5 else "normal"
    )
    return {"api_health": "healthy", "prediction_positive_rate": round(positive_rate, 4),
            "prediction_drift": prediction_drift,
            "note": "Prediction shifts are signals to investigate, not proof of model failure."}


@app.get("/api/drift")
def drift(limit: int = Query(MINIMUM_PRODUCTION_ROWS, ge=MINIMUM_PRODUCTION_ROWS, le=5000)):
    if _reference_profile is None:
        return {"status": "reference_missing", "severity": "unknown", "sample_count": 0,
                "minimum_sample_count": MINIMUM_PRODUCTION_ROWS, "reference_sample_count": 0,
                "source_training_rows": 0, "top_features": [],
                "message": "The KeplerVision training reference profile is missing."}
    with connect() as db:
        deployment = db.execute("SELECT model_id, version FROM deployments ORDER BY model_id LIMIT 1").fetchone()
        if deployment is None:
            return {"status": "no_deployment", "severity": "unknown", "sample_count": 0,
                    "minimum_sample_count": MINIMUM_PRODUCTION_ROWS,
                    "reference_sample_count": _reference_profile.sample_count,
                    "source_training_rows": _reference_profile.source_rows,
                    "top_features": [], "message": "No model version is deployed."}
        rows = db.execute("""SELECT features FROM predictions
                            WHERE model_id=? AND version=? AND status='success'
                            ORDER BY created_at DESC LIMIT ?""",
                          (deployment["model_id"], deployment["version"], limit)).fetchall()
    values = np.asarray([json.loads(row["features"]) for row in reversed(rows)], dtype=np.float64)
    if not len(rows):
        values = np.empty((0, len(_reference_profile.features)), dtype=np.float64)
    report = _reference_profile.score(values)
    report["model_id"] = deployment["model_id"]
    report["version"] = deployment["version"]
    if report["status"] == "ready":
        now = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            previous = db.execute("SELECT severity FROM drift_state WHERE model_id=? AND version=?",
                                  (deployment["model_id"], deployment["version"])).fetchone()
            if report["severity"] == "high" and (previous is None or previous["severity"] != "high"):
                top = report["top_features"][0]
                db.execute("""INSERT OR IGNORE INTO alerts
                          (fingerprint, model_id, version, severity, title, detail, created_at)
                          VALUES (?, ?, ?, 'high', ?, ?, ?)""",
                           (f"feature-drift:{deployment['model_id']}:{deployment['version']}:{top['feature']}:{now[:10]}",
                            deployment["model_id"], deployment["version"],
                            f"High feature drift: {top['feature']}",
                            f"PSI is {top['psi']} across {report['sample_count']} production requests.", now))
            db.execute("""INSERT INTO drift_state (model_id, version, severity, updated_at)
                      VALUES (?, ?, ?, ?) ON CONFLICT(model_id, version) DO UPDATE SET
                      severity=excluded.severity, updated_at=excluded.updated_at""",
                       (deployment["model_id"], deployment["version"], report["severity"], now))
    return report


@app.get("/api/alerts")
def alerts(limit: int = Query(20, ge=1, le=100)):
    with connect() as db:
        rows = db.execute("""SELECT id, model_id, version, severity, title, detail, created_at
                            FROM alerts WHERE resolved_at IS NULL
                            ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
    return {"alerts": [dict(row) for row in rows]}


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    with connect() as db:
        result = db.execute("UPDATE alerts SET resolved_at=? WHERE id=? AND resolved_at IS NULL",
                            (datetime.now(timezone.utc).isoformat(), alert_id))
        if result.rowcount == 0:
            raise HTTPException(404, "Open alert not found")
    return {"alert_id": alert_id, "status": "resolved"}


WEB_DIR = ROOT / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="dashboard")
