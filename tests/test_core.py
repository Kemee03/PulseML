"""Fast core and database behavior checks; TensorFlow is not needed for the suite.

Run from the project root with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from starlette.requests import Request
from starlette.responses import Response

from api import main


class CoreBehaviorTests(unittest.TestCase):
    def test_postgres_adapter_translates_placeholders_and_ignore_inserts(self) -> None:
        connection = main.DatabaseConnection(None, "postgresql")
        self.assertEqual(connection._sql("SELECT * FROM models WHERE id=?"),
                         "SELECT * FROM models WHERE id=%s")
        self.assertEqual(connection._sql("INSERT OR IGNORE INTO models VALUES (?, ?)"),
                         "INSERT INTO models VALUES (%s, %s) ON CONFLICT DO NOTHING")

    def test_postgres_schema_uses_double_precision_for_prediction_values(self) -> None:
        class FakeConnection:
            def __init__(self):
                self.statements = []

            def execute(self, sql, parameters=()):
                self.statements.append(sql)

        raw = FakeConnection()
        main.DatabaseConnection(raw, "postgresql").executescript(
            "CREATE TABLE predictions (probability REAL, latency_ms REAL);"
        )
        self.assertIn("probability DOUBLE PRECISION", raw.statements[0])
        self.assertIn("latency_ms DOUBLE PRECISION", raw.statements[0])

    def test_prediction_request_requires_exact_feature_count(self) -> None:
        request = main.PredictRequest(features=[0.0] * main.FEATURE_COUNT)
        self.assertEqual(len(request.features), 3197)
        with self.assertRaises(ValueError):
            main.PredictRequest(features=[0.0] * (main.FEATURE_COUNT - 1))

    def test_evaluation_metrics_use_planet_as_positive_class(self) -> None:
        summary = main._evaluation_summary(
            "batch-test", "astrolens", "1.2.0",
            predicted=[0, 0, 1, 0], truths=[0, 1, 1, 0],
        )
        self.assertEqual(summary["confusion_matrix"], {
            "true_negative": 2, "false_positive": 0,
            "false_negative": 1, "true_positive": 1,
        })
        self.assertEqual(summary["accuracy"], 0.75)
        self.assertEqual(summary["precision"], 1.0)
        self.assertEqual(summary["recall"], 0.5)
        self.assertAlmostEqual(summary["f1"], 2 / 3)
        self.assertIn("Only 2 planet examples", summary["warning"])

    def test_keras_adapter_scales_features_and_returns_two_probabilities(self) -> None:
        class Scaler:
            feature_names_in_ = None

            def transform(self, values):
                return np.asarray(values)

        class Model:
            def predict(self, values, verbose=0):
                self.input_shape = values.shape
                return np.asarray([[0.2], [0.8]])

        model = Model()
        adapter = main.KerasModelAdapter(model, Scaler())
        probabilities = adapter.predict_proba(np.zeros((2, main.FEATURE_COUNT)))
        self.assertEqual(model.input_shape, (2, main.FEATURE_COUNT, 1))
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities, [[0.8, 0.2], [0.2, 0.8]])

    def test_api_key_middleware_protects_api_and_leaves_health_public(self) -> None:
        async def run_request(path: str, api_key: str | None = None):
            headers = [(b"host", b"test")]
            if api_key:
                headers.append((b"x-api-key", api_key.encode()))
            request = Request({
                "type": "http", "asgi": {"version": "3.0"},
                "http_version": "1.1", "method": "GET", "scheme": "http",
                "path": path, "raw_path": path.encode(), "query_string": b"",
                "headers": headers, "server": ("test", 80),
                "client": ("test", 1234),
            })
            async def next_handler(_request):
                return Response("ok", status_code=200)
            return await main.require_api_key(request, next_handler)

        with patch.dict(os.environ, {"PULSEML_API_KEY": "test-key"}):
            unauthorized = asyncio.run(run_request("/api/models"))
            authorized = asyncio.run(run_request("/api/models", "test-key"))
            health = asyncio.run(run_request("/api/health"))
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(health.status_code, 200)


class TemporaryDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="pulseml-test-")
        self.db_path = Path(self.temp_dir.name) / "pulseml-test.db"
        self.db_path_patch = patch.object(main, "DB_PATH", self.db_path)
        self.database_url_patch = patch.object(main, "DATABASE_URL", None)
        self.db_path_patch.start()
        self.database_url_patch.start()
        main.initialize()

    def tearDown(self) -> None:
        self.database_url_patch.stop()
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_initialization_creates_seed_model_without_touching_real_database(self) -> None:
        registry = main.models()
        self.assertEqual(len(registry["models"]), 1)
        self.assertEqual(registry["models"][0]["version"], main.MODEL_VERSION)
        self.assertEqual(registry["models"][0]["deployed_version"], main.MODEL_VERSION)
        self.assertTrue(self.db_path.is_file())

    def test_metrics_are_zero_for_a_fresh_database(self) -> None:
        summary = main.metrics()
        self.assertEqual(summary["requests"], 0)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["error_rate"], 0)

    def test_latest_batch_includes_row_count_and_saved_evaluation(self) -> None:
        evaluation = main._evaluation_summary(
            "batch-test", "astrolens", "1.0.0", [0, 1], [0, 1]
        )
        db = sqlite3.connect(self.db_path)
        try:
            db.execute(
                "INSERT INTO prediction_batches "
                "(batch_id,model_id,version,row_count,feature_fingerprint,created_at,evaluation_json) "
                "VALUES (?,?,?,?,?,?,?)",
                ("batch-test", "astrolens", "1.0.0", 2, "fingerprint", "2026-10-01T00:00:00+00:00", json.dumps(evaluation)),
            )
            for index, prediction in enumerate((0, 1)):
                db.execute(
                    "INSERT INTO predictions "
                    "(request_id,model_id,version,prediction,probability,latency_ms,status,features,created_at,batch_id,batch_row) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (f"request-{index}", "astrolens", "1.0.0", prediction,
                     float(prediction), 1.0, "success", "[]",
                     f"2026-10-01T00:00:0{index}+00:00", "batch-test", index),
                )
            db.commit()
        finally:
            db.close()
        latest = main.latest_feedback()
        self.assertEqual(latest["batch_id"], "batch-test")
        self.assertEqual(latest["count"], 2)
        self.assertTrue(latest["evaluated"])


class PostgreSQLIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not main.DATABASE_URL:
            raise unittest.SkipTest("Set DATABASE_URL to run PostgreSQL integration checks.")
        main.initialize()

    def test_tiny_prediction_probability_round_trips_as_double_precision(self) -> None:
        request_id = f"ci-{uuid4()}"
        probability = 2.633719084819348e-204
        with main.connect() as db:
            db.execute(
                "INSERT INTO predictions "
                "(request_id,model_id,version,prediction,probability,latency_ms,status,features,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (request_id, "astrolens", "ci", 0, probability, 1.0, "success", "[]", "2026-10-01T00:00:00+00:00"),
            )
        try:
            with main.connect() as db:
                row = db.execute("SELECT probability FROM predictions WHERE request_id=?", (request_id,)).fetchone()
            self.assertEqual(row["probability"], probability)
        finally:
            with main.connect() as db:
                db.execute("DELETE FROM predictions WHERE request_id=?", (request_id,))


if __name__ == "__main__":
    unittest.main()
