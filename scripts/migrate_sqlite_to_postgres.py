"""Merge PulseML's local SQLite history into its configured PostgreSQL database."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dot_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def postgres_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://{user}:{password}@127.0.0.1:5432/pulseml"


def artifact_container_path(stored_path: str) -> str:
    models = ROOT / "models"
    filename = Path(stored_path.replace("\\", "/")).name
    matches = list(models.rglob(filename)) if models.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"Could not find one local model artifact named {filename!r} under {models}.")
    relative = matches[0].relative_to(models).as_posix()
    return f"/app/models/{relative}"


def migrate() -> int:
    load_dot_env(ROOT / ".env")
    source_path = Path(os.getenv("PULSEML_DB_PATH", ROOT / "pulseml.db"))
    if not source_path.is_file():
        raise RuntimeError(f"SQLite database not found: {source_path}")
    url = postgres_url()
    os.environ["DATABASE_URL"] = url

    from api import main as pulseml

    pulseml.DATABASE_URL = url
    pulseml.initialize()

    table_order = (
        "model_versions", "deployments", "prediction_batches", "predictions",
        "prediction_feedback", "deployment_history", "alerts", "drift_state",
    )
    upserts = {
        "model_versions": "ON CONFLICT (model_id, version) DO UPDATE SET name=EXCLUDED.name, algorithm=EXCLUDED.algorithm, artifact_path=EXCLUDED.artifact_path, stage=EXCLUDED.stage, created_at=EXCLUDED.created_at",
        "deployments": "ON CONFLICT (model_id) DO UPDATE SET version=EXCLUDED.version, updated_at=EXCLUDED.updated_at",
        "drift_state": "ON CONFLICT (model_id, version) DO UPDATE SET severity=EXCLUDED.severity, updated_at=EXCLUDED.updated_at",
    }
    counts: dict[str, int] = {}
    with sqlite3.connect(source_path) as source, pulseml.connect() as target:
        source.row_factory = sqlite3.Row
        for table in table_order:
            rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
            if not rows:
                counts[table] = 0
                continue
            columns = rows[0].keys()
            placeholders = ", ".join("?" for _ in columns)
            column_sql = ", ".join(f'"{column}"' for column in columns)
            conflict = upserts.get(table, "ON CONFLICT DO NOTHING")
            statement = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders}) {conflict}'
            values = []
            for row in rows:
                record = list(row)
                if table == "model_versions":
                    artifact_column = columns.index("artifact_path")
                    record[artifact_column] = artifact_container_path(record[artifact_column])
                values.append(tuple(record))
            target.executemany(statement, values)
            counts[table] = len(values)

        for table in ("deployment_history", "alerts"):
            maximum = source.execute(f'SELECT MAX(id) FROM "{table}"').fetchone()[0]
            if maximum is not None:
                target.execute("SELECT setval(pg_get_serial_sequence(?, 'id'), ?, true)", (table, maximum))

    total = sum(counts.values())
    print(f"Merged {total} SQLite records into PostgreSQL.")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    return total


if __name__ == "__main__":
    migrate()
