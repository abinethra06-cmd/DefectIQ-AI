"""SQLite persistence for inspection results and operator feedback."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path("inspections.db")


def _connect(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            profile TEXT,
            decision TEXT,
            max_score REAL,
            reason TEXT,
            defects_json TEXT,
            operator_feedback TEXT
        )
    """)
    return conn


def save_result(image_name, result, profile="Standard", db_path=DB_PATH):
    defects = [
        {k: d.get(k) for k in ("type", "confidence", "score", "decision", "critical")}
        for d in result.get("defects", [])
    ]
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO inspections
               (timestamp, image_name, profile, decision, max_score, reason, defects_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"), image_name, profile,
                result["decision"], result["max_score"], result["reason"], json.dumps(defects)
            ),
        )
        return cur.lastrowid


def save_feedback(inspection_id, feedback, db_path=DB_PATH):
    allowed = {"ACCEPT", "REWORK", "REJECT", "MANUAL_REVIEW"}
    if feedback not in allowed:
        raise ValueError(f"feedback must be one of {sorted(allowed)}")
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE inspections SET operator_feedback = ? WHERE id = ?",
            (feedback, inspection_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No inspection found with id {inspection_id}")


def load_history(db_path=DB_PATH):
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC", conn)


def defect_counts(db_path=DB_PATH):
    rows = []
    for _, row in load_history(db_path).iterrows():
        for defect in json.loads(row["defects_json"] or "[]"):
            rows.append({
                "timestamp": row["timestamp"],
                "type": defect["type"],
                "decision": defect["decision"],
            })
    return pd.DataFrame(rows, columns=["timestamp", "type", "decision"])
