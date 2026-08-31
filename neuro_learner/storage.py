import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from neuro_learner.config import settings
from neuro_learner.models import ConceptMetadata

def get_db_connection(db_path: str | None = None) -> sqlite3.Connection:
    target_path = db_path or settings.persistence_db_path
    settings.ensure_db_dir()
    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn

def _init_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts_index (
            concept_id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            stability REAL NOT NULL,
            difficulty REAL NOT NULL,
            reps INTEGER NOT NULL,
            lapses INTEGER NOT NULL,
            last_review TEXT NOT NULL,
            next_review TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()

def save_concept_index(
    meta: ConceptMetadata,
    thread_id: str,
    db_path: str | None = None,
) -> None:
    conn = get_db_connection(db_path)
    now_str = datetime.now(timezone.utc).isoformat()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO concepts_index (
            concept_id, topic, stability, difficulty, reps, lapses,
            last_review, next_review, thread_id, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(concept_id) DO UPDATE SET
            topic = excluded.topic,
            stability = excluded.stability,
            difficulty = excluded.difficulty,
            reps = excluded.reps,
            lapses = excluded.lapses,
            last_review = excluded.last_review,
            next_review = excluded.next_review,
            thread_id = excluded.thread_id,
            updated_at = excluded.updated_at
    """, (
        meta.concept_id,
        meta.topic,
        meta.stability,
        meta.difficulty,
        meta.reps,
        meta.lapses,
        meta.last_review,
        meta.next_review,
        thread_id,
        now_str,
    ))
    conn.commit()
    conn.close()

def list_saved_concepts(db_path: str | None = None) -> list[dict]:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT concept_id, topic, stability, difficulty, reps, lapses,
               last_review, next_review, thread_id, updated_at
        FROM concepts_index
        ORDER BY updated_at DESC
    """)
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

def get_concept_by_id(concept_id: str, db_path: str | None = None) -> dict | None:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT concept_id, topic, stability, difficulty, reps, lapses,
               last_review, next_review, thread_id, updated_at
        FROM concepts_index
        WHERE concept_id = ? OR topic = ?
    """, (concept_id, concept_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
