import os
import sqlite3
from typing import Dict


class FeedbackDB:
    def __init__(self, path: str):
        self.path = path
        self._ensure()

    def _ensure(self):
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                session_id TEXT,
                realism INTEGER,
                fit INTEGER,
                preferred TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def insert_feedback(self, ts: str, session_id: str, realism: int, fit: int, preferred: str):
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO feedback (ts, session_id, realism, fit, preferred) VALUES (?, ?, ?, ?, ?)",
            (ts, session_id, realism, fit, preferred),
        )
        conn.commit()
        conn.close()

    def summary(self) -> Dict[str, float]:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("SELECT AVG(realism), AVG(fit) FROM feedback")
        row = cur.fetchone()
        avg_realism = float(row[0] or 0.0)
        avg_fit = float(row[1] or 0.0)
        counts = {}
        total = 0
        for pref in ["A", "B", "Both", "None"]:
            cur.execute("SELECT COUNT(*) FROM feedback WHERE preferred=?", (pref,))
            c = cur.fetchone()[0]
            counts[pref] = c
            total += c
        conn.close()
        def pct(x):
            return (counts.get(x, 0) / total * 100.0) if total > 0 else 0.0
        return {
            "avg_realism": avg_realism,
            "avg_fit": avg_fit,
            "pref_A_percent": pct("A"),
            "pref_B_percent": pct("B"),
            "pref_Both_percent": pct("Both"),
            "pref_None_percent": pct("None"),
        }

