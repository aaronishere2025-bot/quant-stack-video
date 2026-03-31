"""
Billing store — SQLite-backed credit ledger.

Credits are denominated in cents (integer). $0.10/second of video = 10 cents/second.

Tables:
  credits  — api_key -> balance_cents
  usage    — per-generation usage log

Thread-safe via sqlite3's check_same_thread=False + a module-level lock.
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

_DB_PATH = Path.home() / ".quant-stack" / "billing.db"
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS credits (
            api_key     TEXT PRIMARY KEY,
            balance_cents INTEGER NOT NULL DEFAULT 0,
            updated_at  REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS usage (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key     TEXT NOT NULL,
            task_id     TEXT,
            seconds     REAL NOT NULL,
            cost_cents  INTEGER NOT NULL,
            created_at  REAL NOT NULL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CENTS_PER_SECOND = 10  # $0.10/second


def get_balance(api_key: str) -> int:
    """Return current balance in cents (0 if never seen)."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT balance_cents FROM credits WHERE api_key = ?", (api_key,)
        ).fetchone()
        return row["balance_cents"] if row else 0


def add_credits(api_key: str, cents: int) -> int:
    """Add cents to balance. Returns new balance."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO credits (api_key, balance_cents, updated_at)
                VALUES (?, ?, ?)
            ON CONFLICT(api_key) DO UPDATE SET
                balance_cents = balance_cents + excluded.balance_cents,
                updated_at = excluded.updated_at
            """,
            (api_key, cents, time.time()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT balance_cents FROM credits WHERE api_key = ?", (api_key,)
        ).fetchone()
        return row["balance_cents"]


def deduct_credits(api_key: str, seconds: float, task_id: str = "") -> dict:
    """
    Deduct cost for `seconds` of generated video.
    Returns {"ok": True, "cost_cents": N, "balance_cents": N}
      or    {"ok": False, "reason": "insufficient_credits", ...}
    Raises ValueError if balance would go negative.
    """
    cost_cents = max(1, round(seconds * CENTS_PER_SECOND))
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT balance_cents FROM credits WHERE api_key = ?", (api_key,)
        ).fetchone()
        balance = row["balance_cents"] if row else 0
        if balance < cost_cents:
            return {
                "ok": False,
                "reason": "insufficient_credits",
                "balance_cents": balance,
                "cost_cents": cost_cents,
            }
        new_balance = balance - cost_cents
        conn.execute(
            "UPDATE credits SET balance_cents = ?, updated_at = ? WHERE api_key = ?",
            (new_balance, time.time(), api_key),
        )
        conn.execute(
            "INSERT INTO usage (api_key, task_id, seconds, cost_cents, created_at) VALUES (?, ?, ?, ?, ?)",
            (api_key, task_id, seconds, cost_cents, time.time()),
        )
        conn.commit()
    return {"ok": True, "cost_cents": cost_cents, "balance_cents": new_balance}


def get_usage(api_key: str, limit: int = 50) -> List[dict]:
    """Return recent usage records, newest first."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT id, task_id, seconds, cost_cents, created_at
            FROM usage WHERE api_key = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (api_key, limit),
        ).fetchall()
        return [dict(r) for r in rows]
