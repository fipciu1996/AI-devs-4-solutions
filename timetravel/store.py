"""SQLite-backed shared state for backend and frontend agents."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import json
import sqlite3
from typing import Any

from .models import DesiredUiState, MissionPhase, MissionStatus, ObservedUiState


class _Unset:
    pass


_UNSET = _Unset()


class SharedStateStore:
    """Persist and exchange mission state between two processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mission_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    flag TEXT,
                    present_date TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS desired_ui (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observed_ui (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backend_snapshot (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )
            timestamp = now_iso()
            connection.execute(
                """
                INSERT OR IGNORE INTO mission_state (
                    id, phase, status, error, flag, present_date, updated_at
                ) VALUES (1, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    MissionPhase.ACQUIRE_BATTERIES.value,
                    MissionStatus.CONFIGURING.value,
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO desired_ui (id, payload, updated_at) VALUES (1, ?, ?)",
                (
                    json.dumps(asdict(DesiredUiState(PTA=False, PTB=False, PWR=0, mode="standby"))),
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO observed_ui (id, payload, updated_at) VALUES (1, ?, ?)",
                (json.dumps(asdict(ObservedUiState())), timestamp),
            )
            connection.execute(
                "INSERT OR IGNORE INTO backend_snapshot (id, payload, updated_at) VALUES (1, ?, ?)",
                (json.dumps({}), timestamp),
            )

    def record_event(self, source: str, level: str, message: str) -> None:
        """Append a human-readable event log row."""

        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO events (created_at, source, level, message)
                VALUES (?, ?, ?, ?)
                """,
                (now_iso(), source, level.upper(), message),
            )

    def set_mission_state(
        self,
        *,
        phase: MissionPhase | None = None,
        status: MissionStatus | None = None,
        error: str | None | object = _UNSET,
        flag: str | None | object = _UNSET,
        present_date: str | None | object = _UNSET,
    ) -> None:
        """Update the single mission_state row."""

        assignments: list[str] = ["updated_at = ?"]
        parameters: list[Any] = [now_iso()]
        if phase is not None:
            assignments.append("phase = ?")
            parameters.append(phase.value)
        if status is not None:
            assignments.append("status = ?")
            parameters.append(status.value)
        if error is not _UNSET:
            assignments.append("error = ?")
            parameters.append(error)
        if flag is not _UNSET:
            assignments.append("flag = ?")
            parameters.append(flag)
        if present_date is not _UNSET:
            assignments.append("present_date = ?")
            parameters.append(present_date)
        parameters.append(1)
        with closing(self._connect()) as connection:
            connection.execute(
                f"UPDATE mission_state SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )

    def get_mission_state(self) -> dict[str, Any]:
        """Read the current mission_state row."""

        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM mission_state WHERE id = 1").fetchone()
        return dict(row) if row is not None else {}

    def set_desired_ui(self, state: DesiredUiState) -> None:
        """Persist the latest desired UI controls."""

        self._set_json_row("desired_ui", asdict(state))

    def get_desired_ui(self) -> DesiredUiState:
        """Read the desired UI state."""

        payload = self._get_json_row("desired_ui")
        return DesiredUiState(**payload)

    def bump_jump_request(self) -> int:
        """Increment the current jump request id and return it."""

        current = self.get_desired_ui()
        next_request_id = current.jump_request_id + 1
        self.set_desired_ui(
            DesiredUiState(
                PTA=current.PTA,
                PTB=current.PTB,
                PWR=current.PWR,
                mode=current.mode,
                jump_request_id=next_request_id,
            )
        )
        return next_request_id

    def set_observed_ui(self, state: ObservedUiState) -> None:
        """Persist the latest observed UI state."""

        self._set_json_row("observed_ui", asdict(state))

    def get_observed_ui(self) -> ObservedUiState:
        """Read the current observed UI state."""

        payload = self._get_json_row("observed_ui")
        return ObservedUiState(**payload)

    def set_backend_snapshot(self, payload: dict[str, Any]) -> None:
        """Persist the last raw backend config snapshot."""

        self._set_json_row("backend_snapshot", payload)

    def get_backend_snapshot(self) -> dict[str, Any]:
        """Read the current backend snapshot payload."""

        return self._get_json_row("backend_snapshot")

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the latest human-readable event rows."""

        normalized_limit = max(1, limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT created_at, source, level, message
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def _set_json_row(self, table_name: str, payload: dict[str, Any]) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                f"UPDATE {table_name} SET payload = ?, updated_at = ? WHERE id = 1",
                (json.dumps(payload, ensure_ascii=False), now_iso()),
            )

    def _get_json_row(self, table_name: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT payload FROM {table_name} WHERE id = 1"
            ).fetchone()
        if row is None:
            return {}
        return json.loads(row["payload"])


def now_iso() -> str:
    """Return a UTC timestamp suitable for the state store."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()
