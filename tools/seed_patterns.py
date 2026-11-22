"""SQLite-backed Threat Pattern Memory Store with Bounded LRU Eviction.

Provides persistent memory of observed attack patterns and behavioral anomalies.
Hard constraint: Bounded table with maximum 10,000 entries and automatic LRU eviction.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import structlog
from backend.config import VerdictEnum, settings
from data.attack_traffic import (
    generate_api_flooding_attack,
    generate_bola_attack,
    generate_bruteforce_login_attack,
    generate_credential_stuffing_attack,
    generate_parameter_tampering_attack,
    generate_sql_injection_attack,
    generate_token_replay_attack,
)
from ml.feature_engineering import extract_features_from_request
from ml.pipeline import get_security_pipeline

logger = structlog.get_logger(__name__)

DEFAULT_PATTERN_DB_PATH = Path("logs/pattern_memory.db")
MAX_PATTERN_ENTRIES = 10_000


class PatternMemoryStore:
    """Thread-safe SQLite store for historical attack patterns with hard LRU cap."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_entries: int = MAX_PATTERN_ENTRIES,
    ) -> None:
        self.db_path = db_path or DEFAULT_PATTERN_DB_PATH
        self.max_entries = max_entries
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize schema and create indexes and LRU enforcement trigger."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_memory (
                    pattern_hash TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    features TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    explanations TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_last_seen ON pattern_memory(last_seen);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_attack_type ON pattern_memory(attack_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_client_id ON pattern_memory(client_id);")

            # Dynamic trigger ensuring bounded storage at SQLite engine level
            conn.execute("DROP TRIGGER IF EXISTS trg_enforce_lru_cap;")
            conn.execute(
                f"""
                CREATE TRIGGER trg_enforce_lru_cap
                AFTER INSERT ON pattern_memory
                WHEN (SELECT COUNT(*) FROM pattern_memory) > {self.max_entries}
                BEGIN
                    DELETE FROM pattern_memory
                    WHERE pattern_hash IN (
                        SELECT pattern_hash FROM pattern_memory
                        ORDER BY last_seen ASC, hit_count ASC
                        LIMIT ((SELECT COUNT(*) FROM pattern_memory) - {self.max_entries})
                    );
                END;
                """
            )
            conn.commit()

    @staticmethod
    def compute_pattern_hash(client_id: str, endpoint: str, attack_type: str, features: Dict[str, float]) -> str:
        """Compute deterministic SHA-256 fingerprint for an observed pattern."""
        serialized = json.dumps(
            {
                "client": client_id,
                "endpoint": endpoint,
                "type": attack_type,
                "features": {k: round(v, 3) for k, v in sorted(features.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def record_pattern(
        self,
        client_id: str,
        endpoint: str,
        attack_type: str,
        features: Dict[str, float],
        verdict: str = VerdictEnum.BLOCK.value,
        anomaly_score: float = 1.0,
        explanations: Optional[List[Dict[str, Any]]] = None,
        observed_time: Optional[datetime] = None,
    ) -> str:
        """Upsert pattern into memory store, updating hit count and enforcing LRU limit."""
        now_iso = (observed_time or datetime.now(timezone.utc)).isoformat()
        pattern_hash = self.compute_pattern_hash(client_id, endpoint, attack_type, features)
        features_json = json.dumps(features)
        explanations_json = json.dumps(explanations or [])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pattern_memory (
                    pattern_hash, client_id, endpoint, attack_type, features,
                    verdict, anomaly_score, explanations, hit_count, created_at, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    hit_count = pattern_memory.hit_count + 1,
                    last_seen = excluded.last_seen,
                    anomaly_score = excluded.anomaly_score,
                    explanations = excluded.explanations;
                """,
                (
                    pattern_hash,
                    client_id,
                    endpoint,
                    attack_type,
                    features_json,
                    verdict,
                    anomaly_score,
                    explanations_json,
                    now_iso,
                    now_iso,
                ),
            )
            # Explicit Python-level check to guarantee hard cap constraint
            self._enforce_lru(conn)
            conn.commit()

        return pattern_hash

    def _enforce_lru(self, conn: sqlite3.Connection) -> None:
        """Strictly prune rows beyond self.max_entries by oldest last_seen timestamp."""
        cursor = conn.execute("SELECT COUNT(*) FROM pattern_memory;")
        count = cursor.fetchone()[0]
        if count > self.max_entries:
            excess = count - self.max_entries
            conn.execute(
                """
                DELETE FROM pattern_memory
                WHERE pattern_hash IN (
                    SELECT pattern_hash FROM pattern_memory
                    ORDER BY last_seen ASC, hit_count ASC
                    LIMIT ?
                );
                """,
                (excess,),
            )

    def lookup_pattern(self, pattern_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve single pattern record by its unique SHA-256 hash."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM pattern_memory WHERE pattern_hash = ?;", (pattern_hash,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "pattern_hash": row["pattern_hash"],
                "client_id": row["client_id"],
                "endpoint": row["endpoint"],
                "attack_type": row["attack_type"],
                "features": json.loads(row["features"]),
                "verdict": row["verdict"],
                "anomaly_score": row["anomaly_score"],
                "explanations": json.loads(row["explanations"]),
                "hit_count": row["hit_count"],
                "created_at": row["created_at"],
                "last_seen": row["last_seen"],
            }

    def count_patterns(self) -> int:
        """Return total active pattern records stored."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pattern_memory;")
            return int(cursor.fetchone()[0])

    def get_recent_patterns(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve most recently observed threat patterns."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pattern_memory
                ORDER BY last_seen DESC
                LIMIT ?;
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "pattern_hash": row["pattern_hash"],
                        "client_id": row["client_id"],
                        "endpoint": row["endpoint"],
                        "attack_type": row["attack_type"],
                        "features": json.loads(row["features"]),
                        "verdict": row["verdict"],
                        "anomaly_score": row["anomaly_score"],
                        "explanations": json.loads(row["explanations"]),
                        "hit_count": row["hit_count"],
                        "created_at": row["created_at"],
                        "last_seen": row["last_seen"],
                    }
                )
            return results

    def clear(self) -> None:
        """Purge all stored patterns."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM pattern_memory;")
            conn.commit()


def seed_initial_threat_patterns(
    store: Optional[PatternMemoryStore] = None,
) -> int:
    """Pre-seed database with signatures of the 7 SBP regulatory attack vectors."""
    target_store = store or PatternMemoryStore()
    pipeline = get_security_pipeline()

    generators = [
        ("sql_injection", generate_sql_injection_attack),
        ("parameter_tampering", generate_parameter_tampering_attack),
        ("brute_force_login", generate_bruteforce_login_attack),
        ("credential_stuffing", generate_credential_stuffing_attack),
        ("bola", generate_bola_attack),
        ("api_flooding", generate_api_flooding_attack),
        ("token_replay", generate_token_replay_attack),
    ]

    total_seeded = 0
    for attack_name, gen_func in generators:
        requests = gen_func()
        for req in requests:
            raw_body = json.dumps(req.body).encode("utf-8") if req.body else b""
            token_val = req.headers.get("authorization", req.headers.get(settings.header_consent_id))
            features = extract_features_from_request(
                endpoint=req.endpoint,
                method=req.method,
                headers=req.headers,
                body_bytes=raw_body,
                client_key=req.client_id,
                account_id=req.account_id,
                token_id=token_val,
            )
            evaluation = pipeline.evaluate_features(
                features=features,
                endpoint=req.endpoint,
                method=req.method,
                client_id=req.client_id,
            )
            explanations_dicts = [
                {"feature": e.feature, "contribution": e.contribution, "description": e.description}
                for e in evaluation.explanations
            ]
            target_store.record_pattern(
                client_id=req.client_id,
                endpoint=req.endpoint,
                attack_type=attack_name,
                features=features,
                verdict=evaluation.verdict.value,
                anomaly_score=evaluation.anomaly_score,
                explanations=explanations_dicts,
            )
            total_seeded += 1

    logger.info("threat_patterns_seeded", total_seeded=total_seeded, db_path=str(target_store.db_path))
    return total_seeded


def main() -> None:
    """CLI utility to seed threat pattern database."""
    print("=" * 70)
    print("  SBP REGULATORY SANDBOX — SEEDING THREAT PATTERN MEMORY STORE")
    print("=" * 70)
    store = PatternMemoryStore()
    count_before = store.count_patterns()
    print(f"  Existing Patterns : {count_before}")
    seeded = seed_initial_threat_patterns(store)
    count_after = store.count_patterns()
    print(f"  New Patterns Seeded : {seeded}")
    print(f"  Total Patterns in DB: {count_after}")
    print(f"  Storage Location    : {store.db_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
