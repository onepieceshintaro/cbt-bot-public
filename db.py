"""DB抽象レイヤー（CBTセルフヘルプ・公開版）。
共有 Supabase 上のテーブル prefix: `cbt_`
"""
import os
from functools import lru_cache

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def _get_database_url() -> str:
    try:
        url = st.secrets.get("DATABASE_URL")
        if url:
            return url
    except Exception:
        pass
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return "sqlite:///cbt.db"


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = _normalize_url(_get_database_url())
    return create_engine(url, pool_pre_ping=True, future=True)


def is_postgres() -> bool:
    return "postgresql" in str(get_engine().url)


def init_db() -> None:
    """3テーブルを作成（冪等）。"""
    engine = get_engine()
    pg = is_postgres()
    pk_id = "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"

    with engine.begin() as conn:
        # cbt_thought_records
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS cbt_thought_records (
                id {pk_id},
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_datetime TEXT,
                situation TEXT,
                emotion_name TEXT,
                intensity_before INTEGER,
                automatic_thought TEXT,
                distortions TEXT,
                adaptive_thought TEXT,
                intensity_after INTEGER,
                raw_conversation TEXT,
                mode TEXT,
                evidence_for TEXT,
                evidence_against TEXT,
                balanced_thought TEXT
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cbt_thought_user "
            "ON cbt_thought_records(user_id, created_at)"
        ))

        # cbt_weekly_reports
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cbt_weekly_reports (
                user_id TEXT NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                n_records INTEGER,
                markdown TEXT NOT NULL,
                PRIMARY KEY (user_id, week_start)
            )
        """))

        # cbt_risk_scores（level 列なし）
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS cbt_risk_scores (
                id {pk_id},
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_message TEXT,
                triggered INTEGER,
                source TEXT,
                overall INTEGER,
                self_harm INTEGER,
                harm_to_others INTEGER,
                abuse INTEGER,
                acute INTEGER,
                reasoning TEXT
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cbt_risk_user "
            "ON cbt_risk_scores(user_id, created_at)"
        ))
