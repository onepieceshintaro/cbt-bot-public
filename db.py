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

        # user_nicknames（3アプリ共通・プレフィックス無し）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_nicknames (
                user_id TEXT PRIMARY KEY,
                nickname TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # cbt_stress_sources（Phase 1: 仕事ストレス源チェックリスト）
        # 何にしんどさを感じているか言語化する第一歩。
        # sources はチェック済み項目を JSON 配列で保存（自由度を保ちつつ集計可能）。
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS cbt_stress_sources (
                id {pk_id},
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sources TEXT,
                free_note TEXT
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cbt_stress_user "
            "ON cbt_stress_sources(user_id, created_at)"
        ))

        # cbt_nonverbal_memos（Phase 1: 「言葉にできないこと」専用メモ）
        # 書ける前提の UI が罠になる場合のフォールバック。
        # state_marker は絵文字や 1 文字シンボルで「今の状態」を選ぶ（任意）。
        # content は自由記述（任意・空でも OK）。
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS cbt_nonverbal_memos (
                id {pk_id},
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                state_markers TEXT,
                content TEXT
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cbt_nonverbal_user "
            "ON cbt_nonverbal_memos(user_id, created_at)"
        ))
