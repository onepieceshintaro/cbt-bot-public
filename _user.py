"""ユーザー識別・復元キー・ニックネーム管理。

- ユーザーID: URL ?u= から取得、無ければ新規生成
- ニックネーム: Supabase の user_nicknames テーブル（3アプリ共有）

※ Streamlit Cloud は1コンテナを複数ユーザーが共有するため、
  ローカルFSは使わない（他ユーザーから見えてしまうため）。
  ユーザーIDの永続化はブラウザのブックマーク（URLの ?u=）で行う。

表示形式：F47A-C10B-58CC-4372-A567-0E02-B2C3-D479（4桁×8・大文字）
"""
import uuid
from datetime import datetime

import streamlit as st
from sqlalchemy import text

from db import get_engine


# ---------------- 取得・生成 ----------------
def get_or_create_user_id() -> str:
    """現在のユーザーIDを返す。なければ生成する。

    注意：ローカルFSからの復元は行わない（Cloud上で他ユーザーのIDが
    漏洩するため）。永続化はURL?u=パラメータのみ。
    """
    try:
        u = st.query_params.get("u")
    except Exception:
        u = None
    if u and _is_valid_hex(u):
        return u

    # ?u= が無い or 無効 → 新規UUIDを発行してURLに書き込む
    uid = uuid.uuid4().hex
    _ensure_query_param(uid)
    return uid


# ---------------- ニックネーム（DB保存・3アプリ共有） ----------------
def get_nickname(uid: str) -> str:
    if not _is_valid_hex(uid):
        return ""
    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                text("SELECT nickname FROM user_nicknames WHERE user_id = :uid"),
                {"uid": uid},
            ).fetchone()
            return (row[0] if row else "") or ""
    except Exception:
        return ""


def set_nickname(uid: str, nickname: str) -> None:
    if not _is_valid_hex(uid):
        return
    nickname = (nickname or "").strip()
    try:
        with get_engine().begin() as conn:
            if not nickname:
                conn.execute(
                    text("DELETE FROM user_nicknames WHERE user_id = :uid"),
                    {"uid": uid},
                )
            else:
                conn.execute(
                    text("""
                        INSERT INTO user_nicknames (user_id, nickname, updated_at)
                        VALUES (:uid, :nick, :now)
                        ON CONFLICT (user_id) DO UPDATE
                        SET nickname = EXCLUDED.nickname,
                            updated_at = EXCLUDED.updated_at
                    """),
                    {
                        "uid": uid,
                        "nick": nickname,
                        "now": datetime.now().isoformat(),
                    },
                )
    except Exception:
        pass


# ---------------- 形式変換 ----------------
def format_restore_key(uid_hex: str) -> str:
    """32文字hex → F47A-C10B-... の8ブロック表示。"""
    s = uid_hex.strip().replace("-", "").upper()
    if len(s) != 32:
        return uid_hex
    return "-".join(s[i:i + 4] for i in range(0, 32, 4))


def parse_restore_key(user_input: str) -> str | None:
    """ユーザー入力 → 32文字小文字hex。不正なら None。"""
    if not user_input:
        return None
    s = "".join(c for c in user_input if c.isalnum()).lower()
    if _is_valid_hex(s):
        return s
    return None


def _is_valid_hex(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 32:
        return False
    return all(c in "0123456789abcdef" for c in s.lower())


def _ensure_query_param(uid: str) -> None:
    try:
        current = st.query_params.get("u")
        if current != uid:
            st.query_params["u"] = uid
    except Exception:
        pass


# ---------------- サイドバーUI ----------------
def render_account_sidebar() -> str:
    """サイドバーに現在のユーザー表示だけ出し、user_id を返す。

    編集・切替UIは HOME ページに集約。
    """
    uid = get_or_create_user_id()
    current_nick = get_nickname(uid)
    display_label = current_nick if current_nick else "（名前未設定）"

    with st.sidebar:
        st.markdown(f"👤 **{display_label}**")
    return uid
