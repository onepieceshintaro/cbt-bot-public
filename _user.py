"""ユーザー識別・復元キー管理。

優先順位：
  1. URLクエリパラメータ ?u=<uuid>
  2. ホームディレクトリの ~/.note_apps_user_id（全アプリ共通）
  3. 新規生成

表示形式：F47A-C10B-58CC-4372-A567-0E02-B2C3-D479（4桁×8・大文字）
ニックネーム：~/.note_apps_nicknames.json に user_id→名前 で保持（ローカルのみ）
"""
import json
import uuid
from pathlib import Path

import streamlit as st

USER_ID_FILE = Path.home() / ".note_apps_user_id"
NICKNAMES_FILE = Path.home() / ".note_apps_nicknames.json"


# ---------------- 取得・生成 ----------------
def get_or_create_user_id() -> str:
    """現在のユーザーIDを返す。なければ生成して保存する。"""
    try:
        u = st.query_params.get("u")
    except Exception:
        u = None
    if u and _is_valid_hex(u):
        _save_local(u)
        return u

    uid = _load_local()
    if uid and _is_valid_hex(uid):
        _ensure_query_param(uid)
        return uid

    uid = uuid.uuid4().hex
    _save_local(uid)
    _ensure_query_param(uid)
    return uid


def switch_user(new_uid_hex: str) -> bool:
    """復元キーで別ユーザーに切り替える。成功すれば True。"""
    if not _is_valid_hex(new_uid_hex):
        return False
    _save_local(new_uid_hex)
    try:
        st.query_params["u"] = new_uid_hex
    except Exception:
        pass
    return True


def create_new_user() -> str:
    """完全に新しいUUIDを生成して切り替える。"""
    new_uid = uuid.uuid4().hex
    _save_local(new_uid)
    try:
        st.query_params["u"] = new_uid
    except Exception:
        pass
    return new_uid


# ---------------- ニックネーム ----------------
def get_nickname(uid: str) -> str:
    try:
        if NICKNAMES_FILE.exists():
            data = json.loads(NICKNAMES_FILE.read_text(encoding="utf-8"))
            return data.get(uid, "") or ""
    except Exception:
        pass
    return ""


def set_nickname(uid: str, nickname: str) -> None:
    try:
        data = {}
        if NICKNAMES_FILE.exists():
            data = json.loads(NICKNAMES_FILE.read_text(encoding="utf-8"))
        nickname = (nickname or "").strip()
        if nickname:
            data[uid] = nickname
        else:
            data.pop(uid, None)
        NICKNAMES_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
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


# ---------------- 永続化 ----------------
def _save_local(uid: str) -> None:
    try:
        USER_ID_FILE.write_text(uid.strip(), encoding="utf-8")
    except Exception:
        pass


def _load_local() -> str | None:
    try:
        if USER_ID_FILE.exists():
            return USER_ID_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _ensure_query_param(uid: str) -> None:
    try:
        current = st.query_params.get("u")
        if current != uid:
            st.query_params["u"] = uid
    except Exception:
        pass


# ---------------- サイドバーUI ----------------
def render_account_sidebar() -> str:
    """サイドバーにアカウント情報を描画し、user_id を返す。"""
    uid = get_or_create_user_id()
    current_nick = get_nickname(uid)
    display_label = current_nick if current_nick else "（名前未設定）"

    # サイドバー幅を少し広げる＋アカウントパネル内の余白を詰める
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        min-width: 320px !important;
        width: 320px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        with st.expander(f"👤 {display_label}", expanded=False):
            # --- ニックネーム ---
            st.caption("名前（任意・この端末にだけ保存）")
            new_nick = st.text_input(
                "名前",
                value=current_nick,
                label_visibility="collapsed",
                placeholder="例：しんたろう",
                key=f"nick_input_{uid[:6]}",
            )
            if new_nick != current_nick:
                set_nickname(uid, new_nick)
                st.rerun()

            # --- 復元キー ---
            st.caption("あなたの復元キー")
            st.code(format_restore_key(uid), language=None)

            # ブックマーク案内
            st.info(
                "💡 **このページをブックマーク**すれば、"
                "次回は開くだけで**自動ログイン**できます。"
            )

            st.markdown("---")

            # --- キー切替 ---
            st.caption("別のキーで読み込む")
            new_key = st.text_input(
                "復元キー",
                key=f"restore_key_input_{uid[:6]}",
                label_visibility="collapsed",
                placeholder="XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX",
            )
            if st.button("このキーで読み込む", key=f"restore_key_btn_{uid[:6]}"):
                parsed = parse_restore_key(new_key)
                if parsed:
                    if switch_user(parsed):
                        st.success("切り替えました。再読み込みします。")
                        st.rerun()
                else:
                    st.error("キーの形式が正しくありません（32文字のhex）")

            # --- 新規アカウント作成 ---
            confirm_key = f"confirm_new_user_{uid[:6]}"
            if st.session_state.get(confirm_key):
                st.warning(
                    "⚠️ 新しいアカウントに切り替えると、**今の復元キーが変わります**。"
                    "上の復元キーをスクショ等で保管してから進めてください"
                    "（あとで同じキーを入れれば戻れます）。"
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 作成する", key=f"yes_new_{uid[:6]}",
                                 use_container_width=True):
                        create_new_user()
                        st.session_state[confirm_key] = False
                        st.rerun()
                with c2:
                    if st.button("キャンセル", key=f"cancel_new_{uid[:6]}",
                                 use_container_width=True):
                        st.session_state[confirm_key] = False
                        st.rerun()
            else:
                if st.button("➕ 新しいアカウントを作る",
                             key=f"new_user_btn_{uid[:6]}"):
                    st.session_state[confirm_key] = True
                    st.rerun()
    return uid
