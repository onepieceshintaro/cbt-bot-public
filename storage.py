"""CBT 記録保存（SQLAlchemy・SQLite/Postgres両対応）。"""
import json
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from db import get_engine, init_db as _init_db
from time_utils import now_jst_naive


def _owner_user_id() -> str:
    from _user import get_or_create_user_id
    return get_or_create_user_id()


def init_db() -> None:
    _init_db()


def _as_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(f"- {v}" for v in value)
    return str(value)


def save_record(
    record: dict, conversation: list,
    event_datetime: str | None = None,
    mode: str | None = None,
    user_id: str | None = None,
) -> int:
    if event_datetime is None:
        event_datetime = now_jst_naive().isoformat()
    if user_id is None:
        user_id = _owner_user_id()

    emotion = record.get("emotion") or {}
    emotion_after = record.get("emotion_after") or {}
    balanced = record.get("balanced_thought")
    adaptive = record.get("adaptive_thought") or balanced
    distortions_val = record.get("distortions") or []

    params = {
        "user_id": user_id,
        "created_at": now_jst_naive().isoformat(),
        "event_datetime": event_datetime,
        "situation": record.get("situation"),
        "emotion_name": emotion.get("name"),
        "intensity_before": emotion.get("intensity_before"),
        "automatic_thought": record.get("automatic_thought"),
        "distortions": json.dumps(distortions_val, ensure_ascii=False),
        "adaptive_thought": adaptive,
        "intensity_after": emotion_after.get("intensity_after"),
        "raw_conversation": json.dumps(conversation, ensure_ascii=False),
        "mode": mode,
        "evidence_for": _as_text(record.get("evidence_for")),
        "evidence_against": _as_text(record.get("evidence_against")),
        "balanced_thought": balanced,
    }

    sql = text("""
        INSERT INTO cbt_thought_records
        (user_id, created_at, event_datetime, situation, emotion_name, intensity_before,
         automatic_thought, distortions, adaptive_thought,
         intensity_after, raw_conversation,
         mode, evidence_for, evidence_against, balanced_thought)
        VALUES
        (:user_id, :created_at, :event_datetime, :situation, :emotion_name, :intensity_before,
         :automatic_thought, :distortions, :adaptive_thought,
         :intensity_after, :raw_conversation,
         :mode, :evidence_for, :evidence_against, :balanced_thought)
        RETURNING id
    """)
    with get_engine().begin() as conn:
        row = conn.execute(sql, params).first()
        return int(row[0])


def update_distortions(record_id: int, distortions: list,
                       user_id: str | None = None):
    """歪み一覧を更新。要素は str（旧形式）でも dict（新形式）でも可。
    新形式: [{"name": "...", "evidence": "...", "dismissed": False}]
    """
    if user_id is None:
        user_id = _owner_user_id()
    sql = text("""
        UPDATE cbt_thought_records
        SET distortions = :distortions
        WHERE id = :id AND user_id = :user_id
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "distortions": json.dumps(distortions, ensure_ascii=False),
            "id": record_id,
            "user_id": user_id,
        })


def dismiss_distortion(record_id: int, distortion_name: str,
                       dismissed: bool = True,
                       user_id: str | None = None) -> int:
    """指定の歪みに対して「これは違いました」フラグを立てる/外す。
    旧形式（list[str]）は dict 化したうえで更新する。
    """
    if user_id is None:
        user_id = _owner_user_id()
    sel = text("""
        SELECT distortions FROM cbt_thought_records
        WHERE id = :id AND user_id = :user_id
    """)
    upd = text("""
        UPDATE cbt_thought_records SET distortions = :distortions
        WHERE id = :id AND user_id = :user_id
    """)
    with get_engine().begin() as conn:
        row = conn.execute(sel, {"id": record_id, "user_id": user_id}).first()
        if not row:
            return 0
        try:
            current = json.loads(row[0] or "[]")
        except Exception:
            current = []
        new_list = []
        _now_iso = now_jst_naive().isoformat()
        for it in current:
            if isinstance(it, dict):
                # _meta sentinel はそのまま保持（dismiss 対象ではない）
                if it.get("_meta"):
                    new_list.append(it)
                    continue
                d = dict(it)
                if d.get("name") == distortion_name:
                    d["dismissed"] = dismissed
                    if dismissed:
                        d["dismissed_at"] = _now_iso
                        # 違和感を押した時点で「ユーザーに見られた」とみなす
                        d["shown"] = True
                    else:
                        # 戻すときは時刻クリア
                        d.pop("dismissed_at", None)
                new_list.append(d)
            elif isinstance(it, str):
                d = {
                    "name": it,
                    "evidence": "",
                    "dismissed": dismissed if it == distortion_name else False,
                }
                if it == distortion_name and dismissed:
                    d["dismissed_at"] = _now_iso
                new_list.append(d)
        conn.execute(upd, {
            "distortions": json.dumps(new_list, ensure_ascii=False),
            "id": record_id,
            "user_id": user_id,
        })
        return 1


def log_engagement(record_id: int, event_name: str,
                   user_id: str | None = None) -> int:
    """ユーザーの engagement イベントを記録に追記する（Phase A・C 用）。

    event_name の例:
      - "other_views_opened" : 「他の見方も見てみる」を開いた
      - "evidence_viewed"    : evidence エキスパンダーを能動的に開いた（任意）

    実装：cbt_thought_records.distortions JSON の中に
        {"_meta": True, "events": {event_name: iso_timestamp, ...}}
    という sentinel item として保存する。スキーマ変更を回避。

    同じイベントは**初回タイムスタンプを保持**（多重クリックで上書きしない）。
    """
    if user_id is None:
        user_id = _owner_user_id()
    sel = text("""
        SELECT distortions FROM cbt_thought_records
        WHERE id = :id AND user_id = :user_id
    """)
    upd = text("""
        UPDATE cbt_thought_records SET distortions = :distortions
        WHERE id = :id AND user_id = :user_id
    """)
    iso_now = now_jst_naive().isoformat()
    with get_engine().begin() as conn:
        row = conn.execute(sel, {"id": record_id, "user_id": user_id}).first()
        if not row:
            return 0
        try:
            current = json.loads(row[0] or "[]")
        except Exception:
            current = []

        meta = None
        for it in current:
            if isinstance(it, dict) and it.get("_meta"):
                meta = it
                break
        if meta is None:
            meta = {"_meta": True, "events": {}}
            current.append(meta)
        else:
            meta.setdefault("events", {})

        # 初回のみ記録（多重クリックの上書き防止）
        if event_name not in meta["events"]:
            meta["events"][event_name] = iso_now

        conn.execute(upd, {
            "distortions": json.dumps(current, ensure_ascii=False),
            "id": record_id,
            "user_id": user_id,
        })
    return 1


def get_distortion_ab_summary(user_id: str | None = None) -> dict:
    """全レコードを走査して、source ごとの impression / dismiss 数を集計。

    Phase A の A/B 観察用。LLM 単独 / RAG 単独（影ログ）/ 両者一致 で
    違和感率の差を見る。

    Returns:
        {
          "impressions": {"llm": N, "both": N, "rag": N},   # 件数（重複あり）
          "dismissals": {"llm": N, "both": N, "rag": N},
          "dismiss_rate": {"llm": 0.xx, "both": 0.xx, "rag": 0.xx},
          "pattern_table": [
            {"source": "llm", "name": "...", "shown": N, "dismissed": N, "rate": 0.xx}
          ],
          "total_records": N,
        }
    """
    if user_id is None:
        user_id = _owner_user_id()

    sel = text("""
        SELECT distortions FROM cbt_thought_records
        WHERE user_id = :uid AND distortions IS NOT NULL
    """)
    impressions: dict[str, int] = {"llm": 0, "both": 0, "rag": 0}
    dismissals: dict[str, int] = {"llm": 0, "both": 0, "rag": 0}
    pattern_counts: dict[tuple[str, str], dict] = {}  # (source, name) -> {"shown", "dismissed"}
    total_records = 0

    with get_engine().begin() as conn:
        rows = conn.execute(sel, {"uid": user_id}).fetchall()

    for row in rows:
        try:
            dists = json.loads(row[0] or "[]")
        except Exception:
            continue
        total_records += 1
        for d in dists:
            if not isinstance(d, dict):
                continue
            if d.get("_meta"):
                continue  # メタ情報は集計対象外
            name = d.get("name")
            if not name:
                continue
            source = d.get("source", "llm")  # 既存データは "llm" 扱い（後方互換）
            if source not in impressions:
                continue
            impressions[source] += 1
            key = (source, name)
            pattern_counts.setdefault(key, {"shown": 0, "dismissed": 0})
            pattern_counts[key]["shown"] += 1
            if d.get("dismissed"):
                dismissals[source] += 1
                pattern_counts[key]["dismissed"] += 1

    dismiss_rate = {
        s: (dismissals[s] / impressions[s] if impressions[s] else 0.0)
        for s in impressions
    }

    pattern_table = [
        {
            "source": s,
            "name": n,
            "shown": v["shown"],
            "dismissed": v["dismissed"],
            "rate": (v["dismissed"] / v["shown"]) if v["shown"] else 0.0,
        }
        for (s, n), v in pattern_counts.items()
    ]
    pattern_table.sort(key=lambda x: (-x["shown"], x["source"], x["name"]))

    return {
        "impressions": impressions,
        "dismissals": dismissals,
        "dismiss_rate": dismiss_rate,
        "pattern_table": pattern_table,
        "total_records": total_records,
    }


# 編集可能なフィールド一覧（DBカラム名と一致）
# 思想：詳しい内容は対話で固めたものなので後から書き換えない。
# 日時のみ修正できれば、書き間違いに対応できる。
_EDITABLE_FIELDS = {
    "event_datetime",
}


def update_record(record_id: int, fields: dict,
                  user_id: str | None = None) -> int:
    """過去の記録を編集する。fields に渡された列だけ更新する。
    distortions はリスト（str/dict 混在可）で渡せばJSON化して保存する。"""
    if user_id is None:
        user_id = _owner_user_id()
    if not fields:
        return 0

    safe: dict = {}
    for k, v in fields.items():
        if k not in _EDITABLE_FIELDS:
            continue
        if k == "distortions" and v is not None and not isinstance(v, str):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = v

    if not safe:
        return 0

    sets = ", ".join(f"{col} = :{col}" for col in safe.keys())
    sql = text(f"""
        UPDATE cbt_thought_records
        SET {sets}
        WHERE id = :id AND user_id = :user_id
    """)
    params = {**safe, "id": record_id, "user_id": user_id}
    with get_engine().begin() as conn:
        result = conn.execute(sql, params)
        return result.rowcount or 0


def load_records(user_id: str | None = None) -> pd.DataFrame:
    if user_id is None:
        user_id = _owner_user_id()
    with get_engine().connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM cbt_thought_records "
                 "WHERE user_id = :user_id ORDER BY created_at"),
            conn, params={"user_id": user_id},
        )


def save_weekly_report(week_start, week_end, markdown: str, n_records: int,
                       user_id: str | None = None):
    if user_id is None:
        user_id = _owner_user_id()
    sql = text("""
        INSERT INTO cbt_weekly_reports
        (user_id, week_start, week_end, generated_at, n_records, markdown)
        VALUES (:user_id, :week_start, :week_end, :generated_at, :n_records, :markdown)
        ON CONFLICT (user_id, week_start) DO UPDATE SET
            week_end = EXCLUDED.week_end,
            generated_at = EXCLUDED.generated_at,
            n_records = EXCLUDED.n_records,
            markdown = EXCLUDED.markdown
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "user_id": user_id,
            "week_start": str(week_start),
            "week_end": str(week_end),
            "generated_at": now_jst_naive().isoformat(),
            "n_records": n_records,
            "markdown": markdown,
        })


def load_weekly_report(week_start, user_id: str | None = None) -> dict | None:
    if user_id is None:
        user_id = _owner_user_id()
    sql = text(
        "SELECT week_start, week_end, generated_at, n_records, markdown "
        "FROM cbt_weekly_reports "
        "WHERE week_start = :week_start AND user_id = :user_id"
    )
    with get_engine().connect() as conn:
        row = conn.execute(sql, {
            "week_start": str(week_start), "user_id": user_id,
        }).first()
    if not row:
        return None
    return {
        "week_start": row[0], "week_end": row[1],
        "generated_at": row[2], "n_records": row[3],
        "markdown": row[4],
    }


def load_all_weekly_reports(user_id: str | None = None) -> pd.DataFrame:
    if user_id is None:
        user_id = _owner_user_id()
    with get_engine().connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM cbt_weekly_reports "
                 "WHERE user_id = :user_id ORDER BY week_start DESC"),
            conn, params={"user_id": user_id},
        )


def save_risk_score(user_message: str, result: dict, user_id: str | None = None):
    if user_id is None:
        user_id = _owner_user_id()
    s = result.get("score", {}) or {}
    sql = text("""
        INSERT INTO cbt_risk_scores
        (user_id, created_at, user_message, triggered, source, overall,
         self_harm, harm_to_others, abuse, acute, reasoning)
        VALUES
        (:user_id, :created_at, :user_message, :triggered, :source, :overall,
         :self_harm, :harm_to_others, :abuse, :acute, :reasoning)
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "user_id": user_id,
            "created_at": now_jst_naive().isoformat(),
            "user_message": user_message,
            "triggered": 1 if result.get("triggered") else 0,
            "source": result.get("source"),
            "overall": s.get("overall", 0),
            "self_harm": s.get("self_harm", 0),
            "harm_to_others": s.get("harm_to_others", 0),
            "abuse": s.get("abuse", 0),
            "acute": s.get("acute", 0),
            "reasoning": s.get("reasoning", ""),
        })


def load_risk_scores(user_id: str | None = None) -> pd.DataFrame:
    if user_id is None:
        user_id = _owner_user_id()
    with get_engine().connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM cbt_risk_scores "
                 "WHERE user_id = :user_id ORDER BY created_at DESC"),
            conn, params={"user_id": user_id},
        )
