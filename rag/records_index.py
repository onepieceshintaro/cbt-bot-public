"""Phase B-α: 過去の自動思考を意味検索する Self-RAG モジュール。

設計方針:
- ユーザーごとにメタデータでフィルタする単一コレクション（user_records）
- 距離指標は cosine
- chromadb の永続化は distortion_index と同じ path を共有
- 失敗はすべてサイレント（フェイルセーフ）

UX 設計上の安全装置:
- min_days_old=14: 直近 2 週間以内の記録は「最近すぎて気付きにならない」ので除外
- min_score: 閾値で品質ゲート
- exclude_record_id: 検索元自身は除外

UI 表示の解放条件は呼び出し側（cbt_engine / app）で記録件数しきい値を見て判断する。
"""
from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional, Iterable, Tuple

# .env を読み込む
try:
    from dotenv import load_dotenv  # noqa: WPS433
    load_dotenv()
except ImportError:
    pass

_COLLECTION = "user_records"
_EMBED_MODEL = "voyage-3"


def _default_chroma_path() -> str:
    explicit = os.environ.get("CBT_CHROMA_PATH")
    if explicit:
        return explicit
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return os.path.join(local_app, "cbt-bot", "chroma_db")
    return os.path.join(os.getcwd(), "chroma_db")


_CHROMA_PATH = _default_chroma_path()


def _resolve_voyage_key() -> Optional[str]:
    key = os.environ.get("VOYAGE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # noqa: WPS433
        return st.secrets.get("VOYAGE_API_KEY")  # type: ignore[union-attr]
    except Exception:
        return None


@lru_cache(maxsize=1)
def _voyage():
    import voyageai  # noqa: WPS433
    return voyageai.Client(api_key=_resolve_voyage_key())


@lru_cache(maxsize=1)
def _collection():
    import chromadb  # noqa: WPS433
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    return client.get_or_create_collection(
        _COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_doc(text: str) -> List[float]:
    res = _voyage().embed([text], model=_EMBED_MODEL, input_type="document")
    return res.embeddings[0]


def _embed_query_one(text: str) -> List[float]:
    res = _voyage().embed([text], model=_EMBED_MODEL, input_type="query")
    return res.embeddings[0]


def _embed_doc_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    res = _voyage().embed(texts, model=_EMBED_MODEL, input_type="document")
    return res.embeddings


def _make_id(user_id: str, record_id: Any) -> str:
    """単一コレクション内での一意 ID。"""
    return f"{user_id}::{record_id}"


def index_record(
    user_id: str,
    record_id: Any,
    automatic_thought: str,
    created_at: str,
) -> bool:
    """1 件をベクタ DB に upsert する。

    Returns:
        True: upsert 成功
        False: フェイルセーフ（API キー欠落 / レート制限 / 空テキスト 等）
    """
    if not user_id or not automatic_thought or not automatic_thought.strip():
        return False
    try:
        emb = _embed_doc(automatic_thought)
        _collection().upsert(
            ids=[_make_id(user_id, record_id)],
            documents=[automatic_thought],
            embeddings=[emb],
            metadatas=[{
                "user_id": user_id,
                "record_id": str(record_id),
                "created_at": created_at or "",
            }],
        )
        return True
    except Exception:
        return False


def ensure_user_indexed(
    user_id: str,
    records: Iterable[Tuple[Any, str, str]],
) -> int:
    """指定ユーザーの記録がコレクションに無ければバルク投入する。

    Streamlit Cloud のエフェメラル領域でコールドスタートした際に呼ぶ想定。
    既に1件でも存在すれば何もしない。

    Args:
        records: [(record_id, automatic_thought, created_at), ...]

    Returns:
        新規 upsert した件数（既存なら 0）
    """
    if not user_id:
        return 0
    try:
        existing = _collection().get(where={"user_id": user_id}, limit=1)
        if existing and existing.get("ids"):
            return 0
    except Exception:
        pass

    items: List[Tuple[Any, str, str]] = []
    for r in records or []:
        try:
            rid, thought, created = r
        except Exception:
            continue
        if not thought or not str(thought).strip():
            continue
        items.append((rid, str(thought), str(created or "")))
    if not items:
        return 0

    try:
        ids = [_make_id(user_id, rid) for rid, _, _ in items]
        docs = [thought for _, thought, _ in items]
        metas = [
            {"user_id": user_id, "record_id": str(rid), "created_at": created}
            for rid, _, created in items
        ]
        embs = _embed_doc_batch(docs)
        _collection().upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        return len(docs)
    except Exception:
        return 0


def find_similar_past_records(
    user_id: str,
    automatic_thought: str,
    *,
    top_k: int = 3,
    min_days_old: int = 14,
    min_score: float = 0.0,
    exclude_record_id: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """指定ユーザーの過去記録から、意味的に近いものを返す（フェイルセーフ）。

    Returns:
        [{"record_id": str, "created_at": str, "text": str,
          "score": float, "age_days": int}, ...]

    安全装置:
        min_days_old: この日数より新しい記録は「最近すぎる」として除外
        min_score: cosine 類似度の最低値
        exclude_record_id: 検索元自身は除外
    """
    if not user_id or not automatic_thought or not automatic_thought.strip():
        return []
    try:
        q = _embed_query_one(automatic_thought)
        res = _collection().query(
            query_embeddings=[q],
            n_results=max(top_k * 3, 10),
            where={"user_id": user_id},
        )
    except Exception:
        return []

    # 現在時刻（JST naive）
    try:
        from time_utils import now_jst_naive
        now = now_jst_naive()
    except Exception:
        now = datetime.now()

    out: List[Dict[str, Any]] = []
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for meta, doc, dist in zip(metas, docs, dists):
        rid = str(meta.get("record_id", ""))
        if exclude_record_id is not None and rid == str(exclude_record_id):
            continue
        score = 1.0 - float(dist)
        if score < min_score:
            continue
        created_str = str(meta.get("created_at", ""))
        age_days = 0
        try:
            created = datetime.fromisoformat(created_str)
            age_days = (now - created).days
        except Exception:
            age_days = 0
        if age_days < min_days_old:
            continue
        out.append({
            "record_id": rid,
            "created_at": created_str,
            "text": doc,
            "score": score,
            "age_days": age_days,
        })
        if len(out) >= top_k:
            break
    return out


def get_user_record_count(user_id: str) -> int:
    """このユーザーの埋め込み済み記録数（しきい値判定用）。"""
    if not user_id:
        return 0
    try:
        res = _collection().get(where={"user_id": user_id})
        ids = res.get("ids") or []
        return len(ids)
    except Exception:
        return 0
