"""Phase A: 認知の歪みパターン辞書を意味検索する RAG モジュール。

設計方針:
- ルールベース（キーワードマッチ）の補完として動かす（置き換えではない）
- chromadb の PersistentClient でローカル保存
- 埋め込みは Voyage AI（Anthropic 推奨パートナー、`voyage-3` を採用）
- 「判定」ではなく「近いパターンの提示」に留める（安全設計 Lv1〜Lv2 範囲）

環境変数:
- VOYAGE_API_KEY : Voyage AI のキー（未設定時は import 時にエラーにせず lazy 初期化）

使い方:
    from rag.distortion_index import index_distortions, find_similar_distortions
    index_distortions()  # 初回のみ
    hits = find_similar_distortions("どうせ無理に決まってる")
    for h in hits:
        print(h["name"], h["score"])
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any, Optional

# .env を読み込む（VOYAGE_API_KEY を環境変数に展開）
try:
    from dotenv import load_dotenv  # noqa: WPS433
    load_dotenv()
except ImportError:
    pass

# 遅延 import（chromadb / voyageai は requirements に追加済みだが、
# ローカル開発で未インストールでもアプリ全体が落ちないように）
def _default_chroma_path() -> str:
    """OS / 実行環境に応じた chroma_db の保存先。

    優先順位:
      1. CBT_CHROMA_PATH 環境変数（明示指定）
      2. Windows ローカル開発: %LOCALAPPDATA%/cbt-bot/chroma_db
      3. Streamlit Cloud / Linux: ./chroma_db（コンテナ内一時領域）
         エフェメラルなので再起動時に自動再インデックスする想定。
    """
    explicit = os.environ.get("CBT_CHROMA_PATH")
    if explicit:
        return explicit
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return os.path.join(local_app, "cbt-bot", "chroma_db")
    # Linux/Streamlit Cloud: アプリ作業ディレクトリ配下
    return os.path.join(os.getcwd(), "chroma_db")


_CHROMA_PATH = _default_chroma_path()
_COLLECTION = "distortions"
_EMBED_MODEL = "voyage-3"


def _resolve_voyage_key() -> Optional[str]:
    """VOYAGE_API_KEY を環境変数 → Streamlit secrets の順で取得。"""
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
    # hnsw:space=cosine で距離を 0〜2 のコサイン距離に固定（1-distance が直感的なスコアになる）
    return client.get_or_create_collection(
        _COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(text: str) -> List[float]:
    """1 件の埋め込みを返す。"""
    res = _voyage().embed([text], model=_EMBED_MODEL, input_type="document")
    return res.embeddings[0]


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """複数テキストを 1 リクエストで埋め込む（レート制限対策）。"""
    if not texts:
        return []
    res = _voyage().embed(texts, model=_EMBED_MODEL, input_type="document")
    return res.embeddings


def _embed_query(text: str) -> List[float]:
    res = _voyage().embed([text], model=_EMBED_MODEL, input_type="query")
    return res.embeddings[0]


def _doc_text(name: str, tip: Dict[str, Any]) -> str:
    """歪みエントリ 1 件をインデックス用テキストに整形。"""
    parts = [
        name,
        tip.get("description", ""),
        tip.get("trap", ""),
        " / ".join(tip.get("actions", [])),
    ]
    return "\n".join(p for p in parts if p)


def index_distortions(distortion_tips: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    """歪み辞書を chromadb に upsert する。

    Returns:
        インデックスしたエントリ数
    """
    if distortion_tips is None:
        from distortion_tips import DISTORTION_TIPS  # ローカル import 循環回避
        distortion_tips = DISTORTION_TIPS

    col = _collection()
    names: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    for name, tip in distortion_tips.items():
        text = _doc_text(name, tip)
        names.append(name)
        docs.append(text)
        metas.append({"name": name})
    # 1 リクエストでバッチ埋め込み（3 RPM 制限対策）
    embs = _embed_batch(docs)

    if names:
        col.upsert(ids=names, documents=docs, embeddings=embs, metadatas=metas)
    return len(names)


def ensure_indexed() -> int:
    """コレクションが空なら歪み辞書をインデックスする（クラウドのコールドスタート対策）。

    Returns:
        現在のコレクションサイズ
    """
    col = _collection()
    try:
        if col.count() > 0:
            return col.count()
    except Exception:
        pass
    return index_distortions()


def find_similar_distortions(thought: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """自動思考に意味的に近い歪みパターンを top_k 件返す。

    Returns:
        [{"name": str, "score": float, "document": str}, ...]
        score は 1 - distance（cosine 想定、大きいほど近い）
    """
    if not thought or not thought.strip():
        return []
    # コレクションが空（コールドスタート）なら自動インデックス
    ensure_indexed()
    q = _embed_query(thought)
    res = _collection().query(query_embeddings=[q], n_results=top_k)
    out: List[Dict[str, Any]] = []
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for meta, doc, dist in zip(metas, docs, dists):
        out.append({
            "name": meta.get("name", ""),
            "score": 1.0 - float(dist),
            "document": doc,
        })
    return out


if __name__ == "__main__":
    # CLI: 初回インデックス + 簡易動作確認
    import time
    n = index_distortions()
    print(f"indexed: {n}")
    samples = ["どうせ無理に決まってる", "全部自分のせいだ", "あの人は私のことを嫌っている"]
    for i, sample in enumerate(samples):
        if i > 0:
            time.sleep(25)  # 無料枠 3 RPM 対策（決済登録後は削除可）
        print(f"\n[{sample}]")
        for hit in find_similar_distortions(sample):
            print(f"  {hit['name']}  score={hit['score']:.3f}")
