import os
import json
import re
from pathlib import Path
from anthropic import Anthropic
from prompts import (
    SYSTEM_PROMPT, CRISIS_KEYWORDS, CRISIS_RESPONSE,
    MODE_CONFIGS, DEFAULT_MODE,
)
from risk import score_risk, is_high_risk, CRISIS_SCORE_THRESHOLD
from dotenv import load_dotenv

# スクリプトと同じディレクトリの .env を確実に読み込む
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH, override=False)

# 優先順位：Streamlit Cloud secrets → 環境変数 → .env
api_key = None
try:
    import streamlit as st  # type: ignore
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
except Exception:
    pass
if not api_key:
    api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY が見つかりません。"
        f"Streamlit の secrets.toml または {ENV_PATH} を確認してください。"
    )

client = Anthropic(api_key=api_key)

MODEL = "claude-sonnet-4-5"


def check_crisis(text: str) -> bool:
    return any(kw in text for kw in CRISIS_KEYWORDS)


def assess_risk(user_message: str) -> dict:
    """発話のリスク評価（キーワード判定 + LLMスコアリングの二層防御）。

    戻り値:
      - triggered: bool（CRISIS_RESPONSE を表示すべきか）
      - source: "keyword" | "llm" | None
      - score: dict（各軸0-10 + overall + reasoning）
    """
    # レイヤー1：キーワード即時判定
    if check_crisis(user_message):
        return {
            "triggered": True,
            "source": "keyword",
            "score": {
                "self_harm": 10, "harm_to_others": 0,
                "abuse": 0, "acute": 10, "overall": 10,
                "reasoning": "危機キーワードが一致しました",
            },
        }

    # レイヤー2：LLMスコアリング（失敗時は0、フェイルセーフ）
    score = score_risk(user_message, client)
    if is_high_risk(score):
        return {"triggered": True, "source": "llm", "score": score}
    return {"triggered": False, "source": None, "score": score}


def chat(messages: list[dict], mode: str = DEFAULT_MODE) -> str:
    """
    messages: [{"role": "user"|"assistant", "content": "..."}, ...]
    mode: "distortion" or "seven_columns"
    戻り値: AIの応答テキスト
    """
    # 直近ユーザー発言で危機ワードをチェック（二重防御の一層目）
    if messages and messages[-1]["role"] == "user":
        if check_crisis(messages[-1]["content"]):
            return CRISIS_RESPONSE

    config = MODE_CONFIGS.get(mode, MODE_CONFIGS[DEFAULT_MODE])
    system_prompt = config["system_prompt"]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )
    return response.content[0].text


# 認知の歪みパターン（推定用）
# 「結論の飛躍」は範囲が広いので、Burns の定義に従い 2 サブタイプを採用：
#   - 結論の飛躍（占い）   = Fortune Telling（未来の悪い結末を先読み）
#   - 結論の飛躍（読心）   = Mind Reading  （相手の内心を勝手に解釈）
DISTORTION_PATTERNS = [
    "全か無か思考", "過度の一般化", "心のフィルター", "マイナス化思考",
    "結論の飛躍（占い）", "結論の飛躍（読心）",
    "拡大解釈と過小評価", "感情的決めつけ", "すべき思考",
    "レッテル貼り", "個人化",
]


def infer_distortions_from_record(record: dict) -> list[dict]:
    """7コラム法のセッション完了後にバックエンドで歪みを推定する。
    各歪みについて「どこからそう判断したか」の根拠も併せて返す。

    戻り値: [{"name": "結論の飛躍（占い）", "evidence": "…と未来を断定的に予測している"}, ...]
    Haikuで1コール。失敗時は空リスト（フェイルセーフ）。"""
    prompt = f"""以下の思考記録から、該当しそうな**認知の歪み**を**最大3つまで**選んでください。
該当なしの場合は空リスト。各歪みについて、なぜそう判断したかの**根拠**も短く添えてください。

# 認知の歪みパターン
{" / ".join(DISTORTION_PATTERNS)}

「結論の飛躍」は2タイプに分かれます：
- 結論の飛躍（占い）：未来の悪い結末を根拠なく先読み（例：「失敗するに決まってる」）
- 結論の飛躍（読心）：相手の内心を根拠なく悪く解釈（例：「嫌われたに違いない」）

# 思考記録
- 状況: {record.get("situation", "")}
- 自動思考: {record.get("automatic_thought", "")}
- 根拠（事実）: {record.get("evidence_for", "")}
- 反証: {record.get("evidence_against", "")}
- バランス思考: {record.get("balanced_thought", "")}

# 出力
必ず以下のJSONのみ（コードブロックや説明文なし）:
{{"distortions": [
  {{"name": "歪み名", "evidence": "自動思考のどこを見て、なぜその歪みと判断したかを1〜2文で"}}
]}}

# 根拠の書き方（重要：トーン）
- 自動思考や記録から具体的な語句を引用しつつ、なぜその歪みパターンに該当しそうかを示す
- **断定形は使わない**。「〜と判断しているように見える」「〜と取れる側面がある」「〜の根拠が示されていない可能性」「〜と捉えてしまう傾向がうかがえる」のような**推察形・控えめな表現**で書く
- 「決めつけ」になる表現（「〜と決めつけている」「〜と断定している」「〜を除外している」）は使わない
- ユーザーが「いや、それは違う」と感じたら外せる前提なので、AIの推察として丁寧に
- 60文字以内が目安。長すぎない

歪み名は上記から正確に選んでください。
"""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        items = data.get("distortions") or []
        out: list[dict] = []
        for d in items:
            # 旧形式（文字列のみ）にも一応対応
            if isinstance(d, str):
                if d in DISTORTION_PATTERNS:
                    out.append({"name": d, "evidence": ""})
                continue
            if not isinstance(d, dict):
                continue
            name = d.get("name")
            if name not in DISTORTION_PATTERNS:
                continue
            ev = (d.get("evidence") or "").strip()
            out.append({"name": name, "evidence": ev})
            if len(out) >= 3:
                break
        return out
    except Exception:
        return []


# ──────────────────────────────────────────────
# RAG 版の歪み推定（Phase A: 意味検索ベース）
# ──────────────────────────────────────────────
# 自動思考のテキストだけを使って、Voyage AI 埋め込み + chromadb で類似歪みを抽出する。
# LLM 版（Haiku で全記録を見て選び evidence も付ける）と並走させ、A/B 比較する。

# RAG ヒットを採用する閾値（cosine 類似度 = 1 - cosine_distance）
_RAG_MIN_SCORE = 0.20


def infer_distortions_via_rag(automatic_thought: str, top_k: int = 3) -> list[dict]:
    """自動思考の意味検索で類似歪みを返す（フェイルセーフ）。

    返り値: [{"name": "...", "evidence": "意味的に近い", "score": 0.xx}, ...]
    - DISTORTION_PATTERNS に含まれる正規名のみ（親カテゴリ「結論の飛躍」は除外）
    - 失敗時は []
    """
    if not automatic_thought or not automatic_thought.strip():
        return []
    try:
        from rag.distortion_index import find_similar_distortions
        hits = find_similar_distortions(automatic_thought, top_k=max(top_k * 2, 5))
    except Exception:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for h in hits:
        name = h.get("name", "")
        score = float(h.get("score", 0.0))
        if score < _RAG_MIN_SCORE:
            continue
        if name not in DISTORTION_PATTERNS:
            continue  # 親カテゴリ「結論の飛躍」などは除外
        if name in seen:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "evidence": f"意味検索で類似（score={score:.2f}）",
            "score": score,
        })
        if len(out) >= top_k:
            break
    return out


def infer_distortions_ab(record: dict) -> dict:
    """LLM 版と RAG 版を並走させ、両方の結果を返す（A/B 比較用）。

    Returns:
        {
          "llm": [{"name": .., "evidence": ..}, ...],          # LLM の生結果
          "rag": [{"name": .., "evidence": .., "score": ..}],  # RAG の生結果
          "intersection": ["name", ...],                       # 両者一致した名前
          "union_names": ["name", ...],                        # 全名前（dedupe）
          "tagged": [                                          # 保存用 tagged list
            {
              "name": str,
              "evidence": str,
              "source": "llm" | "rag" | "both",
              "shown": bool,        # True=ユーザーに表示, False=影ログ
              "dismissed": bool,    # 初期値 False
              "rag_score": float | None,  # RAG ヒット時のみ
            }, ...
          ]
        }
    """
    llm = infer_distortions_from_record(record)
    rag = infer_distortions_via_rag(record.get("automatic_thought", ""))
    llm_names = [d.get("name") for d in llm if d.get("name")]
    rag_names = [d.get("name") for d in rag if d.get("name")]
    inter = [n for n in llm_names if n in rag_names]

    # rag を name でルックアップ可能に
    rag_by_name = {d["name"]: d for d in rag if d.get("name")}

    tagged: list[dict] = []
    seen: set[str] = set()

    # ① LLM が拾った項目（ユーザーに表示する）
    for d in llm:
        name = d.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        source = "both" if name in rag_names else "llm"
        item = {
            "name": name,
            "evidence": d.get("evidence", ""),
            "source": source,
            "shown": True,
            "dismissed": False,
        }
        if source == "both":
            item["rag_score"] = rag_by_name.get(name, {}).get("score")
        tagged.append(item)

    # ② RAG だけが拾った項目（影ログ・表示しない）
    for d in rag:
        name = d.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        tagged.append({
            "name": name,
            "evidence": d.get("evidence", ""),
            "source": "rag",
            "shown": False,
            "dismissed": False,
            "rag_score": d.get("score"),
        })

    union = [d["name"] for d in tagged]

    return {
        "llm": llm,
        "rag": rag,
        "intersection": inter,
        "union_names": union,
        "tagged": tagged,
    }


# バランス思考の提案（7コラム法のバランス思考フェーズでの補助ツール）
BALANCED_THOUGHT_SYSTEM = """あなたはCBTの「7コラム法」でバランス思考を考える補助ツールです。
ユーザーとAIのこれまでの対話（出来事・自動思考・感情・根拠・反証）を踏まえ、
ユーザーが取りうる**バランス思考の例を3つ**提案してください。

# バランス思考に含めたい4要素
1. もともとの感情や反応が**自然だった**という肯定
2. 相手や状況には**別の文脈**があり得たという可能性
3. 自分の価値を否定する必要がない、という**尊厳の確認**
4. 次に試せる**小さな一歩**や工夫（無理に入れなくてよい）

# トーン
- 「〜かもしれない」「〜のような見方もできそう」と控えめに
- 断定せず、ユーザーの言葉を尊重する
- ユーザーが既に出している言葉を引用しながら

# 各提案の長さ
2〜4文。短すぎず、長すぎず。

# 出力形式
必ず以下のJSONのみ（コードブロックや説明文なし）:
{"suggestions": ["提案1の文章", "提案2の文章", "提案3の文章"]}

3つの提案は、それぞれ違う切り口（相手視点／自分への優しさ／小さな行動 など）にしてください。
"""


def suggest_balanced_thoughts(messages: list[dict]) -> list[str]:
    """バランス思考フェーズで、会話コンテキストから3つの提案をHaikuで生成する。
    失敗時は空リスト（フェイルセーフ）。"""
    # 会話末尾に明示のリクエストを足す
    ask = {
        "role": "user",
        "content": "ここまでの対話（出来事・自動思考・感情・根拠・反証）を踏まえて、"
                   "私が取りうるバランス思考の例を3つ提案してください。",
    }
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=BALANCED_THOUGHT_SYSTEM,
            messages=messages + [ask],
        )
        raw = resp.content[0].text if resp.content else ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        items = data.get("suggestions") or []
        return [s.strip() for s in items if isinstance(s, str) and s.strip()][:3]
    except Exception:
        return []
