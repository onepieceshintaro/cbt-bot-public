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


# 認知の歪み10パターン（推定用）
DISTORTION_PATTERNS = [
    "全か無か思考", "過度の一般化", "心のフィルター", "マイナス化思考",
    "結論の飛躍", "拡大解釈と過小評価", "感情的決めつけ", "すべき思考",
    "レッテル貼り", "個人化",
]


def infer_distortions_from_record(record: dict) -> list[str]:
    """7コラム法のセッション完了後にバックエンドで歪みを推定する。
    Haikuで1コール。失敗時は空リスト（フェイルセーフ）。"""
    prompt = f"""以下の思考記録から、該当しそうな**認知の歪み**を**最大3つまで**選んでください。
該当なしの場合は空リスト。

# 認知の歪み10パターン
{" / ".join(DISTORTION_PATTERNS)}

# 思考記録
- 状況: {record.get("situation", "")}
- 自動思考: {record.get("automatic_thought", "")}
- 根拠（事実）: {record.get("evidence_for", "")}
- 反証: {record.get("evidence_against", "")}
- バランス思考: {record.get("balanced_thought", "")}

# 出力
必ず以下のJSONのみ（コードブロックや説明文なし）:
{{"distortions": ["歪み名1", "歪み名2"]}}

歪み名は上記10パターンから正確に選んでください。
"""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        # JSON抽出
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        items = data.get("distortions") or []
        # 有効な歪み名だけに絞る
        return [d for d in items if d in DISTORTION_PATTERNS][:3]
    except Exception:
        return []


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
