"""LLMによる危機スコアリング。キーワード判定を補完する第二の安全網。

- Haiku（安い・速い）で発話を4軸（自傷/他害/虐待/急性）×0-10で評価
- スコア閾値超でキーワード判定と同じく CRISIS_RESPONSE を発火
- 失敗時は 0 を返すフェイルセーフ（既存のキーワード判定は常に効いている）
"""
import json
import re


RISK_SCORING_PROMPT = """あなたは、メンタルヘルスの会話における**リスクスクリーニング**だけを行うアシスタントです。
共感や助言はしません。機械的に評価し、JSONだけを返してください。

ユーザーの発話を読み、以下4軸を 0〜10 で評価してください：

- self_harm: 自傷・自殺に関連する苦悩や言及（希死念慮、自殺計画、自傷行為、「消えたい」などの婉曲表現も含む）
- harm_to_others: 他害リスク（他者への強い暴力衝動、加害の具体性）
- abuse: 虐待の示唆（被虐待・加害・ネグレクト・DV）
- acute: 急性エピソード（解離、幻聴/幻覚、強い衝動制御困難、現実感喪失）

スコア基準:
- 0-2: 関連なし・通常の苦痛表現
- 3-4: 軽度の示唆（ネガティブだが危機的ではない）
- 5-6: 中等度（具体性や継続性あり、専門的支援推奨レベル）
- 7-8: 高度（差し迫った示唆、即時対応推奨）
- 9-10: 緊急（明確な計画・切迫性・生命の危険）

出力は必ず次のJSONのみ（コードブロック・説明文・前置き禁止）:
{"self_harm": N, "harm_to_others": N, "abuse": N, "acute": N, "overall": N, "reasoning": "短い理由（日本語1文）"}

- overall は4軸の最大値
- 比喩・婉曲表現・皮肉も文脈から読み取る（例:「いなくなりたい」「もう無理」）
- 過去の話として語られている場合はやや控えめに、現在進行の訴えは強めに評価
"""

CRISIS_SCORE_THRESHOLD = 7

# Haiku 4.5 （安価・高速、最新のHaiku）
RISK_MODEL = "claude-haiku-4-5-20251001"


def _extract_json(text: str) -> dict | None:
    """LLMの出力からJSONオブジェクトを抽出。説明文が混じっても救済する。"""
    # まず全体を試す
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # { ... } を正規表現で拾う
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _default_score(reason: str = "") -> dict:
    return {
        "self_harm": 0, "harm_to_others": 0, "abuse": 0, "acute": 0,
        "overall": 0, "reasoning": reason,
    }


def score_risk(user_message: str, client, model: str = RISK_MODEL) -> dict:
    """ユーザー発話をリスクスコアリング。失敗時は全0を返す（フェイルセーフ）。"""
    if not user_message or not user_message.strip():
        return _default_score()

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": RISK_SCORING_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )
        raw = resp.content[0].text if resp.content else ""
        data = _extract_json(raw)
        if not data:
            return _default_score(f"parse_error: {raw[:60]}")

        def _clamp(v):
            try:
                return max(0, min(10, int(v)))
            except Exception:
                return 0

        return {
            "self_harm": _clamp(data.get("self_harm")),
            "harm_to_others": _clamp(data.get("harm_to_others")),
            "abuse": _clamp(data.get("abuse")),
            "acute": _clamp(data.get("acute")),
            "overall": _clamp(data.get("overall")),
            "reasoning": str(data.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return _default_score(f"api_error: {type(e).__name__}")


def is_high_risk(score: dict, threshold: int = CRISIS_SCORE_THRESHOLD) -> bool:
    return (score or {}).get("overall", 0) >= threshold
