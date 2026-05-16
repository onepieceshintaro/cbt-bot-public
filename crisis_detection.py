"""3 モード危機検出（統一モジュール・全アプリ共通設計）。

このモジュールは 4 アプリ（cbt-bot / mood-tracker / assertion-bot / self-map）
全てに同じ内容で配置する。改修時は 4 箇所同期。

## 3 モードの定義
- normal       : 通常応答モード
- warning      : 対話を控えるモード（誤検知時の解除導線あり）
- critical     : 緊急窓口モード（解除導線なし）

## 設計原則（hub-public/crisis_response_flow.md と整合）
- キーワード判定 ＋ LLM 判定の二段（フェイルセーフ：LLM失敗時はキーワードだけで判定）
- 「緊急窓口モード」は「警戒キーワード ＋ 具体性キーワード」の AND 設計
- 判定結果は本文に絶対に混ぜない
- LLM 失敗時は全 0 を返すフェイルセーフ
"""
import json
import re
from typing import Literal

Mode = Literal["normal", "warning", "critical"]


# ============================================================
# 警戒キーワード（このリスト hit → warning モード発火候補）
# ============================================================
WARNING_KEYWORDS: list[str] = [
    # 自殺・自傷の直接表現
    "死にたい", "しにたい", "消えたい", "きえたい",
    "いなくなりたい", "終わりにしたい", "おわりにしたい",
    "自殺", "自傷", "リストカット",
    "生きていたくない", "生きてたくない", "生きる意味がない",
    "生きてる意味がない",
    # 自傷・他害のほのめかし
    "自分を傷つけ", "自分を傷つけたい",
    "殺してほしい", "死ぬ方法",
    # 強い無価値感
    "迷惑をかけている", "迷惑かけてる",
    "自分なんていない方が", "自分なんていなくなれば",
    # 解離・現実感喪失
    "現実感がない", "自分が自分じゃない",
    "自分が自分でない",
]


# ============================================================
# 具体性キーワード（WARNING かつ ここ hit → critical 発火）
# ============================================================
CRITICAL_KEYWORDS: list[str] = [
    # 具体的な手段（一部のみ。専門家レビューで拡張）
    "首吊り", "首つり", "飛び降り", "飛びおり",
    "練炭", "首を吊", "薬を集め", "睡眠薬を",
    "包丁", "刃物",
    # 具体的な時期（差し迫った時間表現）
    "今夜", "今日中", "今日には",
    "明日には", "あした",  # 「あした」は誤検知あり得るので AND 設計で守る
    # 完了形・確定表現
    "もう決めた", "もう決まっている", "もうきまっている",
    "準備した", "準備できた", "用意した",
    "もう動けない", "もう止まらない",
    # 終結のニュアンス
    "最後に", "お別れ", "さようなら", "じゃあね",
    "ありがとうございました",  # 文脈次第・LLM で補強
]


def _contains_any(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """text に keywords のいずれかが含まれるか。hit したものを返す。"""
    if not text:
        return False, []
    hits = [kw for kw in keywords if kw in text]
    return bool(hits), hits


# ============================================================
# LLM 判定プロンプト
# ============================================================
LLM_CRISIS_PROMPT = """あなたは、メンタルヘルスの会話における**リスクスクリーニング**だけを行うアシスタントです。
共感や助言はしません。機械的に評価し、JSON だけを返してください。

ユーザーの発話を読み、以下を評価してください：

# 1) overall（0-10）：危機の総合度
- 0-2: 関連なし・通常の苦痛表現
- 3-4: 軽度の示唆（ネガティブだが危機的ではない）
- 5-6: 中等度（自殺/自傷/解離の示唆あり、対話を控えるレベル）
- 7-8: 高度（差し迫った示唆）
- 9-10: 緊急（明確な計画・切迫性・生命の危険）

# 2) specificity（0-10）：具体性
- 0-2: 抽象的・気持ちレベル
- 3-4: 漠然とした手段・時期言及
- 5-6: ある程度具体的（時期は曖昧だが手段に触れる、等）
- 7-8: 手段が具体的または時期が差し迫る
- 9-10: 手段・時期・完了形が揃っている（緊急レベル）

# 3) acute_state（0-10）：急性エピソード
解離・現実感喪失・幻聴/幻覚・強い衝動制御困難など

# 4) abuse（0-10）
被虐待・加害・ネグレクト・DV の示唆

# 出力形式（必ずこの JSON のみ、コードブロック・説明文・前置き禁止）
{"overall": N, "specificity": N, "acute_state": N, "abuse": N, "reasoning": "短い理由（日本語1文）"}

# 注意
- 比喩・婉曲表現も文脈から読み取る（例：「いなくなりたい」「もう無理」）
- 過去の話として語られている場合はやや控えめに、現在進行は強めに評価
- 「ありがとう」「お別れ」など終結ニュアンスは specificity を上げる材料
"""

CRISIS_MODEL = "claude-haiku-4-5-20251001"

# 閾値（β 期で調整可能。レビュー前のたたき台）
WARNING_OVERALL_THRESHOLD = 5
CRITICAL_OVERALL_THRESHOLD = 7
CRITICAL_SPECIFICITY_THRESHOLD = 6


def _extract_json(text: str) -> dict | None:
    """LLM 出力から JSON を抽出。説明文が混じっても救済する。"""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _default_score(reason: str = "") -> dict:
    return {
        "overall": 0, "specificity": 0,
        "acute_state": 0, "abuse": 0,
        "reasoning": reason,
    }


def score_llm(user_text: str, client, model: str = CRISIS_MODEL) -> dict:
    """LLM で危機スコアリング。失敗時は全 0 を返す（フェイルセーフ）。

    Args:
        user_text: 評価対象テキスト
        client: Anthropic client（None で LLM スキップ）
    """
    if not user_text or not user_text.strip():
        return _default_score()
    if client is None:
        return _default_score("client_unavailable")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": LLM_CRISIS_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_text}],
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
            "overall": _clamp(data.get("overall")),
            "specificity": _clamp(data.get("specificity")),
            "acute_state": _clamp(data.get("acute_state")),
            "abuse": _clamp(data.get("abuse")),
            "reasoning": str(data.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return _default_score(f"api_error: {type(e).__name__}")


def detect_mode(
    user_text: str,
    client=None,
    use_llm: bool = True,
) -> dict:
    """3 モードで危機検出する。

    判定ロジック（AND 設計）：
        warning : warning_keyword OR llm.overall >= WARNING_OVERALL_THRESHOLD
        critical: warning AND (critical_keyword OR
                               (llm.overall >= CRITICAL_OVERALL_THRESHOLD
                                AND llm.specificity >= CRITICAL_SPECIFICITY_THRESHOLD))

    Args:
        user_text: 評価するテキスト
        client: Anthropic client。None なら LLM 判定スキップ（キーワードのみ）
        use_llm: False で LLM を呼ばない

    Returns:
        {
            "mode": "normal" | "warning" | "critical",
            "warning_keyword_hits": [...],
            "critical_keyword_hits": [...],
            "llm_score": {...},
            "source": "keyword" | "llm" | "keyword+llm" | "none",
        }
    """
    if not user_text or not user_text.strip():
        return {
            "mode": "normal",
            "warning_keyword_hits": [],
            "critical_keyword_hits": [],
            "llm_score": _default_score(),
            "source": "none",
        }

    has_warning_kw, warning_hits = _contains_any(user_text, WARNING_KEYWORDS)
    has_critical_kw, critical_hits = _contains_any(user_text, CRITICAL_KEYWORDS)

    llm_score = _default_score()
    if use_llm and client is not None:
        llm_score = score_llm(user_text, client)

    has_llm_warning = llm_score.get("overall", 0) >= WARNING_OVERALL_THRESHOLD
    has_llm_critical = (
        llm_score.get("overall", 0) >= CRITICAL_OVERALL_THRESHOLD
        and llm_score.get("specificity", 0) >= CRITICAL_SPECIFICITY_THRESHOLD
    )

    is_warning = has_warning_kw or has_llm_warning
    is_critical = is_warning and (has_critical_kw or has_llm_critical)

    mode: Mode = "normal"
    if is_critical:
        mode = "critical"
    elif is_warning:
        mode = "warning"

    sources = []
    if has_warning_kw or has_critical_kw:
        sources.append("keyword")
    if has_llm_warning or has_llm_critical:
        sources.append("llm")
    source = "+".join(sources) if sources else "none"

    return {
        "mode": mode,
        "warning_keyword_hits": warning_hits,
        "critical_keyword_hits": critical_hits,
        "llm_score": llm_score,
        "source": source,
    }


# ============================================================
# 固定文言（モード別）— hub-public/crisis_response_flow.md と整合
# ============================================================

WARNING_MESSAGE = """\
ここまで書いてくれて、ありがとうございます。

あなたが今、つらい気持ちを抱えていることを受け取りました。

いま、私（このアプリ）が一緒に考えるよりも、
人と話すほうが安全な状態かもしれません。

下の窓口は、どれも無料・匿名で話せます。
うまく話せなくても大丈夫です。
電話をかけて、何も言わずに切ってもいいです。

もしすぐに動けない場合は、誰か近くの人に
「今ちょっとつらい」と一言だけでも伝えられたらと思います。

あなたの安全がいちばん大事です。
"""

CRITICAL_MESSAGE = """\
あなたがいま、自分を傷つけることを
具体的に考えていることを、受け取りました。

このアプリではここから先のお手伝いができません。
私（このアプリ）よりも、人の声を聞いてほしいです。

▶ いますぐ動けるなら：救急（119）／ 警察（110）
▶ 話を聞いてもらいたいなら：下の電話のどれか
▶ 一人にならないでください

電話をかけて、声が出なくても大丈夫です。
「話せないけど、つらい」とだけでも伝わります。
"""


# ============================================================
# 相談窓口（hub-public/crisis_response_flow.md より転載）
# ============================================================
WARNING_HOTLINES = [
    {
        "name": "よりそいホットライン",
        "tel": "0120-279-338",
        "note": "24 時間・無料・匿名",
    },
    {
        "name": "いのちの電話",
        "tel": "0570-783-556",
        "note": "10:00-22:00（毎日）・全国共通",
    },
    {
        "name": "#いのちSOS",
        "tel": "0120-061-338",
        "note": "12:00-22:00（毎日）・通話料無料",
    },
    {
        "name": "こころの耳（厚労省）",
        "tel": None,
        "url": "https://kokoro.mhlw.go.jp/",
        "note": "24 時間（Web）・働く人向け総合ポータル",
    },
]

CRITICAL_HOTLINES = [
    {
        "name": "救急",
        "tel": "119",
        "note": "生命の危険があるとき",
        "priority": True,
    },
    {
        "name": "警察",
        "tel": "110",
        "note": "自他の安全に関わるとき",
        "priority": True,
    },
] + WARNING_HOTLINES
