"""週次レポート生成：1週間の思考記録を Claude で要約する。"""
import json
from datetime import date, timedelta

import pandas as pd


REPORT_SYSTEM_PROMPT = """あなたは、ユーザーの1週間の思考記録を**冷静に振り返る**サポーターです。

目的：
- ユーザーが自分のパターンに気づくのを助ける
- 来週への小さな手がかりを見つける

避けること：
- 感情を煽る言葉（深刻化や悲観的誇張）
- 診断や医療的判断
- 根拠のない励まし（「絶対大丈夫」など）
- 過剰な称賛・おだて

守ること：
- 事実ベースの観察を優先する
- 「〜が○回出てきました」のように数値で示せるところは示す
- 良かったことも難しかったことも**等しい重みで**触れる
- 再評価（新しい見方）で出てきた言葉を、ユーザー自身の財産として引用する

出力フォーマット（Markdown）：

## 📝 今週のまとめ
2〜3文で、今週全体の雰囲気を落ち着いた文体で要約。

## 🎯 繰り返し現れたテーマ
箇条書きで3〜5個。同じような場面が複数回出てきたものに着目。

## 🔍 頻出した認知の歪み
上位3つまで。形式：「歪み名（N回）— どんな場面で出やすかったか」1行ずつ。

## 🛠 よく出ている歪みへの対処
上記の頻出歪み（上位1〜2つ）について、**今週の具体的な記録に紐づけて**、来週試せそうな対処を提案します。
形式：
- **歪み名** — 今週は◯◯の場面で出ていました。試せそうな一歩：△△△。
（ユーザーがすでに出している `adaptive_thought` と被らず、それを補強する形で。決めつけず「〜かもしれません」「〜してみるのも一案です」と控えめに）

## 💡 新しい見方で見つけたこと
`adaptive_thought` から、特に役立ちそうなもの3つを短く引用。ユーザー自身の言葉を尊重する。

## 🌱 来週試してみたい小さな一歩
実行可能な行動アイデアを2〜3個。大きすぎない、今週の記録から自然に導かれるもの。
（上の「🛠 対処」とは違い、歪みに紐付かない生活レベルの行動）

## 💬 ひとこと
1〜2文。労わりと事実確認を兼ねた締めくくり。「〜でしたね」のような事実確認＋「〜でも大丈夫です」のような穏やかな姿勢で。
"""


def current_week_range(today: date | None = None) -> tuple[date, date]:
    """ISO週（月曜始まり）の開始日・終了日を返す。"""
    today = today or date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start, end


def filter_week(df: pd.DataFrame, week_start: date, week_end: date) -> pd.DataFrame:
    """event_datetime が [week_start, week_end] に入る行を返す。"""
    if df.empty:
        return df
    d = df.copy()
    # 列ごとに個別に datetime 変換してから fillna（mixed 型の strptime エラー回避）
    ev = pd.to_datetime(d["event_datetime"], errors="coerce")
    cr = pd.to_datetime(d.get("created_at"), errors="coerce") if "created_at" in d.columns else None
    d["event_datetime"] = ev.fillna(cr) if cr is not None else ev
    d = d.dropna(subset=["event_datetime"])
    if d.empty:
        return d
    mask = (d["event_datetime"].dt.date >= week_start) & \
           (d["event_datetime"].dt.date <= week_end)
    return d[mask].sort_values("event_datetime").reset_index(drop=True)


def _format_records_for_prompt(df: pd.DataFrame) -> str:
    """Claude に渡しやすいように1週間のレコードを整形。"""
    lines = []
    for _, r in df.iterrows():
        dt = r["event_datetime"].strftime("%m/%d %a %H:%M")
        lines.append(f"### {dt}")
        if r.get("situation"):
            lines.append(f"- 場面: {r['situation']}")
        emo = r.get("emotion_name")
        ib = r.get("intensity_before")
        ia = r.get("intensity_after")
        if emo:
            lines.append(f"- 感情: {emo}（前 {ib} → 後 {ia}）")
        if r.get("automatic_thought"):
            lines.append(f"- 自動思考: {r['automatic_thought']}")
        try:
            distortions = json.loads(r.get("distortions") or "[]")
            if distortions:
                lines.append(f"- 認知の歪み: {', '.join(distortions)}")
        except Exception:
            pass
        if r.get("adaptive_thought"):
            lines.append(f"- 新しい見方: {r['adaptive_thought']}")
        lines.append("")
    return "\n".join(lines)


def generate_weekly_report(
    df_week: pd.DataFrame, client, model: str = "claude-sonnet-4-5"
) -> str:
    """1週間分の DataFrame を受けて Markdown 形式のレポートを返す。"""
    if df_week.empty:
        return "_この週はまだ記録がありません。_"

    records_text = _format_records_for_prompt(df_week)
    start = df_week["event_datetime"].min().strftime("%Y/%m/%d")
    end = df_week["event_datetime"].max().strftime("%Y/%m/%d")

    user_prompt = (
        f"期間: {start}〜{end}\n"
        f"記録数: {len(df_week)}件\n\n"
        f"以下が今週の思考記録です。上記のフォーマットで振り返ってください。\n\n"
        f"{records_text}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=[{
            "type": "text",
            "text": REPORT_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text
