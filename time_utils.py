"""日本時間ヘルパー。

Streamlit Cloud のサーバーは UTC で動くため、`datetime.now()` をそのまま使うと
表示・記録のタイムスタンプが 9 時間ズレる。アプリ内では常に JST で扱うため、
タイムスタンプを取る箇所はこのモジュール経由にする。
"""
from datetime import datetime, date
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def now_jst() -> datetime:
    """JST のタイムゾーン付き現在時刻（aware datetime）。"""
    return datetime.now(JST)


def now_jst_naive() -> datetime:
    """JST の現在時刻（naive datetime, tzinfo を落とす）。

    既存コードが naive な isoformat / 比較を前提にしている箇所向け。
    """
    return datetime.now(JST).replace(tzinfo=None)


def today_jst() -> date:
    """JST の今日。"""
    return datetime.now(JST).date()
