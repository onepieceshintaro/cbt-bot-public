"""個人のベースライン学習：感情強度の2σ逸脱検知、頻出歪みランキング。

思想：
- 万人向けの「正常」ではなく、**あなた自身の過去N日間との比較**で逸脱を見る
- mood-tracker と同じ 2σ 検知の思想を CBTボット側にも適用
- 異常検知エンジニアの業務知見をメンタル領域に転用する形
"""
import json
from collections import Counter

import pandas as pd

# ベースラインが意味を持つ最小サンプル数（統計の安定性）
MIN_RECORDS_FOR_BASELINE = 10

# 直近何日の記録を「あなたのいつもの状態」とみなすか
DEFAULT_WINDOW_DAYS = 30

SIGMA_THRESHOLD = 2.0


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    # 列ごとに個別に datetime 変換してから fillna
    ev = pd.to_datetime(d["event_datetime"], errors="coerce")
    cr = pd.to_datetime(d.get("created_at"), errors="coerce") if "created_at" in d.columns else None
    d["event_datetime"] = ev.fillna(cr) if cr is not None else ev
    d = d.dropna(subset=["event_datetime"])
    return d.sort_values("event_datetime").reset_index(drop=True)


def compute_intensity_baseline(
    df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict | None:
    """直近 `window_days` 日の感情強度（前）の平均・標準偏差を返す。

    戻り値（データ不足時は None）:
      - n: サンプル数
      - window_days: 対象窓
      - mean, std: 平均・標準偏差
      - upper_2s, lower_2s: 2σ閾値（上/下）
    """
    d = _prepare_df(df)
    if d.empty:
        return None

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
    recent = d[d["event_datetime"] >= cutoff]

    # 窓内が少なすぎる場合は全体も見る
    if len(recent) < MIN_RECORDS_FOR_BASELINE:
        recent = d  # fallback: 全期間

    if len(recent) < MIN_RECORDS_FOR_BASELINE:
        return {
            "n": len(recent),
            "window_days": window_days,
            "insufficient": True,
            "required": MIN_RECORDS_FOR_BASELINE,
        }

    intensities = pd.to_numeric(recent["intensity_before"], errors="coerce").dropna()
    if len(intensities) < MIN_RECORDS_FOR_BASELINE:
        return {
            "n": len(intensities),
            "window_days": window_days,
            "insufficient": True,
            "required": MIN_RECORDS_FOR_BASELINE,
        }

    mean = float(intensities.mean())
    std = float(intensities.std(ddof=1))
    return {
        "n": len(intensities),
        "window_days": window_days,
        "mean": round(mean, 1),
        "std": round(std, 1),
        "upper_2s": round(mean + SIGMA_THRESHOLD * std, 1),
        "lower_2s": round(mean - SIGMA_THRESHOLD * std, 1),
        "insufficient": False,
    }


def deviation_from_baseline(intensity: int, baseline: dict) -> dict | None:
    """強度がベースラインから何σズレているかを算出。"""
    if not baseline or baseline.get("insufficient"):
        return None
    std = baseline.get("std") or 0
    mean = baseline.get("mean") or 0
    if std == 0:
        sigma = 0.0
    else:
        sigma = (intensity - mean) / std
    return {
        "delta": round(intensity - mean, 1),
        "sigma": round(sigma, 2),
        "is_high": intensity > baseline.get("upper_2s", 999),
        "is_low": intensity < baseline.get("lower_2s", -999),
    }


def _extract_active_names(distortions_json) -> list[str]:
    """distortions JSON から、dismissed=False の歪み名だけ抜き出す。
    旧形式（list[str]）/ 新形式（list[dict{name, evidence, dismissed}]）両対応。"""
    try:
        items = json.loads(distortions_json) if isinstance(distortions_json, str) else distortions_json
    except Exception:
        return []
    if not items:
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("name") and not it.get("dismissed"):
            out.append(it["name"])
        elif isinstance(it, str):
            out.append(it)
    return out


def _count_distortions(period_df: pd.DataFrame) -> tuple[Counter, int]:
    """期間内の歪み件数を集計（dismissed除外）。"""
    counter: Counter = Counter()
    total_records = 0
    for s in period_df["distortions"].dropna():
        names = _extract_active_names(s)
        if not names:
            continue
        total_records += 1
        counter.update(names)
    return counter, total_records


def top_distortions(
    df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    top_n: int = 3,
) -> list[dict]:
    """直近 `window_days` 日で出現した認知の歪み上位N個（dismissed除外）。

    戻り値: [{"name": "最悪化", "count": 5, "rate": 0.42}, ...]
    """
    d = _prepare_df(df)
    if d.empty:
        return []

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
    recent = d[d["event_datetime"] >= cutoff]
    if recent.empty:
        recent = d  # fallback

    counter, total_records = _count_distortions(recent)
    if total_records == 0:
        return []

    return [
        {"name": name, "count": cnt, "rate": round(cnt / total_records, 2)}
        for name, cnt in counter.most_common(top_n)
    ]


def distortion_trend(
    df: pd.DataFrame,
    top_n: int = 3,
    recent_days: int = 15,
) -> list[dict]:
    """各 TOP歪みの「最近増えてる／減ってる」を直近(recent_days)日 vs その前(recent_days)日で比較。

    戻り値: [{
        "name", "count", "rate",
        "recent_count", "older_count", "delta",
        "trend_emoji": "📈"|"📉"|"─",
        "trend_text": str
    }, ...]
    """
    d = _prepare_df(df)
    if d.empty:
        return []

    now = pd.Timestamp.now().normalize()
    recent_cutoff = now - pd.Timedelta(days=recent_days)
    older_cutoff = now - pd.Timedelta(days=recent_days * 2)

    recent_df = d[d["event_datetime"] >= recent_cutoff]
    older_df = d[(d["event_datetime"] < recent_cutoff)
                 & (d["event_datetime"] >= older_cutoff)]

    recent_counts, _ = _count_distortions(recent_df)
    older_counts, _ = _count_distortions(older_df)

    # トップ判定は全期間ベースで取得
    top = top_distortions(df, top_n=top_n, window_days=DEFAULT_WINDOW_DAYS)

    out = []
    for d_info in top:
        name = d_info["name"]
        recent_n = recent_counts.get(name, 0)
        older_n = older_counts.get(name, 0)
        delta = recent_n - older_n
        if delta > 0:
            emoji = "📈"
            text = f"直近{recent_days}日で +{delta}回（増加傾向）"
        elif delta < 0:
            emoji = "📉"
            text = f"直近{recent_days}日で {delta}回（減少傾向）"
        else:
            emoji = "─"
            text = "ほぼ変化なし"
        out.append({
            **d_info,
            "recent_count": recent_n,
            "older_count": older_n,
            "delta": delta,
            "trend_emoji": emoji,
            "trend_text": text,
        })
    return out


def baseline_summary(df: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """1回でベースラインと頻出歪みの両方を返すショートカット。"""
    return {
        "intensity": compute_intensity_baseline(df, window_days),
        "top_distortions": top_distortions(df, window_days),
        "window_days": window_days,
    }
