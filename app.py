import json
import re
from collections import Counter
from datetime import datetime, date, time, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px

from cbt_engine import (
    chat, assess_risk, client as anthropic_client,
    infer_distortions_from_record, infer_distortions_ab, suggest_balanced_thoughts,
    is_phase_b_unlocked, infer_similar_past_records,
)
from storage import (
    init_db, save_record, load_records, update_distortions, update_record,
    save_weekly_report, load_weekly_report, load_all_weekly_reports,
    save_risk_score, load_risk_scores,
)
from prompts import (
    CRISIS_RESPONSE, MODE_CONFIGS, DEFAULT_MODE,
    MODE_DISTORTION, MODE_SEVEN_COLUMNS,
)
from reports import (
    current_week_range, filter_week, generate_weekly_report,
)
from baseline import (
    baseline_summary, compute_intensity_baseline, top_distortions,
    deviation_from_baseline, MIN_RECORDS_FOR_BASELINE, SIGMA_THRESHOLD,
)
from distortion_tips import DISTORTION_TIPS, get_tips_for
from _user import render_account_sidebar
from time_utils import now_jst_naive, today_jst

st.set_page_config(page_title="思考の整理ノート", page_icon="💭", layout="wide")

# ユーザー識別（復元キー・URL ?u= ・ローカルファイルの優先順）
CURRENT_USER_ID = render_account_sidebar()
init_db()

# --- ヘッダー不透明化のみ ---
st.markdown("""
<style>
header[data-testid="stHeader"] { background: white; }
</style>
""", unsafe_allow_html=True)

# --- AI応答からフェーズとJSONを抽出 ---
PHASE_PATTERN = re.compile(r"<!--\s*phase:\s*(\w+)\s*-->")

def parse_phase(text: str) -> str | None:
    """AI応答末尾のフェーズマーカーを抽出"""
    m = PHASE_PATTERN.search(text)
    return m.group(1) if m else None

def strip_meta(text: str) -> str:
    """表示用にメタ情報（phaseマーカー・JSONブロック）を除去"""
    # phaseコメント除去
    text = PHASE_PATTERN.sub("", text)
    # JSONブロックは表示しない（保存用のため）
    text = re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
    return text.strip()

def extract_json(text: str) -> dict | None:
    """完了時のJSON要約を抽出"""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def progress_ratio(phase: str | None, order: list[str]) -> float:
    if phase is None or phase == "crisis":
        return 0.0
    if phase not in order:
        return 0.0
    return (order.index(phase) + 1) / len(order)


def normalize_distortions(raw) -> list[dict]:
    """記録の distortions を、旧形式（list[str]）でも新形式（list[dict]）でも統一形に揃える。

    新フィールド（Phase A）：
      - source: "llm" | "rag" | "both"（既存データは "llm" 扱い）
      - shown: ユーザーに表示するか（既存データは True）
      - dismissed_at: dismiss 時刻（任意）
      - rag_score: RAG ヒット時のみ（任意）
    """
    if not raw:
        return []
    items = raw
    if isinstance(raw, str):
        try:
            items = json.loads(raw or "[]")
        except Exception:
            return []
    out: list[dict] = []
    for it in items or []:
        if isinstance(it, dict) and it.get("_meta"):
            continue  # メタデータは distortions として扱わない
        if isinstance(it, str):
            out.append({
                "name": it,
                "evidence": "",
                "dismissed": False,
                "source": "llm",
                "shown": True,
            })
        elif isinstance(it, dict) and it.get("name"):
            d = {
                "name": str(it["name"]),
                "evidence": str(it.get("evidence") or ""),
                "dismissed": bool(it.get("dismissed", False)),
                "source": str(it.get("source", "llm")),  # 既存は "llm"
                "shown": bool(it.get("shown", True)),    # 既存は True
            }
            if it.get("dismissed_at"):
                d["dismissed_at"] = it["dismissed_at"]
            if it.get("rag_score") is not None:
                d["rag_score"] = it["rag_score"]
            out.append(d)
    return out


def distortion_names(raw) -> list[str]:
    return [d["name"] for d in normalize_distortions(raw)]


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.current_phase = None
    st.session_state.record_saved = False

if "mode" not in st.session_state:
    st.session_state.mode = DEFAULT_MODE

if "balance_suggestions" not in st.session_state:
    st.session_state.balance_suggestions = []

# 直近セッションの歪み（対話後の対処ヒント表示用）
if "last_distortions" not in st.session_state:
    st.session_state.last_distortions = []
if "show_last_tips" not in st.session_state:
    st.session_state.show_last_tips = False


def reset_session():
    """新しいセッションを始めるときのコールバック。
    on_click は次回rerunの前に実行されるので、ここでウィジェットキーを変更できる。"""
    st.session_state.messages = []
    st.session_state.current_phase = None
    st.session_state.record_saved = False
    st.session_state.balance_suggestions = []
    st.session_state.last_distortions = []
    st.session_state.show_last_tips = False
    st.session_state.view_radio = "💬 対話"


def _current_cfg():
    return MODE_CONFIGS.get(st.session_state.mode, MODE_CONFIGS[DEFAULT_MODE])

# --- サイドバー（常に左側に表示・スクロール不要） ---
with st.sidebar:
    _hub_url = "https://app-public-qpy8b2ziwgdf9h2vmu5hqp.streamlit.app/"
    if CURRENT_USER_ID:
        _hub_url += f"?u={CURRENT_USER_ID}"
    st.link_button(
        "🏠 HOME に戻る",
        _hub_url,
        use_container_width=True,
    )
    st.link_button(
        "💬 ご意見・感想",
        "https://docs.google.com/forms/d/e/1FAIpQLSetCb_dHG6JFsUzhK9ZYxydgh5cP8w07Q6NRO4ouEM7BvSTRw/viewform",
        use_container_width=True,
    )
    st.divider()
    # ビュー切替
    view = st.radio(
        "表示",
        ["💬 対話", "📊 傾向を見る", "📝 週次レポート"],
        label_visibility="collapsed",
        key="view_radio",
    )
    st.divider()

    # モード選択（セッション中は変更不可）
    _mode_keys = list(MODE_CONFIGS.keys())
    _mode_display = [MODE_CONFIGS[k]["display_name"] for k in _mode_keys]
    _can_change_mode = len(st.session_state.messages) == 0
    st.markdown("**進め方**")
    if _can_change_mode:
        _selected_display = st.radio(
            "進め方",
            _mode_display,
            index=_mode_keys.index(st.session_state.mode),
            label_visibility="collapsed",
            key="mode_radio",
        )
        st.session_state.mode = _mode_keys[_mode_display.index(_selected_display)]

        # 使い分けのイメージ（両モードを並べて比較できるように）
        with st.expander("💭 どちらを選ぶか迷ったら", expanded=False):
            for _k in _mode_keys:
                _cfg = MODE_CONFIGS[_k]
                st.markdown(f"#### {_cfg['display_name']}")
                _wtu = _cfg.get("when_to_use")
                if _wtu:
                    # 本文は caption で小さめ表示 → タイトルとの階層を出す
                    st.caption(_wtu)
                st.write("")
            st.markdown(
                "どちらも「考え方を変える」ためではなく、"
                "**自分の考えを少し外側から眺めてみる**ための道具です。"
            )
    else:
        st.caption(
            f"現在：**{MODE_CONFIGS[st.session_state.mode]['display_name']}**"
        )
        st.caption("※ 進め方はセッション開始後は固定されます")
    st.divider()

    cfg = _current_cfg()
    phase_labels = cfg["labels"]
    phase_order = cfg["order"]

    st.header("セッションの進捗")
    phase = st.session_state.current_phase

    if phase:
        label = phase_labels.get(phase, phase)
        ratio = progress_ratio(phase, phase_order)
        st.markdown(f"""
        <div style="font-size: 14px; color: #555; margin-bottom: 6px;">
          現在：<b>{label}</b>
        </div>
        <div style="background: #eceff1; border-radius: 6px; height: 10px;
                    overflow: hidden; margin-bottom: 12px;">
          <div style="background: linear-gradient(90deg,#ff8a65,#ff5252);
                      height: 100%; width: {ratio*100:.0f}%;
                      transition: width .4s ease;"></div>
        </div>
        """, unsafe_allow_html=True)

        # 全フェーズの一覧表示
        st.markdown("**ステップ一覧**")
        current_idx = phase_order.index(phase) if phase in phase_order else -1
        for i, p in enumerate(phase_order):
            if p == phase:
                st.markdown(f"▶ **{phase_labels[p]}**")
            elif i < current_idx:
                st.markdown(f"✓ <span style='color:#999'>{phase_labels[p]}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"　<span style='color:#bbb'>{phase_labels[p]}</span>",
                            unsafe_allow_html=True)
    else:
        st.caption("まだ始まっていません。入力欄に、書けるところから自由にどうぞ。")
        _steps = [p for p in phase_order if p != "done"]
        st.markdown(f"**これから進む{len(_steps)}ステップ**")
        for p in _steps:
            st.markdown(f"　<span style='color:#999'>{phase_labels[p]}</span>",
                        unsafe_allow_html=True)

    st.divider()
    st.button(
        "新しいセッションを始める",
        use_container_width=True,
        on_click=reset_session,
    )

    # --- 最近の傾向（ベースライン）---
    # 対話ビュー時のみ、記録が十分あれば表示
    if view == "💬 対話":
        try:
            _bl_df = load_records()
            _summary = baseline_summary(_bl_df)
            _intensity = _summary["intensity"]
            _top_d = _summary["top_distortions"]

            if _intensity and not _intensity.get("insufficient"):
                st.divider()
                st.markdown("**📊 最近のあなた**")
                st.caption(
                    f"直近{_intensity['window_days']}日・{_intensity['n']}件"
                )
                st.markdown(
                    f"感情強度 いつもの範囲：<br>"
                    f"<b>{_intensity['lower_2s']:.0f} 〜 {_intensity['upper_2s']:.0f}</b> "
                    f"（平均{_intensity['mean']:.0f}）",
                    unsafe_allow_html=True,
                )
                if _top_d:
                    st.caption("よく出てくる歪み：")
                    lines = [f"・{d['name']}（{d['count']}回）"
                             for d in _top_d]
                    st.markdown(
                        "<div style='font-size:13px;color:#666'>"
                        + "<br>".join(lines)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass  # ベースライン失敗はサイレントに

    st.divider()
    with st.expander("📚 認知の歪みパターン辞典", expanded=False):
        st.caption(
            "Burns の古典10パターンに準拠（結論の飛躍は2サブタイプに分けて表示）。"
            "気になるものだけ開いて読めば大丈夫です。"
        )
        # 互換用の「結論の飛躍」（無印）は辞書UIには出さない
        for _name, _tip in DISTORTION_TIPS.items():
            if _name == "結論の飛躍":
                continue
            with st.container(border=True):
                st.markdown(f"**{_name}**")
                st.caption(_tip["description"])
                if _tip.get("strength"):
                    st.markdown(f"💪 **良い面**：{_tip['strength']}")
                st.markdown(f"⚠ **ハマりやすいパターン**：{_tip['trap']}")

    st.divider()
    st.caption("※ このBotは医療行為ではありません")
    with st.expander("辛いときの相談窓口"):
        st.markdown("""
        - いのちの電話：0570-783-556
        - よりそいホットライン：0120-279-338
        - こころの健康相談統一ダイヤル：0570-064-556
        """)

    # ── 開発者用：LLM vs RAG 比較パネル（Phase A 検証用） ──
    _ab = st.session_state.get("last_ab_compare")
    if _ab:
        with st.expander("🧪 歪み推定 A/B 比較（dev）", expanded=False):
            st.caption("LLM 版（Haiku, evidence 付き）と RAG 版（Voyage 意味検索）を並走中。")
            _llm_names = [d.get("name", "") for d in _ab.get("llm", [])]
            _rag_items = _ab.get("rag", [])
            _rag_raw = _ab.get("rag_raw", [])
            _threshold = float(_ab.get("rag_threshold", 0.10))

            st.markdown(f"**LLM (Haiku)**：{', '.join(_llm_names) or '（なし）'}")
            if _rag_items:
                _rag_lines = [f"{d['name']} ({d.get('score', 0):.2f})" for d in _rag_items]
                st.markdown(f"**RAG (Voyage)**：{', '.join(_rag_lines)}")
            else:
                st.markdown("**RAG (Voyage)**：（なし）")
            inter = _ab.get("intersection", [])
            if inter:
                st.markdown(f"**両者一致**：{', '.join(inter)}")

            # 生スコア全件（閾値前）
            st.markdown(f"---\n**RAG 全候補**（現在の閾値 ≥ **{_threshold:.2f}**）")
            _rag_error = _ab.get("rag_error")
            if _rag_error:
                st.error(f"⚠️ RAG エラー: `{_rag_error}`")
                st.caption(
                    "→ Streamlit Cloud の **Manage app → Logs** で "
                    "`[RAG ERROR]` を含む行を探すと、より詳細なスタックトレースが見えます。"
                )
            if not _rag_raw and not _rag_error:
                st.caption(
                    "RAG から候補が1件も返っていません。"
                    "VOYAGE_API_KEY 未設定 / インデックス未構築 / レート制限の可能性。"
                )
            elif _rag_raw:
                from cbt_engine import DISTORTION_PATTERNS as _DP
                for h in _rag_raw:
                    _passed = h["score"] >= _threshold
                    _in_dict = h["name"] in _DP
                    if _passed and _in_dict:
                        _mark = "✅ 採用"
                    elif _passed and not _in_dict:
                        _mark = "⚠️ 閾値↑だが辞書外（親カテゴリ等）"
                    else:
                        _mark = "✗ 閾値↓"
                    st.markdown(
                        f"- {_mark}　**{h['name']}**：score = **{h['score']:.3f}**"
                    )

            st.caption("採用は当面 LLM 結果。違いを観察して RAG の精度を見ます。")

    # ── 開発者用：A/B 集計（全期間の dismiss 率） ──
    with st.expander("📊 歪み推定 A/B 集計（dev・全期間）", expanded=False):
        st.caption(
            "違和感ボタンで外された割合を source 別に集計。"
            "数値が低いほど精度が良い（その source の判定が当たっている）。"
        )
        if st.button("集計を更新", key="ab_summary_refresh"):
            try:
                from storage import get_distortion_ab_summary
                _summary = get_distortion_ab_summary()
                st.session_state["_ab_summary_cache"] = _summary
            except Exception as _e:
                st.error(f"集計失敗: {_e}")

        _summary = st.session_state.get("_ab_summary_cache")
        if _summary:
            st.caption(f"対象レコード数：{_summary.get('total_records', 0)}")
            _imps = _summary.get("impressions", {})
            _dis = _summary.get("dismissals", {})
            _rate = _summary.get("dismiss_rate", {})
            _rows = []
            for s in ["llm", "both", "rag"]:
                if _imps.get(s, 0) > 0:
                    _rows.append({
                        "source": {"llm": "LLM 単独", "both": "両者一致", "rag": "RAG 単独（影）"}[s],
                        "提示": _imps[s],
                        "違和感": _dis.get(s, 0),
                        "違和感率": f"{_rate.get(s, 0) * 100:.1f}%",
                    })
            if _rows:
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("まだデータが溜まっていません。")

            # パターン別 top
            _pt = _summary.get("pattern_table", [])
            _pt_shown = [p for p in _pt if p["shown"] >= 2][:10]  # 2 件以上提示されたもの top10
            if _pt_shown:
                st.caption("**パターン別 違和感率（提示2件以上・上位10）**")
                _pt_rows = [
                    {
                        "歪み": p["name"],
                        "source": p["source"],
                        "提示": p["shown"],
                        "違和感": p["dismissed"],
                        "違和感率": f"{p['rate'] * 100:.1f}%",
                    }
                    for p in _pt_shown
                ]
                st.dataframe(pd.DataFrame(_pt_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("「集計を更新」を押すと最新データを取得します。")


# --- メインエリア（サイドバーの選択で切替） ---
if view == "💬 対話":
    st.markdown("### 思考記録")
    st.caption("ゆっくり、あなたのペースで。書けることだけで大丈夫です。")

    # --- いつの出来事か（コンパクトな選択欄） ---
    with st.expander("📅 いつの出来事ですか？（ざっくりでOK）", expanded=False):
        quick = st.radio(
            "クイック選択",
            ["今さっき", "今日の朝", "今日の昼", "今日の夜", "昨日", "日時を指定"],
            horizontal=True,
            label_visibility="collapsed",
            key="quick_when",
        )
        now = now_jst_naive()  # JST に統一（サーバーが UTC でも 9h ズレない）
        if quick == "今さっき":
            event_dt = now
        elif quick == "今日の朝":
            event_dt = datetime.combine(now.date(), time(8, 0))
        elif quick == "今日の昼":
            event_dt = datetime.combine(now.date(), time(12, 0))
        elif quick == "今日の夜":
            event_dt = datetime.combine(now.date(), time(20, 0))
        elif quick == "昨日":
            event_dt = datetime.combine(now.date() - timedelta(days=1), time(12, 0))
        else:  # 日時を指定
            c1, c2 = st.columns(2)
            with c1:
                d = st.date_input("日付", value=now.date(), key="event_date")
            with c2:
                t = st.time_input("時刻", value=now.time().replace(second=0, microsecond=0), key="event_time")
            event_dt = datetime.combine(d, t)

        st.caption(f"記録対象：**{event_dt.strftime('%Y-%m-%d %H:%M')}**")
        st.session_state.event_datetime = event_dt.isoformat()

    # 履歴表示（メタ情報を消して表示）
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                st.write(strip_meta(m["content"]))
            else:
                st.write(m["content"])

    # 入力欄のプレースホルダーをフェーズに応じて
    cfg = _current_cfg()
    placeholder = cfg["hints"].get(phase, "書けることだけで大丈夫ですよ")

    # 現フェーズの注釈を上部にやさしく表示（7コラム法など注釈があるモード）
    note = cfg["notes"].get(phase) if phase else None
    if note and phase != "done":
        st.info(f"**{cfg['labels'].get(phase, phase)}**：{note}")

    # --- バランス思考フェーズの補助ツール（7コラム法のみ） ---
    if (
        st.session_state.mode == MODE_SEVEN_COLUMNS
        and phase == "balanced_thought"
    ):
        st.caption(
            "考えが浮かびづらいときは、AIに例を出してもらえます（あくまで参考まで）"
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button(
                "💡 AIにバランス思考の例をもらう",
                use_container_width=True,
                key="btn_balance_suggest",
            ):
                with st.spinner("例を考えています..."):
                    suggestions = suggest_balanced_thoughts(
                        st.session_state.messages
                    )
                if suggestions:
                    st.session_state.balance_suggestions = suggestions
                else:
                    st.warning("例を出せませんでした。もう一度お試しください。")
        with c2:
            if st.session_state.balance_suggestions:
                if st.button(
                    "🗑 例をしまう",
                    use_container_width=True,
                    key="btn_balance_clear",
                ):
                    st.session_state.balance_suggestions = []
                    st.rerun()

        if st.session_state.balance_suggestions:
            st.markdown("**💡 こんなバランス思考の例があります（参考まで）**")
            for i, s in enumerate(st.session_state.balance_suggestions, 1):
                with st.container(border=True):
                    st.markdown(f"**例 {i}**")
                    st.write(s)
                    if st.button(
                        f"📝 この例（例{i}）で進む",
                        key=f"adopt_balance_{i}",
                        help="この文をそのまま採用して、対話を次に進めます",
                    ):
                        st.session_state.queued_prompt = s
                        st.rerun()
            st.caption(
                "これらはあくまで例です。**あなた自身の言葉で書くのが一番大切**。"
                "ピンと来たものをそのまま採用しても、参考にしながら下の入力欄に"
                "自分の言葉で書き直してもOKです。"
            )

    # 完了後は入力欄を出さない
    if phase != "done":
        # 採用ボタン経由で送られたメッセージがあれば優先
        prompt = None
        if st.session_state.get("queued_prompt"):
            prompt = st.session_state.queued_prompt
            st.session_state.queued_prompt = None
        else:
            prompt = st.chat_input(placeholder)
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("..."):
                    # 危機検知：キーワード + LLMスコアリングの二層チェック
                    risk_result = assess_risk(prompt)
                    save_risk_score(prompt, risk_result)

                    if risk_result["triggered"]:
                        raw_reply = CRISIS_RESPONSE
                    else:
                        raw_reply = chat(
                            st.session_state.messages,
                            mode=st.session_state.mode,
                        )

                st.write(strip_meta(raw_reply))

                # メッセージはメタ情報も含めて保存（次ターンにClaudeが参照するため）
                st.session_state.messages.append(
                    {"role": "assistant", "content": raw_reply}
                )

                # フェーズ更新
                new_phase = parse_phase(raw_reply)
                if new_phase:
                    st.session_state.current_phase = new_phase

                # JSON抽出・保存
                if new_phase == "done" and not st.session_state.record_saved:
                    record = extract_json(raw_reply)
                    if record:
                        try:
                            row_id = save_record(
                                record,
                                st.session_state.messages,
                                event_datetime=st.session_state.get("event_datetime"),
                                mode=st.session_state.mode,
                            )
                            st.success("思考記録を保存しました。お疲れさまでした。")
                            st.session_state.record_saved = True
                            # 伝え方ノートへの誘導判定用に保存
                            st.session_state.last_saved_record = record

                            # 歪みを確定：記録に既にあればそれ、無ければ推定する
                            _captured_distortions = record.get("distortions") or []

                            # 7コラム法のときは、歪みをバックエンドで静かに推定して保存
                            if (
                                st.session_state.mode == MODE_SEVEN_COLUMNS
                                and not _captured_distortions
                            ):
                                try:
                                    with st.spinner("パターンを推定中..."):
                                        # Phase A: LLM と RAG を並走させ、A/B 比較ログを残す
                                        ab = infer_distortions_ab(record)
                                    # tagged 全件を保存（RAG-only も影ログとして残す）
                                    # UI 表示は shown=True のみに後段でフィルタ
                                    tagged = ab.get("tagged", [])
                                    st.session_state.last_ab_compare = ab  # サイドバーに表示
                                    if tagged:
                                        update_distortions(row_id, tagged)
                                        # _captured_distortions はユーザー向け表示専用 → shown=True のみ
                                        _captured_distortions = [d for d in tagged if d.get("shown", True)]
                                except Exception:
                                    pass  # 推定失敗はサイレントに

                            # 対話後のヒント表示用に保存（dict形式に正規化）
                            _normalized = normalize_distortions(_captured_distortions)
                            # 辞書に存在するパターンのみヒント対象に / RAG-only 影ログは除外
                            st.session_state.last_distortions = [
                                d for d in _normalized
                                if d.get("shown", True)
                                and (
                                    d["name"] in DISTORTION_TIPS
                                    or d["name"].startswith("結論の飛躍")
                                )
                            ]
                            # 違和感ボタンで dismiss 反映するために record_id を保持
                            st.session_state.last_record_id = row_id
                            st.session_state.show_last_tips = False
                        except Exception as e:
                            st.warning(f"保存に失敗: {e}")

            st.rerun()  # 進捗バーとヒントを即座に更新
    else:
        st.success("今回のセッションは完了です。お疲れさまでした。")

        # ===== 伝え方ノートへの誘導（言えなかった系のキーワードがあれば自動サジェスト） =====
        ASSERTION_URL = (
            "https://assertion-bot-public-7yjqhpnvshkdkj7avedrml.streamlit.app/"
        )
        # 直近の記録テキストからキーワード検出
        _last_record = st.session_state.get("last_saved_record") or {}
        _text_pool = " ".join([
            str(_last_record.get("situation") or ""),
            str(_last_record.get("automatic_thought") or ""),
            str(_last_record.get("balanced_thought") or ""),
        ])
        _assertion_keywords = [
            "言えなかった", "言えなく", "言えない",
            "伝えられなかった", "伝えられない",
            "黙って", "黙った", "飲み込んだ",
            "我慢", "流された", "反論できなかった",
            "言うべきだった", "強く言えなく", "うまく言えな",
        ]
        _detected = any(kw in _text_pool for kw in _assertion_keywords)
        _u_param = f"?u={CURRENT_USER_ID}" if CURRENT_USER_ID else ""

        if _detected:
            with st.container(border=True):
                st.markdown("💬 **もしかして、伝え方が引っかかった場面でしたか？**")
                st.caption(
                    "「言えなかった」「伝えられなかった」のような表現が記録に出てきました。"
                    "もしよければ、伝え方ノートで**3パターンの文案を一緒に考える**選択肢もあります。"
                    "押しつけではないので、ピンと来なければスルーしてください。"
                )
                st.link_button(
                    "🗣 伝え方ノートで文案を作る",
                    ASSERTION_URL + _u_param,
                )

        # 対話後の対処ヒント（オプトイン）
        if st.session_state.last_distortions:
            st.divider()
            _last = st.session_state.last_distortions  # list[dict]
            # 違和感で外されたものは集計から除外
            _active = [d for d in _last if not d.get("dismissed")]

            if _active:
                names_str = "・".join(d["name"] for d in _active)
                st.caption(
                    f"今回見えてきそうなパターン：**{names_str}**"
                )
            else:
                st.caption(
                    "今回はAIから挙げられたパターンを、すべて違和感ありとして外しました。"
                    "あなたの感覚を信じてください。"
                )

            # 各歪みについて、AIの推察した理由（推察形・違和感ボタン付き）
            _has_evidence = any(d.get("evidence") for d in _active)
            if _has_evidence:
                with st.expander("💭 AIがそう見えた理由（推察・参考まで）", expanded=False):
                    st.caption(
                        "💡 **自分の感覚を一番大事にしてください**。"
                        "ピンと来なかったり違和感があれば、無理に当てはめなくて大丈夫です。"
                        "右の「違和感」を押すと、その項目を外せます。"
                        "あなたが押した違和感は、サービスをあなたの感覚に合わせていく"
                        "ための大事な手がかりにもなります。"
                    )
                    _record_id = st.session_state.get("last_record_id")
                    for i, d in enumerate(_active):
                        if not d.get("evidence"):
                            continue
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**{d['name']}**")
                            st.caption(d["evidence"])
                        with col2:
                            if _record_id and st.button(
                                "違和感",
                                key=f"dismiss_last_{i}_{d['name']}",
                                help="この判定を外します",
                            ):
                                from storage import dismiss_distortion
                                try:
                                    dismiss_distortion(_record_id, d["name"], True)
                                    # session_state も即時反映
                                    for s in st.session_state.last_distortions:
                                        if s.get("name") == d["name"]:
                                            s["dismissed"] = True
                                    st.toast("外しました", icon="🌿")
                                    st.rerun()
                                except Exception as e:
                                    st.warning(f"外す処理に失敗: {e}")
            # ── 他の見方も見てみる（RAG-only の影ログを能動的に展開）──
            # Purpose（新しい視点・選択肢を増やす）に直結する仕組み。
            # 開いた時点で `other_views_opened` イベントを記録（positive engagement シグナル）。
            _ab_for_other = st.session_state.get("last_ab_compare")
            _record_id_for_other = st.session_state.get("last_record_id")
            _rag_only_items = []
            if _ab_for_other:
                for item in _ab_for_other.get("tagged", []):
                    if item.get("source") == "rag" and not item.get("shown", True):
                        _rag_only_items.append(item)
            if _rag_only_items and _record_id_for_other:
                _show_key = f"_show_other_views_{_record_id_for_other}"
                _logged_key = f"_logged_other_views_{_record_id_for_other}"
                if _show_key not in st.session_state:
                    st.session_state[_show_key] = False
                _label = "🌿 他の見方も見てみる" if not st.session_state[_show_key] else "🌿 閉じる"
                if st.button(_label, key=f"btn{_show_key}",
                             help="LLM が拾わなかった、意味的に近い他の候補（参考まで）"):
                    _was_open = st.session_state[_show_key]
                    st.session_state[_show_key] = not _was_open
                    # 初回オープン時のみログ
                    if not _was_open and not st.session_state.get(_logged_key):
                        try:
                            from storage import log_engagement
                            log_engagement(_record_id_for_other, "other_views_opened")
                            st.session_state[_logged_key] = True
                        except Exception:
                            pass
                    st.rerun()
                if st.session_state[_show_key]:
                    st.caption(
                        "💭 LLM が拾わなかったけれど、意味的に近そうな候補です。"
                        "「これかも」と感じたら参考に、ピンと来なければスルーで大丈夫です。"
                    )
                    for _i, _d in enumerate(_rag_only_items):
                        _c1, _c2 = st.columns([5, 1])
                        with _c1:
                            _score = _d.get("rag_score") or _d.get("score") or 0
                            try:
                                _score = float(_score)
                            except Exception:
                                _score = 0.0
                            st.markdown(f"**{_d['name']}**")
                            st.caption(f"意味検索で類似（score = {_score:.2f}）")
                        with _c2:
                            if st.button(
                                "違和感",
                                key=f"dismiss_other_{_record_id_for_other}_{_i}_{_d['name']}",
                                help="この候補を外します",
                            ):
                                from storage import dismiss_distortion
                                try:
                                    dismiss_distortion(_record_id_for_other, _d["name"], True)
                                    st.toast("外しました", icon="🌿")
                                    st.rerun()
                                except Exception as _e:
                                    st.warning(f"外す処理に失敗: {_e}")

            # ── Phase B：過去の似た記録（しきい値で gating）──
            # 記録 30 件未満では UI 自体を表示しない（cold-start 期間の沈黙）
            _record_id_pb = st.session_state.get("last_record_id")
            if CURRENT_USER_ID and is_phase_b_unlocked(CURRENT_USER_ID):
                _last_at = ""
                _saved_record = st.session_state.get("last_saved_record") or {}
                _at_text = str(_saved_record.get("automatic_thought") or "")
                _similar_past = infer_similar_past_records(
                    user_id=CURRENT_USER_ID,
                    automatic_thought=_at_text,
                    exclude_record_id=_record_id_pb,
                    top_k=3,
                )
                if _similar_past:
                    _pb_key = f"_show_phase_b_{_record_id_pb}"
                    if _pb_key not in st.session_state:
                        st.session_state[_pb_key] = False
                    _pb_label = (
                        "🪞 過去の似た記録を見てみる"
                        if not st.session_state[_pb_key]
                        else "🪞 閉じる"
                    )
                    if st.button(
                        _pb_label, key=f"btn{_pb_key}",
                        help="意味的に近い過去の自動思考（参考まで・判定ではありません）",
                    ):
                        st.session_state[_pb_key] = not st.session_state[_pb_key]
                        st.rerun()
                    if st.session_state[_pb_key]:
                        st.caption(
                            "💭 過去の自分が、似た言葉を使っていた記録です。"
                            "「**気付きの鏡**」として眺めるだけで十分。"
                            "似てないと感じたらスルーで大丈夫です。"
                        )
                        for _p in _similar_past:
                            _created = _p.get("created_at", "")[:10]
                            _days = _p.get("age_days", 0)
                            _score = _p.get("score", 0.0)
                            with st.container():
                                st.markdown(
                                    f"**{_created}**　（{_days}日前・似てる度 {_score:.2f}）"
                                )
                                st.markdown(f"> {_p.get('text', '')}")

            # 全部 dismiss されたらヒント表示は出さない
            if not _active:
                pass
            elif not st.session_state.show_last_tips:
                if st.button(
                    "💡 このパターンへの対処のヒントを見る",
                    use_container_width=False,
                    key="btn_show_last_tips",
                ):
                    st.session_state.show_last_tips = True
                    st.rerun()
                st.caption(
                    "※ 押さなくても大丈夫です。今は余韻にひたりたい、という選択もあり。"
                )
            else:
                st.markdown("**🛠 対処のヒント（あくまで参考まで）**")
                # 違和感で外したものはヒント対象から除外
                _names_for_tip = [d["name"] for d in _active]
                for name, tip in get_tips_for(_names_for_tip):
                    with st.container(border=True):
                        st.markdown(f"**💡 {name}**")
                        st.caption(f"特徴：{tip['description']}")
                        if tip.get("strength"):
                            st.markdown(f"💪 **良い面**：{tip['strength']}")
                        st.caption(f"ハマりやすいパターン：{tip['trap']}")
                        st.markdown("**試せそうな一歩**")
                        for a in tip["actions"]:
                            st.markdown(f"- {a}")
                st.caption(
                    "これは一般的な対処例です。"
                    "ピンと来たものだけ、試してみる感じで大丈夫です。"
                )
                if st.button("🗑 ヒントをしまう", key="btn_hide_last_tips"):
                    st.session_state.show_last_tips = False
                    st.rerun()

        # スマホでもサイドバーを開かずに次の行動が取れるよう、
        # メインエリアにも HOME / 新規セッション のボタンを置く
        st.divider()
        _done_hub_url = "https://app-public-qpy8b2ziwgdf9h2vmu5hqp.streamlit.app/"
        if CURRENT_USER_ID:
            _done_hub_url += f"?u={CURRENT_USER_ID}"
        _col_done_1, _col_done_2 = st.columns(2)
        with _col_done_1:
            st.link_button(
                "🏠 HOMEに戻る",
                _done_hub_url,
                use_container_width=True,
            )
        with _col_done_2:
            st.button(
                "🆕 新しいセッション",
                use_container_width=True,
                on_click=reset_session,
                key="btn_new_session_inline",
            )

elif view == "📊 傾向を見る":
    st.markdown("### あなたの傾向")
    df = load_records()

    if df.empty:
        st.info("まだ記録がありません。左側の「💬 対話」から始めてください。")
    else:
        # 列ごとに個別に datetime 変換してから fillna
        _ev = pd.to_datetime(df["event_datetime"], errors="coerce")
        _cr = pd.to_datetime(df["created_at"], errors="coerce")
        df["event_datetime"] = _ev.fillna(_cr)
        df["created_at"] = _cr
        df = df.dropna(subset=["event_datetime"])
        df["hour"] = df["event_datetime"].dt.hour
        df["dow"] = df["event_datetime"].dt.day_name()

        # ===== 🔍 最近の認知の歪み（メイン） =====
        st.subheader("🔍 最近の認知の歪み")
        st.caption(
            "あなたが最近、自分の中でハマりやすかったパターン。"
            "気づいておくと、新しい場面で「また同じパターンかも」と立ち止まりやすくなります。"
        )

        from baseline import distortion_trend
        from distortion_tips import get_tip as _get_tip

        _trend_top = distortion_trend(df, top_n=3, recent_days=15)
        if not _trend_top:
            st.caption("まだ歪みのデータが溜まっていません。記録を続けてみてください。")
        else:
            for d_info in _trend_top:
                with st.container(border=True):
                    _name = d_info["name"]
                    _h1, _h2, _h3 = st.columns([3, 2, 3])
                    _h1.markdown(f"### {_name}")
                    _h2.markdown(
                        f"**{d_info['count']}回**　"
                        f"<span style='color:#888'>({d_info['rate']*100:.0f}%)</span>",
                        unsafe_allow_html=True,
                    )
                    _h3.markdown(
                        f"{d_info['trend_emoji']} {d_info['trend_text']}"
                    )
                    _tip = _get_tip(_name)
                    if _tip and _tip.get("actions"):
                        st.markdown(
                            f"💡 **立ち止まれる一歩**：{_tip['actions'][0]}"
                        )
                    if _tip:
                        with st.expander("もっと詳しく（特徴・良い面・他の試せる一歩）"):
                            st.markdown(f"**特徴**：{_tip['description']}")
                            if _tip.get("strength"):
                                st.markdown(f"💪 **良い面**：{_tip['strength']}")
                            st.markdown(f"**ハマりやすいパターン**：{_tip['trap']}")
                            if len(_tip["actions"]) > 1:
                                st.markdown("**他に試せそうな一歩**")
                                for _a in _tip["actions"][1:]:
                                    st.markdown(f"- {_a}")

        st.divider()

        # ===== 📂 自分の状態の流れ（折りたたみ・任意） =====
        with st.expander(
            "📂 自分の状態の流れ", expanded=False,
        ):
            st.caption(
                "数字で状態を確認したい時だけ開いてください。"
                "毎日見るものではなく、月に1回くらい眺める素材として。"
            )

            chart_df = df.sort_values("event_datetime").copy()
            chart_df["intensity_before"] = pd.to_numeric(
                chart_df["intensity_before"], errors="coerce"
            )
            chart_df["intensity_after"] = pd.to_numeric(
                chart_df["intensity_after"], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=["intensity_before"])

            if len(chart_df) < 3:
                st.info("記録が3件以上たまると、推移のグラフが表示されます。")
            else:
                import plotly.graph_objects as go

                # ----- 強度の推移 -----
                st.markdown("**強度の推移**")
                mean_intensity = float(chart_df["intensity_before"].mean())
                fig_intensity = go.Figure()
                fig_intensity.add_hline(
                    y=mean_intensity, line_dash="dot", line_color="#888",
                    annotation_text=f"平均 {mean_intensity:.0f}",
                    annotation_position="right",
                )
                fig_intensity.add_trace(go.Scatter(
                    x=chart_df["event_datetime"], y=chart_df["intensity_before"],
                    mode="lines+markers", name="強度（前）",
                    line=dict(color="#4a90e2"), marker=dict(size=10),
                ))
                fig_intensity.update_layout(
                    yaxis=dict(range=[0, 100], title="強度（0-100）"),
                    xaxis=dict(title="出来事の日時"),
                    height=300, margin=dict(l=10, r=10, t=10, b=10),
                    hovermode="x unified",
                )
                st.plotly_chart(
                    fig_intensity, use_container_width=True,
                    config={"displayModeBar": False},
                )

                # ----- 強度カテゴリの件数 -----
                st.markdown("**直近30日の強度の内訳**")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                recent = chart_df[chart_df["event_datetime"] >= cutoff]
                if not recent.empty:
                    _i = recent["intensity_before"]
                    n_high = int((_i >= 80).sum())
                    n_mid = int(((_i >= 60) & (_i < 80)).sum())
                    n_low = int((_i < 60).sum())
                    c_b1, c_b2, c_b3 = st.columns(3)
                    c_b1.metric("80以上(強い反応)", f"{n_high}件")
                    c_b2.metric("60-79(中程度)", f"{n_mid}件")
                    c_b3.metric("0-59(軽め)", f"{n_low}件")
                else:
                    st.caption("直近30日の記録はまだありません。")

                # ----- 改善幅（対話の効果） -----
                with_after = chart_df.dropna(subset=["intensity_after"]).copy()
                if len(with_after) >= 3:
                    st.markdown("**対話の効果(改善幅)**")
                    with_after["improvement"] = (
                        with_after["intensity_before"]
                        - with_after["intensity_after"]
                    )
                    mean_imp = float(with_after["improvement"].mean())

                    fig_imp = go.Figure()
                    fig_imp.add_hline(y=0, line_color="#aaa", line_width=1)
                    fig_imp.add_hline(
                        y=mean_imp, line_dash="dot", line_color="#888",
                        annotation_text=f"平均改善幅 {mean_imp:.0f}",
                        annotation_position="right",
                    )
                    fig_imp.add_trace(go.Bar(
                        x=with_after["event_datetime"],
                        y=with_after["improvement"],
                        marker=dict(color="#27ae60"),
                        name="改善幅(前-後)",
                    ))
                    fig_imp.update_layout(
                        yaxis=dict(title="改善幅(前-後)"),
                        xaxis=dict(title="出来事の日時"),
                        height=300, margin=dict(l=10, r=10, t=10, b=10),
                        hovermode="x unified",
                        showlegend=False,
                    )
                    st.plotly_chart(
                        fig_imp, use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.caption(
                        "💡 緑の棒が **高いほど対話で楽になった** サイン。"
                        "**低い／0付近の日が増えてきたら、対話だけでは下がりにくい時期** "
                        "かもしれません(一人で抱え込まない方がいい合図かも)。"
                    )

                # グラフの読み方（インラインで簡潔に）
                st.caption(
                    "📖 **強度(0-100)について**：対話で「そのときの感情の強さは何点？」と聞かれたとき"
                    "あなた自身が答えた数値。"
                    "**改善幅** = 対話前 − 対話後。プラスが大きい = 対話で楽になった。"
                )

        # ===== 📂 認知の歪みの時系列（折りたたみ） =====
        _d_rows = []
        for _, _r in df.iterrows():
            _names = distortion_names(_r.get("distortions"))
            if not _names:
                continue
            for _n in _names:
                _d_rows.append({
                    "event_datetime": _r["event_datetime"],
                    "distortion": _n,
                })

        if len(_d_rows) >= 5:
            with st.expander(
                "📂 認知の歪みの時系列", expanded=False,
            ):
                st.caption(
                    "週ごとにどの歪みがどれくらい出ているか。"
                    "「最近は違うパターンが増えてきた」などの気づきに使えます。"
                )

                _dd = pd.DataFrame(_d_rows)
                _dd["week"] = _dd["event_datetime"].dt.to_period("W-MON").dt.start_time

                _top_names = _dd["distortion"].value_counts().head(6).index.tolist()
                _dd_top = _dd[_dd["distortion"].isin(_top_names)]

                weekly = (
                    _dd_top.groupby(["week", "distortion"])
                    .size().reset_index(name="count")
                )
                n_weeks = weekly["week"].nunique()

                if n_weeks <= 8:
                    fig_ts = px.bar(
                        weekly, x="week", y="count", color="distortion",
                        labels={"week": "週", "count": "出現回数",
                                "distortion": "歪み"},
                        category_orders={"distortion": _top_names},
                    )
                else:
                    fig_ts = px.area(
                        weekly, x="week", y="count", color="distortion",
                        labels={"week": "週", "count": "出現回数",
                                "distortion": "歪み"},
                        category_orders={"distortion": _top_names},
                    )
                fig_ts.update_layout(
                    height=360, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=-0.25),
                    hovermode="x unified",
                )
                st.plotly_chart(
                    fig_ts, use_container_width=True,
                    config={"displayModeBar": False},
                )

                _hidden = _dd["distortion"].nunique() - len(_top_names)
                _note = f"出現回数が多い上位{len(_top_names)}パターンのみ表示"
                if _hidden > 0:
                    _note += f"（残り{_hidden}パターンは省略）"
                st.caption(_note + "。")

        # ===== 📂 自分の使い方を知る（時間帯×曜日・折りたたみ） =====
        if len(df) >= 3:
            with st.expander(
                "📂 利用の傾向", expanded=False,
            ):
                st.caption(
                    "どの曜日・時間帯にこのアプリを使う傾向があるか。"
                    "「平日朝に多い」「金曜夜に多い」といった気づきの素材です。"
                )
                dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                             "Friday", "Saturday", "Sunday"]
                dow_jp = {"Monday": "月", "Tuesday": "火", "Wednesday": "水",
                          "Thursday": "木", "Friday": "金",
                          "Saturday": "土", "Sunday": "日"}
                df["dow_jp"] = df["dow"].map(dow_jp)

                def _bucket(h):
                    if 5 <= h < 11:  return "朝 (5-11)"
                    if 11 <= h < 15: return "昼 (11-15)"
                    if 15 <= h < 19: return "夕 (15-19)"
                    return "夜 (19-5)"
                df["time_bucket"] = df["hour"].apply(_bucket)
                bucket_order = ["朝 (5-11)", "昼 (11-15)",
                                "夕 (15-19)", "夜 (19-5)"]

                pivot = df.pivot_table(
                    index="time_bucket", columns="dow_jp",
                    values="intensity_before", aggfunc="mean"
                ).reindex(index=bucket_order,
                          columns=[dow_jp[d] for d in dow_order])

                fig3 = px.imshow(
                    pivot, text_auto=".0f",
                    labels={"color": "感情強度（平均）"},
                    color_continuous_scale="Reds",
                    aspect="auto",
                )
                st.plotly_chart(
                    fig3, use_container_width=True,
                    config={"displayModeBar": False},
                )

        st.markdown("#### 📂 これまでの記録")
        st.caption(
            "過去の自分がどんな新しい見方にたどり着いたか、見返せます。"
            "気になる時だけスクロールしてください。"
        )

        # 新しい順に表示
        recent = df.sort_values("event_datetime", ascending=False).head(10)

        # 編集モード状態（記録ID → bool）
        if "edit_record_id" not in st.session_state:
            st.session_state.edit_record_id = None

        for _, row in recent.iterrows():
            dt_str = pd.to_datetime(row["event_datetime"]).strftime("%Y-%m-%d %H:%M")
            intensity_change = f"{row['intensity_before']} → {row['intensity_after']}"
            title = f"📝 {dt_str}｜{row['emotion_name']}（{intensity_change}）"
            _row_id = int(row["id"]) if "id" in row.index and pd.notna(row["id"]) else None
            _is_editing = (
                _row_id is not None
                and st.session_state.edit_record_id == _row_id
            )

            with st.expander(title):
                _mode_of_row = row.get("mode") if "mode" in row.index else None
                if _mode_of_row:
                    _mode_label = MODE_CONFIGS.get(
                        _mode_of_row, {}
                    ).get("display_name", _mode_of_row)
                    st.caption(f"進め方：{_mode_label}")

                if _is_editing and _row_id is not None:
                    # ===== 編集フォーム（日時のみ）=====
                    st.caption(
                        "✏️ 編集モード（**出来事の日時のみ修正できます**。"
                        "詳しい内容は対話で固めたものなので、後から書き換えません）"
                    )
                    _form_key = f"edit_form_{_row_id}"
                    with st.form(_form_key):
                        _current_dt = row.get("event_datetime")
                        if isinstance(_current_dt, str):
                            try:
                                _current_dt = pd.to_datetime(_current_dt)
                            except Exception:
                                _current_dt = pd.Timestamp.now()
                        if pd.isna(_current_dt):
                            _current_dt = pd.Timestamp.now()
                        _c_d, _c_t = st.columns(2)
                        with _c_d:
                            _new_date = st.date_input(
                                "出来事の日付",
                                value=_current_dt.date(),
                                key=f"ed_date_{_row_id}",
                            )
                        with _c_t:
                            _new_time = st.time_input(
                                "出来事の時刻",
                                value=_current_dt.time(),
                                key=f"ed_time_{_row_id}",
                            )

                        _c_btn1, _c_btn2 = st.columns(2)
                        with _c_btn1:
                            _saved = st.form_submit_button(
                                "💾 保存", use_container_width=True, type="primary",
                            )
                        with _c_btn2:
                            _cancel = st.form_submit_button(
                                "キャンセル", use_container_width=True,
                            )

                    if _cancel:
                        st.session_state.edit_record_id = None
                        st.rerun()
                    if _saved:
                        from datetime import datetime as _dt
                        _new_event_dt = _dt.combine(_new_date, _new_time).isoformat()
                        _fields = {
                            "event_datetime": _new_event_dt,
                        }
                        try:
                            update_record(_row_id, _fields)
                            st.session_state.edit_record_id = None
                            st.success("更新しました。")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"更新に失敗しました：{_e}")

                else:
                    # ===== 通常表示 =====
                    st.markdown(f"**🌱 状況**")
                    st.write(row["situation"] or "（未記録）")

                    st.markdown(f"**💭 その時の自動思考**")
                    st.write(row["automatic_thought"] or "（未記録）")

                    # 7コラム法の場合は、根拠と反証を表示
                    _ev_for = row.get("evidence_for") if "evidence_for" in row.index else None
                    _ev_against = row.get("evidence_against") if "evidence_against" in row.index else None
                    if _ev_for:
                        st.markdown("**📌 根拠（事実）**")
                        st.write(_ev_for)
                    if _ev_against:
                        st.markdown("**🔄 反証**")
                        st.write(_ev_against)

                    # 歪みをバッジ風に＋根拠（あれば）
                    _dists_all = normalize_distortions(row.get("distortions"))
                    # ユーザー向け表示は shown=True のみ（RAG-only の影ログは除外）
                    _dists = [d for d in _dists_all if d.get("shown", True)]
                    # 違和感ありで外したものは別扱い
                    _active_dists = [d for d in _dists if not d.get("dismissed")]
                    _dismissed_dists = [d for d in _dists if d.get("dismissed")]
                    if _active_dists:
                        st.markdown("**🔍 気づいた認知の歪み**")
                        badges = "　".join([f"`{d['name']}`" for d in _active_dists])
                        st.markdown(badges)
                        _has_ev = any(d.get("evidence") for d in _active_dists)
                        if _has_ev and _row_id is not None:
                            with st.expander("💭 AIがそう見えた理由（推察・参考まで）", expanded=False):
                                st.caption(
                                    "💡 違和感があれば「違和感」を押すと外せます。"
                                    "AIの推察より、あなた自身の感覚を信じてください。"
                                    "押された「違和感」は、サービスをあなたの感覚に合わせていく手がかりにもなります。"
                                )
                                for i, d in enumerate(_active_dists):
                                    if not d.get("evidence"):
                                        continue
                                    _c1, _c2 = st.columns([5, 1])
                                    with _c1:
                                        st.markdown(f"**{d['name']}**")
                                        st.caption(d["evidence"])
                                    with _c2:
                                        if st.button(
                                            "違和感",
                                            key=f"dismiss_past_{_row_id}_{i}_{d['name']}",
                                            help="この判定を外します",
                                        ):
                                            from storage import dismiss_distortion
                                            try:
                                                dismiss_distortion(_row_id, d["name"], True)
                                                st.toast("外しました", icon="🌿")
                                                st.rerun()
                                            except Exception as _e:
                                                st.warning(f"外す処理に失敗: {_e}")
                    elif _dists and not _active_dists:
                        # 全部 dismissed
                        st.caption("（AIが推測した歪みは、すべて違和感ありとして外されています）")

                    # ── 他の見方も見てみる（RAG-only 影ログ）──
                    _rag_only_past = [
                        d for d in _dists_all
                        if d.get("source") == "rag"
                        and not d.get("shown", True)
                        and not d.get("dismissed")
                    ]
                    if _rag_only_past and _row_id is not None:
                        _show_key_past = f"_show_other_views_past_{_row_id}"
                        _logged_key_past = f"_logged_other_views_past_{_row_id}"
                        if _show_key_past not in st.session_state:
                            st.session_state[_show_key_past] = False
                        _label_past = "🌿 他の見方も見てみる" if not st.session_state[_show_key_past] else "🌿 閉じる"
                        if st.button(_label_past, key=f"btn{_show_key_past}",
                                     help="LLM が拾わなかった、意味的に近い他の候補（参考まで）"):
                            _was_open = st.session_state[_show_key_past]
                            st.session_state[_show_key_past] = not _was_open
                            if not _was_open and not st.session_state.get(_logged_key_past):
                                try:
                                    from storage import log_engagement
                                    log_engagement(_row_id, "other_views_opened")
                                    st.session_state[_logged_key_past] = True
                                except Exception:
                                    pass
                            st.rerun()
                        if st.session_state[_show_key_past]:
                            st.caption(
                                "💭 LLM が拾わなかったけれど、意味的に近そうな候補です。"
                                "「これかも」と感じたら参考に、ピンと来なければスルーで大丈夫です。"
                            )
                            for _i, _d in enumerate(_rag_only_past):
                                _c1, _c2 = st.columns([5, 1])
                                with _c1:
                                    _score = _d.get("rag_score") or _d.get("score") or 0
                                    try:
                                        _score = float(_score)
                                    except Exception:
                                        _score = 0.0
                                    st.markdown(f"**{_d['name']}**")
                                    st.caption(f"意味検索で類似（score = {_score:.2f}）")
                                with _c2:
                                    if st.button(
                                        "違和感",
                                        key=f"dismiss_other_past_{_row_id}_{_i}_{_d['name']}",
                                        help="この候補を外します",
                                    ):
                                        from storage import dismiss_distortion
                                        try:
                                            dismiss_distortion(_row_id, _d["name"], True)
                                            st.toast("外しました", icon="🌿")
                                            st.rerun()
                                        except Exception as _e:
                                            st.warning(f"外す処理に失敗: {_e}")

                    # 違和感で外したものを「戻す」導線（折りたたみ）
                    if _dismissed_dists and _row_id is not None:
                        with st.expander(
                            f"🌿 違和感ありとして外した歪み（{len(_dismissed_dists)}件）",
                            expanded=False,
                        ):
                            for j, d in enumerate(_dismissed_dists):
                                _c1, _c2 = st.columns([5, 1])
                                with _c1:
                                    st.markdown(
                                        f"~~**{d['name']}**~~"
                                    )
                                    if d.get("evidence"):
                                        st.caption(f"~~{d['evidence']}~~")
                                with _c2:
                                    if st.button(
                                        "戻す",
                                        key=f"undismiss_past_{_row_id}_{j}_{d['name']}",
                                        help="この判定を再び表示します",
                                    ):
                                        from storage import dismiss_distortion
                                        try:
                                            dismiss_distortion(_row_id, d["name"], False)
                                            st.toast("戻しました", icon="↩️")
                                            st.rerun()
                                        except Exception as _e:
                                            st.warning(f"戻す処理に失敗: {_e}")

                    # バランス思考があればそちらを優先表示、なければ adaptive_thought
                    _balanced = row.get("balanced_thought") if "balanced_thought" in row.index else None
                    if _balanced:
                        st.markdown(f"**✨ バランス思考**")
                        st.info(_balanced)
                    else:
                        st.markdown(f"**✨ 新しい見方（適応的思考）**")
                        st.info(row["adaptive_thought"] or "（未記録）")

                    st.markdown(
                        f"**感情の変化**：{row['emotion_name']}　"
                        f"{row['intensity_before']} → {row['intensity_after']}　"
                        f"（{row['intensity_before'] - row['intensity_after']:+d} 変化）"
                    )

                    if _row_id is not None:
                        if st.button(
                            "✏️ この記録を編集する",
                            key=f"btn_edit_{_row_id}",
                        ):
                            st.session_state.edit_record_id = _row_id
                            st.rerun()

        with st.expander("📋 全記録の一覧（テーブル表示）"):
            display_df = df[["event_datetime", "situation", "emotion_name",
                             "intensity_before", "intensity_after",
                             "adaptive_thought"]].sort_values(
                "event_datetime", ascending=False).copy()
            display_df.columns = ["出来事の日時", "状況", "感情",
                                  "強度（前）", "強度（後）", "新しい見方"]
            st.dataframe(display_df, use_container_width=True)

elif view == "📝 週次レポート":
    st.markdown("### 週次レポート")
    st.caption("1週間の思考記録を Claude に要約してもらいます。冷静に振り返る時間のお供に。")

    df_all = load_records()

    # --- 今週のレポート ---
    week_start, week_end = current_week_range()
    st.subheader(f"📅 今週（{week_start.strftime('%m/%d')}〜{week_end.strftime('%m/%d')}）")

    df_week = filter_week(df_all, week_start, week_end)
    cached = load_weekly_report(week_start)

    if df_week.empty:
        st.info("今週はまだ記録がありません。「💬 対話」から1件だけでも書いてみてください。")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.caption(f"今週の記録数：**{len(df_week)}件**")
        with col_b:
            btn_label = "🔄 再生成" if cached else "📝 レポートを作成"
            if st.button(btn_label, use_container_width=True, type="primary"):
                with st.spinner("Claude が1週間を振り返っています..."):
                    try:
                        md = generate_weekly_report(df_week, anthropic_client)
                        save_weekly_report(week_start, week_end, md, len(df_week))
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成に失敗しました：{e}")

        # 生成済みがあれば表示
        if cached:
            generated = pd.to_datetime(cached["generated_at"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"生成日時：{generated}　｜　対象記録：{cached['n_records']}件")
            st.divider()
            st.markdown(cached["markdown"])
            st.divider()
            with st.expander("📋 このレポートの元になった記録"):
                for _, r in df_week.iterrows():
                    dt_str = r["event_datetime"].strftime("%m/%d %H:%M")
                    st.markdown(
                        f"**{dt_str}** ｜ {r['emotion_name']}（"
                        f"{r['intensity_before']}→{r['intensity_after']}）"
                    )
                    if r.get("situation"):
                        st.caption(f"状況: {r['situation'][:80]}")
        else:
            st.caption("まだ今週のレポートは作成されていません。")

    # --- 過去のレポート ---
    st.divider()
    st.subheader("📚 過去のレポート")
    past = load_all_weekly_reports()
    # 今週以外
    past = past[past["week_start"] != str(week_start)] if not past.empty else past

    if past.empty:
        st.caption("（まだ過去のレポートはありません）")
    else:
        for _, r in past.iterrows():
            ws = pd.to_datetime(r["week_start"]).strftime("%m/%d")
            we = pd.to_datetime(r["week_end"]).strftime("%m/%d")
            with st.expander(f"📝 {ws}〜{we}（{r['n_records']}件の記録から）"):
                st.markdown(r["markdown"])
