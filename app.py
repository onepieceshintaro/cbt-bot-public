import json
import re
from collections import Counter
from datetime import datetime, date, time, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px

from cbt_engine import (
    chat, assess_risk, client as anthropic_client,
    infer_distortions_from_record, suggest_balanced_thoughts,
)
from storage import (
    init_db, save_record, load_records, update_distortions,
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

st.set_page_config(page_title="CBT セルフヘルプ", page_icon="💭", layout="wide")

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
    st.link_button(
        "🏠 HOME に戻る",
        "https://app-public-qpy8b2ziwgdf9h2vmu5hqp.streamlit.app/",
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
        st.caption(MODE_CONFIGS[st.session_state.mode]["description"])
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
    st.caption("※ このBotは医療行為ではありません")
    with st.expander("辛いときの相談窓口"):
        st.markdown("""
        - いのちの電話：0570-783-556
        - よりそいホットライン：0120-279-338
        - こころの健康相談統一ダイヤル：0570-064-556
        """)


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
        now = datetime.now()
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
            st.caption(
                "これらはあくまで例です。**あなた自身の言葉で書くのが一番大切**。"
                "ピンと来たものを参考に、下の入力欄にあなたの言葉を書いてみてください。"
            )

    # 完了後は入力欄を出さない
    if phase != "done":
        if prompt := st.chat_input(placeholder):
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

                            # 歪みを確定：記録に既にあればそれ、無ければ推定する
                            _captured_distortions = record.get("distortions") or []

                            # 7コラム法のときは、歪みをバックエンドで静かに推定して保存
                            if (
                                st.session_state.mode == MODE_SEVEN_COLUMNS
                                and not _captured_distortions
                            ):
                                try:
                                    with st.spinner("パターンを推定中..."):
                                        dists = infer_distortions_from_record(record)
                                    if dists:
                                        update_distortions(row_id, dists)
                                        _captured_distortions = dists
                                except Exception:
                                    pass  # 推定失敗はサイレントに

                            # 対話後のヒント表示用に保存
                            st.session_state.last_distortions = [
                                d for d in _captured_distortions
                                if d in DISTORTION_TIPS
                            ]
                            st.session_state.show_last_tips = False
                        except Exception as e:
                            st.warning(f"保存に失敗: {e}")

            st.rerun()  # 進捗バーとヒントを即座に更新
    else:
        st.success("今回のセッションは完了です。お疲れさまでした。")

        # 対話後の対処ヒント（オプトイン）
        if st.session_state.last_distortions:
            st.divider()
            names_str = "・".join(st.session_state.last_distortions)
            st.caption(
                f"今回出ていたパターン：**{names_str}**"
            )
            if not st.session_state.show_last_tips:
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
                for name, tip in get_tips_for(st.session_state.last_distortions):
                    with st.container(border=True):
                        st.markdown(f"**💡 {name}**")
                        st.caption(f"特徴：{tip['description']}")
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

        st.info("左側の「新しいセッションを始める」ボタンで続けられます。")

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

        # === あなたのベースライン（個人の「いつもの状態」からの逸脱） ===
        st.subheader("📊 あなたのベースライン")
        st.caption(
            "万人向けの「正常」ではなく、あなた自身の直近の状態から"
            "「今回はいつもと違う」を検知します。"
        )

        baseline = compute_intensity_baseline(df)
        if not baseline or baseline.get("insufficient"):
            need = (baseline or {}).get("required", MIN_RECORDS_FOR_BASELINE)
            have = (baseline or {}).get("n", 0)
            st.info(
                f"ベースライン学習は記録 **{need}件** 以上で有効になります。"
                f"現在：**{have}件**（あと {max(0, need - have)}件）"
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("直近の平均強度", f"{baseline['mean']:.0f}")
            c2.metric("標準偏差 σ", f"{baseline['std']:.1f}")
            c3.metric("いつもの上限（+2σ）", f"{baseline['upper_2s']:.0f}")
            c4.metric("サンプル数", f"{baseline['n']}件")

            # 感情強度の時系列＋2σバンド＋逸脱点を赤で
            chart_df = df.sort_values("event_datetime").copy()
            chart_df["intensity_before"] = pd.to_numeric(
                chart_df["intensity_before"], errors="coerce"
            )
            chart_df["is_high"] = chart_df["intensity_before"] > baseline["upper_2s"]

            import plotly.graph_objects as go
            fig_bl = go.Figure()
            fig_bl.add_hrect(
                y0=baseline["lower_2s"], y1=baseline["upper_2s"],
                fillcolor="rgba(100,180,255,0.08)", line_width=0,
                annotation_text=f"いつもの範囲 (±{SIGMA_THRESHOLD:.0f}σ)",
                annotation_position="top left",
            )
            fig_bl.add_hline(
                y=baseline["mean"], line_dash="dot", line_color="#888",
                annotation_text=f"平均 {baseline['mean']:.0f}",
                annotation_position="right",
            )
            normal = chart_df[~chart_df["is_high"]]
            fig_bl.add_trace(go.Scatter(
                x=normal["event_datetime"], y=normal["intensity_before"],
                mode="lines+markers", name="感情強度（前）",
                line=dict(color="#4a90e2"), marker=dict(size=8),
            ))
            high = chart_df[chart_df["is_high"]]
            if not high.empty:
                fig_bl.add_trace(go.Scatter(
                    x=high["event_datetime"], y=high["intensity_before"],
                    mode="markers", name="いつもより強い日",
                    marker=dict(color="#e74c3c", size=14,
                                symbol="circle-open", line=dict(width=3)),
                ))
            fig_bl.update_layout(
                yaxis=dict(range=[0, 110], title="強度（前）"),
                xaxis=dict(title="出来事の日時"),
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_bl, use_container_width=True)

            # 逸脱日の詳細
            if not high.empty:
                with st.expander(f"🔴 いつもより強かったセッション（{len(high)}件）"):
                    for _, r in high.iterrows():
                        dt_s = r["event_datetime"].strftime("%Y-%m-%d %H:%M")
                        dev = deviation_from_baseline(
                            int(r["intensity_before"]), baseline
                        )
                        sigma_str = f"+{dev['sigma']:.1f}σ" if dev else ""
                        st.markdown(
                            f"**{dt_s}** ｜ {r['emotion_name']}（強度{r['intensity_before']}、{sigma_str}）"
                        )
                        if r.get("situation"):
                            st.caption(f"場面: {r['situation'][:100]}")

        # 直近の頻出歪み TOP3
        top_d = top_distortions(df, top_n=3)
        if top_d:
            st.markdown("**🔍 最近よく出ている認知の歪み**")
            cols = st.columns(len(top_d))
            for col, d in zip(cols, top_d):
                col.metric(
                    d["name"],
                    f"{d['count']}回",
                    f"{d['rate']*100:.0f}% のセッションで",
                )
            st.caption(
                "※ これらに気づいておくと、新しいセッションで「また同じパターンかも」と"
                "立ち止まりやすくなります"
            )

            # 各歪みへの対処ヒント（エクスパンダー、興味ある人だけ展開）
            st.markdown("**🛠 対処のヒント（参考まで）**")
            st.caption(
                "以下はあくまで一般的な対処例です。"
                "ピンと来るものがあれば試してみる、くらいの距離感で。"
            )
            for d in top_d:
                tip = DISTORTION_TIPS.get(d["name"])
                if not tip:
                    continue
                with st.expander(f"💡 {d['name']} への対処を見る"):
                    st.markdown(f"**特徴**：{tip['description']}")
                    st.markdown(f"**ハマりやすいパターン**：{tip['trap']}")
                    st.markdown("**試せそうな一歩**")
                    for a in tip["actions"]:
                        st.markdown(f"- {a}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("感情強度の変化（前 vs 後）")
            melt = df.melt(
                id_vars=["event_datetime"],
                value_vars=["intensity_before", "intensity_after"],
                var_name="タイミング", value_name="強度"
            )
            fig = px.line(
                melt, x="event_datetime", y="強度",
                color="タイミング", markers=True,
                labels={"event_datetime": "出来事の日時"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("認知の歪みの出現頻度")
            all_distortions = []
            for s in df["distortions"].dropna():
                try:
                    all_distortions.extend(json.loads(s))
                except Exception:
                    pass
            counts = Counter(all_distortions)
            if counts:
                dist_df = pd.DataFrame(
                    counts.most_common(), columns=["歪み", "回数"]
                )
                fig2 = px.bar(dist_df, x="回数", y="歪み", orientation="h")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("歪みデータがまだありません。")

        # 時間帯・曜日のヒートマップ（3件以上で表示）
        if len(df) >= 3:
            st.subheader("いつ気持ちが揺れやすい？（曜日×時間帯）")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            dow_jp = {"Monday": "月", "Tuesday": "火", "Wednesday": "水",
                      "Thursday": "木", "Friday": "金",
                      "Saturday": "土", "Sunday": "日"}
            df["dow_jp"] = df["dow"].map(dow_jp)
            # 時間帯を4つに束ねる（朝/昼/夕/夜）
            def bucket(h):
                if 5 <= h < 11:  return "朝 (5-11)"
                if 11 <= h < 15: return "昼 (11-15)"
                if 15 <= h < 19: return "夕 (15-19)"
                return "夜 (19-5)"
            df["time_bucket"] = df["hour"].apply(bucket)
            bucket_order = ["朝 (5-11)", "昼 (11-15)", "夕 (15-19)", "夜 (19-5)"]

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
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("あと数件記録が溜まると、時間帯×曜日のパターンも見えてきます。")

        st.subheader("記録の詳細（新しい順）")
        st.caption("過去の自分がどんな新しい見方にたどり着いたか、見返せます。")

        # 新しい順に表示
        recent = df.sort_values("event_datetime", ascending=False).head(10)

        for _, row in recent.iterrows():
            dt_str = pd.to_datetime(row["event_datetime"]).strftime("%Y-%m-%d %H:%M")
            intensity_change = f"{row['intensity_before']} → {row['intensity_after']}"
            title = f"📝 {dt_str}｜{row['emotion_name']}（{intensity_change}）"

            with st.expander(title):
                _mode_of_row = row.get("mode") if "mode" in row.index else None
                if _mode_of_row:
                    _mode_label = MODE_CONFIGS.get(
                        _mode_of_row, {}
                    ).get("display_name", _mode_of_row)
                    st.caption(f"進め方：{_mode_label}")

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

                # 歪みをバッジ風に
                try:
                    distortions = json.loads(row["distortions"] or "[]")
                except Exception:
                    distortions = []
                if distortions:
                    st.markdown("**🔍 気づいた認知の歪み**")
                    badges = "　".join([f"`{d}`" for d in distortions])
                    st.markdown(badges)

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
