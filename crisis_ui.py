"""3 モード危機検出の UI コンポーネント（4 アプリ共通）。

このモジュールも crisis_detection.py と同じく 4 アプリにコピー配置する。
"""
import streamlit as st

from crisis_detection import (
    WARNING_MESSAGE, CRITICAL_MESSAGE,
    WARNING_HOTLINES, CRITICAL_HOTLINES,
)


def render_warning_ui(
    on_resume_key: str = "crisis_warning_resume",
    on_dismiss_callback=None,
) -> None:
    """対話を控えるモード UI。誤検知時の解除導線あり。

    Args:
        on_resume_key: 解除ボタンの session_state キー
        on_dismiss_callback: 解除時に呼ばれる callable（state クリアなど）
    """
    with st.container(border=True):
        st.markdown("### ⚠️ 一度立ち止まりましょう")
        st.markdown(WARNING_MESSAGE)

        st.markdown("---")
        st.markdown("**相談窓口（無料・匿名で話せます）**")
        for h in WARNING_HOTLINES:
            _line = f"- **{h['name']}**"
            if h.get("tel"):
                _line += f"：{h['tel']}"
            if h.get("url"):
                _line += f"：{h['url']}"
            if h.get("note"):
                _line += f"　／　{h['note']}"
            st.markdown(_line)

        st.markdown("---")
        st.caption(
            "💡 **誤検知の場合**：自分の言葉が予想外に検出されたと感じたら、"
            "下のボタンで解除して対話を続けられます。"
        )
        if st.button(
            "🔄 解除して続ける（誤検知だった）",
            key=on_resume_key,
            use_container_width=True,
        ):
            if on_dismiss_callback is not None:
                try:
                    on_dismiss_callback()
                except Exception:
                    pass
            st.rerun()


def render_critical_ui() -> None:
    """緊急窓口モード UI。解除導線なし。

    対話・記録の継続を完全に止め、最上段に 119/110 を表示する。
    """
    # 視覚的に強い境界線
    st.markdown(
        '<div style="border:3px solid #c0392b; padding:16px; border-radius:8px; '
        'background:#fdedec;">',
        unsafe_allow_html=True,
    )
    st.markdown("### 🚨 緊急の窓口")
    st.markdown(CRITICAL_MESSAGE)

    st.markdown("---")
    st.markdown("**最優先の連絡先**")
    for h in CRITICAL_HOTLINES:
        _priority = h.get("priority", False)
        _line = f"- "
        if _priority:
            _line += "🚨 "
        _line += f"**{h['name']}**"
        if h.get("tel"):
            _line += f"：**{h['tel']}**"
        if h.get("url"):
            _line += f"：{h['url']}"
        if h.get("note"):
            _line += f"　／　{h['note']}"
        st.markdown(_line)

    st.markdown(
        "**🌐 こころの耳（厚労省ポータル）**：https://kokoro.mhlw.go.jp/"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        "このセッションは安全のため停止しています。"
        "上記のいずれかにご連絡ください。"
    )


def block_with_critical(mode: str) -> bool:
    """critical モードなら UI を出して True を返す（呼び出し側で stop すべき合図）。"""
    if mode == "critical":
        render_critical_ui()
        return True
    return False
