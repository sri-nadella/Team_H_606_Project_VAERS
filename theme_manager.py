import streamlit as st
from app.utils import apply_theme

# ============================================================
# 🎨 THEME MANAGER — Fully Reactive Version
# ============================================================
def set_theme():
    """Dynamic sidebar theme customization panel 
    with instant UI updates (theme, color, font, compact mode)."""

    # --- Initialize defaults ---
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("accent_color", "#0055aa")  # default blue
    st.session_state.setdefault("font_choice", "Inter")
    st.session_state.setdefault("compact_mode", False)

    # --- Sidebar UI Panel ---
    with st.sidebar.expander("🎨 Personalize Your Experience", expanded=False):
        st.markdown("Tweak the appearance of **VaxShield AI** 👇")

        # --- 1️⃣ Theme Toggle ---
        theme_choice = st.radio(
            "App Theme",
            ["🌙 Dark Mode", "🌞 Light Mode"],
            index=0 if st.session_state.theme == "dark" else 1,
            horizontal=True,
        )
        selected_theme = "dark" if "Dark" in theme_choice else "light"
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.toast(f"✨ Switched to {selected_theme.title()} Theme!", icon="🎨")
            st.rerun()

        # --- 2️⃣ Accent Color Picker ---
        new_color = st.color_picker(
            "Accent Color (buttons & highlights)",
            value=st.session_state.accent_color,
        )
        if new_color != st.session_state.accent_color:
            st.session_state.accent_color = new_color
            st.toast("🌈 Accent color updated!", icon="✅")
            st.rerun()

        # --- 3️⃣ Font Selector ---
        font_options = ["Inter", "Roboto", "Open Sans", "Poppins", "Lato", "Source Sans Pro"]
        selected_font = st.selectbox(
            "Choose App Font",
            font_options,
            index=font_options.index(st.session_state.font_choice)
            if st.session_state.font_choice in font_options else 0,
        )
        if selected_font != st.session_state.font_choice:
            st.session_state.font_choice = selected_font
            st.toast(f"🖋 Font changed to {selected_font}", icon="🧠")
            st.rerun()

        # --- 4️⃣ Compact Mode ---
        compact_toggle = st.checkbox(
            "Enable Compact Mode (denser layout)",
            value=st.session_state.compact_mode,
        )
        if compact_toggle != st.session_state.compact_mode:
            st.session_state.compact_mode = compact_toggle
            st.toast(
                "📏 Compact mode toggled "
                + ("on" if compact_toggle else "off") + "!",
                icon="🪄"
            )
            st.rerun()

    # --- Apply Visual Theme ---
    apply_theme(
        theme=st.session_state.theme,
        accent=st.session_state.accent_color,
        font=st.session_state.font_choice,
        compact=st.session_state.compact_mode
    )
