import streamlit as st
import time
import random
from datetime import datetime
from app.utils import predict_seriousness

# ============================================================
# 💬 VaxShield AI Chatbot (Final Model-Driven Version)
# ============================================================

def run_chatbot():
    st.title("💬 VaxShield AI — Model Validation Chatbot")
    st.caption("Analyze real-world vaccine reaction reports using trained CatBoost models.")

    # --- Initialize Session State ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_chat_update" not in st.session_state:
        st.session_state.last_chat_update = datetime.now()
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = True
    if "compact_ui" not in st.session_state:
        st.session_state.compact_ui = False

    # ========================================================
    # ⚙️ SIDEBAR CONTROLS
    # ========================================================
    with st.sidebar.expander("⚙️ Chat Controls", expanded=False):
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.toast("🧹 Chat cleared!", icon="✅")
            st.session_state.last_chat_update = datetime.now()
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🔁 Reanalyze Last"):
                last_msg = st.session_state.chat_history[-1]["content"]
                handle_user_message(last_msg, reanalyze=True)

        if st.session_state.get("show_debug", False) and "debug_info" in st.session_state:
            dbg = st.session_state["debug_info"]

            with st.sidebar.expander("🧩 Extracted Entities", expanded=True):
                for k, v in dbg["entities"].items():
                    st.write(f"**{k}:** {v}")
    
            with st.sidebar.expander("🧠 Model Info", expanded=False):
                st.write(f"**Loaded Model:** `{dbg['model']}`")

            with st.sidebar.expander("📊 Prediction Confidence", expanded=False):
                st.metric("Seriousness Probability", f"{dbg['probability']*100:.2f}%")
                st.metric("Adjusted Score", f"{dbg['score']} %")


        st.markdown("💡 **Try these examples:**")
        st.code("A 70-year-old man was hospitalized after a COVID vaccine", language="text")
        st.code("A child developed fever and rash after a flu shot", language="text")
        st.code("An elderly woman fainted after taking the PPV shot", language="text")

        st.divider()

        # ✅ Debug toggle (controls sidebar panels in utils)
        st.session_state.show_debug = st.checkbox("🧩 Show Model Debug Info", value=st.session_state.show_debug)

        # ✅ Compact UI toggle
        st.session_state.compact_ui = st.checkbox("🪶 Compact Chat UI", value=st.session_state.compact_ui)

        st.caption("_Toggle debug panels and UI layout above._")

    # ========================================================
    # 💬 DISPLAY CHAT HISTORY
    # ========================================================
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("time", "")

        compact = st.session_state.get("compact_ui", False)
        pad = "8px 10px" if compact else "10px 15px"
        font_size = "14px" if compact else "16px"

        if role == "user":
            text_color = "#fff" if st.session_state.get("theme", "dark") == "dark" else "#000"
            bubble_bg = "#004aad" if st.session_state.get("theme", "dark") == "dark" else "#d0e6ff"
            st.markdown(
                f"""
                <div style='background-color:{bubble_bg};
                            color:{text_color};
                            border-radius:12px;
                            padding:{pad};
                            margin:6px 0;
                            text-align:right;
                            font-size:{font_size};'>
                    🧑‍💬 <b>You</b> ({timestamp}):<br>{content}
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif role == "assistant":
            bg_color = "#262730" if st.session_state.get("theme", "dark") == "dark" else "#f5f6f9"
            st.markdown(
                f"""
                <div style='background-color:{bg_color};
                            border-radius:12px;
                            padding:{pad};
                            margin:6px 0;
                            text-align:left;
                            font-size:{font_size};'>
                    🤖 <b>VaxShield AI</b> ({timestamp}):<br>{content}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # 🧠 USER CHAT INPUT
    # ========================================================
    user_input = st.chat_input("Describe your vaccination experience...")
    if user_input:
        handle_user_message(user_input)


# ============================================================
# 🧠 Message Handling Logic
# ============================================================
def handle_user_message(user_input, reanalyze=False):
    """Process user message and route to appropriate intent."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    if not reanalyze:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "time": timestamp}
        )

    lowered = user_input.lower()

    # --- Intent Detection ---
    if any(w in lowered for w in ["took", "developed", "hospital", "reaction", "side effect", "vaccine"]):
        intent = "prediction"
    elif any(w in lowered for w in ["how serious", "severity", "serious"]):
        intent = "prediction"
    elif any(w in lowered for w in ["hello", "hi", "hey", "good morning", "who are you"]):
        intent = "greeting"
    elif any(w in lowered for w in ["what", "why", "common", "symptom"]):
        intent = "info"
    else:
        intent = "fallback"

    # --- Intent Routing ---
    if intent == "greeting":
        reply = random.choice([
            "👋 Hi there! I'm VaxShield AI — your vaccine safety validation assistant.",
            "Hello! You can describe a vaccine reaction, and I'll analyze its seriousness.",
            "Hey! Try something like: 'A 45-year-old woman took the flu shot and felt dizzy.'"
        ])

    elif intent == "info":
        reply = (
            "💡 Common mild reactions: fatigue, soreness, fever, or headache.\n"
            "Severe ones (hospitalization, paralysis, death) are rare — our CatBoost models estimate these probabilities."
        )

    elif intent == "prediction":
        with st.spinner("🔍 Analyzing report using trained CatBoost model..."):
            time.sleep(1.2)
            raw = predict_seriousness(user_input, interactive=False)
            summary = summarize_prediction(raw)
            reply = f"{raw}\n\n{summary}"

    else:
        reply = (
            "🤔 I didn’t quite catch that.\n"
            "Try describing a case — for example:\n"
            "_‘A 50-year-old woman took the flu shot and developed dizziness and nausea.’_"
        )

    # --- Save Assistant Reply ---
    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply, "time": timestamp}
    )
    st.session_state.last_chat_update = datetime.now()

    st.rerun()


# ============================================================
# 🧾 Summary Logic (Model Output Condenser)
# ============================================================
def summarize_prediction(prediction_text):
    """Condense model output into a short clinical summary."""
    if "Severe" in prediction_text or "High Seriousness" in prediction_text:
        tone = random.choice([
            "🚨 Signs suggest a **high-severity** reaction — please consult a doctor immediately.",
            "⚠️ This case appears serious and may require medical review.",
            "❗ The model flagged this as severe — human review recommended."
        ])
    elif "Moderate" in prediction_text:
        tone = random.choice([
            "🟠 The model predicts **moderate seriousness** — monitor symptoms closely.",
            "⚠️ Reaction appears moderate — consult a healthcare provider if it worsens."
        ])
    elif "Mild" in prediction_text:
        tone = random.choice([
            "🟢 This appears to be a **low seriousness** reaction — typical mild case.",
            "✅ Likely mild reaction — rest, hydration, and observation are enough."
        ])
    else:
        tone = "🤔 The model couldn’t confidently classify this report. Try rephrasing the description."

    return tone


