import streamlit as st
from app import dashboard, chatbot, auth
from app.theme_manager import set_theme

st.set_page_config(page_title="VaxShield AI", layout="wide", page_icon="💉")

# Styling
st.markdown("""
<style>
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
div[data-testid="stAppViewContainer"] { animation: fadeIn 1.2s ease-in; }
body { font-family: 'Inter', sans-serif; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "first_visit" not in st.session_state:
    st.toast("👋 Welcome back to VaxShield AI!", icon="💉")
    st.session_state.first_visit = True

set_theme()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    auth.login_signup()
else:
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to:", ["📊 Dashboard", "💬 Chatbot", "ℹ️ About", "🚪 Logout"])

    if page == "📊 Dashboard":
        st.toast("📈 Loading Dashboard...", icon="📊")
        dashboard.run_dashboard()
    elif page == "💬 Chatbot":
        st.toast("🤖 Chatbot active", icon="💬")
        chatbot.run_chatbot()
    elif page == "ℹ️ About":
        st.title("ℹ️ About VaxShield AI")
        st.markdown("""
        **VaxShield AI** predicts vaccine adverse event seriousness using CatBoost models  
        trained across multiple vaccine datasets: **COVID**, **FLU**, **VARZOS**, and **PPV**.
        """, unsafe_allow_html=True)
    elif page == "🚪 Logout":
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.success("✅ Logged out successfully.")
        st.rerun()
