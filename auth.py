
import streamlit as st, hashlib, json, os
USER_FILE = "app/users.json"

def hash_password(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def load_users(): 
    return json.load(open(USER_FILE)) if os.path.exists(USER_FILE) else {}

def save_users(u): 
    json.dump(u, open(USER_FILE,"w"), indent=4)

def login_signup():
    st.title("🔐 Welcome to VaxShield AI")
    tab1, tab2 = st.tabs(["Login","Sign Up"])
    users = load_users()

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in users and users[u]["password"]==hash_password(p):
                st.session_state.authenticated=True
                st.success(f"Welcome {u}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("Choose username")
        npw = st.text_input("Choose password", type="password")
        if st.button("Create Account"):
            if nu in users:
                st.warning("Username already exists.")
            else:
                users[nu]={"password":hash_password(npw)}
                save_users(users)
                st.success("Account created! Please login.")
