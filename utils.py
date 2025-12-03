import pickle, re, pandas as pd, numpy as np, os, json, hashlib
from catboost import Pool
import streamlit as st
from rapidfuzz import fuzz, process

# ============================================================
# ⚙️ Global Config
# ============================================================
DEBUG_MODE = False
def log(msg):
    if DEBUG_MODE:
        st.write(f"🪵 DEBUG: {msg}")


# ============================================================
# 🔹 Model Loader
# ============================================================
def load_model(path):
    """Safely load a serialized CatBoost model (.sav)."""
    if not os.path.exists(path):
        log(f"Model file missing: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 🔹 Entity Extraction (robust + gender/age fixes)
# ============================================================
def extract_entities(text):
    text_lower = text.lower().strip()

    # --- Vaccine detection ---
    vaccine_aliases = {
        "COVID": ["covid", "covid19", "pfizer", "moderna", "booster"],
        "FLU": ["flu", "influenza"],
        "VARZOS": ["zoster", "shingles", "herpes zoster"],
        "PPV": ["ppv", "pneumococcal", "pneumonia"],
    }
    vaccine = None
    for key, patterns in vaccine_aliases.items():
        result = process.extractOne(text_lower, patterns, scorer=fuzz.partial_ratio)
        if result:
            _, score, _ = result
            if score >= 70:
                vaccine = key
                break

    # --- Gender detection ---
    if re.search(r"\b(female|woman|lady|girl|madam|mrs|ms|aunt|mother|grandmother)\b", text_lower):
        sex = "F"
    elif re.search(r"\b(male|man|boy|guy|gentleman|father|grandfather)\b", text_lower):
        sex = "M"
    else:
        sex = "U"

    # --- Age detection ---
    age_flag = "exact"
    match = re.search(r"\b(\d{1,3})\s*(?:year|yr|years|yrs|old)\b", text_lower)
    if match:
        age = int(match.group(1))
    elif "young" in text_lower:
        age_flag, age = "approximate", 25
    elif "middle" in text_lower:
        age_flag, age = "approximate", 45
    elif re.search(r"\bold\b|\belderly\b|\bsenior\b|\baged\b", text_lower):
        age_flag, age = "approximate", 70
    elif any(k in text_lower for k in ["child", "boy", "girl"]):
        age_flag, age = "approximate", 10
    else:
        age_flag, age = "missing", 40

    # cross-inference (safe now)
    if sex == "F" and age < 50 and re.search(r"\bold\b|\belderly\b|\bgrandmother\b", text_lower):
        age = 70
    if sex == "M" and age < 50 and re.search(r"\bold\b|\belderly\b|\bgrandfather\b", text_lower):
        age = 70

    # --- Symptom detection ---
    symptom_map = {
        "fever": ["fever", "temperature", "hot"],
        "chills": ["chills", "cold", "shivering"],
        "headache": ["headache", "migraine"],
        "nausea": ["nausea", "vomiting", "sick", "upset stomach"],
        "fatigue": ["tired", "weak", "fatigue", "exhausted"],
        "pain": ["pain", "ache", "soreness", "body ache"],
        "rash": ["rash", "itch", "redness"],
        "dizziness": ["dizzy", "faint", "lightheaded"],
        "swelling": ["swelling", "inflamed"],
        "breath": ["shortness of breath", "difficulty breathing", "breathless"],
        "cough": ["cough", "congestion"]
    }

    detected = set()
    words = text_lower.split()

    for base, synonyms in symptom_map.items():
        found = any(re.search(rf"\b{s}\b", text_lower) for s in synonyms)
        if found:
            detected.add(base)
            continue

        # fuzzy fallback with stricter threshold
        for s in synonyms:
            if len(s) > 3:
                score = fuzz.partial_ratio(s, text_lower)
                if score >= 90 and not any(ex in text_lower for ex in ["no " + s, "not " + s]):
                    detected.add(base)
                    break

    # correction: if "cold" or "chills" detected, remove fever false positives
    if ("chills" in detected or "cough" in detected or "cold" in text_lower) and "fever" in detected:
        detected.discard("fever")

    return vaccine, age, sex, sorted(list(detected)), age_flag

# ============================================================
# 🔹 Feature Builder
# ============================================================
def build_feature_row(model, age, sex, symptoms, text):
    text_lower = text.lower()
    cols = model.feature_names_
    data = {}

    hosp_kw = ["hospital", "hospitalized", "admitted", "icu", "er", "emergency"]
    died_kw = ["died", "death", "deceased", "passed away", "fatal"]
    disable_kw = ["disabled", "paralyzed", "unable to move", "bedridden"]
    threat_kw = ["life threatening", "critical", "coma", "unresponsive", "collapsed"]

    for c in cols:
        cu = c.upper()
        if any(k in cu for k in ["TEXT", "NOTE", "COMMENT", "DESC", "HISTORY", "OBS", "DETAIL"]):
            data[c] = str(text)
        elif "SYMPTOM" in cu:
            data[c] = ", ".join(symptoms) if symptoms else str(text)
        elif "AGE" in cu:
            data[c] = age
        elif "SEX" in cu:
            data[c] = {"M": 0, "F": 1, "U": 2}.get(sex, 2)
        elif "HOSP" in cu:
            data[c] = int(any(k in text_lower for k in hosp_kw))
        elif "DIED" in cu:
            data[c] = int(any(k in text_lower for k in died_kw))
        elif "DISABLE" in cu:
            data[c] = int(any(k in text_lower for k in disable_kw))
        elif "THREAT" in cu:
            data[c] = int(any(k in text_lower for k in threat_kw))
        elif "SERIOUS" in cu or "SEVERE" in cu:
            data[c] = int("severe" in text_lower or "serious" in text_lower)
        else:
            data[c] = 0

    df = pd.DataFrame([data])
    for col in df.columns:
        if any(k in col.upper() for k in ["TEXT", "NOTE", "COMMENT", "DESC", "HISTORY", "OBS", "DETAIL", "SYMPTOM"]):
            df[col] = df[col].astype(str)
    return df


# ============================================================
# 🔹 Prediction Logic with Advice
# ============================================================
def predict_seriousness(text, interactive=False):
    """Predict seriousness using the trained CatBoost model, and show live debug info."""
    vaccine, age, sex, symptoms, age_flag = extract_entities(text)

    # Sidebar debug panel — visible during model validation
    with st.sidebar.expander("🧩 Model Debug Panel", expanded=True):
        st.markdown("**🔍 Extracted Entities:**")
        st.write({
            "Vaccine": vaccine or "❌ Not detected",
            "Age": age,
            "Sex": sex,
            "Symptoms": ", ".join(symptoms) if symptoms else "N/A",
            "Age Confidence": age_flag,
        })

    # Interactive age prompt if missing
    if age_flag != "exact" and interactive:
        st.warning("🧠 Please specify your exact age (e.g., 25, 40–50) for better accuracy.")
        user_input = st.text_input("Enter your age:")
        if user_input:
            m = re.search(r"\d{1,3}", user_input)
            if m:
                age = int(m.group())
                st.info(f"✅ Age set to {age}")
            else:
                st.error("⚠️ Couldn't parse age — using default 40.")
        else:
            st.stop()

    if not vaccine:
        st.error("⚠️ Could not identify vaccine type (mention COVID, FLU, VARZOS, or PPV).")
        return "⚠️ Could not identify vaccine type (mention COVID, FLU, VARZOS, or PPV)."

    # --- Load trained CatBoost model ---
    model_map = {
        "COVID": "covid_catboost_model.sav",
        "FLU": "flu_catboost_model.sav",
        "VARZOS": "varzos_catboost_model.sav",
        "PPV": "ppv_catboost_model.sav",
    }
    model_path = os.path.join("MODELS", model_map.get(vaccine))
    model = load_model(model_path)
    if not model:
        st.error(f"❌ Model file missing: {model_path}")
        return f"❌ Model file missing: {model_path}"

    # Display model loaded in debug panel
    with st.sidebar.expander("🧠 Model Info", expanded=True):
        st.markdown(f"**Model Loaded:** `{model_path}`")

    # Identify text features (fallback for older CatBoost)
    try:
        text_feats = model.get_text_feature_names()
    except Exception:
        text_feats = [
            f for f in model.feature_names_
            if any(k in f.upper() for k in
                   ["TEXT", "NOTE", "COMMENT", "DESC", "ILL", "MEDS", "ALLERG", "OTHER", "HISTORY"])
        ]

    # Build feature row
    X = build_feature_row(model, age, sex, symptoms, text)
    for c in model.feature_names_:
        if c not in X.columns:
            X[c] = ""
    for col in text_feats:
        if col in X.columns:
            X[col] = X[col].astype(str)
    cat_feats = [c for c in X.columns if X[c].dtype == "object" and c not in text_feats]

    # --- Predict probability ---
    try:
        pool = Pool(X, text_features=text_feats, cat_features=cat_feats)
        pred_proba = model.predict_proba(pool)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return f"❌ Prediction error: {e}"

    # Add keyword weighting (for qualitative emphasis)
    score = pred_proba
    for word in ["hospital", "died", "life threatening", "coma", "critical"]:
        if word in text.lower():
            score += 0.15
    score = min(1.0, score)
    pct = round(score * 100, 2)

    # Determine seriousness level
    if score >= 0.7:
        label = "🔴 **Severe (High Seriousness)**"
        advice = "🚨 Seek immediate medical attention."
    elif score >= 0.4:
        label = "🟠 **Moderate Seriousness**"
        advice = "⚠️ Monitor closely and consult a doctor if symptoms persist."
    else:
        label = "🟢 **Mild (Low Seriousness)**"
        advice = "🙂 Rest, stay hydrated, and monitor symptoms."

    matched_kw = ", ".join([w for w in ["hospital", "died", "critical", "coma"] if w in text.lower()]) or "None"

    # --- Debug Info (Model Output) ---
    with st.sidebar.expander("📊 Model Output Confidence", expanded=True):
        st.metric(label="Predicted Probability (Serious Case)", value=f"{pct} %")
        st.write(f"Model raw probability: `{pred_proba:.4f}`")

    # --- Final formatted response (for chatbot or UI) ---
    return (
        f"**Prediction for {vaccine} vaccine:** {label}\n\n"
        f"**Seriousness Score:** {pct}%\n"
        f"**Detected Info:** Age={age}, Sex={sex}, Symptoms={', '.join(symptoms) if symptoms else 'N/A'}\n"
        f"**Keywords Detected:** {matched_kw}\n\n"
        f"{advice}"
    )


# ============================================================
# 🔹 Chat + Auth Utilities
# ============================================================
def get_user_file(username):
    os.makedirs("USER_DATA", exist_ok=True)
    return f"USER_DATA/{username}_history.json"

def save_chat(username, history):
    with open(get_user_file(username), "w") as f:
        json.dump(history, f)

def load_chat(username):
    p = get_user_file(username)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# ============================================================
# 🔹 Theme Utility (stable + persistent)
# ============================================================
def apply_theme(theme="dark", accent="#0055aa", font="Inter", compact=False):
    """Enhanced reactive theme injection with accent + font + compact mode (fixed for visible <style> bug)."""
    padding = "0.6rem 0.8rem" if compact else "0.9rem 1.2rem"

    # --- Base Styles ---
    css = f"""
    <style>
    footer {{visibility:hidden;}}
    html, body, [class*="stApp"] {{
        font-family: '{font}', sans-serif;
        transition: all 0.6s ease-in-out;
    }}
    div[data-testid="stAppViewContainer"] {{
        transition: background-color 0.6s ease, color 0.6s ease;
    }}
    .stButton>button {{
        background-color: {accent} !important;
        color: white !important;
        border-radius: 10px !important;
        padding: {padding};
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        opacity: 0.9 !important;
        transform: scale(1.03);
    }}
    .stSelectbox, .stRadio, .stTextInput>div>div>input {{
        transition: background-color 0.5s ease, color 0.5s ease;
    }}
    """

    # --- Theme Specific ---
    if theme == "dark":
        css += """
        body {background-color:#0E1117; color:#ffffff;}
        .stTextInput>div>div>input {background-color:#1E2229; color:white;}
        .stSidebar {background-color:#111418 !important;}
        .stMarkdown, .stCode, .stExpander {color:white;}
        """
    else:
        css += """
        body {background-color:#FAFBFD; color:#111111;}
        .stTextInput>div>div>input {background-color:#ffffff; color:#111;}
        .stSidebar {background-color:#F5F7FA !important;}
        .stMarkdown, .stCode, .stExpander {color:#111;}
        """

    css += "</style>"

    st.markdown(css, unsafe_allow_html=True)

