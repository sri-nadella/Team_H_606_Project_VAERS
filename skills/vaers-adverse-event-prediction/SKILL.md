---
name: vaers-adverse-event-prediction
description: >
  End-to-end data science workflow for predicting serious adverse events from
  healthcare surveillance or reporting datasets (like VAERS). Use this skill
  whenever the user is working on: adverse event classification, VAERS data
  analysis, imbalanced binary classification on medical/clinical data, per-group
  or per-category ML model strategies, mixed structured + text feature pipelines,
  CatBoost on healthcare data, or deploying a Streamlit dashboard + chatbot for
  model validation. Trigger even if the user just mentions "VAERS", "adverse
  events", "serious vs non-serious classification", "imbalanced healthcare data",
  or "per-vaccine modeling".
---

# VAERS Adverse Event Prediction Skill

End-to-end workflow for predicting serious outcomes from healthcare surveillance
data. Based on VaxShield (DS606 capstone): VAERS 2015–2025, 4 vaccine groups,
CatBoost per-group models, Streamlit deployment.

---

## Phase 1: Data Ingestion & Merging

VAERS (and similar systems) split data across multiple files. Always merge on the
shared ID before anything else.

```python
import pandas as pd
import glob

def load_vaers_years(data_dir, years):
    frames = {"data": [], "vax": [], "symptoms": []}
    for year in years:
        for suffix, key in [("DATA", "data"), ("VAX", "vax"), ("SYMPTOMS", "symptoms")]:
            files = glob.glob(f"{data_dir}/{year}VAERS{suffix}.csv")
            if files:
                frames[key].append(pd.read_csv(files[0], encoding="latin1", low_memory=False))
    data = pd.concat(frames["data"])
    vax  = pd.concat(frames["vax"])
    symp = pd.concat(frames["symptoms"])
    merged = data.merge(vax, on="VAERS_ID", how="left") \
                 .merge(symp, on="VAERS_ID", how="left")
    return merged
```

**Key decisions:**
- Use `encoding="latin1"` — VAERS CSVs are not UTF-8.
- Merge `left` on VAERS_ID to keep all reports even if VAX/SYMPTOMS are missing.
- After merging, drop obvious duplicates: `df.drop_duplicates(subset="VAERS_ID")`.

---

## Phase 2: Target Label Definition

Binary classification: **Serious** vs **Non-Serious**.

```python
SERIOUS_COLS = ["DIED", "L_THREAT", "HOSPITAL", "X_STAY", "DISABLE", "BIRTH_DEFECT"]

def make_target(df):
    # Any serious outcome = 1
    df["SERIOUS"] = df[SERIOUS_COLS].apply(
        lambda row: 1 if row.astype(str).str.upper().isin(["Y", "1"]).any() else 0,
        axis=1
    )
    return df
```

**Watch out for:** `Y/N` strings vs `1/0` integers — VAERS mixes both. Always
normalize before applying the rule.

---

## Phase 3: Filtering to Vaccine Groups

Do NOT train one global model. Focus on specific vaccine groups with adequate
volume. Each group has different demographics, symptom profiles, and serious-rate.

```python
VACCINE_MAP = {
    "COVID": ["COVID19", "COVID19-2"],
    "FLU":   ["FLU", "FLUN", "FLUN3", "FLUN4", "FLU3", "FLU4"],
    "VARZOS": ["VARZOS"],
    "PPV":   ["PPV23", "PPV"],
}

def filter_vaccine_group(df, vax_name, vax_types):
    return df[df["VAX_TYPE"].isin(vax_types)].copy()
```

**Why per-group?** COVID reports dominate VAERS volume. A global model learns
COVID-specific decision boundaries. Vaccine-specific models capture distinct
symptom/demographic patterns per group.

---

## Phase 4: Cleaning & Feature Engineering

### Structured features
- **Age**: numeric; impute missing with median per vaccine group
- **Sex**: categorical (`M/F/U`); fill missing with `"U"`
- **State**: categorical; fill missing with `"Unknown"`
- **Manufacturer**: categorical; fill missing with `"Unknown"`

### Text features
VAERS has multiple narrative fields. Treat each separately, not concatenated.

```python
TEXT_COLS = ["SYMPTOM_TEXT", "HISTORY", "OTHER_MEDS", "CUR_ILL"]
STRUCT_COLS = ["AGE_YRS", "SEX", "STATE", "VAX_MANU"]
```

Fill text NaNs with `""` before vectorizing.

### TF-IDF pipeline (baseline)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(text_cols, cat_cols, num_cols):
    transformers = []
    for col in text_cols:
        transformers.append((f"tfidf_{col}", TfidfVectorizer(max_features=500, ngram_range=(1,2)), col))
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    return ColumnTransformer(transformers)
```

---

## Phase 5: Train/Test Split

Always stratify to preserve the serious/non-serious ratio in both sets.

```python
from sklearn.model_selection import train_test_split

X = df[TEXT_COLS + STRUCT_COLS]
y = df["SERIOUS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
```

---

## Phase 6: Modeling

### Baseline (Phase 2 equivalent)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

baselines = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "rf":     RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "xgb":    XGBClassifier(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
}
```

### Final model: CatBoost (Phase 3 equivalent)

CatBoost handles mixed feature types natively and is robust to missing values.
Train one model per vaccine group.

```python
from catboost import CatBoostClassifier

def train_catboost(X_train, y_train, cat_features_idx):
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        class_weights={0: 1, 1: (y_train==0).sum()/(y_train==1).sum()},
        cat_features=cat_features_idx,
        verbose=100,
        random_seed=42,
    )
    model.fit(X_train, y_train)
    return model
```

Save each model:
```python
import pickle
with open("covid_catboost_model.sav", "wb") as f:
    pickle.dump(model, f)
```

---

## Phase 7: Evaluation (Imbalanced Classification)

**Never rely on accuracy alone.** The minority class (Serious) is what matters.

```python
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, matthews_corrcoef
)

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_test, y_proba):.4f}")
    print(f"MCC:      {matthews_corrcoef(y_test, y_pred):.4f}")
```

**Key metrics to watch:**
- `PR-AUC` — best single metric for imbalanced problems
- `Recall` (Serious class) — missing a serious event is costly
- `MCC` — balanced measure that accounts for all four confusion matrix cells
- Watch `PPV` results carefully — near-perfect metrics often signal data leakage
  (outcome fields included as features). Run feature ablation if suspicious.

---

## Phase 8: Streamlit Deployment (VaxShield Pattern)

Structure the app as: dashboard tab + chatbot tab.

```python
import streamlit as st
import pickle

# Load all models at startup
models = {
    "COVID": pickle.load(open("covid_catboost_model.sav", "rb")),
    "FLU":   pickle.load(open("flu_catboost_model.sav", "rb")),
    # ...
}

tab1, tab2 = st.tabs(["Dashboard", "Chatbot"])

with tab1:
    # Show per-vaccine metrics from a summary CSV
    import pandas as pd
    metrics = pd.read_csv("catboost_vaccine_metrics_summary.csv")
    st.dataframe(metrics)

with tab2:
    user_input = st.text_area("Describe the adverse event:")
    if st.button("Predict"):
        vaccine = extract_vaccine(user_input)   # lightweight NLP
        age     = extract_age(user_input)
        sex     = extract_sex(user_input)
        prob    = models[vaccine].predict_proba([build_features(user_input)])[0][1]
        severity = "HIGH" if prob > 0.5 else "LOW"
        st.write(f"Seriousness probability: {prob:.2f} — {severity}")
        st.caption("This is decision support only, not medical diagnosis.")
```

**Entity extraction tips:**
- Detect vaccine type with keyword matching before regex
- For age: if text says "elderly" → display `>50`, not a specific number
- Always show extracted entities back to the user (transparency panel)
- Never claim more precision than the input supports

---

## Common Pitfalls

| Problem | Fix |
|---|---|
| Accuracy looks great but Recall on Serious is ~0 | Model predicts all Non-Serious. Use `class_weight="balanced"` or `scale_pos_weight` |
| PPV model shows near-perfect metrics | Likely leakage from outcome indicator columns. Remove `DIED`, `HOSPITAL`, etc. from features |
| Single global model performs worse on small vaccine groups | Switch to per-group models |
| VAERS CSV encoding errors | Use `encoding="latin1"` |
| Text columns have NaN | Fill with `""` before TF-IDF, not `"Unknown"` |
| Age field is string or mixed type | `pd.to_numeric(df["AGE_YRS"], errors="coerce")` then impute |

---

## Validation Checklist Before Deployment

- [ ] Stratified train/test split used
- [ ] Target label uses outcome columns, NOT features
- [ ] Serious-class Recall > 0.5 on test set
- [ ] PR-AUC reported alongside ROC-AUC
- [ ] Per-group models trained separately
- [ ] Outcome indicator columns removed from feature set
- [ ] App shows extracted entities to user
- [ ] App includes disclaimer: "decision support, not diagnosis"

