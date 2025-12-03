
import streamlit as st, pandas as pd, plotly.express as px, seaborn as sns, matplotlib.pyplot as plt, os
from app.utils import load_model

def run_dashboard():
    st.title("📊 VaxShield Model Dashboard")
    p = os.path.join("OUTPUTS","catboost_vaccine_metrics_summary.csv")
    if not os.path.exists(p):
        st.warning("⚠️ Metrics file missing in OUTPUTS/")
        return

    df = pd.read_csv(p)
    st.dataframe(df.style.background_gradient(cmap="Blues"))

    metric = st.selectbox("Metric to compare",["Accuracy","F1-Score","ROC-AUC","Precision","Recall"])
    st.plotly_chart(px.bar(df,x="Vaccine",y=metric,color="Vaccine",text_auto=".2f"), use_container_width=True)

    st.subheader("🧩 Feature Importance")
    vac = st.selectbox("Select Model", df["Vaccine"].unique())
    mpath = os.path.join("MODELS", f"{vac.lower()}_catboost_model.sav")

    if os.path.exists(mpath):
        m = load_model(mpath)
        if hasattr(m,"feature_importances_"):
            im = pd.DataFrame({
                "Feature": m.feature_names_,
                "Importance": m.feature_importances_
            }).sort_values("Importance",ascending=False).head(10)
            plt.figure(figsize=(6,4))
            sns.barplot(y="Feature",x="Importance",data=im,palette="crest")
            st.pyplot(plt.gcf()); plt.clf()
        else:
            st.info("No feature importance available for this model.")
