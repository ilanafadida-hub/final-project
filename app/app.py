"""
Streamlit Web Application — AI Product Workflow Dashboard

Pages:
  1. Run Flow    — trigger the full pipeline with a button
  2. EDA Report  — display eda_report.html inline
  3. Predict     — input RFM values → customer cluster prediction
  4. Downloads   — links to all output artifacts
"""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

CLEAN_DATA   = OUTPUTS_DIR / "clean_data.csv"
EDA_REPORT   = OUTPUTS_DIR / "eda_report.html"
INSIGHTS     = OUTPUTS_DIR / "insights.md"
CONTRACT     = OUTPUTS_DIR / "dataset_contract.json"
FEATURES     = OUTPUTS_DIR / "features.csv"
MODEL_PATH   = OUTPUTS_DIR / "model.pkl"
EVAL_REPORT  = OUTPUTS_DIR / "evaluation_report.md"
MODEL_CARD   = OUTPUTS_DIR / "model_card.md"

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Product Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Run Flow", "EDA Report", "Predict", "Downloads"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Output Files**")
for artifact in [CLEAN_DATA, EDA_REPORT, INSIGHTS, CONTRACT,
                 FEATURES, MODEL_PATH, EVAL_REPORT, MODEL_CARD]:
    icon = "✅" if artifact.exists() else "❌"
    st.sidebar.markdown(f"{icon} `{artifact.name}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Run Flow
# ══════════════════════════════════════════════════════════════════════════════
if page == "Run Flow":
    st.title("Run Full Pipeline")
    st.markdown(
        """
        Click **Run Pipeline** to trigger the complete AI workflow:

        1. **Analyst Crew** — validate, clean, EDA, contract
        2. **Validation Gate 1** — contract vs. clean data
        3. **Scientist Crew** — validate, features, model, model card
        4. **Validation Gate 2** — required features check
        """
    )

    if st.button("▶ Run Pipeline", type="primary"):
        st.info("Pipeline started — this takes a few minutes…")
        log_box = st.empty()

        flow_script = PROJECT_ROOT / "flow" / "main_flow.py"
        python_exe  = sys.executable

        with st.spinner("Running…"):
            proc = subprocess.run(
                [python_exe, str(flow_script)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                encoding="utf-8",
                errors="replace",
            )

        if proc.returncode == 0:
            st.success("Pipeline completed successfully!")
        else:
            st.error("Pipeline failed. See logs below.")

        with st.expander("Pipeline output", expanded=True):
            combined = proc.stdout + ("\n\nSTDERR:\n" + proc.stderr if proc.stderr.strip() else "")
            st.code(combined, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA Report
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Report":
    st.title("Exploratory Data Analysis Report")

    tab1, tab2 = st.tabs(["Interactive Report", "Business Insights"])

    with tab1:
        if EDA_REPORT.exists():
            html_content = EDA_REPORT.read_text(encoding="utf-8", errors="replace")
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.warning("EDA report not found. Run the pipeline first.")

    with tab2:
        if INSIGHTS.exists():
            st.markdown(INSIGHTS.read_text(encoding="utf-8", errors="replace"))
        else:
            st.warning("Insights file not found. Run the pipeline first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Customer Segment Prediction")
    st.markdown(
        """
        Enter a customer's RFM metrics to predict which segment they belong to.
        The model was trained on **scaled** features; raw values are transformed
        automatically using the feature distribution from `features.csv`.
        """
    )

    # Load model
    if not MODEL_PATH.exists():
        st.error("Model not found. Run the pipeline first.")
        st.stop()
    if not FEATURES.exists():
        st.error("features.csv not found. Run the pipeline first.")
        st.stop()

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    @st.cache_data
    def load_feature_stats():
        """Return (mean, std) per feature column from the scaled features.csv.
        Since features.csv holds StandardScaler output (mean≈0, std≈1) we
        re-fit a scaler from clean_data.csv to recover the original scale."""
        df_clean = pd.read_csv(CLEAN_DATA, parse_dates=["InvoiceDate"], low_memory=False)
        df_clean["Revenue"] = df_clean["Quantity"] * df_clean["Price"]
        ref_date = df_clean["InvoiceDate"].max()

        rfm = df_clean.groupby("Customer ID").agg(
            Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        ).reset_index()
        rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
        rfm["UniqueProducts"] = df_clean.groupby("Customer ID")["StockCode"].nunique().values

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(rfm[FEATURE_COLS])
        return scaler

    model  = load_model()

    with st.spinner("Loading feature statistics…"):
        scaler = load_feature_stats()

    st.subheader("Enter Customer RFM Values")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=0, max_value=3650, value=30, step=1,
        )
        frequency = st.number_input(
            "Frequency (number of orders)",
            min_value=1, max_value=10000, value=5, step=1,
        )
    with col2:
        monetary = st.number_input(
            "Monetary (total spend, GBP)",
            min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0,
        )
        avg_order = st.number_input(
            "Avg Order Value (GBP)",
            min_value=0.0, max_value=100_000.0, value=100.0, step=10.0,
        )
    with col3:
        unique_products = st.number_input(
            "Unique Products purchased",
            min_value=1, max_value=5000, value=10, step=1,
        )

    if st.button("Predict Segment", type="primary"):
        raw = np.array([[recency, frequency, monetary, avg_order, unique_products]])
        scaled = scaler.transform(raw)
        cluster = int(model.predict(scaled)[0])

        st.success(f"Predicted Customer Segment: **Cluster {cluster}**")

        # Simple segment descriptions
        descriptions = {
            0: "High-value / Champions — frequent buyers with high spend.",
            1: "Loyal Customers — regular buyers with moderate spend.",
            2: "At-Risk / Lapsed — used to buy but recent activity is low.",
            3: "New / Low-Engagement — few purchases, low monetary value.",
        }
        st.info(descriptions.get(cluster, "Segment description not available."))

        # Show model card if available
        if MODEL_CARD.exists():
            with st.expander("View Model Card"):
                st.markdown(MODEL_CARD.read_text(encoding="utf-8", errors="replace"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Downloads
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Downloads":
    st.title("Output Artifacts")
    st.markdown("Download any of the files produced by the pipeline.")

    artifacts = [
        (CLEAN_DATA,    "clean_data.csv",       "Cleaned dataset (CSV)"),
        (EDA_REPORT,    "eda_report.html",       "Interactive EDA report (HTML)"),
        (INSIGHTS,      "insights.md",           "Business insights (Markdown)"),
        (CONTRACT,      "dataset_contract.json", "Dataset contract (JSON)"),
        (FEATURES,      "features.csv",          "Engineered RFM features (CSV)"),
        (MODEL_PATH,    "model.pkl",             "Best clustering model (pickle)"),
        (EVAL_REPORT,   "evaluation_report.md",  "Model evaluation report (Markdown)"),
        (MODEL_CARD,    "model_card.md",         "Model card (Markdown)"),
    ]

    for path, fname, desc in artifacts:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"**{fname}** — {desc}")
        with col2:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                st.write(f"`{size_kb:,.1f} KB`")
            else:
                st.write("*not found*")
        with col3:
            if path.exists():
                data = path.read_bytes()
                mime = (
                    "text/html" if fname.endswith(".html") else
                    "application/json" if fname.endswith(".json") else
                    "text/markdown" if fname.endswith(".md") else
                    "text/csv" if fname.endswith(".csv") else
                    "application/octet-stream"
                )
                st.download_button(
                    label="Download",
                    data=data,
                    file_name=fname,
                    mime=mime,
                    key=fname,
                )
