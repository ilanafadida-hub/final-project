"""
Data Scientist Crew — Tool implementations.
Each tool does the real ML/data processing work.
Tools use result_as_answer=True so agents return tool output directly,
minimising LLM token consumption.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from crewai.tools import tool
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLEAN_DATA_PATH = OUTPUTS_DIR / "clean_data.csv"
CONTRACT_PATH = OUTPUTS_DIR / "dataset_contract.json"
FEATURES_PATH = OUTPUTS_DIR / "features.csv"
MODEL_PATH = OUTPUTS_DIR / "model.pkl"
EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.md"
MODEL_CARD_PATH = OUTPUTS_DIR / "model_card.md"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tool 1 — Validate contract
# ---------------------------------------------------------------------------
@tool("validate_contract", result_as_answer=True)
def validate_contract(input: str = "") -> str:
    """
    Load outputs/clean_data.csv and outputs/dataset_contract.json, then verify
    that all required columns exist with the correct dtypes as specified in the
    contract. Returns a pass/fail validation report.
    """
    required_cols = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ]

    # Load data
    try:
        df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False, nrows=1000)
    except Exception as e:
        return f"CONTRACT VALIDATION FAILED\n  Could not load clean_data.csv: {e}"

    # Load contract
    try:
        with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)
    except Exception as e:
        return f"CONTRACT VALIDATION FAILED\n  Could not load dataset_contract.json: {e}"

    report_lines = [
        "CONTRACT VALIDATION REPORT",
        "==========================",
        f"Validated at: {datetime.now().isoformat()}",
        f"Data file: {CLEAN_DATA_PATH}",
        f"Contract file: {CONTRACT_PATH}",
        f"Contract version: {contract.get('version', 'unknown')}",
        "",
    ]

    # Check required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    present_cols = [c for c in required_cols if c in df.columns]

    report_lines.append("Column presence check:")
    for col in required_cols:
        status = "PASS" if col in df.columns else "FAIL (missing)"
        report_lines.append(f"  {col}: {status}")

    report_lines.append("")

    # Check dtypes against contract
    contract_cols = contract.get("columns", {})
    dtype_issues = []

    dtype_map = {
        "int64": ["int64", "int32", "int"],
        "float64": ["float64", "float32", "float"],
        "object": ["object", "string"],
    }

    report_lines.append("Dtype check (contract vs actual):")
    for col in present_cols:
        if col in contract_cols:
            expected_dtype = contract_cols[col].get("dtype", "unknown")
            actual_dtype = str(df[col].dtype)
            # Normalize for comparison
            expected_base = expected_dtype.split("[")[0].rstrip("0123456789")
            actual_base = actual_dtype.split("[")[0].rstrip("0123456789")
            compatible = (
                expected_base == actual_base
                or (expected_base in ("int", "float") and actual_base in ("int", "float"))
                or expected_dtype == actual_dtype
            )
            status = "PASS" if compatible else f"WARN (expected {expected_dtype}, got {actual_dtype})"
            report_lines.append(f"  {col}: {status}")
            if not compatible:
                dtype_issues.append(col)
        else:
            report_lines.append(f"  {col}: PASS (not in contract, present in data)")

    report_lines.append("")

    # Row count check
    contract_rows = contract.get("row_count", 0)
    try:
        actual_rows = sum(1 for _ in open(CLEAN_DATA_PATH, encoding="utf-8")) - 1  # minus header
    except Exception:
        actual_rows = "unknown"

    report_lines.append(f"Row count — contract: {contract_rows:,}, actual: {actual_rows:,}" if isinstance(actual_rows, int) else f"Row count — contract: {contract_rows:,}, actual: {actual_rows}")

    # Overall result
    overall = "PASS" if not missing_cols and not dtype_issues else "FAIL"
    report_lines += [
        "",
        f"Missing required columns: {missing_cols if missing_cols else 'none'}",
        f"Dtype mismatches: {dtype_issues if dtype_issues else 'none'}",
        "",
        f"Overall result: {overall}",
    ]

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Tool 2 — Engineer features
# ---------------------------------------------------------------------------
@tool("engineer_features", result_as_answer=True)
def engineer_features(input: str = "") -> str:
    """
    From clean_data.csv, compute RFM features per customer:
      - Recency: days since last purchase (reference = max InvoiceDate)
      - Frequency: number of unique invoices per customer
      - Monetary: total spend (Quantity * Price) per customer
      - AvgOrderValue: Monetary / Frequency
      - UniqueProducts: distinct StockCodes per customer
    Scale all features with StandardScaler (random_state=42) and save
    outputs/features.csv (one row per Customer ID).
    Returns a summary string.
    """
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate"], low_memory=False)
    df["Revenue"] = df["Quantity"] * df["Price"]

    reference_date = df["InvoiceDate"].max()

    # Recency
    last_purchase = df.groupby("Customer ID")["InvoiceDate"].max()
    recency = (reference_date - last_purchase).dt.days.rename("Recency")

    # Frequency
    frequency = df.groupby("Customer ID")["Invoice"].nunique().rename("Frequency")

    # Monetary
    monetary = df.groupby("Customer ID")["Revenue"].sum().rename("Monetary")

    # AvgOrderValue
    order_value = df.groupby(["Customer ID", "Invoice"])["Revenue"].sum()
    avg_order_value = order_value.groupby("Customer ID").mean().rename("AvgOrderValue")

    # UniqueProducts
    unique_products = df.groupby("Customer ID")["StockCode"].nunique().rename("UniqueProducts")

    # Combine
    features = pd.concat([recency, frequency, monetary, avg_order_value, unique_products], axis=1)
    features = features.dropna()
    features.index.name = "Customer ID"
    features = features.reset_index()

    n_customers = len(features)

    # Scale
    feature_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
    scaler = StandardScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols])

    features.to_csv(FEATURES_PATH, index=False)

    return (
        "FEATURE ENGINEERING COMPLETE\n"
        "=============================\n"
        f"Reference date (max InvoiceDate): {reference_date.date()}\n"
        f"Customers processed: {n_customers:,}\n"
        f"Features: {feature_cols}\n"
        f"Scaler: StandardScaler (fitted on all customers)\n"
        f"Saved to: {FEATURES_PATH}\n\n"
        "Feature stats (scaled):\n"
        f"{features[feature_cols].describe().round(3).to_string()}\n"
    )


# ---------------------------------------------------------------------------
# Tool 3a — Train models and evaluate
# ---------------------------------------------------------------------------
@tool("train_models", result_as_answer=True)
def train_models(input: str = "") -> str:
    """
    Load outputs/features.csv, train KMeans (n_clusters=4) and
    AgglomerativeClustering (n_clusters=4), compute silhouette scores and
    KMeans inertia, pick the best model by silhouette score, save it as
    outputs/model.pkl (joblib), save outputs/evaluation_report.md, and
    return a metrics comparison string.
    """
    features_df = pd.read_csv(FEATURES_PATH)
    feature_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
    X = features_df[feature_cols].values
    n_customers = len(X)

    # --- KMeans ---
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_silhouette = float(silhouette_score(X, km_labels))
    km_inertia = float(km.inertia_)

    # --- AgglomerativeClustering ---
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(X)
    agg_silhouette = float(silhouette_score(X, agg_labels))

    # Cluster size distributions
    km_sizes = dict(zip(*np.unique(km_labels, return_counts=True)))
    agg_sizes = dict(zip(*np.unique(agg_labels, return_counts=True)))
    km_size_str = ", ".join(f"Cluster {k}: {v}" for k, v in sorted(km_sizes.items()))
    agg_size_str = ", ".join(f"Cluster {k}: {v}" for k, v in sorted(agg_sizes.items()))

    # Select best model
    if km_silhouette >= agg_silhouette:
        best_name = "KMeans"
        best_model = km
        best_labels = km_labels
        best_silhouette = km_silhouette
    else:
        best_name = "AgglomerativeClustering"
        best_model = agg
        best_labels = agg_labels
        best_silhouette = agg_silhouette

    # Save best model
    joblib.dump(best_model, MODEL_PATH)

    # Compute per-cluster RFM profile (unscaled interpretation via sign)
    cluster_profiles = []
    for c in range(4):
        mask = best_labels == c
        profile = features_df[feature_cols][mask].mean()
        cluster_profiles.append(
            f"  Cluster {c} (n={mask.sum():,}): "
            f"Recency={profile['Recency']:.3f}, "
            f"Frequency={profile['Frequency']:.3f}, "
            f"Monetary={profile['Monetary']:.3f}, "
            f"AvgOrderValue={profile['AvgOrderValue']:.3f}, "
            f"UniqueProducts={profile['UniqueProducts']:.3f}"
        )

    metrics_str = (
        "MODEL TRAINING & EVALUATION\n"
        "============================\n"
        f"Dataset: {FEATURES_PATH}\n"
        f"Customers: {n_customers:,}\n"
        f"Features: {feature_cols}\n"
        f"n_clusters: 4\n\n"
        "Results:\n"
        f"  KMeans          — silhouette={km_silhouette:.4f}, inertia={km_inertia:,.2f}\n"
        f"  KMeans cluster sizes: {km_size_str}\n"
        f"  Agglomerative   — silhouette={agg_silhouette:.4f}, inertia=N/A\n"
        f"  Agglomerative cluster sizes: {agg_size_str}\n\n"
        f"Best model: {best_name} (silhouette={best_silhouette:.4f})\n"
        f"Saved to: {MODEL_PATH}\n\n"
        f"Best model cluster profiles (scaled features):\n"
        + "\n".join(cluster_profiles)
    )

    # Save evaluation report
    eval_md = f"""# Evaluation Report — Customer Segmentation

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset
- Source: `outputs/features.csv`
- Customers: {n_customers:,}
- Features: {feature_cols}
- n_clusters: 4

## Model Comparison

| Model | Silhouette Score | Inertia |
|---|---|---|
| KMeans (n_clusters=4, random_state=42) | {km_silhouette:.4f} | {km_inertia:,.2f} |
| AgglomerativeClustering (n_clusters=4) | {agg_silhouette:.4f} | N/A |

## Best Model: {best_name}
- Silhouette Score: {best_silhouette:.4f}
- Model saved to: `outputs/model.pkl`

## Cluster Sizes

### KMeans
{chr(10).join(f'- {k}: {v} customers' for k, v in sorted(km_sizes.items()))}

### AgglomerativeClustering
{chr(10).join(f'- {k}: {v} customers' for k, v in sorted(agg_sizes.items()))}

## Best Model Cluster Profiles (scaled features)
{chr(10).join(cluster_profiles)}

## Notes
- Features were standardized with StandardScaler before training.
- Silhouette score ranges from -1 (poor) to 1 (perfect); higher is better.
- Inertia (KMeans only): sum of squared distances of samples to their nearest cluster centre.
"""
    EVAL_REPORT_PATH.write_text(eval_md, encoding="utf-8")

    return metrics_str


# ---------------------------------------------------------------------------
# Tool 3b — Save evaluation report (standalone)
# ---------------------------------------------------------------------------
@tool("save_evaluation_report", result_as_answer=True)
def save_evaluation_report(report_text: str = "") -> str:
    """
    Save the provided report_text to outputs/evaluation_report.md.
    If no text is provided, reads the existing report from disk and confirms it.
    """
    if report_text.strip():
        EVAL_REPORT_PATH.write_text(report_text, encoding="utf-8")
        return f"Evaluation report saved to {EVAL_REPORT_PATH} ({len(report_text):,} chars)."
    elif EVAL_REPORT_PATH.exists():
        size = EVAL_REPORT_PATH.stat().st_size
        return f"Evaluation report already exists at {EVAL_REPORT_PATH} ({size:,} bytes)."
    else:
        return "No report text provided and no existing report found."


# ---------------------------------------------------------------------------
# Tool 4 — Save model card (generates content from evaluation report)
# ---------------------------------------------------------------------------
@tool("save_model_card", result_as_answer=True)
def save_model_card(card_text: str = "") -> str:
    """
    Generate and save outputs/model_card.md. If card_text is provided, save it
    directly. Otherwise, read outputs/evaluation_report.md and automatically
    generate a comprehensive model card.
    """
    if not card_text.strip():
        # Auto-generate model card from evaluation report
        eval_content = ""
        if EVAL_REPORT_PATH.exists():
            eval_content = EVAL_REPORT_PATH.read_text(encoding="utf-8")

        # Parse key values from eval report
        best_model = "KMeans"
        best_silhouette = "N/A"
        n_customers = "N/A"
        for line in eval_content.splitlines():
            if "Best Model:" in line:
                best_model = line.split("Best Model:")[-1].strip().split()[0]
            if "Silhouette Score:" in line and "Best" not in line:
                try:
                    best_silhouette = line.split("Silhouette Score:")[-1].strip()
                except Exception:
                    pass
            if "Customers:" in line:
                n_customers = line.split("Customers:")[-1].strip()

        card_text = f"""# Model Card — Customer Segmentation (RFM-based Clustering)

Generated: {datetime.now().strftime('%Y-%m-%d')}

---

## Model Overview

| Field | Value |
|---|---|
| **Task** | Unsupervised customer segmentation |
| **Algorithm** | {best_model} (n_clusters=4) |
| **Selection criterion** | Highest silhouette score |
| **Input features** | Recency, Frequency, Monetary, AvgOrderValue, UniqueProducts |
| **Preprocessing** | StandardScaler |
| **Output** | Cluster label (0–3) per customer |
| **Saved artifact** | `outputs/model.pkl` (joblib) |

---

## Intended Use

This model segments retail customers into 4 behavioural groups based on their
purchasing history (RFM analysis). It is designed to:

- Enable targeted marketing campaigns for each customer segment.
- Identify high-value, at-risk, and dormant customer groups.
- Support personalised product recommendations and loyalty programme design.

**In-scope:** UK-based and international retail customers with at least one transaction.
**Out-of-scope:** New customers with no transaction history; real-time scoring without
feature recomputation.

---

## Training Data

- **Source:** `outputs/clean_data.csv` (derived from Online Retail II dataset)
- **Customers:** {n_customers}
- **Features engineered:** 5 RFM-derived features per customer
- **Feature scaling:** StandardScaler (mean=0, std=1)

### Feature Descriptions

| Feature | Description |
|---|---|
| Recency | Days since last purchase (lower = more recent) |
| Frequency | Number of unique invoices (higher = more frequent buyer) |
| Monetary | Total spend in GBP (higher = bigger spender) |
| AvgOrderValue | Mean revenue per invoice (higher = larger baskets) |
| UniqueProducts | Count of distinct StockCodes purchased |

---

## Model Performance

{eval_content if eval_content else "See outputs/evaluation_report.md for full metrics."}

---

## Ethical Considerations

- Customers are segmented on **behavioural signals only** — no demographic, geographic,
  or protected-class attributes are used.
- Segment labels are anonymous (0–3) and carry no value judgement.
- This model should not be used to deny service or make credit decisions.

---

## Limitations

- RFM features reflect historical behaviour; the model may not generalise to new
  market conditions or seasonal shifts outside the training window.
- AgglomerativeClustering does not natively support predicting labels for new data
  points; the saved KMeans model supports `.predict()` on new observations.
- Cluster count (k=4) was fixed by design; optimal k was not searched automatically.

---

## How to Use

```python
import joblib, pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load("outputs/model.pkl")
features = pd.read_csv("outputs/features.csv")
feature_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
labels = model.predict(features[feature_cols].values)
features["Segment"] = labels
```

---

*Model card generated automatically by the Data Scientist Crew.*
"""

    MODEL_CARD_PATH.write_text(card_text, encoding="utf-8")
    lines = card_text.strip().split("\n")
    return (
        f"Model card saved to {MODEL_CARD_PATH} "
        f"({len(lines)} lines, {len(card_text):,} chars)."
    )
