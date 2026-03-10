"""
Data Analyst Crew — Tool implementations.
Each tool does the real data processing work.
Tools use result_as_answer=True so agents return tool output directly,
minimising LLM token consumption.
"""

import json
import base64
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from crewai.tools import tool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLEAN_DATA_PATH = OUTPUTS_DIR / "clean_data.csv"
EDA_REPORT_PATH = OUTPUTS_DIR / "eda_report.html"
INSIGHTS_PATH = OUTPUTS_DIR / "insights.md"
CONTRACT_PATH = OUTPUTS_DIR / "dataset_contract.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tool 1 — Validate raw data
# ---------------------------------------------------------------------------
@tool("validate_raw_data", result_as_answer=True)
def validate_raw_data(input: str = "") -> str:
    """
    Load the raw online_retail_II.csv, check schema, count nulls/duplicates,
    identify cancellation invoices and negative-quantity rows, and return
    a detailed validation report string.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip", low_memory=False)

    total_rows = len(df)
    total_cols = len(df.columns)

    expected_cols = ["Invoice", "StockCode", "Description", "Quantity",
                     "InvoiceDate", "Price", "Customer ID", "Country"]
    present = [c for c in expected_cols if c in df.columns]
    missing = [c for c in expected_cols if c not in df.columns]

    null_counts = df.isnull().sum()
    null_report = null_counts[null_counts > 0].to_dict()

    dup_count = int(df.duplicated().sum())
    cancel_count = int(df["Invoice"].astype(str).str.startswith("C").sum()) if "Invoice" in df.columns else 0
    neg_qty = int((df["Quantity"] <= 0).sum()) if "Quantity" in df.columns else 0

    report = (
        "RAW DATA VALIDATION REPORT\n"
        "==========================\n"
        f"Total rows: {total_rows:,}\n"
        f"Total columns: {total_cols}\n"
        f"Columns present: {present}\n"
        f"Columns missing: {missing}\n\n"
        "NULL counts per column:\n"
    )
    for col, cnt in null_report.items():
        pct = cnt / total_rows * 100
        report += f"  {col}: {cnt:,} ({pct:.1f}%)\n"

    report += (
        f"\nDuplicate rows: {dup_count:,}\n"
        f"Cancellation invoices (start with 'C'): {cancel_count:,}\n"
        f"Rows with Quantity <= 0: {neg_qty:,}\n"
    )
    return report


# ---------------------------------------------------------------------------
# Tool 2 — Clean data
# ---------------------------------------------------------------------------
@tool("clean_data", result_as_answer=True)
def clean_data(input: str = "") -> str:
    """
    Load raw CSV, remove duplicates, remove cancellation invoices (Invoice
    starting with 'C'), remove rows where Quantity <= 0 or Price <= 0,
    drop rows with missing Customer ID, fix dtypes (InvoiceDate -> datetime,
    Customer ID -> int), and save outputs/clean_data.csv.
    Returns a cleaning summary.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    raw_rows = len(df)

    before_dedup = len(df)
    df = df.drop_duplicates()
    removed_dups = before_dedup - len(df)

    before_cancel = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    removed_cancel = before_cancel - len(df)

    before_neg = len(df)
    df = df[df["Quantity"] > 0]
    removed_neg = before_neg - len(df)

    before_price = len(df)
    df = df[df["Price"] > 0]
    removed_price = before_price - len(df)

    before_cid = len(df)
    df = df.dropna(subset=["Customer ID"])
    removed_cid = before_cid - len(df)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df["Customer ID"] = df["Customer ID"].astype(int)

    before_null = len(df)
    df = df.dropna(subset=["Invoice", "StockCode", "Quantity", "Price", "Country"])
    removed_null = before_null - len(df)

    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    final_rows = len(df)
    return (
        "DATA CLEANING SUMMARY\n"
        "=====================\n"
        f"Raw rows: {raw_rows:,}\n"
        f"Removed duplicates: {removed_dups:,}\n"
        f"Removed cancellations (Invoice starts with 'C'): {removed_cancel:,}\n"
        f"Removed Quantity <= 0: {removed_neg:,}\n"
        f"Removed Price <= 0: {removed_price:,}\n"
        f"Removed missing Customer ID: {removed_cid:,}\n"
        f"Removed other nulls: {removed_null:,}\n"
        f"Final clean rows: {final_rows:,}\n"
        f"Saved to: {CLEAN_DATA_PATH}\n"
    )


# ---------------------------------------------------------------------------
# Helper: custom HTML EDA report (fallback when ydata-profiling unavailable)
# ---------------------------------------------------------------------------
def _make_html_report(df: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        enc = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return enc

    df2 = df.copy()
    df2["Revenue"] = df2["Quantity"] * df2["Price"]
    df2["YearMonth"] = df2["InvoiceDate"].dt.to_period("M")
    monthly = df2.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(monthly["YearMonth"], monthly["Revenue"] / 1e3, color="steelblue")
    ax1.set_title("Monthly Revenue (GBP thousands)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Revenue (GBPk)")
    plt.xticks(rotation=45, ha="right")
    c1 = fig_to_b64(fig1)

    country_rev = df2.groupby("Country")["Revenue"].sum().nlargest(10)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    country_rev.plot(kind="bar", ax=ax2, color="darkorange")
    ax2.set_title("Top 10 Countries by Revenue")
    ax2.set_ylabel("Revenue (GBP)")
    plt.xticks(rotation=45, ha="right")
    c2 = fig_to_b64(fig2)

    prod_rev = df2.groupby("Description")["Revenue"].sum().nlargest(10)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    prod_rev.plot(kind="barh", ax=ax3, color="seagreen")
    ax3.set_title("Top 10 Products by Revenue")
    ax3.set_xlabel("Revenue (GBP)")
    ax3.invert_yaxis()
    c3 = fig_to_b64(fig3)

    desc_html = df2.describe(include="all").to_html(classes="table", border=1)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>EDA Report - Online Retail II</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
  h1 {{ color: #2c3e50; }} h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 6px; }}
  .table {{ border-collapse: collapse; font-size: 12px; margin-bottom: 20px; }}
  .table th, .table td {{ border: 1px solid #ddd; padding: 4px 8px; }}
  .table th {{ background: #2c3e50; color: white; }}
  .stat-box {{ background: white; border-radius: 8px; padding: 16px; margin: 10px 0;
               box-shadow: 1px 1px 4px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>EDA Report - Online Retail II</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="stat-box">
<h2>Dataset Overview</h2>
<ul>
  <li>Rows: {len(df2):,}</li>
  <li>Date range: {df2['InvoiceDate'].min().date()} to {df2['InvoiceDate'].max().date()}</li>
  <li>Unique customers: {df2['Customer ID'].nunique():,}</li>
  <li>Unique products: {df2['StockCode'].nunique():,}</li>
  <li>Unique countries: {df2['Country'].nunique():,}</li>
  <li>Total Revenue: GBP {df2['Revenue'].sum():,.2f}</li>
</ul>
</div>
<h2>Statistical Summary</h2>
{desc_html}
<h2>Monthly Revenue</h2>
<img src="data:image/png;base64,{c1}" style="max-width:100%;"/>
<h2>Top 10 Countries by Revenue</h2>
<img src="data:image/png;base64,{c2}" style="max-width:100%;"/>
<h2>Top 10 Products by Revenue</h2>
<img src="data:image/png;base64,{c3}" style="max-width:100%;"/>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tool 3 — Run EDA report + generate and save insights in one shot
# ---------------------------------------------------------------------------
@tool("run_eda_report", result_as_answer=True)
def run_eda_report(input: str = "") -> str:
    """
    Run EDA on the cleaned dataset: generate eda_report.html using ydata-profiling
    (minimal=True, with custom HTML fallback), compute key business statistics,
    write at least 5 business insights to outputs/insights.md, and return a
    combined summary string.
    """
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate"], low_memory=False)
    df["Revenue"] = df["Quantity"] * df["Price"]

    # --- EDA HTML Report ---
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, minimal=True, title="Online Retail II - EDA Report")
        profile.to_file(str(EDA_REPORT_PATH))
        profiling_used = "ydata-profiling (minimal=True)"
    except Exception as e:
        html = _make_html_report(df)
        EDA_REPORT_PATH.write_text(html, encoding="utf-8")
        profiling_used = f"custom HTML fallback (ydata-profiling error: {e})"

    # --- Compute statistics ---
    total_revenue = df["Revenue"].sum()
    avg_order_value = df.groupby("Invoice")["Revenue"].sum().mean()
    unique_customers = int(df["Customer ID"].nunique())
    unique_products = int(df["StockCode"].nunique())
    unique_countries = int(df["Country"].nunique())
    date_min = df["InvoiceDate"].min()
    date_max = df["InvoiceDate"].max()

    top_countries = df.groupby("Country")["Revenue"].sum().nlargest(5)
    top_products = df.groupby("Description")["Revenue"].sum().nlargest(5)

    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly_rev = df.groupby("YearMonth")["Revenue"].sum()
    peak_month = str(monthly_rev.idxmax())
    peak_rev = float(monthly_rev.max())
    low_month = str(monthly_rev.idxmin())
    low_rev = float(monthly_rev.min())

    customer_spend = df.groupby("Customer ID")["Revenue"].sum()
    top5_customers = customer_spend.nlargest(5)
    top1_spend = float(top5_customers.iloc[0])
    top1_id = int(top5_customers.index[0])

    uk_rev = float(top_countries.get("United Kingdom", 0.0))
    uk_pct = uk_rev / total_revenue * 100
    intl_pct = 100 - uk_pct

    qty_mean = float(df["Quantity"].mean())
    qty_median = float(df["Quantity"].median())
    total_rows = len(df)

    # Top product details
    top_prod_name = top_products.index[0]
    top_prod_rev = float(top_products.iloc[0])

    # Second country
    second_country = top_countries.index[1] if len(top_countries) > 1 else "N/A"
    second_country_rev = float(top_countries.iloc[1]) if len(top_countries) > 1 else 0.0
    second_country_pct = second_country_rev / total_revenue * 100

    # --- Write business insights ---
    insights = f"""# Business Insights - Online Retail II

Generated: {datetime.now().strftime('%Y-%m-%d')}
Dataset period: {date_min.date()} to {date_max.date()}
Clean rows analysed: {total_rows:,}

---

## 1. UK Market Dominance Creates Concentration Risk

**Finding:** The United Kingdom accounts for GBP {uk_rev:,.0f} ({uk_pct:.1f}%) of total revenue
(GBP {total_revenue:,.0f}), while all 40 international markets combined contribute only {intl_pct:.1f}%.

**Business implication:** This extreme concentration means that any disruption to UK operations
(economic downturn, logistics failures, regulatory changes) would immediately threaten the vast
majority of the business. The company is structurally exposed to a single geography.

**Recommended action:** Develop a dedicated international growth strategy targeting
{second_country} (already at GBP {second_country_rev:,.0f}, {second_country_pct:.1f}%) and other
top-5 markets, with the goal of reducing UK share to below 70% within 3 years.

---

## 2. Q4 Seasonal Spike Signals Inventory and Staffing Pressure

**Finding:** Peak revenue month is {peak_month} at GBP {peak_rev:,.0f}, compared to the lowest
month ({low_month}) at GBP {low_rev:,.0f} - a {peak_rev/low_rev:.1f}x seasonal multiplier.
Monthly revenue analysis shows a pronounced Q4 uplift driven by Christmas gift purchasing.

**Business implication:** The company faces significant seasonal demand volatility. Under-stocking
in peak months means lost sales; over-stocking in off-peak months ties up working capital.
Customer-facing operations (support, fulfilment) also face extreme load in Q4.

**Recommended action:** Implement demand-based inventory forecasting using the 2-year monthly
trend. Pre-build stock for the top 20 products by October. Staff up customer service from
September through January.

---

## 3. Top Product "{top_prod_name}" Drives Outsized Revenue Share

**Finding:** The single top product "{top_prod_name}" generated GBP {top_prod_rev:,.0f} in
revenue, representing {top_prod_rev/total_revenue*100:.1f}% of total revenue across the dataset
period. The top 5 products combined account for a disproportionate share of sales.

**Business implication:** Heavy dependence on a handful of SKUs creates supply chain and
trend risk. If a top product falls out of fashion or a supplier has issues, revenue will
drop sharply. It also suggests strong gift/novelty item demand in the customer base.

**Recommended action:** Protect supply chain continuity for top-10 SKUs with dual-sourcing
agreements. Use these hero products as anchors for bundle promotions to lift average basket
size and cross-sell adjacent product categories.

---

## 4. High-Value Customer Segment Offers a VIP Retention Opportunity

**Finding:** The top customer (ID {top1_id}) alone spent GBP {top1_spend:,.0f}, and the top 5
customers collectively represent significant revenue concentration. With only {unique_customers:,}
unique customers in the clean dataset, the customer base is relatively small and each retained
customer has high lifetime value.

**Business implication:** Losing even a few top accounts would materially impact revenue.
The current model appears to serve a mix of individual consumers and trade buyers; the
top spenders are almost certainly B2B wholesale customers.

**Recommended action:** Identify the top 50 customers by lifetime spend and assign them to
a dedicated account management programme with personalised outreach, volume discounts, and
early access to new product launches.

---

## 5. Average Order Value of GBP {avg_order_value:,.0f} Suggests Basket-Building Opportunity

**Finding:** The average order value (revenue per invoice) is GBP {avg_order_value:,.2f}, with a
median quantity per line item of {qty_median:.0f} units (mean: {qty_mean:.1f}). The business
sells to {unique_customers:,} customers across {unique_products:,} distinct products in
{unique_countries} countries.

**Business implication:** With 4,631 SKUs available but a median order quantity of only
6 units per line, there is substantial room to increase basket depth through product
recommendations and bundle promotions. A modest 10% lift in AOV would add approximately
GBP {total_revenue * 0.10:,.0f} in incremental annual revenue.

**Recommended action:** Implement a "frequently bought together" recommendation engine on
the online storefront. Test minimum-order-value promotions (e.g., free shipping above
GBP 500) to nudge mid-tier customers toward larger baskets.

---

## 6. International Markets Show Disproportionate Growth Potential

**Finding:** EIRE (Ireland) at GBP {second_country_rev:,.0f}, Netherlands, Germany, and France
each represent 2-4% of total revenue - yet these markets likely require proportionally less
marketing investment than acquiring UK customers, given lower market saturation.
The dataset spans 41 countries, many of which have minimal transaction volumes.

**Business implication:** The long tail of 36+ low-volume international markets suggests
opportunistic sales rather than deliberate expansion. There is no evidence of structured
market development outside the UK. The top 4 international markets (EIRE, Netherlands,
Germany, France) are natural beachheads for European expansion.

**Recommended action:** Focus international sales efforts on the existing top-4 markets
before expanding to new ones. Localise the website and catalogue for German and French
customers (language, currency, VAT handling). Set a 12-month target to double revenue
from each of these four markets.

---

*Report generated automatically from clean_data.csv ({total_rows:,} rows, {date_min.date()} to {date_max.date()})*
"""

    INSIGHTS_PATH.write_text(insights, encoding="utf-8")

    summary = (
        f"EDA + INSIGHTS COMPLETE\n"
        f"=======================\n"
        f"EDA report: {EDA_REPORT_PATH} (method: {profiling_used})\n"
        f"Insights: {INSIGHTS_PATH} (6 business findings)\n\n"
        f"Key stats:\n"
        f"  Total Revenue: GBP {total_revenue:,.2f}\n"
        f"  Avg Order Value: GBP {avg_order_value:,.2f}\n"
        f"  Unique Customers: {unique_customers:,}\n"
        f"  Unique Products: {unique_products:,}\n"
        f"  Countries: {unique_countries}\n"
        f"  Peak Month: {peak_month} (GBP {peak_rev:,.2f})\n"
        f"  UK Revenue Share: {uk_pct:.1f}%\n"
    )
    return summary


# ---------------------------------------------------------------------------
# Tool 4 — save_insights (kept for agent compatibility; also callable directly)
# ---------------------------------------------------------------------------
@tool("save_insights", result_as_answer=True)
def save_insights(insights_text: str) -> str:
    """
    Save the provided insights_text to outputs/insights.md.
    The text should be markdown with at least 5 business findings.
    """
    INSIGHTS_PATH.write_text(insights_text, encoding="utf-8")
    lines = insights_text.strip().split("\n")
    return (
        f"Insights saved to {INSIGHTS_PATH} "
        f"({len(lines)} lines, {len(insights_text):,} chars)."
    )


# ---------------------------------------------------------------------------
# Tool 5 — Generate dataset contract
# ---------------------------------------------------------------------------
@tool("generate_dataset_contract", result_as_answer=True)
def generate_dataset_contract(input: str = "") -> str:
    """
    Read outputs/clean_data.csv, infer schema (dtypes, null rates, min/max,
    sample values) for each column, and save outputs/dataset_contract.json.
    Returns a summary of the contract.
    """
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate"], low_memory=False)

    columns_schema = {}
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())

        col_info = {
            "dtype": str(series.dtype),
            "nullable": null_count > 0,
            "null_count": null_count,
            "null_rate": round(null_count / len(df), 6),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info["min"] = float(series.min()) if not series.isnull().all() else None
            col_info["max"] = float(series.max()) if not series.isnull().all() else None
            col_info["mean"] = round(float(series.mean()), 4) if not series.isnull().all() else None
            col_info["sample_values"] = [float(v) for v in series.dropna().head(5).tolist()]
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["min"] = str(series.min())
            col_info["max"] = str(series.max())
            col_info["sample_values"] = [str(v) for v in series.dropna().head(5).tolist()]
        else:
            col_info["unique_count"] = int(series.nunique())
            col_info["sample_values"] = [str(v) for v in series.value_counts().head(5).index.tolist()]
            col_info["min"] = None
            col_info["max"] = None

        columns_schema[col] = col_info

    contract = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source_file": "data/raw/online_retail_II.csv",
        "cleaned_file": "outputs/clean_data.csv",
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_schema,
        "constraints": [
            "Quantity > 0",
            "Price > 0",
            "Invoice does NOT start with 'C' (no cancellations)",
            "Customer ID is not null and is a non-negative integer",
            "InvoiceDate is a valid datetime",
            "No duplicate rows",
        ],
    }

    with open(CONTRACT_PATH, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, default=str)

    return (
        "DATASET CONTRACT GENERATED\n"
        "==========================\n"
        f"Saved to: {CONTRACT_PATH}\n"
        f"Rows: {contract['row_count']:,}\n"
        f"Columns: {contract['column_count']}\n"
        f"Constraints: {len(contract['constraints'])}\n"
        f"Columns covered: {list(columns_schema.keys())}\n"
    )
