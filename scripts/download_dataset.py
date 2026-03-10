"""
Phase 1 - Dataset Selection & Ingestion
Searches Kaggle for a retail/sales dataset and downloads it to data/raw/
Uses the kaggle CLI binary from the venv.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env credentials BEFORE anything touches kaggle
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Ensure kaggle env vars are set (kaggle v2 reads these directly)
username = os.environ.get("KAGGLE_USERNAME")
key      = os.environ.get("KAGGLE_KEY")
if not username or not key:
    sys.exit("KAGGLE_USERNAME or KAGGLE_KEY missing in .env")

# Config
DATASET_REF = "mashlyn/online-retail-ii-uci"   # well-known UK retail/sales dataset
RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
DOCS_DIR    = Path(__file__).parent.parent / "docs"
KAGGLE_CLI  = Path(sys.executable).parent / "kaggle"

def run(cmd, **kwargs):
    """Run a shell command, raise on failure."""
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", **kwargs)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result.stdout.strip()

# Search for datasets
print("Searching Kaggle for retail/sales datasets...")
search_out = run(
    [str(KAGGLE_CLI), "datasets", "list",
     "--search", "online retail sales customer", "-v"],
    env={**os.environ},
)
print("\nTop results:")
lines = search_out.splitlines()
for line in lines[:11]:   # header + 10 rows
    print(" ", line)

# Download
RAW_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nDownloading: {DATASET_REF} -> {RAW_DIR}")
run(
    [str(KAGGLE_CLI), "datasets", "download",
     "--dataset", DATASET_REF,
     "--path", str(RAW_DIR),
     "--unzip"],
    env={**os.environ},
)

# Report downloaded files
files = [f for f in RAW_DIR.glob("*") if f.name != ".gitkeep"]
print(f"\nFiles in data/raw/:")
for f in files:
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"   * {f.name}  ({size_mb:.1f} MB)")

# Write docs/dataset_description.md
DOCS_DIR.mkdir(parents=True, exist_ok=True)
desc_path = DOCS_DIR / "dataset_description.md"

file_list = "\n".join(
    f"- `{f.name}` ({f.stat().st_size/1024/1024:.1f} MB)" for f in files
)

description = f"""# Dataset Description

## Source
- **Platform**: Kaggle
- **Dataset**: `{DATASET_REF}`
- **URL**: https://www.kaggle.com/datasets/{DATASET_REF}

## Overview
The **Online Retail II** dataset contains all transactions occurring between
01/12/2009 and 09/12/2011 for a UK-based, registered online retail company.
The company mainly sells unique all-occasion giftware, with many customers
being wholesalers.

## Columns
| Column       | Type    | Description                                       |
|--------------|---------|---------------------------------------------------|
| Invoice      | str     | Invoice number (prefix 'C' = cancellation)        |
| StockCode    | str     | Product / item code                               |
| Description  | str     | Product name                                      |
| Quantity     | int     | Units per transaction (negative = return/cancel)  |
| InvoiceDate  | datetime| Date and time of invoice                          |
| Price        | float   | Unit price in sterling (GBP)                      |
| Customer ID  | float   | Unique customer identifier (nullable)             |
| Country      | str     | Country of the customer                           |

## Why This Dataset?
- **Domain match**: Retail / sales / customer -- exactly what Phase 2 EDA targets
- **Rich for ML**: Supports customer segmentation (RFM), churn prediction, and sales forecasting
- **Real-world quality**: Contains nulls, duplicates, cancellations -- realistic for the cleaning crew
- **Size**: ~1 M rows across two years -- statistically significant without being unwieldy

## Downloaded Files
{file_list}
"""

desc_path.write_text(description, encoding="utf-8")
print(f"\nDataset description -> {desc_path}")
print("\nPhase 1 complete! Dataset ready for Crew 1 (Data Analyst).")
