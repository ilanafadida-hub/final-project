# Dataset Description

## Source
- **Platform**: Kaggle
- **Dataset**: `mashlyn/online-retail-ii-uci`
- **URL**: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

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
- `online_retail_II.csv` (90.5 MB)
