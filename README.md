# Customer Segmentation Engine

A machine learning web app that groups customers into segments using **RFM analysis** (Recency, Frequency, Monetary) and **K-Means clustering**, built with Python, scikit-learn and Streamlit.

## Overview

Businesses cannot treat every customer the same. This project analyses purchase history and automatically divides customers into behavioural segments, so each group can be targeted with the right marketing action (rewards, win-back offers, onboarding and so on).

## Features

- **Single customer prediction:** enter Recency, Frequency and Monetary values with sliders or exact inputs and get the predicted segment instantly, with a suggested business action.
- **Bulk upload:** upload a CSV of many customers, filter by segment, view sizes and averages, and download the results.
- **Segment explorer:** segment profiles, customers and revenue share per segment, and an interactive customer map.
- **Model details sidebar:** algorithm information and evaluation scores.

## Dataset

[Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI Machine Learning Repository): transactions from a UK-based online retailer.

## Methodology

1. **Data cleaning**
   - Removed rows with missing `CustomerID`
   - Removed cancelled invoices (starting with `C`) and zero or negative quantities and prices
   - Removed duplicate rows
   - Created `TotalSpend = Quantity x UnitPrice` and capped outliers using the IQR method
2. **RFM feature engineering** (one row per customer)
   - **Recency:** days since the last purchase
   - **Frequency:** number of unique invoices
   - **Monetary:** total spend
3. **Preprocessing:** log transform (`log1p`) to reduce skew, then standard scaling
4. **Modelling:** K-Means clustering, with k chosen using the elbow method and silhouette score
5. **Evaluation:** silhouette score, Davies-Bouldin index and Calinski-Harabasz index, plus a test-set silhouette from an 80/20 split

## Segments

| Segment | Name | Avg Recency | Avg Frequency | Avg Monetary | Suggested action |
|---|---|---|---|---|---|
| 3 | Champions | 11.2 | 13.7 | 4,527 | VIP perks and early access |
| 0 | Regular Customers | 62.9 | 4.2 | 1,384 | Bundles and loyalty points |
| 1 | New / Potential | 22.4 | 1.9 | 403 | Follow-up offers for repeat purchases |
| 2 | Lost / Dormant | 191.4 | 1.4 | 297 | Win-back campaign |

## Evaluation

K-Means is unsupervised, so there is no accuracy score. Quality is measured with clustering metrics:

| Metric | Meaning | Better |
|---|---|---|
| Silhouette | How well points fit their own cluster versus others (-1 to 1) | Higher |
| Davies-Bouldin | Average similarity between clusters | Lower |
| Calinski-Harabasz | Between-cluster versus within-cluster spread | Higher |

Exact scores are shown in the app sidebar (loaded from `metrics.json`).

## Project Structure

```
customer_segmentation/
├── app.py                   # Streamlit application
├── kmeans.pkl               # Trained K-Means model
├── scaler.pkl               # Fitted StandardScaler
├── customer_segments.csv    # Segmented customers (used by the explorer)
├── metrics.json             # Evaluation scores (shown in sidebar)
├── requirements.txt         # Python dependencies
├── README.md
└── .streamlit/
    └── config.toml          # Light theme settings
```

## Installation

```bash
# 1. Clone or download the project, then enter the folder
cd customer_segmentation

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Bulk upload format:** a CSV with the columns `CustomerID, Recency, Frequency, Monetary`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, joblib, Streamlit, Matplotlib and Seaborn (analysis in Google Colab).

## Limitations

- Segments are specific to this dataset and this trained model. Cluster numbers change if the model is retrained, so `SEGMENT_NAMES` in `app.py` must be updated to match.
- RFM captures purchase behaviour only, not product preferences or demographics.
- The dataset covers a single retailer over a limited period.

## Future Improvements

- Accept raw transaction files and compute RFM inside the app
- Add product-level or demographic features
- Compare with other clustering methods (Gaussian Mixture, Hierarchical)
- Deploy on Streamlit Community Cloud

## Author

Elikem, Computer Engineering, University of Mines and Technology (UMaT), Tarkwa, Ghana
