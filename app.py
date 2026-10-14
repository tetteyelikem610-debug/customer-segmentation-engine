import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Segmentation Engine", page_icon="📊", layout="wide")

# ---------------- SEGMENT SETTINGS (match your k=4 model) ----------------
SEGMENT_NAMES = {
    0: "Regular Customers",
    1: "New / Potential Customers",
    2: "Lost / Dormant Customers",
    3: "Champions",
}
SEGMENT_ACTIONS = {
    0: "Offer bundles and loyalty points to increase order frequency.",
    1: "Send a follow-up offer to encourage a second and third purchase.",
    2: "Run a win-back campaign with a discount or reminder.",
    3: "Reward with VIP perks and early access to new products.",
}
# -------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #F3F6FA; border-radius: 10px; padding: 12px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #1F2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- LOADERS ----------------
@st.cache_resource
def load_model():
    return joblib.load("scaler.pkl"), joblib.load("kmeans.pkl")


@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


@st.cache_data
def load_customers():
    if os.path.exists("customer_segments.csv"):
        data = pd.read_csv("customer_segments.csv")
        data["Segment_Name"] = data["Segment"].map(SEGMENT_NAMES)
        return data
    return None


scaler, model = load_model()
metrics = load_metrics()
customers = load_customers()
K = model.n_clusters


def to_scaled(df_rfm):
    X = np.log1p(df_rfm[["Recency", "Frequency", "Monetary"]])
    return scaler.transform(X)


def predict(df_rfm):
    return model.predict(to_scaled(df_rfm))


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📊 Model Details")
    st.write(
        "This engine groups customers into segments based on their purchase "
        "behaviour so businesses can target each group differently."
    )

    st.subheader("About the model")
    st.markdown(
        f"""
- **Algorithm:** K-Means clustering
- **Segments (k):** {K}
- **Features:** Recency, Frequency, Monetary (RFM)
- **Preprocessing:** log transform + standard scaling
- **Dataset:** Online Retail (UCI)
"""
    )
    if metrics:
        st.markdown(f"- **Customers analysed:** {metrics.get('n_customers', 'N/A'):,}")

    # st.subheader("Evaluation")
    # if metrics:
    #     st.metric("Silhouette (test set)", f"{metrics['test_silhouette']:.3f}")
    #     st.metric("Silhouette (all data)", f"{metrics['silhouette']:.3f}")
    #     st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}")
    #     st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:,.0f}")
    # else:
    #     st.info("Add metrics.json to show evaluation scores.")

    st.caption(
        "K-Means is unsupervised, so there is no 'accuracy'. Quality is measured "
        "with clustering metrics. Silhouette ranges from -1 to 1 (higher is better); "
        "Davies-Bouldin is better when lower."
    )

# ---------------- MAIN ----------------
st.title("Customer Segmentation Engine")
st.write("Predict a customer's segment from their RFM values, or analyse many customers at once.")

tab1, tab2, tab3 = st.tabs(["🎯 Single Customer", "📁 Bulk Upload", "📈 Segment Explorer"])

# ---------- TAB 1 ----------
with tab1:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Customer details")
        cust_name = st.text_input("Customer name / ID (optional)")
        mode = st.radio("Input method", ["Sliders", "Exact values"], horizontal=True)

        if mode == "Sliders":
            r = st.slider("Recency (days since last purchase)", 0, 400, 30)
            f = st.slider("Frequency (number of orders)", 1, 50, 5)
            m = st.slider("Monetary (total spend)", 0, 20000, 500, step=50)
        else:
            r = st.number_input("Recency (days since last purchase)", 0, 1000, 30)
            f = st.number_input("Frequency (number of orders)", 1, 500, 5)
            m = st.number_input("Monetary (total spend)", 0.0, 1_000_000.0, 500.0, step=50.0)

    row = pd.DataFrame([[r, f, m]], columns=["Recency", "Frequency", "Monetary"])
    seg = int(predict(row)[0])

    with right:
        st.subheader("Prediction")
        who = f" for **{cust_name}**" if cust_name else ""
        st.success(f"Segment {seg}: **{SEGMENT_NAMES.get(seg, 'Unnamed')}**{who}")
        st.info(f"💡 Suggested action: {SEGMENT_ACTIONS.get(seg, '')}")

        if customers is not None:
            avg = customers[customers["Segment"] == seg][["Recency", "Frequency", "Monetary"]].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Recency", f"{r} days", f"{r - avg['Recency']:+.0f} vs segment avg", delta_color="off")
            c2.metric("Frequency", f"{f}", f"{f - avg['Frequency']:+.1f} vs segment avg", delta_color="off")
            c3.metric("Monetary", f"{m:,.0f}", f"{m - avg['Monetary']:+,.0f} vs segment avg", delta_color="off")

        st.markdown("**Distance to each segment centre** (smaller = closer match)")
        dist = model.transform(to_scaled(row))[0]
        dist_df = pd.DataFrame(
            {"Distance": dist},
            index=[SEGMENT_NAMES.get(i, f"Segment {i}") for i in range(K)],
        )
        st.bar_chart(dist_df)

# ---------- TAB 2 ----------
with tab2:
    st.write("Upload a CSV with columns: **CustomerID, Recency, Frequency, Monetary**")
    file = st.file_uploader("Choose CSV", type="csv")

    if file is not None:
        data = pd.read_csv(file)
        if not {"Recency", "Frequency", "Monetary"}.issubset(data.columns):
            st.error("CSV must contain Recency, Frequency and Monetary columns.")
        else:
            data["Segment"] = predict(data)
            data["Segment_Name"] = data["Segment"].map(SEGMENT_NAMES)

            m1, m2, m3 = st.columns(3)
            m1.metric("Customers", f"{len(data):,}")
            m2.metric("Segments found", data["Segment"].nunique())
            m3.metric("Largest segment", data["Segment_Name"].value_counts().idxmax())

            pick = st.multiselect(
                "Filter segments", sorted(data["Segment_Name"].unique()),
                default=sorted(data["Segment_Name"].unique()),
            )
            view = data[data["Segment_Name"].isin(pick)]

            st.subheader("Segment sizes")
            st.bar_chart(view["Segment_Name"].value_counts())

            st.subheader("Segment averages")
            st.dataframe(
                view.groupby("Segment_Name")[["Recency", "Frequency", "Monetary"]].mean().round(1),
                use_container_width=True,
            )

            st.subheader("Results")
            st.dataframe(view, use_container_width=True)
            st.download_button(
                "Download results", view.to_csv(index=False).encode("utf-8"),
                "segmented_customers.csv", "text/csv",
            )

# ---------- TAB 3 ----------
with tab3:
    if customers is None:
        st.info("Place customer_segments.csv in the project folder to see the explorer.")
    else:
        st.subheader("Segment profiles")
        prof = customers.groupby("Segment_Name").agg(
            Customers=("Segment", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        ).round(1)
        prof["Revenue_%"] = (prof["Total_Revenue"] / prof["Total_Revenue"].sum() * 100).round(1)
        st.dataframe(prof, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Customers per segment**")
            st.bar_chart(prof["Customers"])
        with c2:
            st.markdown("**Revenue share (%) per segment**")
            st.bar_chart(prof["Revenue_%"])

        st.subheader("Customer map")
        x_axis = st.selectbox("X axis", ["Recency", "Frequency", "Monetary"], index=0)
        y_axis = st.selectbox("Y axis", ["Recency", "Frequency", "Monetary"], index=2)
        sample = customers.sample(min(2000, len(customers)), random_state=1)
        st.scatter_chart(sample, x=x_axis, y=y_axis, color="Segment_Name")
