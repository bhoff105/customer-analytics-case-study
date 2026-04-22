import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEGMENT_NAMES = {
    0: "Champions",
    1: "Loyalists",
    2: "Promising",
    3: "Price Hunters",
    4: "At Risk",
}

SEGMENT_COLORS = {
    "Champions":    "#1D4ED8",
    "Loyalists":    "#059669",
    "Promising":    "#7C3AED",
    "Price Hunters":"#DC2626",
    "At Risk":      "#F59E0B",
}


def build_rfm(customers: pd.DataFrame) -> pd.DataFrame:
    rfm = customers[["customer_id", "days_since_last_order", "total_orders", "total_revenue"]].copy()
    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

    # Score 1–5: recency lower is better, freq/monetary higher is better
    rfm["R"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["R"] + rfm["F"] + rfm["M"]
    return rfm


def run_segmentation(customers: pd.DataFrame) -> pd.DataFrame:
    rfm = build_rfm(customers)

    features = rfm[["R", "F", "M"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(scaled)

    # Map clusters to human names by RFM score centroid
    cluster_scores = rfm.groupby("cluster")["rfm_score"].mean().sort_values(ascending=False)
    rank_to_name = {cluster: SEGMENT_NAMES[i] for i, cluster in enumerate(cluster_scores.index)}
    rfm["segment"] = rfm["cluster"].map(rank_to_name)

    result = customers.merge(rfm[["customer_id", "R", "F", "M", "rfm_score", "segment"]], on="customer_id")
    return result


def segment_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    summary = segmented.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_revenue=("total_revenue", "mean"),
        avg_orders=("total_orders", "mean"),
        avg_recency=("days_since_last_order", "mean"),
        churn_rate=("is_churned", "mean"),
    ).reset_index()
    summary["avg_revenue"] = summary["avg_revenue"].round(2)
    summary["avg_orders"] = summary["avg_orders"].round(1)
    summary["avg_recency"] = summary["avg_recency"].round(0).astype(int)
    summary["churn_rate"] = (summary["churn_rate"] * 100).round(1)
    return summary.sort_values("avg_revenue", ascending=False)


def segment_scatter(segmented: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        segmented,
        x="days_since_last_order",
        y="total_orders",
        color="segment",
        size="total_revenue",
        color_discrete_map=SEGMENT_COLORS,
        title="Customer Segments — Recency vs. Order Frequency",
        labels={
            "days_since_last_order": "Days Since Last Order (Recency)",
            "total_orders": "Total Orders (Frequency)",
            "total_revenue": "Total Revenue",
        },
        size_max=30,
        opacity=0.75,
    )
    fig.update_layout(
        height=420,
        margin=dict(t=50, b=40, l=50, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="Segment"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig
