import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.eda import _apply_base_layout, KIN_BG, KIN_TEXT_DIM, KIN_MUTED, KIN_BORDER

SEGMENT_NAMES = {
    0: "Champions",
    1: "Loyalists",
    2: "Promising",
    3: "Price Hunters",
    4: "At Risk",
}

# Kinetric segment palette — teal reserved for Champions, amber signals risk
SEGMENT_COLORS = {
    "Champions":     "#2DD4BF",  # teal — primary
    "Loyalists":     "#22C55E",  # green — healthy
    "Promising":     "#A78BFA",  # violet — emerging
    "Price Hunters": "#60A5FA",  # blue — distinct but not-warning
    "At Risk":       "#F59E0B",  # amber — warning
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
        title="Champions Cluster Tight and Recent. At Risk Drifts Right",
        labels={
            "days_since_last_order": "Days Since Last Order (Recency)",
            "total_orders": "Total Orders (Frequency)",
            "total_revenue": "Total Revenue",
        },
        size_max=26,
        opacity=0.78,
    )
    fig.update_traces(
        marker=dict(line=dict(color=KIN_BG, width=1)),
    )
    fig.update_layout(
        legend=dict(title=dict(text=""), orientation="h", y=1.05, x=0.01),
    )
    fig = _apply_base_layout(fig, height=440)
    return fig
