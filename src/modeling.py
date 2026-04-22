import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


FEATURES = [
    "total_orders",
    "total_revenue",
    "avg_order_value",
    "days_since_last_order",
    "days_as_customer",
    "email_opens_30d",
    "email_clicks_30d",
    "discount_rate",
]


def train_churn_model(customers: pd.DataFrame):
    df = customers.dropna(subset=FEATURES + ["is_churned"]).copy()

    X = df[FEATURES]
    y = df["is_churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    return clf


def score_customers(customers: pd.DataFrame, model) -> pd.DataFrame:
    df = customers.copy()
    df["churn_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    df["churn_score"] = (df["churn_probability"] * 100).round(1)
    return df


def feature_importance_chart(model, feature_names: list) -> go.Figure:
    importances = model.feature_importances_
    idx = np.argsort(importances)
    labels = [feature_names[i].replace("_", " ").title() for i in idx]

    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=labels,
        orientation="h",
        marker_color="#3B82F6",
    ))
    fig.update_layout(
        title="Churn Model — Feature Importance",
        xaxis_title="Importance",
        height=360,
        margin=dict(t=50, b=40, l=160, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig


def score_distribution_chart(scored: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        scored, x="churn_score", nbins=20,
        title="Churn Score Distribution",
        color_discrete_sequence=["#3B82F6"],
        labels={"churn_score": "Churn Risk Score (0–100)"},
    )
    fig.add_vline(x=70, line_dash="dash", line_color="#DC2626",
                  annotation_text="High Risk Threshold", annotation_position="top right")
    fig.update_layout(
        height=340,
        margin=dict(t=50, b=40, l=50, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="Customers",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig


def intervention_list(scored: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top N customers ranked by LTV x churn risk — highest intervention priority."""
    df = scored.copy()
    df["priority_score"] = df["total_revenue"] * df["churn_probability"]
    top = df.nlargest(n, "priority_score")[[
        "customer_id", "total_revenue", "total_orders",
        "avg_order_value", "days_since_last_order", "churn_score"
    ]].reset_index(drop=True)
    top.index += 1
    top.columns = ["Customer ID", "Lifetime Revenue", "Orders", "Avg Order Value", "Days Since Purchase", "Churn Score"]
    top["Lifetime Revenue"] = top["Lifetime Revenue"].map("${:,.0f}".format)
    top["Avg Order Value"] = top["Avg Order Value"].map("${:,.0f}".format)
    return top
