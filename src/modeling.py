import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.eda import (
    _apply_base_layout,
    KIN_ACCENT, KIN_ACCENT_SOFT, KIN_BG, KIN_RED, KIN_AMBER, KIN_BLUE, KIN_BLUE_SOFT,
)


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
    values = importances[idx]
    max_val = float(values.max()) if len(values) else 1.0

    # Highlight the top feature in teal; others in neutral grey-blue
    bar_fill = [
        "rgba(45, 212, 191, 0.35)" if v == max_val else "rgba(126, 149, 176, 0.22)"
        for v in values
    ]
    bar_line = [
        KIN_ACCENT if v == max_val else "#7E95B0"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=bar_fill, line=dict(color=bar_line, width=1)),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Email Engagement Leads, Recency and Frequency Close Behind",
        xaxis_title="Relative Importance",
        yaxis_title="",
        showlegend=False,
        bargap=0.3,
    )
    fig = _apply_base_layout(fig, height=400)
    # Left margin for long feature labels
    fig.update_layout(margin=dict(t=56, b=56, l=170, r=32))
    fig.update_xaxes(tickformat=".0%", range=[0, max(max_val * 1.1, 0.01)])
    fig.update_yaxes(tickfont=dict(family="DM Sans, sans-serif", size=12, color="#E8EDF5"))
    return fig


def score_distribution_chart(scored: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        scored, x="churn_score", nbins=20,
        title="Most Customers Sit Low on Risk. The Right Tail Is Where Money Is Made",
        color_discrete_sequence=[KIN_BLUE],
        labels={"churn_score": "Churn Risk Score"},
    )
    fig.update_traces(
        marker=dict(color=KIN_BLUE_SOFT, line=dict(color=KIN_BLUE, width=1)),
        hovertemplate="<b>Score %{x}</b><br>%{y} customers<extra></extra>",
    )
    fig.add_vline(
        x=70, line_dash="dot", line_color=KIN_RED, line_width=1.25,
        annotation_text="High risk",
        annotation_position="top right",
        annotation_font=dict(family="DM Sans, sans-serif", size=11, color=KIN_RED),
    )
    fig.update_layout(
        yaxis_title="Customers",
        xaxis_title="Churn Risk Score",
        showlegend=False,
        bargap=0.08,
    )
    fig = _apply_base_layout(fig, height=400)
    return fig


def _recommended_action(row: pd.Series) -> str:
    """
    Generate a specific one-line recommended action for each customer.
    Logic is driven by segment, recency, top_category, and discount_rate.
    Output is imperative, specific, and includes the "why" in the fewest possible words.
    """
    segment = row.get("segment", "")
    days = int(row.get("days_since_last_order", 0))
    orders = int(row.get("total_orders", 1))
    discount_rate = float(row.get("discount_rate", 0.0))
    top_category = str(row.get("top_category", "")).strip()
    avg_order_value = float(row.get("avg_order_value", 0.0))

    high_discount = discount_rate >= 0.5   # 50%+ of orders used a discount
    gifts_buyer = top_category.lower() == "gifts"
    high_value = avg_order_value >= 80

    if segment == "At Risk":
        if high_discount:
            return (
                f"Lapsed At Risk customer ({days}d quiet) with high discount dependency — "
                "send a personalized win-back referencing their last purchase; omit any discount offer "
                "to avoid reinforcing price-driven behaviour"
            )
        elif gifts_buyer:
            return (
                f"At Risk buyer ({days}d quiet) whose purchases concentrated in Gifts — "
                "send a 3-email cross-category sequence introducing Home & Kitchen or Beauty & Wellness; "
                "no discount, the goal is category bridge not price incentive"
            )
        else:
            return (
                f"High-LTV At Risk customer, {days} days since last purchase, {orders} prior orders — "
                "send a personalized win-back referencing purchase history; "
                "no blanket discount; target reactivation within 21 days"
            )

    elif segment == "Price Hunters":
        return (
            f"Price Hunter: {int(discount_rate * 100)}% of orders used a discount, now {days}d lapsed — "
            "do not offer further discounts; send educational content (how-to, product story) "
            "to shift the relationship pattern before re-engagement"
        )

    elif segment == "Loyalists":
        if days > 60:
            return (
                f"Loyalist with {orders} orders, unusually quiet for {days} days — "
                "send a personal check-in (not a blast); reference their purchase history; "
                "consider an early-access or new-arrival preview to re-engage without a discount"
            )
        else:
            return (
                f"Loyalist, {orders} orders, recently active — "
                "invite into referral program; do not include in discount promotions"
            )

    elif segment == "Promising":
        if gifts_buyer:
            return (
                f"Promising customer whose first purchase was in Gifts ({days}d ago) — "
                "send a cross-category intro sequence to Home & Kitchen within 14 days of that order; "
                "first repeat purchase is the critical conversion"
            )
        else:
            return (
                f"Promising buyer with {orders} order(s), {days}d since last purchase — "
                "send a second-purchase sequence featuring highest-retention categories; "
                "goal is first repeat purchase before the 90-day drop-off window"
            )

    elif segment == "Champions":
        return (
            f"Champion customer, {orders} orders, active — "
            "invite into referral program; exclude from discount campaigns; "
            "surface new arrivals or exclusive access as re-engagement lever"
        )

    else:
        return (
            f"Customer {days}d since last purchase, {orders} orders — "
            "review manually; segment signal unclear"
        )


def intervention_list(scored: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Top N customers ranked by LTV x churn risk — highest intervention priority.
    Each row includes a specific one-line Recommended Action derived from segment,
    recency, category, and discount behaviour.
    """
    df = scored.copy()
    df["priority_score"] = df["total_revenue"] * df["churn_probability"]
    df["recommended_action"] = df.apply(_recommended_action, axis=1)

    # Include customer_email if present (added in Phase 1 data refresh)
    select_cols = ["customer_id"]
    if "customer_email" in df.columns:
        select_cols.append("customer_email")
    select_cols += [
        "segment",
        "total_revenue", "total_orders", "avg_order_value",
        "days_since_last_order", "churn_score",
        "recommended_action",
    ]

    top = df.nlargest(n, "priority_score")[select_cols].reset_index(drop=True)
    top.index += 1

    rename_map = {
        "customer_id":          "Customer ID",
        "customer_email":       "Email",
        "segment":              "Segment",
        "total_revenue":        "Lifetime Revenue",
        "total_orders":         "Orders",
        "avg_order_value":      "Avg Order Value",
        "days_since_last_order":"Days Since Purchase",
        "churn_score":          "Churn Score",
        "recommended_action":   "Recommended Action",
    }
    top = top.rename(columns=rename_map)
    top["Lifetime Revenue"] = top["Lifetime Revenue"].map("${:,.0f}".format)
    top["Avg Order Value"] = top["Avg Order Value"].map("${:,.0f}".format)
    return top
