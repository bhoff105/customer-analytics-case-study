import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def revenue_concentration_chart(customers: pd.DataFrame) -> go.Figure:
    """Pareto: cumulative revenue share by customer percentile."""
    sorted_rev = customers["total_revenue"].sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_rev.cumsum() / sorted_rev.sum() * 100
    pct_customers = (sorted_rev.index + 1) / len(sorted_rev) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pct_customers, y=cumulative,
        mode="lines", fill="tozeroy",
        line=dict(color="#3B82F6", width=2),
        fillcolor="rgba(59,130,246,0.1)",
        name="Cumulative Revenue"
    ))
    fig.add_shape(type="line", x0=20, x1=20, y0=0, y1=100,
                  line=dict(color="#F59E0B", dash="dash", width=1.5))
    fig.add_shape(type="line", x0=0, x1=20, y0=cumulative.iloc[int(len(cumulative)*0.2)-1],
                  y1=cumulative.iloc[int(len(cumulative)*0.2)-1],
                  line=dict(color="#F59E0B", dash="dash", width=1.5))
    fig.update_layout(
        title="Revenue Concentration — Top 20% of Customers",
        xaxis_title="Cumulative % of Customers",
        yaxis_title="Cumulative % of Revenue",
        height=380,
        margin=dict(t=50, b=40, l=50, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", range=[0, 100])
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", range=[0, 101])
    return fig


def acquisition_retention_chart(customers: pd.DataFrame) -> go.Figure:
    """Volume vs. retention rate by acquisition source."""
    grouped = customers.groupby("acquisition_source").agg(
        volume=("customer_id", "count"),
        retention_rate=("is_churned", lambda x: 1 - x.mean())
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["acquisition_source"],
        y=grouped["volume"],
        name="Customer Volume",
        marker_color="#BFDBFE",
        yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=grouped["acquisition_source"],
        y=grouped["retention_rate"] * 100,
        name="Retention Rate (%)",
        mode="lines+markers",
        marker=dict(size=9, color="#F59E0B"),
        line=dict(color="#F59E0B", width=2),
        yaxis="y2"
    ))
    fig.update_layout(
        title="Acquisition Volume vs. Retention Rate by Channel",
        yaxis=dict(title="Customers Acquired", showgrid=True, gridcolor="#F3F4F6"),
        yaxis2=dict(title="Retention Rate (%)", overlaying="y", side="right", range=[0, 100]),
        height=380,
        margin=dict(t=50, b=40, l=50, r=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
        barmode="group",
    )
    return fig


def monthly_revenue_chart(orders: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend over the full date range."""
    orders = orders.copy()
    orders["month"] = orders["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly = orders.groupby("month")["revenue"].sum().reset_index()

    fig = px.bar(
        monthly, x="month", y="revenue",
        title="Monthly Revenue — 18-Month Trend",
        color_discrete_sequence=["#3B82F6"],
    )
    fig.update_layout(
        xaxis_title="", yaxis_title="Revenue ($)",
        height=360, margin=dict(t=50, b=40, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig


def category_performance_chart(customers: pd.DataFrame, orders: pd.DataFrame) -> go.Figure:
    """Repeat rate and AOV by product category."""
    cat_orders = orders.groupby(["customer_id", "product_category"]).agg(
        cat_orders=("order_id", "count"),
        cat_revenue=("revenue", "sum")
    ).reset_index()
    top_cat = cat_orders.loc[cat_orders.groupby("customer_id")["cat_revenue"].idxmax()][["customer_id", "product_category"]]
    merged = top_cat.merge(customers[["customer_id", "total_orders", "avg_order_value"]], on="customer_id")
    grouped = merged.groupby("product_category").agg(
        repeat_rate=("total_orders", lambda x: (x > 1).mean()),
        avg_aov=("avg_order_value", "mean")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["product_category"], y=grouped["repeat_rate"] * 100,
        name="Repeat Purchase Rate (%)",
        marker_color="#3B82F6",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["product_category"], y=grouped["avg_aov"],
        name="Avg Order Value ($)",
        mode="lines+markers",
        marker=dict(size=9, color="#F59E0B"),
        line=dict(color="#F59E0B", width=2),
        yaxis="y2"
    ))
    fig.update_layout(
        title="Category Performance — Repeat Rate vs. AOV",
        yaxis=dict(title="Repeat Purchase Rate (%)", showgrid=True, gridcolor="#F3F4F6"),
        yaxis2=dict(title="Avg Order Value ($)", overlaying="y", side="right"),
        height=380,
        margin=dict(t=50, b=40, l=50, r=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
