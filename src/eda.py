import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ── Kinetric chart palette (mirrors assets/style.css) ────────────────────────
KIN_BG         = "#101626"
KIN_PAPER      = "#101626"
KIN_TEXT       = "#E8EDF5"
KIN_TEXT_DIM   = "#B5C2D6"
KIN_MUTED      = "#7E95B0"
KIN_BORDER     = "#243350"
KIN_GRID       = "rgba(126, 149, 176, 0.14)"
KIN_ACCENT     = "#2DD4BF"
KIN_ACCENT_SOFT= "rgba(45, 212, 191, 0.18)"
KIN_BLUE       = "#60A5FA"
KIN_BLUE_SOFT  = "rgba(96, 165, 250, 0.35)"
KIN_AMBER      = "#F59E0B"
KIN_RED        = "#EF4444"
KIN_VIOLET     = "#A78BFA"

KIN_FONT = dict(family="DM Sans, system-ui, sans-serif", color=KIN_TEXT_DIM, size=13)
# Softer gridline for a cleaner business-chart look
KIN_GRID_SOFT = "rgba(126, 149, 176, 0.08)"

# Chart config applied globally — hides modebar
CHART_CONFIG = {"displayModeBar": False, "displaylogo": False, "staticPlot": False}


def _apply_base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    """
    Shared base styling for every chart in the case study.
    Business-dashboard aesthetic: sans-serif tick labels, soft gridlines,
    generous margins, no modebar, no techy axis rules.
    """
    fig.update_layout(
        height=height,
        margin=dict(t=56, b=56, l=68, r=32),
        plot_bgcolor=KIN_BG,
        paper_bgcolor=KIN_PAPER,
        font=KIN_FONT,
        title=dict(
            font=dict(family="DM Sans, sans-serif", size=15, color=KIN_TEXT, weight=500),
            x=0.0, xanchor="left", y=0.97,
            pad=dict(l=0, r=0, t=0, b=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=KIN_BORDER,
            borderwidth=0,
            font=dict(family="DM Sans, sans-serif", size=12, color=KIN_TEXT_DIM),
        ),
        hoverlabel=dict(
            bgcolor="#0B0F1A",
            bordercolor=KIN_ACCENT,
            font=dict(family="DM Sans, sans-serif", size=12, color=KIN_TEXT),
        ),
        modebar=dict(remove=[
            "zoom", "pan", "select", "lasso2d", "zoomIn", "zoomOut",
            "autoScale", "resetScale", "toImage"
        ]),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(family="DM Sans, sans-serif", size=12, color=KIN_TEXT_DIM),
        title_font=dict(family="DM Sans, sans-serif", size=12, color=KIN_MUTED),
        title_standoff=14,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=KIN_GRID_SOFT,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(family="DM Sans, sans-serif", size=12, color=KIN_TEXT_DIM),
        title_font=dict(family="DM Sans, sans-serif", size=12, color=KIN_MUTED),
        title_standoff=14,
    )
    return fig


def revenue_concentration_chart(customers: pd.DataFrame) -> go.Figure:
    """Pareto: cumulative revenue share by customer percentile."""
    sorted_rev = customers["total_revenue"].sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_rev.cumsum() / sorted_rev.sum() * 100
    pct_customers = (sorted_rev.index + 1) / len(sorted_rev) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pct_customers, y=cumulative,
        mode="lines", fill="tozeroy",
        line=dict(color=KIN_ACCENT, width=2.25),
        fillcolor=KIN_ACCENT_SOFT,
        name="Cumulative Revenue",
        hovertemplate="<b>%{x:.0f}%</b> of customers<br>%{y:.1f}% of revenue<extra></extra>",
    ))
    # 20% marker lines
    y_at_20 = cumulative.iloc[int(len(cumulative) * 0.2) - 1]
    fig.add_shape(type="line", x0=20, x1=20, y0=0, y1=100,
                  line=dict(color=KIN_AMBER, dash="dot", width=1.25))
    fig.add_shape(type="line", x0=0, x1=20, y0=y_at_20, y1=y_at_20,
                  line=dict(color=KIN_AMBER, dash="dot", width=1.25))

    fig.update_layout(
        title="Top 20% of Customers Drive ~53% of Revenue",
        xaxis_title="Cumulative % of Customers",
        yaxis_title="Cumulative % of Revenue",
        showlegend=False,
    )
    fig = _apply_base_layout(fig, height=400)
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(range=[0, 101], ticksuffix="%")
    return fig


def acquisition_retention_chart(customers: pd.DataFrame) -> go.Figure:
    """Volume vs. retention rate by acquisition source."""
    grouped = customers.groupby("acquisition_source").agg(
        volume=("customer_id", "count"),
        retention_rate=("is_churned", lambda x: 1 - x.mean())
    ).reset_index()
    grouped = grouped.sort_values("volume", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["acquisition_source"],
        y=grouped["volume"],
        name="Customer Volume",
        marker=dict(color=KIN_BLUE_SOFT, line=dict(color=KIN_BLUE, width=1)),
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["acquisition_source"],
        y=grouped["retention_rate"] * 100,
        name="Retention Rate (%)",
        mode="lines+markers",
        marker=dict(size=10, color=KIN_AMBER, line=dict(color=KIN_BG, width=1)),
        line=dict(color=KIN_AMBER, width=2.25),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Retention: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Paid Social Acquires the Most Customers and Retains the Fewest",
        yaxis=dict(title="Customers Acquired"),
        yaxis2=dict(
            title="Retention Rate (%)", overlaying="y", side="right",
            range=[0, 100], showgrid=False, zeroline=False, showline=False, ticks="",
            tickfont=dict(family="DM Sans, sans-serif", size=12, color=KIN_AMBER),
            title_font=dict(family="DM Sans, sans-serif", size=12, color=KIN_AMBER),
            title_standoff=14,
        ),
        # Legend pinned above the plot area so bars and line can never overlap it
        legend=dict(
            orientation="h",
            x=0, xanchor="left",
            y=1.14, yanchor="bottom",
        ),
        barmode="group",
    )
    fig = _apply_base_layout(fig, height=440)
    fig.update_layout(margin=dict(t=90, b=56, l=68, r=68))
    fig.update_yaxes(tickformat=",d")
    return fig


def monthly_revenue_chart(orders: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend over the full date range."""
    orders = orders.copy()
    orders["month"] = orders["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly = orders.groupby("month")["revenue"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["revenue"],
        marker=dict(color=KIN_ACCENT_SOFT, line=dict(color=KIN_ACCENT, width=1)),
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Off-Peak Revenue Is Flat: No Compounding from the Existing Base",
        xaxis_title="",
        yaxis_title="Monthly Revenue",
        showlegend=False,
        bargap=0.25,
    )
    fig = _apply_base_layout(fig, height=380)
    fig.update_xaxes(showgrid=False, dtick="M3", tickformat="%b %Y")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
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
    ).reset_index().sort_values("repeat_rate", ascending=False)

    # Highlight any category whose repeat rate is meaningfully below the others (the Gifts trough)
    median_rate = grouped["repeat_rate"].median()
    bar_colors = [
        KIN_RED if r < median_rate * 0.5 else KIN_ACCENT
        for r in grouped["repeat_rate"]
    ]
    bar_line_colors = [
        KIN_RED if r < median_rate * 0.5 else KIN_ACCENT
        for r in grouped["repeat_rate"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["product_category"], y=grouped["repeat_rate"] * 100,
        name="Repeat Purchase Rate (%)",
        marker=dict(color=bar_colors, line=dict(color=bar_line_colors, width=1)),
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Repeat rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["product_category"], y=grouped["avg_aov"],
        name="Avg Order Value ($)",
        mode="lines+markers",
        marker=dict(size=10, color=KIN_AMBER, line=dict(color=KIN_BG, width=1)),
        line=dict(color=KIN_AMBER, width=2.25),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>AOV: $%{y:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Gifts Buyers Repeat at ~17%. Every Other Category Clears 70%",
        yaxis=dict(title="Repeat Purchase Rate (%)", range=[0, 100]),
        yaxis2=dict(
            title="Avg Order Value ($)", overlaying="y", side="right",
            showgrid=False, zeroline=False, showline=False, ticks="",
            tickfont=dict(family="DM Sans, sans-serif", size=12, color=KIN_AMBER),
            title_font=dict(family="DM Sans, sans-serif", size=12, color=KIN_AMBER),
            title_standoff=14,
        ),
        # Legend pinned above the plot area — prevents overlap with tall bars
        legend=dict(
            orientation="h",
            x=0, xanchor="left",
            y=1.14, yanchor="bottom",
        ),
        bargap=0.35,
    )
    fig = _apply_base_layout(fig, height=440)
    fig.update_layout(margin=dict(t=90, b=56, l=68, r=68))
    fig.update_yaxes(ticksuffix="%")
    return fig
