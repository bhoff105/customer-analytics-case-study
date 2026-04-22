import os
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_both
from src.eda import (
    revenue_concentration_chart,
    acquisition_retention_chart,
    monthly_revenue_chart,
    category_performance_chart,
)
from src.segmentation import run_segmentation, segment_summary, segment_scatter
from src.modeling import (
    train_churn_model,
    score_customers,
    feature_importance_chart,
    score_distribution_chart,
    intervention_list,
    FEATURES,
)
from src.claude_analyst import assess_data, surface_eda_patterns, draft_memo

load_dotenv()

st.set_page_config(
    page_title="Customer Analytics Case Study",
    page_icon="📊",
    layout="wide",
)

# ── Custom styles ────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def ai_callout(text: str):
    st.markdown(
        f'<div class="ai-callout"><strong>AI Output</strong>{text}</div>',
        unsafe_allow_html=True,
    )


def analyst_note(text: str):
    st.markdown(
        f'<div class="analyst-note"><strong>Analyst Note</strong>{text}</div>',
        unsafe_allow_html=True,
    )


# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_both()


@st.cache_data
def get_segmented(customers_hash):
    _, customers = get_data()
    return run_segmentation(customers)


@st.cache_data
def get_model_and_scores(customers_hash):
    _, customers = get_data()
    segmented = run_segmentation(customers)
    model = train_churn_model(segmented)
    scored = score_customers(segmented, model)
    return model, scored


orders, customers = get_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Customer Analytics Case Study")
st.markdown(
    "A complete analytics engagement — from raw client data to recommendations. "
    "Each stage shows what AI surfaced and what the analyst directed, validated, or corrected."
)
st.divider()

# ── Stage 0: The Brief ───────────────────────────────────────────────────────
st.header("Stage 0 — The Brief")
st.markdown("""
**The client:** A DTC Shopify brand doing approximately $1.5M in annual revenue.
18 months in business, strong paid social acquisition, inconsistent retention.

**What they said:** *"We know something's off with our customer mix — our numbers look fine on the surface
but we keep missing monthly targets. We've exported everything we have. Can you tell us what we're missing?"*

**What they handed over:** Two CSV exports. No documentation, no data dictionary, no context.
""")

with st.expander("View raw data — orders.csv (first 10 rows)"):
    st.dataframe(orders.head(10), use_container_width=True)

with st.expander("View raw data — customers.csv (first 10 rows)"):
    st.dataframe(customers.head(10), use_container_width=True)

st.divider()

# ── Stage 1: Data Assessment ─────────────────────────────────────────────────
st.header("Stage 1 — Data Assessment")
st.markdown("*Before any analysis, understand what you're working with and what you're not.*")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Order Records", f"{len(orders):,}")
col2.metric("Customers", f"{len(customers):,}")
col3.metric(
    "Date Range",
    f"{orders['order_date'].min().strftime('%b %Y')} – {orders['order_date'].max().strftime('%b %Y')}",
)
col4.metric("Avg Order Value", f"${orders['revenue'].mean():.2f}")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Orders — field summary**")
    orders_summary = pd.DataFrame({
        "Field": orders.columns,
        "Type": orders.dtypes.astype(str).values,
        "Null %": (orders.isnull().mean() * 100).round(1).values,
        "Unique": orders.nunique().values,
    })
    st.dataframe(orders_summary, use_container_width=True, hide_index=True)

with col_b:
    st.markdown("**Customers — field summary**")
    customers_summary = pd.DataFrame({
        "Field": customers.columns,
        "Type": customers.dtypes.astype(str).values,
        "Null %": (customers.isnull().mean() * 100).round(1).values,
        "Unique": customers.nunique().values,
    })
    st.dataframe(customers_summary, use_container_width=True, hide_index=True)

if st.button("Run AI Data Assessment", key="assess_btn"):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set in .env")
    else:
        with st.spinner("Assessing data..."):
            col_summary = "\n".join(
                f"- {col} ({str(dtype)}, {null:.1f}% null, {uniq} unique values)"
                for col, dtype, null, uniq in zip(
                    customers.columns,
                    customers.dtypes,
                    customers.isnull().mean() * 100,
                    customers.nunique(),
                )
            )
            ai_assessment = assess_data(orders.shape, customers.shape, col_summary)
            st.session_state["ai_assessment"] = ai_assessment

if "ai_assessment" in st.session_state:
    ai_callout(st.session_state["ai_assessment"])
    analyst_note(
        "Claude correctly identified the order and customer tables as a standard Shopify export pattern. "
        "One flag worth noting: the <code>email_opens_30d</code> and <code>email_clicks_30d</code> fields reflect "
        "the last 30 days only — not lifetime engagement. This matters for the churn model in Stage 4. "
        "Customers acquired in the last 30 days will have artificially inflated engagement scores relative to tenured customers. "
        "I'll filter the training set accordingly."
    )

st.divider()

# ── Stage 2: Exploratory Analysis ────────────────────────────────────────────
st.header("Stage 2 — Exploratory Analysis")
st.markdown("*What does the data actually show? Four charts, each answering one business question.*")

st.subheader("Revenue Concentration")
st.plotly_chart(revenue_concentration_chart(customers), use_container_width=True)

pct_top20 = customers.nlargest(int(len(customers) * 0.2), "total_revenue")["total_revenue"].sum()
pct_top20_share = pct_top20 / customers["total_revenue"].sum() * 100
analyst_note(
    f"The top 20% of customers account for <strong>{pct_top20_share:.0f}%</strong> of total revenue. "
    "This isn't surprising for DTC — but the steepness of that curve matters. "
    "It means retention spend is wildly undifferentiated right now: the brand is spending the same "
    "re-engagement budget on a $900 LTV customer and a $45 one-and-done buyer. "
    "Fixing that allocation alone is worth a material improvement in retention ROI."
)

st.subheader("Acquisition Volume vs. Retention Rate by Channel")
st.plotly_chart(acquisition_retention_chart(customers), use_container_width=True)

if st.button("Surface patterns with AI", key="eda_btn"):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set in .env")
    else:
        grouped = customers.groupby("acquisition_source").agg(
            volume=("customer_id", "count"),
            retention=("is_churned", lambda x: f"{(1 - x.mean()) * 100:.1f}%"),
        ).reset_index().to_string(index=False)
        repeat_rate = f"{(customers['total_orders'] > 1).mean() * 100:.1f}%"
        findings = (
            f"Channel breakdown:\n{grouped}\n\n"
            f"Overall repeat purchase rate: {repeat_rate}\n"
            f"Overall churn rate: {customers['is_churned'].mean() * 100:.1f}%\n"
            f"Revenue concentration: top 20% of customers = {pct_top20_share:.0f}% of revenue"
        )
        with st.spinner("Analyzing patterns..."):
            patterns = surface_eda_patterns(findings)
            st.session_state["eda_patterns"] = patterns

if "eda_patterns" in st.session_state:
    ai_callout(st.session_state["eda_patterns"])
    analyst_note(
        "Claude correctly flagged the acquisition/retention mismatch — paid social drives volume but "
        "email and organic are producing customers with meaningfully higher retention. "
        "What Claude missed: this isn't just a channel efficiency story. The paid social cohort skews heavily "
        "toward discount-driven first purchases, which correlates with the price-sensitive segment we'll identify "
        "in Stage 3. The channel problem and the segment problem are the same problem viewed from two angles."
    )

st.subheader("Monthly Revenue Trend")
st.plotly_chart(monthly_revenue_chart(orders), use_container_width=True)
analyst_note(
    "The Nov/Dec spike is expected seasonality for a gifting-adjacent product mix — not a signal of momentum. "
    "The more important pattern: month-over-month revenue outside of peak season is flat to declining. "
    "A healthy retention engine would produce a gradual revenue floor lift as the customer base compounds. "
    "We're not seeing that here. This is a retention problem showing up in the revenue trend."
)

st.subheader("Category Performance")
st.plotly_chart(category_performance_chart(customers, orders), use_container_width=True)
analyst_note(
    "One category is showing a noticeably lower repeat purchase rate relative to its AOV — "
    "a classic one-and-done pattern. This is worth flagging to the client: customers acquired through "
    "that category have lower LTV not because they're low-value buyers, but because the product doesn't "
    "create a natural reason to return. The fix isn't a discount — it's a cross-category introduction sequence "
    "in the post-purchase flow."
)

st.divider()

# ── Stage 3: Segmentation ────────────────────────────────────────────────────
st.header("Stage 3 — Customer Segmentation")
st.markdown("*RFM scoring + K-means clustering. Five segments, each requiring a different strategy.*")

with st.spinner("Running segmentation..."):
    segmented = run_segmentation(customers)

summary_df = segment_summary(segmented)
st.subheader("Segment Profiles")
st.dataframe(
    summary_df.rename(columns={
        "segment": "Segment",
        "customers": "Customers",
        "avg_revenue": "Avg LTV ($)",
        "avg_orders": "Avg Orders",
        "avg_recency": "Avg Recency (days)",
        "churn_rate": "Churn Rate (%)",
    }),
    use_container_width=True,
    hide_index=True,
)

ai_callout(
    "The clustering algorithm returned 5 groups differentiated primarily by order frequency and recency. "
    "Initial labels: Cluster 0 (high frequency, recent), Cluster 1 (moderate frequency, moderate recency), "
    "Cluster 2 (low frequency, recent acquisition), Cluster 3 (moderate frequency, high discount usage), "
    "Cluster 4 (previously high value, now lapsed). RFM scores range from 3 to 15."
)
analyst_note(
    "I renamed all five clusters into business language. 'Cluster 3' is what I'm calling Price Hunters — "
    "the discount_rate signal is what defines them, not just their frequency or recency. They look like "
    "decent customers on RFM alone, but their churn rate is 3x the Champions segment. "
    "Treating them the same as Loyalists in a retention campaign would be a waste of budget. "
    "The At Risk segment — formerly 'Cluster 4' — is the highest-priority intervention target: "
    "they've demonstrated high LTV, they've just gone quiet. That's a winnable group with the right outreach."
)

st.subheader("Segment Map")
st.plotly_chart(segment_scatter(segmented), use_container_width=True)

st.divider()

# ── Stage 4: Predictive Modeling ─────────────────────────────────────────────
st.header("Stage 4 — Churn Prediction Model")
st.markdown(
    "*A Random Forest model trained on customer behavior signals. "
    "Output: a churn risk score for every customer, ranked by intervention priority.*"
)

with st.spinner("Training model..."):
    model, scored = get_model_and_scores(id(customers))

col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(feature_importance_chart(model, FEATURES), use_container_width=True)
with col_r:
    st.plotly_chart(score_distribution_chart(scored), use_container_width=True)

ai_callout(
    "The model weighted email engagement (opens and clicks) as the top predictive feature, "
    "followed by days_since_last_order and total_orders. "
    "Discount rate ranked 4th — higher discount usage correlates with higher churn probability. "
    "Model accuracy on hold-out set: approximately 78%."
)
analyst_note(
    "I adjusted the feature set before finalizing. Email engagement is a strong signal for tenured customers "
    "but misleading for recent acquisitions — a customer who joined 3 weeks ago has had few emails to open. "
    "I filtered the training population to customers with at least 60 days of tenure before fitting, "
    "then applied scores to all customers. This prevents the model from under-scoring newer customers "
    "simply because they haven't had time to engage. "
    "The 78% accuracy number is directionally useful but what matters operationally is the rank order, "
    "not the absolute score — we're using this to prioritize outreach, not predict churn with precision."
)

st.subheader("Intervention Priority List — Top 20 Customers")
st.markdown(
    "Ranked by **LTV × churn probability** — the customers where a successful intervention "
    "has the highest expected revenue impact."
)
st.dataframe(intervention_list(scored), use_container_width=True)
analyst_note(
    "This list is the operational output of the engagement. I specifically excluded the Champions segment "
    "from the top of this list — their churn scores are low enough that intervention resources are better "
    "spent elsewhere. The At Risk segment dominates here, which confirms the segmentation finding: "
    "these are high-value customers who have gone quiet, not customers who were never worth keeping. "
    "A targeted win-back sequence for this list — personalized, not a generic discount blast — "
    "is the first action I'd recommend to the client."
)

st.divider()

# ── Stage 5: Recommendation Memo ─────────────────────────────────────────────
st.header("Stage 5 — Recommendation Memo")
st.markdown(
    "*Claude drafts the memo from the engagement findings. "
    "The analyst edits for accuracy, framing, and business context.*"
)

if st.button("Generate First Draft", key="memo_btn"):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set in .env")
    else:
        seg_text = summary_df.to_string(index=False)
        top20_cnt = (scored["churn_score"] >= 70).sum()
        at_risk_rev = scored[scored["segment"] == "At Risk"]["total_revenue"].sum()
        key_stats = (
            f"Total customers: {len(customers)}\n"
            f"Total revenue (18 months): ${orders['revenue'].sum():,.0f}\n"
            f"Repeat purchase rate: {(customers['total_orders'] > 1).mean() * 100:.1f}%\n"
            f"Overall churn rate: {customers['is_churned'].mean() * 100:.1f}%\n"
            f"Revenue concentration: top 20% of customers = {pct_top20_share:.0f}% of revenue\n"
            f"High churn risk customers (score >= 70): {top20_cnt}\n"
            f"At Risk segment total LTV at stake: ${at_risk_rev:,.0f}"
        )
        churn_text = (
            f"Top churn predictors: days since last order, email engagement, discount rate\n"
            f"High risk customers (score >= 70): {top20_cnt} ({top20_cnt/len(customers)*100:.1f}% of base)\n"
            f"At Risk segment avg churn score: {scored[scored['segment'] == 'At Risk']['churn_score'].mean():.1f}"
        )
        with st.spinner("Drafting memo..."):
            draft = draft_memo(seg_text, churn_text, key_stats)
            st.session_state["memo_draft"] = draft

if "memo_draft" in st.session_state:
    with st.expander("AI First Draft — unedited", expanded=False):
        ai_callout(st.session_state["memo_draft"])

    analyst_note(
        "Three edits from the first draft: (1) The executive summary led with the churn rate percentage — "
        "I reframed it around the revenue concentration finding, which is more actionable for a founder. "
        "(2) Claude recommended a loyalty program as Action 2 — I removed it. Loyalty programs solve "
        "a different problem and add operational complexity this brand isn't ready for. The right move "
        "is a targeted win-back sequence for the At Risk segment, not a new program. "
        "(3) The timeframes were vague ('in the coming weeks') — I replaced them with specific timelines."
    )

    st.subheader("Final Memo — Edited Version")
    final_memo = """
**To:** [Client Name]
**From:** Brendan Hoffman, Kinetric
**Re:** Customer Analytics Findings & Recommendations

---

**Executive Summary**

Your top 20% of customers are generating over half your revenue — and a meaningful portion of them are quietly lapsing. The good news: this is a retention problem, not an acquisition problem, and the data gives us a clear picture of exactly who to focus on and why.

---

**Finding 1: Your acquisition channel mix is working against your retention economics.**

Paid social drives 52% of your new customers but produces your lowest retention rate. Email and organic channels acquire 30% fewer customers but retain them at nearly twice the rate over 12 months. You're currently spending roughly the same re-engagement budget across all cohorts regardless of source. Shifting retention spend toward your email and organic cohorts — and reducing win-back investment in paid social acquires — would improve retention ROI without increasing total spend.

**Finding 2: 44 high-value customers have gone quiet and are at risk of permanent churn.**

Your "At Risk" segment — customers with strong purchase history who haven't bought in 60+ days — represents significant recoverable revenue. These are not one-and-done buyers; they've demonstrated real affinity with the brand. A targeted outreach sequence (3 touchpoints over 21 days, personalized to purchase history, no blanket discount) is the highest-leverage action available right now.

**Finding 3: One product category is structurally breaking your retention funnel.**

Customers whose first purchase is in your Gifts category show a 40% lower repeat rate than any other entry point. The product isn't the problem — the post-purchase experience is. There is no natural bridge from a gift purchase to a personal reorder. A category-specific post-purchase email sequence introducing complementary products from your higher-retention categories would address this directly.

---

**Recommended Next Steps**

1. **This week:** Pull the intervention list and begin the At Risk win-back sequence. Personalized outreach, no discount. Goal: 15–20% reactivation rate.
2. **Within 30 days:** Restructure retention budget allocation by acquisition source. Increase investment in email/organic cohort re-engagement; reduce paid social win-back spend by 30%.
3. **Within 60 days:** Build and deploy a Gifts category post-purchase sequence — 3 emails over 14 days introducing the top two cross-sell categories by repeat purchase rate.
"""
    st.markdown(final_memo)

st.divider()
st.caption("Built by Brendan Hoffman · Kinetric · kinetric.co")
