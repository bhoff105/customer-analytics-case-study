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
from src.pdf_export import build_memo_pdf

load_dotenv()

# Streamlit Cloud stores secrets in st.secrets; inject into os.environ so
# os.getenv("ANTHROPIC_API_KEY") works identically in local and cloud environments.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass  # st.secrets not available in local dev without a secrets.toml — .env handles it

st.set_page_config(
    page_title="Customer Analytics Case Study — Kinetric",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom styles ────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Layout helpers ───────────────────────────────────────────────────────────
PLOTLY_CFG = {"displayModeBar": False, "displaylogo": False}


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


def stage_marker(label: str, sub: str = ""):
    """Compact stage-progress badge above each stage heading."""
    st.markdown(
        f'''
        <div class="stage-marker">
          <span class="stage-marker__badge">{label}</span>
          <span class="stage-marker__rule"></span>
          <span class="stage-marker__label">{sub}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def section_divider():
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)


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

# ── Hero block ───────────────────────────────────────────────────────────────
st.markdown(
    '''
    <div class="hero-block">
      <div class="hero-block__eyebrow">Kinetric · Portfolio Case Study</div>
      <h1 class="hero-block__title">Customer Analytics Case Study</h1>
      <p class="hero-block__frame">
        A full analytics engagement for a $1.5M DTC brand, from raw CSVs to a founder-ready memo.
        Every stage shows what AI produced, what the analyst corrected, and why the correction mattered.
      </p>
      <div class="hero-block__scenario">
        <strong>Client Scenario</strong>
        A DTC Shopify brand doing approximately $1.5M in annual revenue. 18 months in market,
        strong paid social acquisition, inconsistent retention. Two CSV exports handed over,
        no documentation, no data dictionary, no context. The brief: "tell us what we're missing."
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Stage 0: The Brief ───────────────────────────────────────────────────────
stage_marker("Stage 0 of 5", "The Brief")
st.markdown("## The Brief")
st.markdown("""
Every engagement starts the same way: the founder hands you data and a vague worry. The work is to turn that worry into a diagnosis, then the diagnosis into a decision. Before touching the data, pin down what the client is actually asking and what shape an answer has to take to be useful.

**The client:** A DTC Shopify brand doing approximately $1.5M in annual revenue. 18 months in business, strong paid social acquisition, inconsistent retention.

**What they said:** *"We know something's off with our customer mix. Our numbers look fine on the surface but we keep missing monthly targets. We've exported everything we have. Can you tell us what we're missing?"*

**What they handed over:** Two CSV exports. No documentation, no data dictionary, no context.
""")

with st.expander("View raw data: orders.csv (first 10 rows)"):
    st.dataframe(orders.head(10), use_container_width=True)

with st.expander("View raw data: customers.csv (first 10 rows)"):
    st.dataframe(customers.head(10), use_container_width=True)

section_divider()

# ── Stage 1: Data Assessment ─────────────────────────────────────────────────
stage_marker("Stage 1 of 5", "Data Assessment")
st.markdown("## Data Assessment")
st.markdown(
    "Before any chart gets built, you have to know what the data can and cannot answer. "
    "This stage is about reading the schema like a domain expert: which fields are trustworthy, "
    "which are point-in-time snapshots masquerading as lifetime values, and which gaps will force a caveat "
    "later in the engagement. A good assessment prevents a finding being overturned in week three because "
    "the underlying field was being misread from day one."
)

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
    st.markdown("**Orders: field summary**")
    orders_summary = pd.DataFrame({
        "Field": orders.columns,
        "Type": orders.dtypes.astype(str).values,
        "Null %": (orders.isnull().mean() * 100).round(1).values,
        "Unique": orders.nunique().values,
    })
    st.dataframe(orders_summary, use_container_width=True, hide_index=True)

with col_b:
    st.markdown("**Customers: field summary**")
    customers_summary = pd.DataFrame({
        "Field": customers.columns,
        "Type": customers.dtypes.astype(str).values,
        "Null %": (customers.isnull().mean() * 100).round(1).values,
        "Unique": customers.nunique().values,
    })
    st.dataframe(customers_summary, use_container_width=True, hide_index=True)

if st.button("Run AI Data Assessment", key="assess_btn"):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not configured. Set it in Streamlit Cloud secrets or a local .env file.")
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
        "Claude read the schema correctly as a standard Shopify order-and-customer export, and it flagged the "
        "right business context (DTC, transactional, suited to retention work). That framing is useful and I'd keep it. "
        "What it missed is a trap in the engagement columns: <code>email_opens_30d</code> and <code>email_clicks_30d</code> "
        "are a 30-day snapshot, not a lifetime signal. A customer who signed up last week has had almost nothing to open, "
        "so feeding those fields into a churn model unfiltered will punish new customers for not having had time to engage. "
        "The fix is to restrict model training to customers with 60+ days of tenure, then score the full population. "
        "That single correction changes who ends up on the intervention list in Stage 4, which is the only output the "
        "client is going to act on. Worth naming now so nothing downstream has to be redone."
    )

section_divider()

# ── Stage 2: Exploratory Analysis ────────────────────────────────────────────
stage_marker("Stage 2 of 5", "Exploratory Analysis")
st.markdown("## Exploratory Analysis")
st.markdown(
    "EDA is where you earn the right to make a recommendation. Four questions drive this stage: "
    "where does the revenue actually come from, which acquisition channels are producing customers worth keeping, "
    "what does the revenue trend look like once seasonality is stripped out, and which product entry points predict a second purchase. "
    "A good EDA produces two or three findings sharp enough to bet the memo on. A bad one produces ten charts and no point of view."
)

st.markdown("### Top 20% of Customers Drive 53% of Revenue")
st.plotly_chart(revenue_concentration_chart(customers), use_container_width=True, config=PLOTLY_CFG)

pct_top20 = customers.nlargest(int(len(customers) * 0.2), "total_revenue")["total_revenue"].sum()
pct_top20_share = pct_top20 / customers["total_revenue"].sum() * 100
analyst_note(
    f"The top 20% of customers account for <strong>{pct_top20_share:.0f}%</strong> of total revenue. "
    "A 50/20 split isn't unusual for DTC, so the shape of the curve is consistent with a healthy repeat-buyer base. "
    "What the chart doesn't say on its own is what the client is doing with that information today. "
    "Right now the brand runs a single re-engagement campaign across its entire file: a $900 LTV customer and a "
    "$45 one-and-done buyer receive the same email, the same discount, the same cadence. "
    "The domain logic is straightforward: retention spend scales with expected future revenue, not with list size. "
    "For the engagement, this becomes the first input to the segmentation work in Stage 3 and the intervention "
    "priority ranking in Stage 4. The memo will recommend a reallocation, not a new program."
)

st.markdown("### Paid Social Buys 52% of Customers and Keeps 22% of Them")
st.plotly_chart(acquisition_retention_chart(customers), use_container_width=True, config=PLOTLY_CFG)

if st.button("Surface patterns with AI", key="eda_btn"):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not configured. Set it in Streamlit Cloud secrets or a local .env file.")
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
        "Claude surfaced the volume/retention inversion correctly: paid_social is the largest channel by customer "
        "count and the worst by retention (roughly 22% retained vs. 70% for email and 59% for organic). "
        "That framing is right and worth keeping in the memo verbatim. "
        "Where Claude stopped short: it treated this as a channel efficiency story. It isn't, or rather it isn't only that. "
        "The paid social cohort skews heavily toward discount-driven first purchases and concentrates in the Gifts category, "
        "which means the channel issue is the same population as the Price Hunter segment we'll isolate in Stage 3 "
        "and the Gifts retention trough we'll see in the next chart. Three findings, one underlying customer. "
        "For the engagement, the implication is that pulling paid social spend without fixing the downstream "
        "experience trades one problem for another. The memo needs to recommend both moves together."
    )

st.markdown("### Outside the Holiday Window, Revenue Is Flat")
st.plotly_chart(monthly_revenue_chart(orders), use_container_width=True, config=PLOTLY_CFG)
analyst_note(
    "The Nov/Dec peak is the expected gifting-season lift and not a signal of underlying growth, so stripping "
    "it out is the honest way to read the chart. With the peak removed, monthly revenue is flat to modestly "
    "declining across the 18-month window. The chart is accurate as far as it goes. "
    "What it doesn't show is the compounding effect a healthy retention engine should produce. "
    "Eighteen months into a DTC brand, a working retention motion should be lifting the monthly floor as the "
    "customer base accumulates, even in off-peak months. "
    "We're seeing the opposite pattern, which means new acquisitions are replacing lapsed customers rather "
    "than stacking on top of them. That's the definition of a retention problem showing up in the top line, "
    "and it sets up the At Risk segment work in Stage 3 as the highest-return intervention in the engagement."
)

st.markdown("### Gifts Buyers Don't Come Back, Everyone Else Does")
st.plotly_chart(category_performance_chart(customers, orders), use_container_width=True, config=PLOTLY_CFG)
analyst_note(
    "Customers whose first purchase is in the Gifts category show a repeat rate near 17%, versus roughly "
    "71% across every other category. That's a structural gap, not a rounding difference, and it's the single "
    "cleanest finding in the EDA. The chart on its own could be read as 'Gifts customers are low-value,' which "
    "would be the wrong conclusion. AOV on the Gifts cohort is in line with the rest of the file, so these "
    "customers are spending when they convert. The problem is that a gift purchase creates no natural reason to "
    "return. Without a post-purchase flow that bridges into categories the customer actually uses themselves, "
    "the relationship ends at the first transaction. "
    "For the memo, this becomes Finding 3: the fix isn't a discount or a new product line, it's a category-specific "
    "email sequence introducing the two highest-retention categories to every Gifts-first buyer within 14 days of their order."
)

section_divider()

# ── Stage 3: Segmentation ────────────────────────────────────────────────────
stage_marker("Stage 3 of 5", "Customer Segmentation")
st.markdown("## Customer Segmentation")
st.markdown(
    "Segmentation is where the engagement stops describing the business and starts giving the client someone "
    "to talk to. RFM scoring and K-means cluster the file into five groups that each deserve a different "
    "conversation: who to protect, who to grow, who to win back, who to stop subsidizing. "
    "A good segmentation is legible to the founder in one read and maps cleanly to specific retention actions. "
    "A bad one produces five clusters of statistical distinctions that nobody can operationalize."
)

with st.spinner("Running segmentation..."):
    segmented = run_segmentation(customers)

summary_df = segment_summary(segmented)
st.markdown("### Segment Profiles")

# Segment profile card data — one row per segment, written for a founder
_SEGMENT_META = {
    "Champions": {
        "description": "Your highest-value customers, active in the last 30 days, buying at or near full price. The engine of the business.",
        "strategy": "Invite into a referral program. Exclude from discount promotions entirely. Surface new arrivals and early access as engagement levers.",
        "color": "#2DD4BF",
    },
    "Loyalists": {
        "description": "Consistent repeat buyers with strong order frequency. Not as recent as Champions but reliably active across the year.",
        "strategy": "Maintain cadence with product-led content. Do not lead with discounts. Candidates for loyalty tiers once the brand has scale to support them.",
        "color": "#22C55E",
    },
    "Promising": {
        "description": "Newer customers who have purchased once or twice. High potential, not yet habit. Concentrated in single-category purchases including Gifts.",
        "strategy": "Send a second-purchase sequence within 14 days of first order. Cross-sell into high-retention categories. The 90-day window is the conversion window; don't let it close.",
        "color": "#A78BFA",
    },
    "Price Hunters": {
        "description": "Repeat buyers whose purchases are heavily discount-driven. Churn rate runs roughly 3x Champions. Discount exposure reinforces the pattern.",
        "strategy": "Do not offer further discounts. Send educational and brand-story content to shift the relationship. Measure 60-day repeat rate without a coupon before any additional spend.",
        "color": "#60A5FA",
    },
    "At Risk": {
        "description": "Previously high-LTV buyers who have gone quiet in the last 60 to 150 days. Not churned yet. The highest-return intervention target in the engagement.",
        "strategy": "Launch a personalized 3-touch win-back sequence referencing each customer's purchase history. No blanket discount. Target 15 to 20% reactivation in 30 days.",
        "color": "#F59E0B",
    },
}

for _, row in summary_df.iterrows():
    seg = row["segment"]
    meta = _SEGMENT_META.get(seg, {})
    color = meta.get("color", "#7E95B0")
    description = meta.get("description", "")
    strategy = meta.get("strategy", "")
    customers_n = int(row["customers"])
    avg_ltv = row["avg_revenue"]
    avg_orders = row["avg_orders"]

    st.markdown(
        f'''
        <div style="
            background: #101626;
            border: 1px solid #1A2840;
            border-left: 4px solid {color};
            border-radius: 6px;
            padding: 20px 24px;
            margin: 10px 0;
            box-shadow: 0 2px 12px rgba(0,0,0,0.18);
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                margin-bottom: 8px;
                flex-wrap: wrap;
                gap: 8px;
            ">
                <span style="
                    font-family: \'DM Sans\', sans-serif;
                    font-size: 17px;
                    font-weight: 600;
                    color: {color};
                    letter-spacing: -0.01em;
                ">{seg}</span>
                <div style="
                    display: flex;
                    gap: 20px;
                    font-family: \'IBM Plex Mono\', monospace;
                    font-size: 12px;
                    color: #B5C2D6;
                ">
                    <span><strong style="color: #E8EDF5;">{customers_n:,}</strong> customers</span>
                    <span><strong style="color: #E8EDF5;">${avg_ltv:,.0f}</strong> avg LTV</span>
                    <span><strong style="color: #E8EDF5;">{avg_orders:.1f}</strong> avg orders</span>
                </div>
            </div>
            <p style="
                font-family: Georgia, \'Times New Roman\', serif;
                font-size: 14px;
                color: #B5C2D6;
                line-height: 1.65;
                margin: 0 0 10px 0;
            ">{description}</p>
            <div style="
                font-family: \'DM Sans\', sans-serif;
                font-size: 13px;
                color: #7E95B0;
                border-top: 1px solid #1A2840;
                padding-top: 10px;
                margin-top: 4px;
            ">
                <span style="
                    font-family: \'IBM Plex Mono\', monospace;
                    font-size: 10px;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    color: {color};
                    font-weight: 600;
                    margin-right: 8px;
                ">Strategy</span>{strategy}
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

ai_callout(
    "The clustering algorithm returned 5 groups differentiated primarily by order frequency and recency. "
    "Initial labels: Cluster 0 (high frequency, recent), Cluster 1 (moderate frequency, moderate recency), "
    "Cluster 2 (low frequency, recent acquisition), Cluster 3 (moderate frequency, high discount usage), "
    "Cluster 4 (previously high value, now lapsed). RFM scores range from 3 to 15."
)
analyst_note(
    "Claude got the math right: five clusters separated cleanly on RFM, and the centroid descriptions match "
    "what's in the data. The issue is that 'Cluster 3' and 'Cluster 4' are not segments a founder can act on. "
    "I relabeled them into business language and added discount_rate as a defining dimension for Cluster 3, "
    "which becomes Price Hunters. On pure RFM they look like decent customers. Their churn rate, however, runs "
    "roughly 3x the Champions segment, and almost every repeat purchase they make carries a discount. "
    "Treating them as Loyalists in a retention campaign is a negative-margin move. "
    "Cluster 4 becomes At Risk, which is where the money is in this engagement: these are previously high-LTV "
    "customers who have gone quiet, not customers who were never worth keeping. "
    "The Stage 4 intervention list leans hard on this segment because it's the winnable group, and the Stage 5 "
    "memo leads with it for the same reason."
)

st.markdown("### Champions and At Risk Sit on Opposite Ends of the Same Curve")
st.plotly_chart(segment_scatter(segmented), use_container_width=True, config=PLOTLY_CFG)

section_divider()

# ── Stage 4: Predictive Modeling ─────────────────────────────────────────────
stage_marker("Stage 4 of 5", "Churn Prediction Model")
st.markdown("## Churn Prediction Model")
st.markdown(
    "The model has one job: turn the qualitative segmentation into a ranked list of customers the retention "
    "team can work through on Monday morning. A Random Forest scores every customer on churn probability, "
    "which gets multiplied by LTV to produce an intervention priority. What matters operationally isn't the "
    "model's accuracy score, it's whether the top 20 names on the list are actually the 20 the client should "
    "call first. That's the bar this stage has to clear."
)

with st.spinner("Training model..."):
    model, scored = get_model_and_scores(id(customers))

col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(feature_importance_chart(model, FEATURES), use_container_width=True, config=PLOTLY_CFG)
with col_r:
    st.plotly_chart(score_distribution_chart(scored), use_container_width=True, config=PLOTLY_CFG)

ai_callout(
    "The model weighted email engagement (opens and clicks) as the top predictive feature, "
    "followed by days_since_last_order and total_orders. "
    "Discount rate ranked 4th: higher discount usage correlates with higher churn probability. "
    "Model accuracy on hold-out set: approximately 78%."
)
analyst_note(
    "The feature ranking is defensible, and flagging discount_rate as a churn signal is the right call: "
    "price-sensitive customers churn faster, which is consistent with the Price Hunter profile from Stage 3. "
    "The correction is the one previewed in Stage 1. Email engagement as the top feature is true on this training set, "
    "but it's partially an artifact of the 30-day window in <code>email_opens_30d</code> and <code>email_clicks_30d</code>. "
    "Newer customers haven't had time to rack up opens, so the model reads them as disengaged when they're simply new. "
    "I restricted training to customers with at least 60 days of tenure, then applied the fitted model to the full "
    "population. That stabilizes the ranking and stops the model from mislabeling last month's acquisitions as churn risks. "
    "The 78% accuracy figure is directionally fine, but don't sell it as the headline. The operational value here is the "
    "rank order, which is what the Stage 4 intervention list is actually built from."
)

st.markdown("### Where to Spend the Next Retention Dollar: Top 20 Customers")
st.markdown(
    "Ranked by **LTV x churn probability**. These are the 20 customers where a successful "
    "intervention has the highest expected revenue impact. Each row includes a specific recommended "
    "action derived from that customer's segment, recency, category history, and discount behaviour."
)
_ilist = intervention_list(scored)
st.dataframe(
    _ilist,
    use_container_width=True,
    column_config={
        "Recommended Action": st.column_config.TextColumn(
            "Recommended Action",
            width="large",
        ),
        "Email": st.column_config.TextColumn("Email", width="medium"),
        "Lifetime Revenue": st.column_config.TextColumn("Lifetime Revenue", width="small"),
        "Avg Order Value": st.column_config.TextColumn("Avg Order Value", width="small"),
        "Churn Score": st.column_config.NumberColumn("Churn Score", format="%.1f"),
    },
)
analyst_note(
    "This table is the operational deliverable of the engagement. Champions are excluded from the top of the list "
    "by design: their churn probability is low enough that intervention dollars spent here have a low expected return, "
    "even with high LTV. The list is dominated by At Risk customers, which confirms the segmentation: these are "
    "high-LTV buyers who have gone quiet, not customers who were never going to stick. "
    "The recommended action is a three-touch, 21-day personalized win-back sequence keyed to each customer's purchase "
    "history. No blanket discount. A discount on this cohort teaches exactly the wrong behavior and converts your "
    "highest-value buyers into Price Hunters. "
    "Success looks like a 15-20% reactivation rate on this list inside 30 days, which on the At Risk LTV base is the "
    "single largest revenue lever available to the brand this quarter."
)

section_divider()

# ── Stage 5: Recommendation Memo ─────────────────────────────────────────────
stage_marker("Stage 5 of 5", "Recommendation Memo")
st.markdown("## Recommendation Memo")
st.markdown(
    "The memo is the only artifact the client remembers. Everything before this stage exists to justify three "
    "findings, three numbers, and three actions with timelines. Claude drafts the first version from the engagement "
    "outputs. The analyst rewrites it, reorders it, and strips out anything that isn't a decision. A good memo "
    "is one page and one Monday morning away from being executed."
)

if st.button("Generate First Draft", key="memo_btn"):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not configured. Set it in Streamlit Cloud secrets or a local .env file.")
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
    with st.expander("AI First Draft (unedited)", expanded=False):
        ai_callout(st.session_state["memo_draft"])

    analyst_note(
        "Three edits from the first draft, each worth naming. "
        "First, the draft led with overall churn rate. That's a description, not a decision. I reordered to lead "
        "with the At Risk win-back because that's the action the client can take on Monday. "
        "Second, Claude recommended a loyalty program as Action 2. I cut it. Loyalty programs solve a different "
        "problem, add operational overhead this brand can't absorb at $1.5M in revenue, and distract from the "
        "three moves that actually fix the retention economics. "
        "Third, the timelines were written as 'in the coming weeks,' which is not a timeline. I replaced every "
        "vague phrase with a specific window (this week, 30 days, 60 days) tied to a specific owner action. "
        "The result is a memo a founder can forward to their retention lead without editing."
    )

    st.markdown("### Final Memo (Edited Version)")
    memo_body_md = """**Executive Summary**

You have a retention problem, not an acquisition problem, and the data points cleanly at where to start. Your top 20% of customers generate 53% of your revenue, and a recoverable group of them has gone quiet in the last 60 to 150 days. Three moves over the next 60 days, sequenced below, address the bulk of the gap.

---

**Finding 1: Your highest-value lapsed customers are a winnable group, and no one is working them.**

Your "At Risk" segment is a set of previously high-LTV buyers who haven't purchased in 60 or more days. These aren't discount-chasers. They're customers who already demonstrated affinity, then went quiet. A three-touch, 21-day personalized win-back sequence keyed to each customer's purchase history is the highest-return move available to you right now. No blanket discount. A discount here trains your best customers to wait for the next one, which is exactly the pattern you're trying to reverse with the Price Hunter segment.

**Finding 2: Your acquisition mix is at war with your retention economics.**

Paid social is acquiring roughly 52% of your new customers and retaining about 22% of them at 12 months. Email and organic acquire fewer customers but retain them near 70% and 59% respectively. You're currently funding re-engagement against paid social cohorts at roughly the same rate as organic cohorts, which is subsidizing the channel that produces your worst customers. Shifting 30% of paid social win-back budget into email and organic cohort re-engagement improves retention ROI without raising total spend.

**Finding 3: One category is structurally breaking your retention funnel.**

Customers whose first purchase is in the Gifts category repeat at roughly 17% versus 71% for every other category. The problem isn't the product or the price. A gift purchase creates no natural reason for the buyer to return, so without a bridge into a category they'd use themselves, the relationship ends at the first order. A category-specific post-purchase sequence introducing the two highest-retention categories (Home & Kitchen, Beauty & Wellness) closes the gap.

---

**Recommended Next Steps**

1. **This week, by Friday:** Pull the 20-name intervention list from Stage 4. Assign it to one owner. Launch the At Risk win-back sequence with personalized outreach. Target: 15–20% reactivation inside 30 days.
2. **Within 30 days:** Reallocate retention spend by acquisition source. Reduce paid social win-back budget by 30%. Redirect to email and organic cohort re-engagement. Measure 60-day repeat rate against baseline before making it permanent.
3. **Within 60 days:** Ship a Gifts-entry post-purchase sequence. Three emails over 14 days. Cross-sell into Home & Kitchen and Beauty & Wellness. Success metric: 30-day second-order rate on Gifts-first buyers lifted from 17% toward the 40% mark.
"""

    # Render memo in document-style container
    st.markdown(
        f'''
        <div class="memo-container">
          <div class="memo-header">
            <div class="memo-header__brand">
              <span>Kinetric</span>
              <span class="memo-header__brand-tag">Advisory Memorandum</span>
            </div>
            <div class="memo-header__label">To</div>
            <div class="memo-header__value">[Client Name], Founder</div>
            <div class="memo-header__label">From</div>
            <div class="memo-header__value">Brendan Hoffman, Kinetric</div>
            <div class="memo-header__label">Date</div>
            <div class="memo-header__value">Engagement Closeout</div>
            <div class="memo-header__label">Re</div>
            <div class="memo-header__value">Customer Analytics Findings &amp; Recommendations</div>
          </div>
          <div class="memo-body">
        ''',
        unsafe_allow_html=True,
    )
    st.markdown(memo_body_md)
    st.markdown(
        '''
          </div>
          <div class="memo-footer">Kinetric · kinetric.co · Prepared for client use</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ── Memo PDF download ────────────────────────────────────────────────────
    try:
        _pdf_bytes = build_memo_pdf(
            memo_body_md=memo_body_md,
            to_="[Client Name], Founder",
            from_="Brendan Hoffman, Kinetric",
            date_="Engagement Closeout",
            re_="Customer Analytics Findings & Recommendations",
        )
        st.download_button(
            label="Download memo as PDF",
            data=_pdf_bytes,
            file_name="kinetric_customer_analytics_memo.pdf",
            mime="application/pdf",
            use_container_width=False,
        )
    except Exception as _pdf_err:
        st.caption(f"PDF export unavailable: {_pdf_err}")

section_divider()

# ── Close: the sales argument ────────────────────────────────────────────────
# Computed figures for the leave-behind, driven by the same data the memo uses.
at_risk_count = int((scored["segment"] == "At Risk").sum())
at_risk_ltv = float(scored.loc[scored["segment"] == "At Risk", "total_revenue"].sum())
top20_share_int = int(round(pct_top20_share))

st.markdown(
    f'''
    <div class="close-section">
      <div class="close-section__eyebrow">Engagement Close</div>
      <h2 class="close-section__title">Where this leaves the client</h2>

      <div class="close-beat">
        <div class="close-beat__label">Where we started</div>
        <p class="close-beat__body">
          Two CSV exports, no documentation, no view of who the best customers were or which ones were slipping.
          The founder could see the top-line number moving sideways and could not tell you why.
        </p>
      </div>

      <div class="close-beat">
        <div class="close-beat__label">What you now have</div>
        <ul class="close-beat__list">
          <li><strong>A segmented customer base.</strong> Five named segments (Champions, Loyalists, Promising, Price Hunters, At Risk) with a plain-English playbook for each.</li>
          <li><strong>A churn-risk score for every customer.</strong> 0 to 100, refreshed from the same model, ranked by intervention priority so retention spend goes where it earns.</li>
          <li><strong>A prioritized win-back list.</strong> {at_risk_count} At Risk customers, {top20_share_int}% revenue concentration at the top of the file, specific names and emails to work this week with a recommended action per customer.</li>
          <li><strong>A written recommendation memo.</strong> Three findings, three numbers, three actions with timelines, formatted for a founder to forward to their retention lead without editing.</li>
        </ul>
      </div>

      <div class="close-beat">
        <div class="close-beat__label">What happens next</div>
        <div class="close-paths">
          <div class="close-path">
            <div class="close-path__tag">Option A</div>
            <h3 class="close-path__name">One-time audit</h3>
            <p class="close-path__body">
              Kinetric delivers every output above as a fixed-scope engagement. The client owns the segment playbooks,
              the scoring model, and the intervention list, and operates them independently from day one.
            </p>
            <div class="close-path__fit">Fit: brands with a retention owner on staff</div>
          </div>
          <div class="close-path">
            <div class="close-path__tag">Option B</div>
            <h3 class="close-path__name">Ongoing retainer</h3>
            <p class="close-path__body">
              Kinetric refreshes the segmentation and churn scores monthly, owns the At Risk intervention tracking
              end to end, and expands the analysis as new data sources come online (Klaviyo, ads, reviews).
            </p>
            <div class="close-path__fit">Fit: brands without analytical capacity in-house</div>
          </div>
        </div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Leave-behind: one-page summary ───────────────────────────────────────────
st.markdown(
    f'''
    <div class="leavebehind">
      <div class="leavebehind__eyebrow">Client Leave-Behind</div>
      <h3 class="leavebehind__title">One page. Four numbers. Three moves.</h3>

      <div class="leavebehind__numbers">
        <div class="leavebehind__num">
          <div class="leavebehind__num-value">{top20_share_int}%</div>
          <div class="leavebehind__num-label">of revenue from the top 20% of customers. Retention spend currently flat across the file.</div>
        </div>
        <div class="leavebehind__num">
          <div class="leavebehind__num-value">${at_risk_ltv/1000:,.0f}K</div>
          <div class="leavebehind__num-label">of lifetime revenue sitting in {at_risk_count} At Risk customers who have gone quiet.</div>
        </div>
        <div class="leavebehind__num">
          <div class="leavebehind__num-value">2.2×</div>
          <div class="leavebehind__num-label">retention gap between paid social (22%) and email/organic cohorts. Subsidized by current spend mix.</div>
        </div>
      </div>

      <div class="leavebehind__actions-title">Three moves, sequenced</div>
      <ul class="leavebehind__actions">
        <li><span class="when">This week</span><span>Launch the At Risk win-back sequence against the 20-name intervention list. Personalized, no blanket discount. Target 15 to 20% reactivation in 30 days.</span></li>
        <li><span class="when">Within 30 days</span><span>Reallocate 30% of paid social win-back budget into email and organic cohort re-engagement. Measure 60-day repeat rate before making it permanent.</span></li>
        <li><span class="when">Within 60 days</span><span>Ship the Gifts-entry post-purchase sequence cross-selling into Home & Kitchen and Beauty & Wellness. Lift 30-day second-order rate on Gifts-first buyers from 17% toward 40%.</span></li>
      </ul>

      <div class="leavebehind__contact">
        Kinetric · <a href="https://kinetric.co">kinetric.co</a> · <a href="mailto:brendan@kinetric.co">brendan@kinetric.co</a> · <a href="https://calendly.com/brendan-kinetric/30min">Schedule a 30-minute call</a>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Soft CTA: two paths forward ──────────────────────────────────────────────
st.markdown(
    '''
    <div class="soft-cta">
      <a class="soft-cta__item" href="https://data-discovery-tool-3tvfke3anwvtyj4c7nr9uq.streamlit.app" target="_blank" rel="noopener noreferrer">
        <span class="soft-cta__prompt">Want to see this run on your own data?</span>
        <span class="soft-cta__target">Open the Data Discovery Tool → kinetric.co</span>
      </a>
      <a class="soft-cta__item" href="https://calendly.com/brendan-kinetric/30min" target="_blank" rel="noopener noreferrer">
        <span class="soft-cta__prompt">Ready to start a conversation?</span>
        <span class="soft-cta__target">Book 30 minutes → calendly.com/brendan-kinetric</span>
      </a>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="kin-footer">Built by Brendan Hoffman · '
    '<a href="https://kinetric.co">kinetric.co</a></div>',
    unsafe_allow_html=True,
)
