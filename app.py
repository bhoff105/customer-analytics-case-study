from pathlib import Path
import pandas as pd
import streamlit as st

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
        f'''
        <div class="ai-callout">
          <div class="callout-eyebrow">AI Output</div>
          <div class="callout-body"><p>{text}</p></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def analyst_note(text: str):
    st.markdown(
        f'''
        <div class="analyst-note">
          <div class="callout-eyebrow">Analyst Note</div>
          <div class="callout-body"><p>{text}</p></div>
        </div>
        ''',
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
        This is a complete Kinetric engagement, end to end, for a DTC brand shaped like yours. Every stage shows
        what we actually delivered: what your data says, where AI gets it close but not quite, and the domain
        judgment that turns the analysis into a decision you can execute on Monday.
      </p>
      <div class="hero-block__scenario">
        <strong>Is this shaped like you?</strong>
        If you're running a DTC brand doing roughly $1M to $10M in annual revenue, with strong paid social acquisition
        and inconsistent retention, and your monthly targets keep slipping despite a healthy top of funnel, this
        engagement is shaped like you. The example below is a $1.5M Shopify brand, 18 months in market, who
        handed us two CSVs and one question: "tell us what we're missing."
      </div>
      <div class="hero-block__scenario">
        <strong>What this is not</strong>
        Not a dashboarding project. Not a data engineering build. Not a fractional data team on retainer.
        One focused engagement, one decision-ready memo, on your desk in a matter of weeks.
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Executive Summary (deliverable preview) ──────────────────────────────────
st.markdown(
    '''
<div class="exec-summary">
<div class="exec-summary__eyebrow">Executive summary</div>
<h2 class="exec-summary__headline">Here is what you walk away with after a Kinetric engagement.</h2>
<p class="exec-summary__lede">You hand us two CSV exports and the question you have been sitting on: where is the revenue actually leaking. You walk away with a named segmentation of your customer base, a churn score for every account, a ranked list of the 20 customers your retention lead calls first, and a founder memo that reallocates retention spend toward the group that actually moves your top line. The dollar shape of your opportunity is retention, not acquisition, and this page is the full engagement you receive, end to end.</p>
<div class="exec-summary__grid">
<div class="exec-summary__col">
<div class="exec-summary__col-label">What you hand over</div>
<div class="exec-summary__inputs">
<div class="exec-input"><div class="exec-input__icon">CSV</div><div class="exec-input__meta"><div class="exec-input__name">orders.csv</div><div class="exec-input__detail">18 months of transactions. Order dates, revenue, product category, discount flags. No documentation.</div></div></div>
<div class="exec-input"><div class="exec-input__icon">CSV</div><div class="exec-input__meta"><div class="exec-input__name">customers.csv</div><div class="exec-input__detail">Customer file with acquisition source, email engagement, lifetime totals. No data dictionary.</div></div></div>
</div>
</div>
<div class="exec-summary__col">
<div class="exec-summary__col-label">What you receive</div>
<div class="exec-summary__outputs">
<div class="exec-output"><div class="exec-output__num">01</div><div class="exec-output__name">A segmented customer base</div><div class="exec-output__detail">Your customers grouped into five named segments (Champions, Loyalists, Promising, Price Hunters, At Risk) with a plain-English playbook for each.</div></div>
<div class="exec-output"><div class="exec-output__num">02</div><div class="exec-output__name">A churn-risk score on every customer</div><div class="exec-output__detail">A 0 to 100 score on every account in your file, ranked by intervention priority so your retention spend goes where it earns.</div></div>
<div class="exec-output"><div class="exec-output__num">03</div><div class="exec-output__name">A prioritized intervention list</div><div class="exec-output__detail">The 20 customers your retention lead works first, ranked by LTV times churn probability, with a specific recommended action on every row.</div></div>
<div class="exec-output"><div class="exec-output__num">04</div><div class="exec-output__name">A one-page founder memo</div><div class="exec-output__detail">Three findings, three numbers, three actions with specific timelines. Written for you to forward to your retention lead without editing.</div></div>
</div>
</div>
</div>
<div class="exec-summary__use">
<div class="exec-summary__use-label">What you do with them</div>
<p class="exec-summary__use-body">On Monday morning, your retention lead pulls the 20-name intervention list and owns it end to end. By Friday, they have launched a personalized three-touch win-back sequence against your At Risk cohort, targeting 15 to 20% reactivation inside 30 days. Inside that same 30-day window, you reallocate 30% of your paid social win-back budget into email and organic cohort re-engagement, then measure 60-day repeat rate against baseline before making the change permanent. Within 60 days, you ship a Gifts-entry post-purchase sequence that cross-sells into Home &amp; Kitchen and Beauty &amp; Wellness, lifting 30-day second-order rate on Gifts-first buyers from 17% toward 40%. From that point forward, your segmentation and churn scores refresh monthly and drive every retention decision you make. No new tools. No new hires.</p>
</div>
<div class="exec-summary__use">
<div class="exec-summary__use-label">Cost of waiting a quarter</div>
<p class="exec-summary__use-body">Your At Risk 20 keep slipping further from recoverable every week no one works the list. Your paid social cohort keeps spending as it always has, acquiring the same low-retention customers on top of a churning base. Your Gifts-first buyers keep arriving and exiting without a bridge into the categories that would have kept them. The math of the opportunity gets worse, not better, while the outputs above sit unused.</p>
</div>
</div>
    ''',
    unsafe_allow_html=True,
)

# ── Stage 0: The Brief ───────────────────────────────────────────────────────
stage_marker("Stage 0 of 5", "The Brief")
st.markdown("## The Brief")
st.markdown("""
You hand us two CSVs and a vague worry. Before we touch the data, we pin down what you are actually asking and what shape an answer has to take to be useful to you. Everything downstream, the segmentation in Stage 3, the model in Stage 4, the memo in Stage 5, traces back to what we agreed to here.

**Your business:** A DTC Shopify brand doing approximately $1.5M in annual revenue. 18 months in market. Strong paid social acquisition. Inconsistent retention.

**What you told us:** *"We know something's off with our customer mix. Our numbers look fine on the surface but we keep missing monthly targets. We've exported everything we have. Can you tell us what we're missing?"*

**What you handed over:** Two CSV exports. No documentation, no data dictionary, no context.
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
    "Before we build anything, we read your schema like a domain expert. Which of your fields are trustworthy at face "
    "value, which are point-in-time snapshots masquerading as lifetime signals, and where are the gaps that will force "
    "a caveat later in your engagement. Below is the AI's read of your two tables, then the correction we made. The "
    "correction matters because one of the trustworthy-looking fields, `email_opens_30d`, will otherwise quietly bias "
    "your churn model in Stage 4 against every customer you acquired in the last two months."
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

ai_callout(
    "The two tables match a standard Shopify DTC export: an <code>orders</code> fact table keyed on "
    "<code>order_id</code> with <code>customer_id</code>, <code>order_date</code>, <code>revenue</code>, "
    "<code>product_category</code>, and a discount flag, joined to a <code>customers</code> dimension carrying "
    "<code>acquisition_source</code>, lifetime totals, and recent email engagement. Null rates are low across both "
    "tables and the 18-month date range is long enough to capture a full holiday cycle plus two shoulder quarters, "
    "which makes this schema well suited to retention analysis (RFM segmentation, churn modeling, cohort comparison). "
    "The <code>email_opens_30d</code> and <code>email_clicks_30d</code> fields are strong engagement features and "
    "should rank among the top predictors in the churn model. Recommended next step: proceed to EDA with the full "
    "feature set and treat email engagement as a core signal."
)
analyst_note(
    "Claude read your schema correctly as a standard Shopify order-and-customer export, and it framed your business "
    "context accurately: DTC, transactional, suited to retention work. We kept that framing. "
    "What it missed is a trap in your engagement columns. <code>email_opens_30d</code> and <code>email_clicks_30d</code> "
    "are a 30-day snapshot, not a lifetime signal. A customer you acquired last week has had almost nothing to open, "
    "so feeding those fields into your churn model unfiltered will punish your newest customers for not having had "
    "time to engage. The fix is to restrict model training to customers with 60+ days of tenure, then score your full "
    "population. That single correction changes who ends up on your intervention list in Stage 4, which is the only "
    "output you are actually going to act on. Worth naming now so nothing downstream has to be redone."
)

section_divider()

# ── Stage 2: Exploratory Analysis ────────────────────────────────────────────
stage_marker("Stage 2 of 5", "Exploratory Analysis")
st.markdown("## Exploratory Analysis")
st.markdown(
    "Four questions drive this stage, all about your business: where your revenue actually comes from, which of your "
    "acquisition channels produce customers worth keeping, what your revenue trend looks like once seasonality is "
    "stripped out, and which product entry points predict a second purchase. The four charts below answer each in "
    "turn. Three findings are strong enough to anchor your final memo, and by the end of this stage the thread "
    "connecting them (one underlying customer segment driving all three) is visible without us labeling it yet."
)

st.markdown("### Top 20% of Customers Drive 53% of Revenue")
st.plotly_chart(revenue_concentration_chart(customers), use_container_width=True, config=PLOTLY_CFG)

pct_top20 = customers.nlargest(int(len(customers) * 0.2), "total_revenue")["total_revenue"].sum()
pct_top20_share = pct_top20 / customers["total_revenue"].sum() * 100
analyst_note(
    f"Your top 20% of customers account for <strong>{pct_top20_share:.0f}%</strong> of your total revenue. "
    "A 50/20 split isn't unusual for DTC, so the shape of your curve is consistent with a healthy repeat-buyer base. "
    "That's the part worth keeping. "
    "What the chart doesn't say on its own is what you are doing with that information today. "
    "Today you run a single re-engagement campaign across your entire file: a $900 LTV customer and a $45 "
    "one-and-done buyer receive the same email, the same discount, the same cadence. "
    "The logic here is straightforward: your retention spend should scale with expected future revenue, not with list size. "
    "For your engagement, this becomes the first input to the segmentation in Stage 3 and the intervention "
    "priority ranking in Stage 4. Your memo recommends a reallocation of what you already spend, not a new program."
)

st.markdown("### Paid Social Buys 52% of Customers and Keeps 22% of Them")
st.plotly_chart(acquisition_retention_chart(customers), use_container_width=True, config=PLOTLY_CFG)

ai_callout(
    "Three patterns stand out in your EDA. First, <strong>revenue concentration</strong>: your top 20% of customers "
    "account for roughly 53% of total revenue, a healthy Pareto shape for DTC. Second, <strong>channel inversion</strong>: "
    "<code>paid_social</code> is your largest acquisition channel at about 52% of customer volume but retains only around "
    "22% at 12 months, while <code>email</code> and <code>organic</code> acquire fewer customers and retain them near 70% "
    "and 59% respectively. Third, <strong>category gap</strong>: customers whose first purchase is in the <code>Gifts</code> "
    "category repeat at roughly 17% versus 71% for every other category, a materially lower rate. Each pattern is worth "
    "investigating further as a potential driver of the retention softness you flagged in your brief."
)
analyst_note(
    "Claude surfaced your volume/retention inversion correctly: paid social is your largest channel by customer count "
    "and your worst by retention (roughly 22% retained vs. 70% for email and 59% for organic). That framing is right "
    "and it lands in your memo verbatim. "
    "Where Claude stopped short: it treated this as a channel efficiency story. It isn't, or rather it isn't only that. "
    "Your paid social cohort skews heavily toward discount-driven first purchases and concentrates in the Gifts "
    "category, which means your channel issue is the same population as the Price Hunter segment we'll isolate in "
    "Stage 3 and the Gifts retention trough we'll see in the next chart. Three findings, one underlying customer. "
    "For your engagement, the implication is that pulling paid social spend without fixing the downstream experience "
    "trades one problem for another. Your memo recommends both moves together."
)

st.markdown("### Outside the Holiday Window, Revenue Is Flat")
st.plotly_chart(monthly_revenue_chart(orders), use_container_width=True, config=PLOTLY_CFG)
analyst_note(
    "Your Nov/Dec peak is the expected gifting-season lift, not a signal of underlying growth, so stripping it out is "
    "the honest way to read this chart. With the peak removed, your monthly revenue is flat to modestly declining across "
    "the 18-month window. That reading is accurate as far as it goes. "
    "What the chart doesn't show is the compounding effect a healthy retention engine should produce. "
    "Eighteen months into a DTC brand, a working retention motion should be lifting your monthly floor as your customer "
    "base accumulates, even in off-peak months. "
    "You are seeing the opposite pattern, which means your new acquisitions are replacing lapsed customers rather than "
    "stacking on top of them. That's the definition of a retention problem showing up in your top line, and it sets up "
    "the At Risk segment work in Stage 3 as the highest-return intervention in your engagement."
)

st.markdown("### Gifts Buyers Don't Come Back, Everyone Else Does")
st.plotly_chart(category_performance_chart(customers, orders), use_container_width=True, config=PLOTLY_CFG)
analyst_note(
    "Your customers whose first purchase is in the Gifts category repeat at roughly 17%, versus near 71% across every "
    "other category. That's a structural gap, not a rounding difference, and it's the single cleanest finding in your "
    "EDA. The chart on its own could be read as 'Gifts customers are low-value,' which would be the wrong conclusion. "
    "AOV on your Gifts cohort is in line with the rest of your file, so these customers are spending when they convert. "
    "The problem is that a gift purchase creates no natural reason to return. Without a post-purchase flow that bridges "
    "into a category the buyer actually uses themselves, the relationship ends at the first transaction. "
    "For your memo, this becomes Finding 3: the fix isn't a discount or a new product line, it's a category-specific "
    "email sequence introducing your two highest-retention categories to every Gifts-first buyer within 14 days of their order."
)

section_divider()

# ── Stage 3: Segmentation ────────────────────────────────────────────────────
stage_marker("Stage 3 of 5", "Customer Segmentation")
st.markdown("## Customer Segmentation")
st.markdown(
    "The segmentation gives you five named groups to talk to, each with a different retention action: who to protect, "
    "who to grow, who to win back, who to stop subsidizing. The underlying method is RFM (recency, frequency, monetary) "
    "scoring run through K-means to land on five clusters, then relabeled from cluster numbers into business language "
    "your retention lead can use without translation. Below are the five segment profiles as we delivered them to you, "
    "followed by the scatter that shows how your segments separate in the data."
)

with st.spinner("Running segmentation..."):
    segmented = run_segmentation(customers)

summary_df = segment_summary(segmented)
st.markdown("### Segment Profiles")

# Segment profile card data — one row per segment, written for a founder
_SEGMENT_META = {
    "Champions": {
        "description": "Your highest-value customers, active in the last 30 days, buying at or near full price. The engine of your business.",
        "strategy": "Invite them into a referral program. Exclude them from discount promotions entirely. Surface new arrivals and early access as your engagement levers.",
        "color": "#2DD4BF",
    },
    "Loyalists": {
        "description": "Your consistent repeat buyers. Strong order frequency, not as recent as Champions, but reliably active across the year.",
        "strategy": "Maintain cadence with product-led content. Do not lead with discounts. These are the candidates for a loyalty tier once your brand has the scale to support one.",
        "color": "#22C55E",
    },
    "Promising": {
        "description": "Your newer customers, one or two purchases in. High potential, not yet habit. Concentrated in single-category purchases including Gifts.",
        "strategy": "Send them a second-purchase sequence within 14 days of first order. Cross-sell into your highest-retention categories. Your 90-day window is the conversion window. Don't let it close.",
        "color": "#A78BFA",
    },
    "Price Hunters": {
        "description": "Repeat buyers whose purchases are heavily discount-driven. Their churn rate runs roughly 3x your Champions. Every discount you send reinforces the pattern.",
        "strategy": "Do not offer them further discounts. Send educational and brand-story content to shift the relationship. Measure 60-day repeat rate without a coupon before you spend another dollar here.",
        "color": "#60A5FA",
    },
    "At Risk": {
        "description": "Your previously high-LTV buyers who have gone quiet in the last 60 to 150 days. Not churned yet. The highest-return intervention target in your engagement.",
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
    "The clustering algorithm returned 5 groups in your file, differentiated primarily by order frequency and recency. "
    "Initial labels: Cluster 0 (high frequency, recent), Cluster 1 (moderate frequency, moderate recency), "
    "Cluster 2 (low frequency, recent acquisition), Cluster 3 (moderate frequency, high discount usage), "
    "Cluster 4 (previously high value, now lapsed). RFM scores range from 3 to 15."
)
analyst_note(
    "Claude got the math right: your file separated cleanly into five clusters on RFM, and the centroid descriptions "
    "match what's actually there. The issue is that 'Cluster 3' and 'Cluster 4' are not segments you can act on. "
    "We relabeled them into business language and added discount_rate as a defining dimension for Cluster 3, which "
    "becomes your Price Hunters. On pure RFM they look like decent customers. Their churn rate, however, runs roughly "
    "3x your Champions, and almost every repeat purchase they make carries a discount. Treating them as Loyalists in "
    "a retention campaign is a negative-margin move for you. "
    "Cluster 4 becomes At Risk, which is where your money is in this engagement: these are previously high-LTV "
    "customers who have gone quiet, not customers who were never worth keeping. "
    "Your Stage 4 intervention list leans hard on this segment because it's your winnable group, and your Stage 5 "
    "memo leads with it for the same reason."
)

st.markdown("### Champions and At Risk Sit on Opposite Ends of the Same Curve")
st.markdown(
    '<div class="chart-explainer-wrap">',
    unsafe_allow_html=True,
)
st.plotly_chart(segment_scatter(segmented), use_container_width=True, config=PLOTLY_CFG)
analyst_note(
    "We shaped your segmentation around RFM, not a fancier clustering exercise, for one reason: actionability. "
    "RFM uses three fields you already understand (how recently they bought, how often, and how much), and the output "
    "is a handful of named groups your team can act on without a data scientist in the room. A higher-dimensional "
    "cluster analysis would produce statistically tighter groupings that nobody on your team could translate into a "
    "Monday morning decision. Read the scatter left to right on recency and bottom to top on frequency, with bubble "
    "size encoding lifetime revenue: your <strong>Champions</strong> cluster tight in the upper-left corner (bought "
    "recently, bought often, highest LTV bubbles), your <strong>Loyalists</strong> sit just below them (consistent "
    "frequency, still recent), your <strong>Promising</strong> hug the left edge low on the frequency axis (recent "
    "first purchase, not yet habit), your <strong>Price Hunters</strong> look mid-pack on RFM but carry a distinct "
    "discount signature underneath the dots, and your <strong>At Risk</strong> drift to the right (previously active, "
    "now quiet, still carrying large revenue bubbles). "
    "The shape that matters most is the distance between your Champions and your At Risk: those are the same kind of "
    "customer at two different moments in their lifecycle, and the gap between them is the recoverable revenue this "
    "engagement is built around. "
    "These named labels beat a raw customer-to-cluster mapping because they give your ops team a shared vocabulary. "
    "Your retention lead and your founder can argue about what to do with At Risk customers without ever looking at a "
    "cluster ID. The labels feed directly into Stage 4, where your intervention list ranks customers by segment plus "
    "churn probability, and into Stage 5, where your final memo leads with the At Risk cohort as the first Monday "
    "morning move."
)
st.markdown('</div>', unsafe_allow_html=True)

section_divider()

# ── Stage 4: Predictive Modeling ─────────────────────────────────────────────
stage_marker("Stage 4 of 5", "Churn Prediction Model")
st.markdown("## Churn Prediction Model")
st.markdown(
    "The model turns your segmentation into a ranked list. A Random Forest scores every customer in your file on "
    "churn probability, then multiplies that probability by lifetime revenue to produce an intervention priority, the "
    "20 names you should call first. The two charts below show which features the model weighed most heavily and how "
    "your resulting risk scores distribute across the file. The operational output you actually use is the intervention "
    "table further down, with a specific recommended action on every row."
)

with st.spinner("Training model..."):
    model, scored = get_model_and_scores(id(customers))

col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(feature_importance_chart(model, FEATURES), use_container_width=True, config=PLOTLY_CFG)
with col_r:
    st.plotly_chart(score_distribution_chart(scored), use_container_width=True, config=PLOTLY_CFG)

ai_callout(
    "The model weighted email engagement (opens and clicks) as the top predictive feature in your file, followed by "
    "<code>days_since_last_order</code> and <code>total_orders</code>. Discount rate ranked 4th: higher discount usage "
    "correlates with higher churn probability. Model accuracy on the hold-out set: approximately 78%."
)
analyst_note(
    "The feature ranking is defensible, and flagging discount_rate as a churn signal is the right call for your file: "
    "price-sensitive customers churn faster, which is consistent with the Price Hunter profile we named in Stage 3. "
    "The correction is the one we previewed in Stage 1. Email engagement as the top feature is true on this training "
    "set, but it's partially an artifact of the 30-day window in <code>email_opens_30d</code> and "
    "<code>email_clicks_30d</code>. Your newer customers haven't had time to rack up opens, so the model reads them as "
    "disengaged when they're simply new. We restricted training to customers with at least 60 days of tenure, then "
    "applied the fitted model to your full population. That stabilizes the ranking and stops the model from mislabeling "
    "last month's acquisitions as your churn risks. "
    "The 78% accuracy figure is directionally fine, but don't let it become the headline. The operational value for you "
    "is the rank order, which is what your intervention list below is actually built from."
)

st.markdown("### Where to spend your next retention dollar: your top 20 customers")
st.markdown(
    "Ranked by **LTV x churn probability**. These are the 20 customers where a successful intervention delivers the "
    "highest expected revenue impact for you. Each row includes a specific recommended action derived from that "
    "customer's segment, recency, category history, and discount behavior."
)
_ilist = intervention_list(scored)
st.markdown('<div class="intervention-table">', unsafe_allow_html=True)
st.dataframe(
    _ilist,
    use_container_width=True,
    hide_index=False,
    height=520,
    column_config={
        "Customer ID": st.column_config.TextColumn("Customer ID", width="small"),
        "Email": st.column_config.TextColumn("Email", width="medium"),
        "Segment": st.column_config.TextColumn("Segment", width="small"),
        "Lifetime Revenue": st.column_config.TextColumn("Lifetime Revenue", width="small"),
        "Orders": st.column_config.NumberColumn("Orders", width="small", format="%d"),
        "Avg Order Value": st.column_config.TextColumn("AOV", width="small"),
        "Days Since Purchase": st.column_config.NumberColumn("Days Since", width="small", format="%d"),
        "Churn Score": st.column_config.ProgressColumn(
            "Churn Score",
            width="small",
            format="%.0f",
            min_value=0,
            max_value=100,
        ),
        "Recommended Action": st.column_config.TextColumn(
            "Recommended Action",
            width="large",
        ),
    },
)
st.markdown('</div>', unsafe_allow_html=True)
analyst_note(
    "This table is the operational deliverable of your engagement. Your Champions are excluded from the top of the "
    "list by design: their churn probability is low enough that intervention dollars spent here have a low expected "
    "return, even with their high LTV. The list is dominated by At Risk customers, which confirms your segmentation: "
    "these are your high-LTV buyers who have gone quiet, not customers who were never going to stick. "
    "The recommended action on each row is a three-touch, 21-day personalized win-back sequence keyed to that customer's "
    "purchase history. No blanket discount. A discount on this cohort teaches exactly the wrong behavior and converts "
    "your highest-value buyers into Price Hunters. "
    "Success looks like a 15 to 20% reactivation rate on this list inside 30 days, which on your At Risk LTV base is "
    "the single largest revenue lever available to you this quarter."
)

section_divider()

# ── Stage 5: Recommendation Memo ─────────────────────────────────────────────
stage_marker("Stage 5 of 5", "Recommendation Memo")
st.markdown("## Recommendation Memo")
st.markdown(
    "The memo is the deliverable you keep. It consolidates the entire engagement into three findings, three numbers, "
    "and three actions with specific timelines, written for you to forward to your retention lead without editing a "
    "word. Below, in order: the unedited AI first draft (collapsed), the three edits we made to that draft, and the "
    "final memo as delivered to you."
)

with st.expander("AI first draft (unedited), expand to view", expanded=False):
    ai_callout(
        "<strong>Executive Summary</strong></p>"
        "<p>The customer file shows an overall churn rate of roughly 38% across the 18-month window, which is a meaningful "
        "risk to revenue stability. Analysis of the segmentation and predictive model points to three findings worth "
        "addressing in the coming weeks.</p>"
        "<p><strong>Finding 1: Overall churn rate is elevated.</strong></p>"
        "<p>Approximately 38% of the customer base is flagged as churned under the current definition. This rate is above "
        "typical DTC benchmarks and suggests retention work should be a priority going forward. Reducing this rate by even "
        "a few percentage points would have a meaningful effect on lifetime value across the customer base.</p>"
        "<p><strong>Finding 2: Channel mix shows retention variance.</strong></p>"
        "<p>Paid social is the largest acquisition channel but shows the lowest retention, while email and organic show "
        "materially higher retention rates. This suggests the current mix could be reconsidered, and budget could be "
        "rebalanced in favor of the higher-retaining channels in the coming weeks.</p>"
        "<p><strong>Finding 3: A subset of customers concentrated in the Gifts category repeats at a materially lower rate.</strong></p>"
        "<p>Customers whose first purchase is in Gifts repeat at roughly 17% versus 71% for other categories. This gap is "
        "large enough to warrant a targeted intervention.</p>"
        "<p><strong>Recommended Actions</strong></p>"
        "<p>1. Launch a win-back campaign targeting lapsed customers in the coming weeks to address the elevated churn rate. "
        "Consider a discount offer as an incentive to re-engage. 2. Stand up a loyalty program to reward repeat purchasers "
        "and improve retention across the customer base over time. A points-based or tiered structure is worth evaluating. "
        "3. Develop a post-purchase email sequence for Gifts-first buyers soon, introducing them to other product "
        "categories to encourage repeat purchase."
    )

analyst_note(
    "Three edits we made to the first draft before it reached you, each worth naming. "
    "First, the draft led with your overall churn rate (38%). That's a description, not a decision. We reordered to "
    "lead with your At Risk win-back, because the winnable group inside that rate is the action you can take on Monday, "
    "not the headline number itself. "
    "Second, Claude recommended a loyalty program as Action 2. We cut it. Loyalty programs solve a different problem, "
    "add operational overhead your business can't absorb at this stage, and distract from the three moves that actually "
    "fix your retention economics. Reallocating your paid social win-back budget is the higher-leverage Action 2. "
    "Third, the timelines read as 'in the coming weeks,' which is not a timeline. Every vague phrase got replaced with "
    "a specific window (this week, 30 days, 60 days) tied to a specific owner action you can assign. "
    "The final memo below is the version we delivered to you."
)

st.markdown("### Final memo (as delivered)")
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

# Render memo inline as a document-styled HTML block — the deliverable
# rendered as if it were the printed page, inside the app.
st.markdown(
    '<div class="memo-container"><div class="memo-header">'
    '<div class="memo-header__brand"><span>Kinetric</span>'
    '<span class="memo-header__brand-tag">Advisory Memorandum</span></div>'
    '<div class="memo-header__label">To</div>'
    '<div class="memo-header__value">[Client Name], Founder</div>'
    '<div class="memo-header__label">From</div>'
    '<div class="memo-header__value">Brendan Hoffman, Kinetric</div>'
    '<div class="memo-header__label">Date</div>'
    '<div class="memo-header__value">Engagement Closeout</div>'
    '<div class="memo-header__label">Re</div>'
    '<div class="memo-header__value">Customer Analytics Findings &amp; Recommendations</div>'
    '</div><div class="memo-body">',
    unsafe_allow_html=True,
)
st.markdown(memo_body_md)
st.markdown(
    '</div>'
    '<div class="memo-footer">Kinetric · kinetric.co · Prepared for client use</div>'
    '</div>',
    unsafe_allow_html=True,
)

section_divider()

# ── Close: the sales argument ────────────────────────────────────────────────
# Computed figures for the leave-behind, driven by the same data the memo uses.
at_risk_count = int((scored["segment"] == "At Risk").sum())
at_risk_ltv = float(scored.loc[scored["segment"] == "At Risk", "total_revenue"].sum())
top20_share_int = int(round(pct_top20_share))

st.markdown(
    f'''
    <div class="close-section">
      <div class="close-section__eyebrow">If this is your data</div>
      <h2 class="close-section__title">Here is what to do with it</h2>

      <div class="close-beat">
        <div class="close-beat__label">You have now seen the whole engagement</div>
        <p class="close-beat__body">
          Two CSVs in, a named segmentation, a churn score on every account, a 20-name intervention list,
          and a one-page memo out. If that maps to a decision sitting on your desk this quarter, the rest of
          this section is how you put it on your calendar.
        </p>
      </div>

      <div class="close-beat">
        <div class="close-beat__label">What lands on your desk</div>
        <p class="close-beat__body close-beat__body--lede">
          Four deliverables. Each one a working tool your retention lead picks up Monday morning and operates
          without a translation layer.
        </p>
        <ul class="close-beat__list">
          <li><strong>A segmented customer base.</strong> Five named segments (Champions, Loyalists, Promising, Price Hunters, At Risk) with a plain-English playbook per segment covering message, offer, cadence, and the outcome you are paying for.</li>
          <li><strong>A churn-risk score for every customer.</strong> 0 to 100, trained on your own repeat behavior, ranked by intervention priority so your retention spend lands on the accounts where it earns and leaves the rest alone.</li>
          <li><strong>A prioritized win-back list.</strong> {at_risk_count} At Risk customers in your file, {top20_share_int}% of revenue concentrated at the top, specific names and emails for your retention lead to work this week, a recommended action per customer, an expected reactivation rate to measure against.</li>
          <li><strong>A one-page founder memo.</strong> Three findings, three numbers, three actions with owner-named timelines. Forward-ready. You send it to your retention lead without editing.</li>
        </ul>
      </div>

      <div class="close-beat">
        <div class="close-beat__label">Two ways to run it. Pick the one that describes you.</div>
        <div class="close-paths">
          <div class="close-path">
            <div class="close-path__tag">Option A · $1,500 fixed</div>
            <h3 class="close-path__name">One-time audit</h3>
            <p class="close-path__body">
              You get every output above as a single fixed-scope engagement, roughly four weeks from kickoff
              to handoff. You own the segment playbooks, the scoring model, and the intervention list from
              day one and operate them yourselves from that point forward. Refreshes happen when you want
              them to happen, at whatever cadence you decide.
            </p>
            <div class="close-path__fit-label">You are a fit if</div>
            <ul class="close-path__fit-list">
              <li>You have a retention or lifecycle lead on payroll already</li>
              <li>You run a Klaviyo or Braze instance and someone is accountable for it</li>
              <li>You want the framework installed once, not an ongoing vendor relationship</li>
              <li>Your data stack is stable enough that monthly refreshes are not the gating constraint</li>
            </ul>
          </div>
          <div class="close-path">
            <div class="close-path__tag">Option B · $2,000 to $3,500 / month</div>
            <h3 class="close-path__name">Ongoing retainer</h3>
            <p class="close-path__body">
              Kinetric refreshes your segmentation and churn scores monthly, owns your At Risk intervention
              tracking end to end, and expands the analysis as new data sources come online. Klaviyo flows
              and open rates month two. Paid media attribution month three. Review sentiment and post-purchase
              survey signal after that. You get a monthly memo; your retention lead gets a working list
              every Monday.
            </p>
            <div class="close-path__fit-label">You are a fit if</div>
            <ul class="close-path__fit-list">
              <li>Analytics capacity is thin and you are the de facto analyst today</li>
              <li>You have the data but no one whose job is to turn it into a weekly decision</li>
              <li>You want the outputs maintained, not just delivered once and left to rot</li>
              <li>You would rather expand coverage over time than scope a second engagement from scratch</li>
            </ul>
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
      <h3 class="leavebehind__title">One page. Three numbers. Three moves.</h3>

      <div class="leavebehind__numbers">
        <div class="leavebehind__num">
          <div class="leavebehind__num-value">{top20_share_int}%</div>
          <div class="leavebehind__num-label">of your revenue comes from the top 20% of your customers. Your retention spend is currently flat across the file.</div>
        </div>
        <div class="leavebehind__num">
          <div class="leavebehind__num-value">${at_risk_ltv/1000:,.0f}K</div>
          <div class="leavebehind__num-label">of lifetime revenue sitting in your {at_risk_count} At Risk customers who have gone quiet.</div>
        </div>
        <div class="leavebehind__num">
          <div class="leavebehind__num-value">2.2×</div>
          <div class="leavebehind__num-label">retention gap between your paid social (22%) and email/organic cohorts. Subsidized by your current spend mix.</div>
        </div>
      </div>

      <div class="leavebehind__actions-title">Three moves, sequenced. The same plan the memo lands on.</div>
      <ul class="leavebehind__actions">
        <li><span class="when">This week, by Friday</span><span>Pull the 20-name At Risk intervention list. Assign it to one owner. Launch a personalized three-touch win-back sequence, no blanket discount. Target 15 to 20% reactivation inside 30 days.</span></li>
        <li><span class="when">Within 30 days</span><span>Reallocate 30% of paid social win-back budget into email and organic cohort re-engagement. Measure 60-day repeat rate against baseline before making the shift permanent.</span></li>
        <li><span class="when">Within 60 days</span><span>Ship the Gifts-entry post-purchase sequence. Three emails over 14 days, cross-selling into Home &amp; Kitchen and Beauty &amp; Wellness. Lift 30-day second-order rate on Gifts-first buyers from 17% toward 40%.</span></li>
      </ul>

      <div class="leavebehind__contact">
        <strong>Two ways to start.</strong>
        <a href="https://calendly.com/brendan-kinetric/30min">Book a 30-minute working call →</a>
        or read more at <a href="https://kinetric.co">kinetric.co</a> · <a href="mailto:brendan@kinetric.co">brendan@kinetric.co</a>
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
        <span class="soft-cta__prompt">Ready to put this engagement on your calendar?</span>
        <span class="soft-cta__target">Book 30 minutes to scope it → calendly.com/brendan-kinetric</span>
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
