# Customer Analytics Case Study

## The Problem
A DTC Shopify brand doing ~$1.5M/year handed over two CSV exports and said: *"We know something's off with our customer mix — can you tell us what we're missing?"* No documentation, no data dictionary, no context.

## The Hypothesis
The right first move isn't to clean the data or build a dashboard. It's to follow a structured engagement arc: understand what the data captures, surface what it shows, segment the customer base, score for risk, and produce a recommendation the client can act on this week — not a report they'll file away.

## The Approach
A five-stage analytics engagement, built as an interactive walkthrough. Each stage shows two things: what AI surfaced or generated, and what the analyst directed, validated, or corrected. The dual-voice format makes the collaboration visible — and makes clear where domain expertise is doing the work that AI cannot.

Three decisions shaped the design:
1. **Fixed data, not a generic tool** — the walkthrough uses a single synthetic dataset with realistic DTC patterns baked in. The analysis tells a coherent story, not a generic one.
2. **Analyst notes are the point** — the AI callouts show what Claude produced. The analyst notes show what changed, why, and what domain knowledge informed the correction. That's where the expertise lives.
3. **Output that a client would actually receive** — the final memo is written the way you'd write it after a real engagement: specific findings, specific actions, specific timelines.

## What It Produces
- **Stage 0:** The brief — raw data, no documentation
- **Stage 1:** Data assessment — schema profiling, quality flags, AI assessment + analyst correction
- **Stage 2:** Exploratory analysis — revenue concentration, acquisition/retention mismatch, seasonal trends, category performance
- **Stage 3:** Customer segmentation — RFM scoring, K-means clustering, five named segments with distinct strategies
- **Stage 4:** Churn prediction — Random Forest model, feature importance, score distribution, intervention priority list
- **Stage 5:** Recommendation memo — AI first draft, analyst edits documented, final client-ready memo

## The Outcome
The At Risk segment — high-value customers who've gone quiet — represents the highest-leverage intervention available. The acquisition channel mismatch explains why retention economics are broken. One product category is structurally breaking the repeat purchase funnel. All three findings are specific, actionable, and grounded in the data.

---

## Setup

```bash
git clone https://github.com/bhoff105/customer-analytics-case-study
cd customer-analytics-case-study
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
python generate_data.py
streamlit run app.py
```

The app runs fully without an API key — AI-powered steps (data assessment, EDA pattern surfacing, memo generation) are triggered on-demand via buttons. Everything else runs on pre-computed outputs.

## Tech Stack
Python · Streamlit · pandas · scikit-learn · Plotly · Anthropic Claude API
