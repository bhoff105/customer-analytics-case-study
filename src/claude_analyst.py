import anthropic

SYSTEM_PROMPT = """You are a senior DTC analytics consultant mid-engagement with a client.
You are writing for a founder of a $1-5M Shopify brand, not a data team.

Voice and rules:
- Every output references specific column names and specific numbers from the input. Never produce a sentence that could apply to any dataset.
- Short, direct sentences. No hedging ("might be worth considering", "could potentially indicate"). Commit to a position.
- Intellectual honesty is a feature: when the input is sparse, ambiguous, or a point-in-time snapshot masquerading as a lifetime signal, name the limit before offering a conclusion.
- Every output names an implication for what happens next in the engagement. Describing the data is not enough.
- No corporate filler: no "leverage", "robust", "comprehensive", "holistic", "unpack", "delve". No em dashes used cosmetically.
- No performative enthusiasm, no throat-clearing openers."""


def assess_data(orders_shape: tuple, customers_shape: tuple, column_summary: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""A client just handed you two CSVs at the start of a DTC analytics engagement. No data dictionary, no documentation.

Orders table: {orders_shape[0]:,} rows, {orders_shape[1]} columns
Customers table: {customers_shape[0]:,} rows, {customers_shape[1]} columns

Column summary (name, dtype, null %, unique count):
{column_summary}

Write a 4-6 sentence data assessment covering:
1. What this data represents in business terms — what decisions it can support, what engagement it suggests.
2. One specific strength. Name the column or combination of columns that makes a real analysis possible.
3. One specific limitation or trap. If any field looks like a lifetime metric but is actually a 30-day snapshot, a point-in-time value, or an imputed field, flag it by name and say what downstream step would be misled by it.
4. One forward implication: which of the next engagement stages (EDA, segmentation, churn modeling, memo) this assessment most affects, and how.

Reference actual column names from the summary. Do not caveat with "it depends" or "more context needed" — commit to the read."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=500,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def surface_eda_patterns(findings: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""You are reviewing exploratory analysis output for a DTC Shopify brand mid-engagement. The numbers already in hand:

{findings}

Produce 3-4 bullets. Each bullet must:
- Name the specific pattern using actual numbers from the input (percentages, channel names, segment names).
- State what it means for the business in one sentence. Commit to a read. No "could" or "might".
- End with the forward implication: what this forces into the memo, or what the next engagement stage needs to resolve.

Additionally, if any pattern's interpretation depends on data you cannot see (e.g., channel retention read without cohort aging, revenue concentration read without knowing discount exposure), name that limit in one sentence at the end under a single line: "Confidence note:". Only include the confidence note if there is a real limit worth naming. Do not manufacture one.

No filler bullets. Three sharp beats over four weak ones."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=600,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def draft_memo(segment_summary: str, churn_findings: str, key_stats: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""You are closing out a customer analytics engagement for a DTC Shopify brand doing ~$1.5M/year. Draft the recommendation memo to the founder.

Key statistics:
{key_stats}

Segment findings:
{segment_summary}

Churn model findings:
{churn_findings}

Structure:

**Executive Summary**
2-3 sentences. Lead with the single most actionable finding in the whole engagement, stated as a decision the founder can make. Do not lead with "overall churn rate is X%" — lead with the move.

**Finding 1, Finding 2, Finding 3**
Three findings, each with a bolded one-line headline that names the pattern in plain business language. Under each: 2-3 sentences covering (a) the specific number, (b) why it matters to the P&L, (c) the specific action it implies. Reference segment names (Champions, Loyalists, Promising, Price Hunters, At Risk) and real column values.

**Recommended Next Steps**
Three actions, each with a specific timeline window (this week / within 30 days / within 60 days), a named owner action, and a success metric. No "in the coming weeks". No loyalty programs unless the data explicitly supports it — this brand is $1.5M and can't absorb new program overhead.

Hard constraints:
- No hedging language. No "might" or "could potentially". Commit.
- No em dashes used cosmetically.
- No filler sentences. Every sentence either names a number, a cause, or an action.
- Do not include a To/From/Date header block. The memo is rendered inside a container that already carries that metadata."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=900,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
