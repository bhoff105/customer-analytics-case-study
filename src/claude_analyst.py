import anthropic

SYSTEM_PROMPT = """You are a senior DTC analytics consultant assisting with a client engagement.
You produce clear, specific, business-focused outputs — written for a founder, not a data team.
Never use generic statements. Always reference specific data patterns, column names, or numbers provided."""


def assess_data(orders_shape: tuple, customers_shape: tuple, column_summary: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""A client has shared two data tables with you at the start of an analytics engagement.

Orders table: {orders_shape[0]:,} rows, {orders_shape[1]} columns
Customers table: {customers_shape[0]:,} rows, {customers_shape[1]} columns

Column summary:
{column_summary}

Write a brief data assessment (3–4 sentences) covering:
1. What this data represents and what business context it captures
2. One specific strength of the dataset — what it will allow you to do well
3. One specific limitation or gap — what it cannot tell you without additional data

Be specific. Reference actual column names."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=400,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def surface_eda_patterns(findings: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""You are reviewing exploratory analysis results for a DTC Shopify brand. Here are the key findings:

{findings}

Write 3–4 bullet points identifying the most important patterns in this data.
For each: name the pattern, explain what it might mean for the business, and suggest one follow-up question it raises.
Be specific — reference the actual numbers provided."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=500,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def draft_memo(segment_summary: str, churn_findings: str, key_stats: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""You have completed a customer analytics engagement for a DTC Shopify brand doing ~$1.5M/year in revenue.

Key statistics:
{key_stats}

Segment findings:
{segment_summary}

Churn model findings:
{churn_findings}

Write a recommendation memo addressed to the founder. Structure:
1. Executive Summary (2–3 sentences — what we found and what it means)
2. Three specific findings (each with a number, a so-what, and one action)
3. Recommended next steps (prioritized list of 3 actions with a timeframe each)

Write as a trusted advisor. Be direct. No filler sentences."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=800,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
