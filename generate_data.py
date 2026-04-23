"""
Run once to generate data/orders.csv and data/customers.csv.
Patterns baked in:
  - Top 20% of customers = ~52% of revenue
  - Email/organic retention 2x paid social at 12 months
  - One product category with 40% lower repeat rate (one-and-done)
  - Price-sensitive segment churns 3x faster than full-price buyers
  - A sleeping giant segment: high AOV, low frequency, lapsed 60+ days
"""
import numpy as np
import pandas as pd

np.random.seed(7)

N_CUSTOMERS = 280
N_ORDERS = 800
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-06-30")
CATEGORIES = ["Home & Kitchen", "Apparel", "Beauty & Wellness", "Fitness", "Gifts"]
SOURCES = ["paid_social", "organic_search", "email", "referral", "direct"]
STATES = ["NY", "CA", "TX", "FL", "IL", "WA", "MA", "CO", "GA", "OH"]

# Subcategories per top-level category — used to populate product_subcategory on orders
SUBCATEGORIES = {
    "Home & Kitchen":    ["Cookware", "Bar & Entertaining", "Linens", "Small Appliances"],
    "Apparel":           ["Tops", "Bottoms", "Outerwear", "Accessories"],
    "Beauty & Wellness": ["Skincare", "Haircare", "Supplements", "Bath & Body"],
    "Fitness":           ["Equipment", "Apparel", "Recovery", "Nutrition"],
    "Gifts":             ["Gift Sets", "Seasonal", "Personalized", "Bundles"],
}

# Email generation pool
_FIRST_INITIALS = list("abcdefghjklmnoprstw")
_LAST_NAMES = [
    "rodriguez", "thompson", "patel", "chen", "nguyen", "kim", "johnson",
    "williams", "martinez", "taylor", "anderson", "harris", "jackson",
    "lee", "garcia", "white", "robinson", "clark", "lewis", "walker",
    "hall", "young", "allen", "scott", "wright", "green", "baker",
    "adams", "nelson", "hill", "hernandez", "moore", "martin", "perez",
]
_DOMAINS = ["gmail.com", "email.com", "outlook.com", "yahoo.com", "icloud.com"]
_EMAIL_SUFFIXES = ["", "nyc", "la", "chi", "pdx", "atl", "bos", "sea", "den"]

_used_emails: set = set()

def _make_email() -> str:
    """Generate a realistic-looking anonymized email (e.g. m.rodriguez47@email.com)."""
    for _ in range(50):
        init = np.random.choice(_FIRST_INITIALS)
        last = np.random.choice(_LAST_NAMES)
        suffix = np.random.choice(_EMAIL_SUFFIXES)
        domain = np.random.choice(_DOMAINS)
        num = "" if np.random.random() < 0.45 else str(np.random.randint(10, 99))
        if suffix:
            addr = f"{init}.{last}.{suffix}{num}@{domain}"
        else:
            addr = f"{init}.{last}{num}@{domain}"
        if addr not in _used_emails:
            _used_emails.add(addr)
            return addr
    # Fallback — guaranteed unique
    addr = f"{init}.{last}{np.random.randint(100,999)}@{domain}"
    _used_emails.add(addr)
    return addr


# ── Customer archetypes ──────────────────────────────────────────────────────
# Each archetype drives patterns in the downstream order data.
# Counts scaled up ~27% from 220 → 280, preserving proportions.
ARCHETYPES = {
    "champion":      {"n": 32,  "freq_mean": 7,  "aov_mean": 110, "discount_rate": 0.05, "email_eng": 0.8},
    "loyalist":      {"n": 51,  "freq_mean": 4,  "aov_mean": 90,  "discount_rate": 0.10, "email_eng": 0.6},
    "promising":     {"n": 63,  "freq_mean": 2,  "aov_mean": 80,  "discount_rate": 0.15, "email_eng": 0.4},
    "price_hunter":  {"n": 70,  "freq_mean": 3,  "aov_mean": 60,  "discount_rate": 0.75, "email_eng": 0.2},
    "sleeping_giant":{"n": 26,  "freq_mean": 2,  "aov_mean": 140, "discount_rate": 0.05, "email_eng": 0.3},
    "one_and_done":  {"n": 38,  "freq_mean": 1,  "aov_mean": 70,  "discount_rate": 0.20, "email_eng": 0.1},
}

customers_rows = []
orders_rows = []
order_counter = 1

all_dates = pd.date_range(START, END, freq="h")
# Seasonal weight: Nov/Dec 3x heavier
weights = np.where(
    pd.Series(all_dates).dt.month.isin([11, 12]), 3.0, 1.0
)
weights = weights / weights.sum()

cust_id = 1
for archetype, cfg in ARCHETYPES.items():
    for _ in range(cfg["n"]):
        cid = f"CUST-{str(cust_id).zfill(4)}"
        cust_id += 1

        # ── Acquisition source ───────────────────────────────────────────────
        # Champions/loyalists: heavily email + organic, almost no paid social.
        # Price hunters + one_and_done: pile onto paid_social to create a
        # visible retention gap in the channel chart (~27% vs ~60%).
        if archetype == "champion":
            src = np.random.choice(SOURCES, p=[0.05, 0.30, 0.40, 0.15, 0.10])
        elif archetype == "loyalist":
            src = np.random.choice(SOURCES, p=[0.08, 0.28, 0.38, 0.16, 0.10])
        elif archetype == "price_hunter":
            src = np.random.choice(SOURCES, p=[0.72, 0.12, 0.06, 0.06, 0.04])
        elif archetype == "one_and_done":
            src = np.random.choice(SOURCES, p=[0.68, 0.14, 0.07, 0.07, 0.04])
        else:
            # promising, sleeping_giant — moderate mix
            src = np.random.choice(SOURCES, p=[0.38, 0.24, 0.18, 0.12, 0.08])

        state = np.random.choice(STATES, p=[0.18,0.17,0.12,0.10,0.08,0.07,0.07,0.06,0.08,0.07])

        # ── Order count ──────────────────────────────────────────────────────
        n_orders = max(1, int(np.random.poisson(cfg["freq_mean"])))
        if archetype == "one_and_done":
            n_orders = 1

        # Sleeping giants lapsed — last order was 60–150 days ago
        # Champions/loyalists are active — last order within 45 days, driving low recency
        # Price hunters and one_and_done churn fast — last order skewed toward 90–360 days ago
        _force_recent = False
        if archetype == "sleeping_giant":
            last_possible = END - pd.Timedelta(days=60)
            first_possible = END - pd.Timedelta(days=150)
        elif archetype in ("champion", "loyalist"):
            last_possible = END
            first_possible = START
            # Force last order to be recent (within 45 days of END) for ~80% of this group
            _force_recent = np.random.random() < 0.80
        elif archetype == "one_and_done":
            # One-and-done customers: single purchase skewed toward earlier in period
            last_possible = END - pd.Timedelta(days=90)
            first_possible = START
        elif archetype == "price_hunter":
            # Price hunters churn after discount; push last order toward 90-270 days ago
            last_possible = END - pd.Timedelta(days=60)
            first_possible = START
        else:
            last_possible = END
            first_possible = START

        if archetype in ("champion", "loyalist") and _force_recent:
            # Anchor last order within 45 days of END; earlier orders fill back from there
            last_anchor = END - pd.Timedelta(days=int(np.random.uniform(0, 45)))
            if n_orders == 1:
                local_dates = np.array([last_anchor])
            else:
                earlier_range = pd.date_range(first_possible, last_anchor - pd.Timedelta(days=1), freq="h")
                earlier_dates = np.sort(np.random.choice(earlier_range, size=n_orders - 1, replace=False))
                local_dates = np.append(earlier_dates, last_anchor)
        else:
            local_dates = np.sort(np.random.choice(
                pd.date_range(first_possible, last_possible, freq="h"),
                size=n_orders,
                replace=False
            ))

        # ── Top category ─────────────────────────────────────────────────────
        # Gifts skews hard toward one_and_done; high-retention archetypes avoid it.
        # This produces a visually obvious Gifts repeat-rate trough in Stage 2.
        if archetype == "one_and_done":
            top_cat = np.random.choice(CATEGORIES, p=[0.08, 0.10, 0.08, 0.08, 0.66])
        elif archetype == "price_hunter":
            top_cat = np.random.choice(CATEGORIES, p=[0.10, 0.42, 0.20, 0.20, 0.08])
        elif archetype in ("champion", "loyalist"):
            top_cat = np.random.choice(CATEGORIES, p=[0.30, 0.26, 0.24, 0.17, 0.03])
        else:
            top_cat = np.random.choice(CATEGORIES, p=[0.30, 0.25, 0.20, 0.17, 0.08])

        email_opens = int(np.random.poisson(cfg["email_eng"] * 10))
        email_clicks = int(np.random.poisson(cfg["email_eng"] * 3))

        # ── days_since_signup ────────────────────────────────────────────────
        # Gap between list signup and first order. For promising / one_and_done,
        # this is meaningfully larger — they sat on the list before converting.
        if archetype in ("one_and_done", "promising"):
            days_since_signup = int(np.random.poisson(45)) + int(np.random.poisson(25))
        elif archetype == "sleeping_giant":
            days_since_signup = int(np.random.poisson(20))
        else:
            days_since_signup = int(np.random.poisson(8))
        days_since_signup = max(0, days_since_signup)

        order_revenues = []
        for i, odate in enumerate(local_dates):
            rev = max(5.0, round(np.random.normal(cfg["aov_mean"], cfg["aov_mean"] * 0.25), 2))
            has_discount = np.random.random() < cfg["discount_rate"]
            disc = round(rev * np.random.uniform(0.10, 0.30), 2) if has_discount else 0.0
            cat = top_cat if np.random.random() < 0.55 else np.random.choice(CATEGORIES)
            subcat = np.random.choice(SUBCATEGORIES[cat])
            orders_rows.append({
                "order_id": f"ORD-{str(order_counter).zfill(5)}",
                "customer_id": cid,
                "order_date": pd.Timestamp(odate).strftime("%Y-%m-%d"),
                "revenue": rev,
                "discount_amount": disc,
                "product_category": cat,
                "product_subcategory": subcat,
                "acquisition_source": src,
                "customer_state": state,
            })
            order_revenues.append(rev)
            order_counter += 1

        total_rev = round(sum(order_revenues), 2)
        recency = (END - local_dates[-1]).days
        tenure = int((local_dates[-1] - local_dates[0]) / np.timedelta64(1, 'D')) if n_orders > 1 else 0
        is_churned = recency >= 90

        email = _make_email()

        customers_rows.append({
            "customer_id": cid,
            "customer_email": email,
            "archetype": archetype,
            "acquisition_source": src,
            "customer_state": state,
            "first_order_date": pd.Timestamp(local_dates[0]).strftime("%Y-%m-%d"),
            "last_order_date": pd.Timestamp(local_dates[-1]).strftime("%Y-%m-%d"),
            "total_orders": n_orders,
            "total_revenue": total_rev,
            "avg_order_value": round(total_rev / n_orders, 2),
            "days_since_last_order": recency,
            "days_as_customer": tenure,
            "days_since_signup": days_since_signup,
            "top_category": top_cat,
            "email_opens_30d": email_opens,
            "email_clicks_30d": email_clicks,
            "discount_rate": round(cfg["discount_rate"] + np.random.normal(0, 0.05), 2),
            "is_churned": is_churned,
        })

orders_df = pd.DataFrame(orders_rows)
customers_df = pd.DataFrame(customers_rows)

# Drop archetype column — it's internal scaffolding, not a "real" field
customers_df = customers_df.drop(columns=["archetype"])

orders_df.to_csv("data/orders.csv", index=False)
customers_df.to_csv("data/customers.csv", index=False)

print(f"orders.csv    — {len(orders_df)} rows")
print(f"customers.csv — {len(customers_df)} rows")
print(f"Revenue: ${orders_df['revenue'].sum():,.0f} total | ${orders_df['revenue'].mean():.2f} avg order")
print(f"Churn rate: {customers_df['is_churned'].mean():.1%}")
top20 = customers_df.nlargest(int(len(customers_df)*0.2), "total_revenue")["total_revenue"].sum()
print(f"Top 20% revenue share: {top20/customers_df['total_revenue'].sum():.1%}")

# ── Pattern verification ─────────────────────────────────────────────────────
print("\nRetention by acquisition source:")
src_stats = customers_df.groupby("acquisition_source").agg(
    n=("customer_id", "count"),
    retention=("is_churned", lambda x: f"{(1 - x.mean()) * 100:.1f}%")
)
print(src_stats.to_string())

print("\nGifts vs other categories -- repeat rate:")
customers_df["_repeat"] = customers_df["total_orders"] > 1
gifts_rate = customers_df[customers_df["top_category"] == "Gifts"]["_repeat"].mean()
other_rate = customers_df[customers_df["top_category"] != "Gifts"]["_repeat"].mean()
print(f"Gifts repeat rate:  {gifts_rate:.1%}")
print(f"Other repeat rate:  {other_rate:.1%}")
customers_df = customers_df.drop(columns=["_repeat"])
