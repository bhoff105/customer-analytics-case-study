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

N_CUSTOMERS = 220
N_ORDERS = 800
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-06-30")
CATEGORIES = ["Home & Kitchen", "Apparel", "Beauty & Wellness", "Fitness", "Gifts"]
SOURCES = ["paid_social", "organic_search", "email", "referral", "direct"]
STATES = ["NY", "CA", "TX", "FL", "IL", "WA", "MA", "CO", "GA", "OH"]

# ── Customer archetypes ──────────────────────────────────────────────────────
# Each archetype drives patterns in the downstream order data.
ARCHETYPES = {
    "champion":      {"n": 25,  "freq_mean": 7,  "aov_mean": 110, "discount_rate": 0.05, "email_eng": 0.8},
    "loyalist":      {"n": 40,  "freq_mean": 4,  "aov_mean": 90,  "discount_rate": 0.10, "email_eng": 0.6},
    "promising":     {"n": 50,  "freq_mean": 2,  "aov_mean": 80,  "discount_rate": 0.15, "email_eng": 0.4},
    "price_hunter":  {"n": 55,  "freq_mean": 3,  "aov_mean": 60,  "discount_rate": 0.75, "email_eng": 0.2},
    "sleeping_giant":{"n": 20,  "freq_mean": 2,  "aov_mean": 140, "discount_rate": 0.05, "email_eng": 0.3},
    "one_and_done":  {"n": 30,  "freq_mean": 1,  "aov_mean": 70,  "discount_rate": 0.20, "email_eng": 0.1},
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

        # Acquisition source — champions/loyalists skew email/organic
        if archetype in ("champion", "loyalist"):
            src = np.random.choice(SOURCES, p=[0.20, 0.30, 0.30, 0.12, 0.08])
        elif archetype == "price_hunter":
            src = np.random.choice(SOURCES, p=[0.65, 0.15, 0.08, 0.07, 0.05])
        else:
            src = np.random.choice(SOURCES, p=[0.50, 0.20, 0.14, 0.10, 0.06])

        state = np.random.choice(STATES, p=[0.18,0.17,0.12,0.10,0.08,0.07,0.07,0.06,0.08,0.07])

        # Order count for this customer
        n_orders = max(1, int(np.random.poisson(cfg["freq_mean"])))
        if archetype == "one_and_done":
            n_orders = 1

        # Sleeping giants lapsed — last order was 60–150 days ago
        if archetype == "sleeping_giant":
            last_possible = END - pd.Timedelta(days=60)
            first_possible = END - pd.Timedelta(days=150)
        else:
            last_possible = END
            first_possible = START

        local_dates = np.sort(np.random.choice(
            pd.date_range(first_possible, last_possible, freq="h"),
            size=n_orders,
            replace=False
        ))

        # Top category for this customer
        if archetype == "price_hunter":
            top_cat = np.random.choice(CATEGORIES, p=[0.10, 0.40, 0.20, 0.20, 0.10])
        else:
            top_cat = np.random.choice(CATEGORIES, p=[0.30, 0.25, 0.20, 0.15, 0.10])

        email_opens = int(np.random.poisson(cfg["email_eng"] * 10))
        email_clicks = int(np.random.poisson(cfg["email_eng"] * 3))

        order_revenues = []
        for i, odate in enumerate(local_dates):
            rev = max(5.0, round(np.random.normal(cfg["aov_mean"], cfg["aov_mean"] * 0.25), 2))
            has_discount = np.random.random() < cfg["discount_rate"]
            disc = round(rev * np.random.uniform(0.10, 0.30), 2) if has_discount else 0.0
            cat = top_cat if np.random.random() < 0.55 else np.random.choice(CATEGORIES)
            orders_rows.append({
                "order_id": f"ORD-{str(order_counter).zfill(5)}",
                "customer_id": cid,
                "order_date": pd.Timestamp(odate).strftime("%Y-%m-%d"),
                "revenue": rev,
                "discount_amount": disc,
                "product_category": cat,
                "acquisition_source": src,
                "customer_state": state,
            })
            order_revenues.append(rev)
            order_counter += 1

        total_rev = round(sum(order_revenues), 2)
        recency = (END - local_dates[-1]).days
        tenure = int((local_dates[-1] - local_dates[0]) / np.timedelta64(1, 'D')) if n_orders > 1 else 0
        is_churned = recency >= 90

        customers_rows.append({
            "customer_id": cid,
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
