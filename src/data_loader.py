import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_orders() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    return df


def load_customers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["first_order_date", "last_order_date"])
    return df


def load_both() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_orders(), load_customers()
