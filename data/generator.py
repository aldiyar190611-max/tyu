import pandas as pd
import numpy as np
from datetime import date, timedelta

ACCOUNTS = [
    {
        "id": "USD_CITI_SWIFT",
        "name": "Citibank USD (SWIFT)",
        "currency": "USD",
        "payment_system": "SWIFT",
        "min_balance": 500_000,
        "target_balance": 2_000_000,
        "daily_volume": 850_000,
    },
    {
        "id": "USD_STRIPE_CARD",
        "name": "Stripe USD (Card)",
        "currency": "USD",
        "payment_system": "CARD",
        "min_balance": 300_000,
        "target_balance": 1_500_000,
        "daily_volume": 600_000,
    },
    {
        "id": "EUR_DB_SEPA",
        "name": "Deutsche Bank EUR (SEPA)",
        "currency": "EUR",
        "payment_system": "SEPA",
        "min_balance": 400_000,
        "target_balance": 1_800_000,
        "daily_volume": 720_000,
    },
    {
        "id": "EUR_ADYEN_CARD",
        "name": "Adyen EUR (Card)",
        "currency": "EUR",
        "payment_system": "CARD",
        "min_balance": 250_000,
        "target_balance": 1_200_000,
        "daily_volume": 480_000,
    },
    {
        "id": "GBP_BARCLAYS_LOCAL",
        "name": "Barclays GBP (Local)",
        "currency": "GBP",
        "payment_system": "LOCAL",
        "min_balance": 200_000,
        "target_balance": 800_000,
        "daily_volume": 320_000,
    },
    {
        "id": "GBP_HSBC_SWIFT",
        "name": "HSBC GBP (SWIFT)",
        "currency": "GBP",
        "payment_system": "SWIFT",
        "min_balance": 150_000,
        "target_balance": 600_000,
        "daily_volume": 240_000,
    },
]

CLEARING_DAYS = {"LOCAL": 0, "SEPA": 1, "SWIFT": 3, "CARD": 5}

FX_RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}

BANK_HOLIDAYS = {
    date(2024, 1, 1), date(2024, 3, 29), date(2024, 4, 1),
    date(2024, 5, 1), date(2024, 5, 27), date(2024, 12, 25), date(2024, 12, 26),
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 1), date(2025, 12, 25), date(2025, 12, 26),
    date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 6),
    date(2026, 5, 1), date(2026, 12, 25), date(2026, 12, 26),
}


def is_business_day(d):
    return d.weekday() < 5 and d not in BANK_HOLIDAYS


def _day_cashflow(account, d, seed):
    rng = np.random.default_rng(seed)
    dow = d.weekday()
    dom = d.day

    if not is_business_day(d):
        inflow = rng.exponential(account["daily_volume"] * 0.04)
        outflow = rng.exponential(account["daily_volume"] * 0.02)
        return inflow, outflow

    dow_mult = [1.20, 1.10, 1.00, 0.95, 0.85][dow]
    month_mult = {11: 1.25, 12: 1.35, 1: 0.80, 8: 0.85}.get(d.month, 1.0)
    period_mult = 1.5 if dom >= 25 else (0.90 if dom <= 5 else 1.0)

    vol = account["daily_volume"] * dow_mult * month_mult * period_mult
    inflow = max(0.0, rng.normal(vol * 0.53, vol * 0.14))
    outflow = max(0.0, rng.normal(vol * 0.47, vol * 0.12))
    return inflow, outflow


def generate_historical_data(months: int = 12, end_date: date | None = None) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=months * 30)

    records = []
    for ai, acc in enumerate(ACCOUNTS):
        rng0 = np.random.default_rng(ai * 999)
        balance = acc["target_balance"] * rng0.uniform(0.85, 1.15)
        clearing_delay = CLEARING_DAYS[acc["payment_system"]]
        pending_queue = []  # (value_date, amount)

        d = start_date
        while d <= end_date:
            seed = ai * 100_000 + (d - start_date).days
            inflow_raw, outflow = _day_cashflow(acc, d, seed)

            # Inflows with clearing delay
            if clearing_delay > 0:
                value_d = d + timedelta(days=clearing_delay)
                while not is_business_day(value_d):
                    value_d += timedelta(days=1)
                pending_queue.append((value_d, inflow_raw))
                settled_inflow = sum(amt for vd, amt in pending_queue if vd <= d)
                pending_queue = [(vd, amt) for vd, amt in pending_queue if vd > d]
            else:
                settled_inflow = inflow_raw

            pending_total = sum(amt for _, amt in pending_queue)
            net = settled_inflow - outflow
            balance = max(0.0, balance + net)

            # Occasional large inter-account transfers (treasury ops)
            rng_shock = np.random.default_rng(seed + 77)
            if rng_shock.random() < 0.015:
                shock = acc["target_balance"] * rng_shock.uniform(0.08, 0.25)
                balance = max(0.0, balance + rng_shock.choice([-1, 1]) * shock)

            records.append({
                "date": pd.Timestamp(d),
                "account_id": acc["id"],
                "account_name": acc["name"],
                "currency": acc["currency"],
                "payment_system": acc["payment_system"],
                "inflow": settled_inflow,
                "outflow": outflow,
                "net_flow": net,
                "pending_inflow": pending_total,
                "balance": balance,
                "min_balance": acc["min_balance"],
                "target_balance": acc["target_balance"],
                "is_business_day": is_business_day(d),
            })
            d += timedelta(days=1)

    return pd.DataFrame(records)


def get_current_state(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = generate_historical_data()
    last_date = df["date"].max()
    today = df[df["date"] == last_date].copy()

    rows = []
    for _, row in today.iterrows():
        acc = next(a for a in ACCOUNTS if a["id"] == row["account_id"])
        bal = row["balance"]
        rows.append({
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "currency": row["currency"],
            "payment_system": row["payment_system"],
            "balance": bal,
            "pending_inflow": row["pending_inflow"],
            "available_balance": bal + row["pending_inflow"],
            "min_balance": acc["min_balance"],
            "target_balance": acc["target_balance"],
            "excess": max(0.0, bal - acc["target_balance"]),
            "deficit": max(0.0, acc["min_balance"] - bal),
            "usd_equivalent": bal * FX_RATES[acc["currency"]],
        })
    return pd.DataFrame(rows)
