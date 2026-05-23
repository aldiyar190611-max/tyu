from __future__ import annotations

import pandas as pd
import numpy as np
from data.generator import ACCOUNTS, FX_RATES, CLEARING_DAYS


TRANSFER_COST_BPS = {
    ("USD", "USD"): 2,
    ("EUR", "EUR"): 1,
    ("GBP", "GBP"): 1,
    ("USD", "EUR"): 15,
    ("USD", "GBP"): 18,
    ("EUR", "GBP"): 12,
    ("EUR", "USD"): 15,
    ("GBP", "USD"): 18,
    ("GBP", "EUR"): 12,
}

TRANSFER_TIME_DAYS = {
    "LOCAL": 0,
    "SEPA": 1,
    "SWIFT": 2,
    "CARD": 0,
}


class LiquidityOptimizer:
    def recommend(
        self,
        current_state: pd.DataFrame,
        forecasts: pd.DataFrame | None = None,
        horizon_days: int = 3,
    ) -> list[dict]:
        recommendations = []

        # Find surplus and deficit accounts
        surplus = []
        deficit = []

        for _, row in current_state.iterrows():
            acc_cfg = next(a for a in ACCOUNTS if a["id"] == row["account_id"])
            buffer = acc_cfg["min_balance"] * 0.15  # safety buffer

            # Incorporate forecast if available
            projected_bal = row["balance"]
            if forecasts is not None and not forecasts.empty:
                acc_fc = forecasts[forecasts["account_id"] == row["account_id"]]
                if not acc_fc.empty:
                    projected_bal = acc_fc.iloc[-1]["predicted_balance"]

            excess = projected_bal - (acc_cfg["target_balance"] + buffer)
            shortfall = (acc_cfg["min_balance"] + buffer) - projected_bal

            if excess > 50_000:
                surplus.append({
                    "account_id": row["account_id"],
                    "account_name": row["account_name"],
                    "currency": row["currency"],
                    "payment_system": row["payment_system"],
                    "available_excess": excess,
                    "current_balance": projected_bal,
                })
            elif shortfall > 50_000:
                deficit.append({
                    "account_id": row["account_id"],
                    "account_name": row["account_name"],
                    "currency": row["currency"],
                    "payment_system": row["payment_system"],
                    "needed": shortfall,
                    "current_balance": projected_bal,
                })

        # Greedy matching: match surplus to deficit
        for d in deficit:
            needed = d["needed"]
            for s in surplus:
                if s["available_excess"] < 1_000:
                    continue

                transfer_amount = min(needed, s["available_excess"])
                cost_bps = TRANSFER_COST_BPS.get(
                    (s["currency"], d["currency"]),
                    TRANSFER_COST_BPS.get((d["currency"], s["currency"]), 20),
                )
                transfer_time = max(
                    TRANSFER_TIME_DAYS[s["payment_system"]],
                    TRANSFER_TIME_DAYS[d["payment_system"]],
                )

                # FX conversion if needed
                if s["currency"] != d["currency"]:
                    amount_in_dest = transfer_amount * FX_RATES[s["currency"]] / FX_RATES[d["currency"]]
                    fx_note = f" (конвертация {s['currency']} → {d['currency']})"
                else:
                    amount_in_dest = transfer_amount
                    fx_note = ""

                cost = transfer_amount * cost_bps / 10_000
                urgency = "НЕМЕДЛЕННО" if d["needed"] >= d["current_balance"] * 0.5 else "В ТЕЧЕНИЕ 24Ч"

                recommendations.append({
                    "from_account": s["account_name"],
                    "from_id": s["account_id"],
                    "to_account": d["account_name"],
                    "to_id": d["account_id"],
                    "amount": transfer_amount,
                    "amount_dest": amount_in_dest,
                    "currency_from": s["currency"],
                    "currency_to": d["currency"],
                    "transfer_time_days": transfer_time,
                    "cost_bps": cost_bps,
                    "estimated_cost": cost,
                    "urgency": urgency,
                    "reason": (
                        f"{d['account_name']} дефицит {d['needed']:,.0f} {d['currency']}{fx_note}. "
                        f"Перевод займёт {transfer_time} дн., стоимость ~{cost:,.0f} {s['currency']}"
                    ),
                    "roi": (d["needed"] * 0.05 / 365) / max(cost, 1),  # vs overdraft cost
                })

                s["available_excess"] -= transfer_amount
                needed -= transfer_amount
                if needed <= 0:
                    break

        # Sort by urgency and ROI
        recommendations.sort(key=lambda x: (x["urgency"], -x["roi"]))

        # --- Idle capital recommendations ---
        # Даже если дефицитных счетов нет, показываем куда деть избыток
        ANNUAL_RATE = 0.045  # 4.5% risk-free rate
        for s in surplus:
            idle_usd = s["available_excess"] * FX_RATES[s["currency"]]
            # Пропускаем если этот излишек уже был использован для покрытия дефицита
            if s["available_excess"] < 50_000:
                continue
            annual_income = idle_usd * ANNUAL_RATE
            recommendations.append({
                "from_account": s["account_name"],
                "from_id": s["account_id"],
                "to_account": "💰 Доходный депозит / Money Market",
                "to_id": "MONEY_MARKET",
                "amount": s["available_excess"],
                "amount_dest": s["available_excess"],
                "currency_from": s["currency"],
                "currency_to": s["currency"],
                "transfer_time_days": 1,
                "cost_bps": 1,
                "estimated_cost": s["available_excess"] * 0.0001,
                "urgency": "ОПТИМИЗАЦИЯ",
                "reason": (
                    f"Idle-капитал {s['available_excess']:,.0f} {s['currency']} "
                    f"сверх целевого баланса. При размещении под 4.5% годовых — "
                    f"+${annual_income:,.0f} дохода в год."
                ),
                "roi": annual_income / max(s["available_excess"] * 0.0001, 1),
                "type": "IDLE_OPTIMIZATION",
                "id": f"idle_{s['account_id']}_MONEY_MARKET",
            })

        return recommendations

    def idle_capital_report(self, current_state: pd.DataFrame) -> dict:
        total_idle_usd = 0.0
        total_liquidity_usd = 0.0
        details = []

        for _, row in current_state.iterrows():
            acc_cfg = next(a for a in ACCOUNTS if a["id"] == row["account_id"])
            usd = row["balance"] * FX_RATES[row["currency"]]
            target_usd = acc_cfg["target_balance"] * FX_RATES[row["currency"]]
            idle = max(0.0, usd - target_usd)
            total_idle_usd += idle
            total_liquidity_usd += usd
            details.append({
                "account": row["account_name"],
                "balance_usd": usd,
                "idle_usd": idle,
                "idle_pct": (idle / usd * 100) if usd > 0 else 0,
            })

        return {
            "total_liquidity_usd": total_liquidity_usd,
            "total_idle_usd": total_idle_usd,
            "idle_pct": total_idle_usd / total_liquidity_usd * 100 if total_liquidity_usd > 0 else 0,
            "annual_opportunity_cost_usd": total_idle_usd * 0.045,  # ~4.5% risk-free rate
            "details": details,
        }
