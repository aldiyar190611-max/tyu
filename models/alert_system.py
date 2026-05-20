from __future__ import annotations

import pandas as pd
from datetime import timedelta
from data.generator import ACCOUNTS, FX_RATES


SEVERITY_COLORS = {
    "CRITICAL": "#FF4444",
    "HIGH": "#FF8800",
    "MEDIUM": "#FFCC00",
    "LOW": "#44AA44",
}


def _severity(deficit_pct: float) -> str:
    if deficit_pct >= 0.5:
        return "CRITICAL"
    if deficit_pct >= 0.25:
        return "HIGH"
    if deficit_pct >= 0.10:
        return "MEDIUM"
    return "LOW"


class AlertSystem:
    def generate(
        self,
        current_state: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> list[dict]:
        alerts = []

        for _, cs in current_state.iterrows():
            acc_id = cs["account_id"]
            acc_cfg = next(a for a in ACCOUNTS if a["id"] == acc_id)
            min_bal = acc_cfg["min_balance"]

            # Current balance alert
            if cs["balance"] < min_bal:
                deficit = min_bal - cs["balance"]
                alerts.append({
                    "account_id": acc_id,
                    "account_name": cs["account_name"],
                    "currency": cs["currency"],
                    "severity": _severity(deficit / min_bal),
                    "type": "CURRENT_DEFICIT",
                    "message": f"Баланс ниже минимума: {cs['balance']:,.0f} < {min_bal:,.0f} {cs['currency']}",
                    "deficit": deficit,
                    "time_to_breach_h": 0,
                    "recommended_action": f"Немедленный перевод {deficit:,.0f} {cs['currency']} на счёт",
                })

            # Forecast alerts
            acc_fc = forecasts[forecasts["account_id"] == acc_id].sort_values("date")
            for _, fc_row in acc_fc.iterrows():
                proj_bal = fc_row["predicted_balance"]
                if proj_bal < min_bal:
                    deficit = min_bal - proj_bal
                    delta_days = (fc_row["date"] - forecasts["date"].min()).days + 1
                    alerts.append({
                        "account_id": acc_id,
                        "account_name": cs["account_name"],
                        "currency": cs["currency"],
                        "severity": _severity(deficit / min_bal),
                        "type": "FORECAST_DEFICIT",
                        "message": (
                            f"Прогноз дефицита через {delta_days} дн.: "
                            f"{proj_bal:,.0f} < {min_bal:,.0f} {cs['currency']}"
                        ),
                        "deficit": deficit,
                        "time_to_breach_h": delta_days * 24,
                        "recommended_action": (
                            f"Подготовить перевод {deficit:,.0f} {cs['currency']} "
                            f"не позднее чем через {max(0, delta_days - 1)} дн."
                        ),
                    })

            # Clearing risk: high pending inflows = risk if they don't arrive
            if cs["pending_inflow"] > cs["balance"] * 0.8 and cs["balance"] < min_bal * 1.2:
                alerts.append({
                    "account_id": acc_id,
                    "account_name": cs["account_name"],
                    "currency": cs["currency"],
                    "severity": "MEDIUM",
                    "type": "CLEARING_RISK",
                    "message": (
                        f"Высокая зависимость от клиринга: ожидается "
                        f"{cs['pending_inflow']:,.0f} {cs['currency']} в обработке"
                    ),
                    "deficit": 0,
                    "time_to_breach_h": 24,
                    "recommended_action": "Проверить статус клиринга и держать резерв",
                })

            # Excess alert
            if cs["excess"] > acc_cfg["target_balance"] * 0.5:
                alerts.append({
                    "account_id": acc_id,
                    "account_name": cs["account_name"],
                    "currency": cs["currency"],
                    "severity": "LOW",
                    "type": "EXCESS_IDLE",
                    "message": (
                        f"Избыточный остаток: {cs['excess']:,.0f} {cs['currency']} "
                        f"заморожено без использования"
                    ),
                    "deficit": -cs["excess"],
                    "time_to_breach_h": None,
                    "recommended_action": f"Перевести излишек на доходный счёт или в другой пул",
                })

        alerts.sort(key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x["severity"]])
        return alerts

    def summary(self, alerts: list[dict]) -> dict:
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in alerts:
            counts[a["severity"]] = counts.get(a["severity"], 0) + 1
        return counts
