from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import timedelta, date
from data.generator import ACCOUNTS, FX_RATES, CLEARING_DAYS, is_business_day

SCENARIOS = {
    "swift_delay": {
        "name": "Задержка SWIFT (+2 дня)",
        "description": "Все SWIFT-переводы задерживаются на 2 дополнительных дня из-за технического сбоя или санкционных проверок.",
        "icon": "🔴",
    },
    "eu_holiday": {
        "name": "Праздник в ЕС (SEPA стоит 2 дня)",
        "description": "Банковские праздники в Германии и Франции блокируют SEPA-расчёты на 2 рабочих дня.",
        "icon": "🟡",
    },
    "volume_spike": {
        "name": "Пиковая нагрузка (+80% объём)",
        "description": "Резкий рост транзакционного объёма — промо-акция, вирусный эффект, массовые выплаты.",
        "icon": "🔶",
    },
    "card_delay": {
        "name": "Задержка карточного клиринга (+3 дня)",
        "description": "Adyen/Stripe задерживают клиринг карточных транзакций из-за технического сбоя процессора.",
        "icon": "🟠",
    },
    "fx_shock": {
        "name": "Валютный шок EUR -8%, GBP -5%",
        "description": "Резкое падение EUR и GBP против USD из-за геополитических событий или рыночного стресса.",
        "icon": "💱",
    },
    "multi_crisis": {
        "name": "Кризисный сценарий (всё сразу)",
        "description": "Комбинация задержки SWIFT, падения валют и пика объёма — наихудший сценарий.",
        "icon": "🚨",
    },
}


class StressTester:
    def run(
        self,
        scenario_key: str,
        current_state: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> dict:
        scenario_meta = SCENARIOS.get(scenario_key, {})
        stressed = self._apply_scenario(scenario_key, current_state.copy(), forecasts.copy())

        impact_rows = []
        for _, base in current_state.iterrows():
            acc_id = base["account_id"]
            stressed_row = stressed[stressed["account_id"] == acc_id]
            if stressed_row.empty:
                continue
            s = stressed_row.iloc[0]

            base_end = forecasts[forecasts["account_id"] == acc_id]["predicted_balance"].iloc[-1] if not forecasts[forecasts["account_id"] == acc_id].empty else base["balance"]
            stressed_end = s.get("stressed_balance_end", base_end)

            delta = stressed_end - base_end
            delta_usd = delta * FX_RATES[base["currency"]]
            acc_cfg = next(a for a in ACCOUNTS if a["id"] == acc_id)
            at_risk = stressed_end < acc_cfg["min_balance"]

            impact_rows.append({
                "account_id": acc_id,
                "account_name": base["account_name"],
                "currency": base["currency"],
                "base_balance_end": base_end,
                "stressed_balance_end": stressed_end,
                "delta": delta,
                "delta_usd": delta_usd,
                "at_risk": at_risk,
                "min_balance": acc_cfg["min_balance"],
                "shortfall": max(0.0, acc_cfg["min_balance"] - stressed_end),
            })

        impact_df = pd.DataFrame(impact_rows)
        total_delta_usd = impact_df["delta_usd"].sum()
        accounts_at_risk = impact_df["at_risk"].sum()
        total_shortfall_usd = (impact_df["shortfall"] * impact_df["currency"].map(FX_RATES)).sum()

        return {
            "scenario_key": scenario_key,
            "scenario_name": scenario_meta.get("name", scenario_key),
            "scenario_description": scenario_meta.get("description", ""),
            "icon": scenario_meta.get("icon", "⚠️"),
            "impact_df": impact_df,
            "total_delta_usd": total_delta_usd,
            "accounts_at_risk": int(accounts_at_risk),
            "total_shortfall_usd": total_shortfall_usd,
            "severity": self._severity(accounts_at_risk, total_shortfall_usd),
        }

    def _apply_scenario(
        self,
        scenario_key: str,
        current_state: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        result = current_state.copy()

        def get_end_balance(acc_id):
            fc = forecasts[forecasts["account_id"] == acc_id]
            return fc.iloc[-1]["predicted_balance"] if not fc.empty else current_state[current_state["account_id"] == acc_id]["balance"].iloc[0]

        for idx, row in result.iterrows():
            acc_id = row["account_id"]
            acc_cfg = next(a for a in ACCOUNTS if a["id"] == acc_id)
            base_end = get_end_balance(acc_id)

            stressed_end = base_end

            if scenario_key == "swift_delay":
                if acc_cfg["payment_system"] == "SWIFT":
                    daily_in = forecasts[forecasts["account_id"] == acc_id]["predicted_inflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else 0
                    stressed_end -= daily_in * 2  # lose 2 days of inflows

            elif scenario_key == "eu_holiday":
                if acc_cfg["payment_system"] == "SEPA":
                    daily_in = forecasts[forecasts["account_id"] == acc_id]["predicted_inflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else 0
                    stressed_end -= daily_in * 2  # 2 days blocked

            elif scenario_key == "volume_spike":
                daily_out = forecasts[forecasts["account_id"] == acc_id]["predicted_outflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else acc_cfg["daily_volume"] * 0.48
                stressed_end -= daily_out * 0.8 * 3  # +80% outflows for 3 days

            elif scenario_key == "card_delay":
                if acc_cfg["payment_system"] == "CARD":
                    daily_in = forecasts[forecasts["account_id"] == acc_id]["predicted_inflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else 0
                    stressed_end -= daily_in * 3  # lose 3 days inflows

            elif scenario_key == "fx_shock":
                if row["currency"] == "EUR":
                    stressed_end *= 0.92  # EUR -8%
                elif row["currency"] == "GBP":
                    stressed_end *= 0.95  # GBP -5%

            elif scenario_key == "multi_crisis":
                if acc_cfg["payment_system"] == "SWIFT":
                    daily_in = forecasts[forecasts["account_id"] == acc_id]["predicted_inflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else 0
                    stressed_end -= daily_in * 2
                if acc_cfg["payment_system"] == "SEPA":
                    daily_in = forecasts[forecasts["account_id"] == acc_id]["predicted_inflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else 0
                    stressed_end -= daily_in * 2
                daily_out = forecasts[forecasts["account_id"] == acc_id]["predicted_outflow"].mean() if not forecasts[forecasts["account_id"] == acc_id].empty else acc_cfg["daily_volume"] * 0.48
                stressed_end -= daily_out * 0.5 * 2
                if row["currency"] == "EUR":
                    stressed_end *= 0.92
                elif row["currency"] == "GBP":
                    stressed_end *= 0.95

            result.at[idx, "stressed_balance_end"] = max(0.0, stressed_end)

        return result

    def _severity(self, accounts_at_risk: int, shortfall_usd: float) -> str:
        if accounts_at_risk >= 3 or shortfall_usd > 2_000_000:
            return "КРИТИЧЕСКИЙ"
        if accounts_at_risk >= 2 or shortfall_usd > 500_000:
            return "ВЫСОКИЙ"
        if accounts_at_risk >= 1 or shortfall_usd > 100_000:
            return "СРЕДНИЙ"
        return "НИЗКИЙ"
