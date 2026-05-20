from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from data.generator import generate_historical_data, get_current_state, ACCOUNTS
from models.forecaster import CashFlowForecaster
from models.alert_system import AlertSystem
from models.optimizer import LiquidityOptimizer
from models.stress_tester import StressTester, SCENARIOS

app = FastAPI(
    title="FinTech Liquidity Management API",
    description="Предиктивное управление ликвидностью",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production — replace with DB + cache)
_df: pd.DataFrame | None = None
_forecaster: CashFlowForecaster | None = None


def _get_df():
    global _df
    if _df is None:
        _df = generate_historical_data(months=12)
    return _df


def _get_forecaster():
    global _forecaster
    if _forecaster is None:
        _forecaster = CashFlowForecaster()
        _forecaster.train(_get_df())
    return _forecaster


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "fintech-liquidity"}


@app.get("/api/accounts")
def get_accounts():
    state = get_current_state(_get_df())
    return state.to_dict(orient="records")


@app.get("/api/forecast")
def get_forecast(days: int = 3, account_id: str | None = None):
    forecaster = _get_forecaster()
    state = get_current_state(_get_df())
    balances = dict(zip(state["account_id"], state["balance"]))

    if account_id:
        fc = forecaster.forecast(account_id, days=days, current_balance=balances.get(account_id))
    else:
        fc = forecaster.forecast_all(days=days, current_balances=balances)

    if fc.empty:
        raise HTTPException(404, "Account not found or no forecast available")

    fc["date"] = fc["date"].astype(str)
    return fc.to_dict(orient="records")


@app.get("/api/alerts")
def get_alerts():
    state = get_current_state(_get_df())
    forecaster = _get_forecaster()
    balances = dict(zip(state["account_id"], state["balance"]))
    forecasts = forecaster.forecast_all(days=3, current_balances=balances)

    alerts = AlertSystem().generate(state, forecasts)
    summary = AlertSystem().summary(alerts)
    return {"alerts": alerts, "summary": summary, "total": len(alerts)}


@app.get("/api/recommendations")
def get_recommendations():
    state = get_current_state(_get_df())
    forecaster = _get_forecaster()
    balances = dict(zip(state["account_id"], state["balance"]))
    forecasts = forecaster.forecast_all(days=3, current_balances=balances)

    optimizer = LiquidityOptimizer()
    recs = optimizer.recommend(state, forecasts)
    idle = optimizer.idle_capital_report(state)
    return {"recommendations": recs, "idle_capital": idle}


@app.get("/api/scenarios")
def list_scenarios():
    return SCENARIOS


class StressTestRequest(BaseModel):
    scenario: str
    horizon_days: int = 3


@app.post("/api/stress-test")
def run_stress_test(req: StressTestRequest):
    if req.scenario not in SCENARIOS:
        raise HTTPException(400, f"Unknown scenario: {req.scenario}. Available: {list(SCENARIOS.keys())}")

    state = get_current_state(_get_df())
    forecaster = _get_forecaster()
    balances = dict(zip(state["account_id"], state["balance"]))
    forecasts = forecaster.forecast_all(days=req.horizon_days, current_balances=balances)

    result = StressTester().run(req.scenario, state, forecasts)
    result["impact_df"] = result["impact_df"].to_dict(orient="records")
    return result


@app.get("/api/dashboard")
def get_dashboard_data():
    state = get_current_state(_get_df())
    forecaster = _get_forecaster()
    balances = dict(zip(state["account_id"], state["balance"]))
    forecasts = forecaster.forecast_all(days=3, current_balances=balances)

    alert_sys = AlertSystem()
    alerts = alert_sys.generate(state, forecasts)
    summary = alert_sys.summary(alerts)

    optimizer = LiquidityOptimizer()
    recs = optimizer.recommend(state, forecasts)
    idle = optimizer.idle_capital_report(state)

    fc_str = forecasts.copy()
    fc_str["date"] = fc_str["date"].astype(str)

    return {
        "accounts": state.to_dict(orient="records"),
        "forecasts": fc_str.to_dict(orient="records"),
        "alerts": alerts,
        "alert_summary": summary,
        "recommendations": recs,
        "idle_capital": idle,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
