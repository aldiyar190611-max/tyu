from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from datetime import date, timedelta

from data.generator import ACCOUNTS, CLEARING_DAYS, is_business_day


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_end"] = (df["dom"] >= 25).astype(int)
    df["is_month_start"] = (df["dom"] <= 5).astype(int)
    df["is_biz"] = df["is_business_day"].astype(int)

    for lag in [1, 2, 3, 7, 14]:
        df[f"net_lag{lag}"] = df["net_flow"].shift(lag)
        df[f"in_lag{lag}"] = df["inflow"].shift(lag)
        df[f"out_lag{lag}"] = df["outflow"].shift(lag)

    df["net_roll7"] = df["net_flow"].shift(1).rolling(7).mean()
    df["net_roll14"] = df["net_flow"].shift(1).rolling(14).mean()
    df["net_roll30"] = df["net_flow"].shift(1).rolling(30).mean()
    df["out_roll7"] = df["outflow"].shift(1).rolling(7).mean()
    df["in_roll7"] = df["inflow"].shift(1).rolling(7).mean()
    df["bal_lag1"] = df["balance"].shift(1)

    return df.dropna()


FEATURE_COLS = [
    "dow", "dom", "month", "is_weekend", "is_month_end", "is_month_start", "is_biz",
    "net_lag1", "net_lag2", "net_lag3", "net_lag7", "net_lag14",
    "in_lag1", "in_lag2", "in_lag3", "in_lag7",
    "out_lag1", "out_lag2", "out_lag3", "out_lag7",
    "net_roll7", "net_roll14", "net_roll30",
    "out_roll7", "in_roll7", "bal_lag1",
]


class CashFlowForecaster:
    def __init__(self):
        self._models: dict[str, RandomForestRegressor] = {}
        self._last_rows: dict[str, pd.DataFrame] = {}
        self.trained = False

    def train(self, df: pd.DataFrame) -> None:
        for acc in ACCOUNTS:
            acc_df = df[df["account_id"] == acc["id"]].copy()
            feat_df = _make_features(acc_df)
            if len(feat_df) < 30:
                continue

            X = feat_df[FEATURE_COLS]
            y_net = feat_df["net_flow"]
            y_in = feat_df["inflow"]
            y_out = feat_df["outflow"]

            model_net = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
            model_net.fit(X, y_net)

            model_in = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            model_in.fit(X, y_in)

            model_out = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            model_out.fit(X, y_out)

            self._models[acc["id"]] = (model_net, model_in, model_out)
            self._last_rows[acc["id"]] = feat_df.tail(30)

        self.trained = True

    def forecast(self, acc_id: str, days: int = 3, current_balance: float | None = None) -> pd.DataFrame:
        if acc_id not in self._models:
            return pd.DataFrame()

        model_net, model_in, model_out = self._models[acc_id]
        history = self._last_rows[acc_id].copy()
        acc_cfg = next(a for a in ACCOUNTS if a["id"] == acc_id)
        bal = current_balance if current_balance is not None else history["balance"].iloc[-1]

        results = []
        last_date = history["date"].iloc[-1]

        for i in range(1, days + 1):
            forecast_date = last_date + timedelta(days=i)
            is_biz = is_business_day(forecast_date.date())

            row = {
                "date": forecast_date,
                "account_id": acc_id,
                "account_name": acc_cfg["name"],
                "currency": acc_cfg["currency"],
                "payment_system": acc_cfg["payment_system"],
                "is_business_day": is_biz,
            }

            # Build feature vector from history tail
            tail = history.tail(14)
            net_vals = tail["net_flow"].values
            in_vals = tail["inflow"].values
            out_vals = tail["outflow"].values

            feats = {
                "dow": forecast_date.dayofweek,
                "dom": forecast_date.day,
                "month": forecast_date.month,
                "is_weekend": int(forecast_date.dayofweek >= 5),
                "is_month_end": int(forecast_date.day >= 25),
                "is_month_start": int(forecast_date.day <= 5),
                "is_biz": int(is_biz),
                "net_lag1": net_vals[-1] if len(net_vals) >= 1 else 0,
                "net_lag2": net_vals[-2] if len(net_vals) >= 2 else 0,
                "net_lag3": net_vals[-3] if len(net_vals) >= 3 else 0,
                "net_lag7": net_vals[-7] if len(net_vals) >= 7 else 0,
                "net_lag14": net_vals[-14] if len(net_vals) >= 14 else 0,
                "in_lag1": in_vals[-1] if len(in_vals) >= 1 else 0,
                "in_lag2": in_vals[-2] if len(in_vals) >= 2 else 0,
                "in_lag3": in_vals[-3] if len(in_vals) >= 3 else 0,
                "in_lag7": in_vals[-7] if len(in_vals) >= 7 else 0,
                "out_lag1": out_vals[-1] if len(out_vals) >= 1 else 0,
                "out_lag2": out_vals[-2] if len(out_vals) >= 2 else 0,
                "out_lag3": out_vals[-3] if len(out_vals) >= 3 else 0,
                "out_lag7": out_vals[-7] if len(out_vals) >= 7 else 0,
                "net_roll7": np.mean(net_vals[-7:]) if len(net_vals) >= 7 else np.mean(net_vals),
                "net_roll14": np.mean(net_vals[-14:]) if len(net_vals) >= 14 else np.mean(net_vals),
                "net_roll30": np.mean(net_vals) if len(net_vals) >= 30 else np.mean(net_vals),
                "out_roll7": np.mean(out_vals[-7:]) if len(out_vals) >= 7 else np.mean(out_vals),
                "in_roll7": np.mean(in_vals[-7:]) if len(in_vals) >= 7 else np.mean(in_vals),
                "bal_lag1": bal,
            }

            X = pd.DataFrame([feats])[FEATURE_COLS]
            pred_net = float(model_net.predict(X)[0])
            pred_in = float(model_in.predict(X)[0])
            pred_out = float(model_out.predict(X)[0])

            # Confidence interval via individual tree variance (use .values to avoid feature name warnings)
            X_arr = X.values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree_preds = np.array([t.predict(X_arr)[0] for t in model_net.estimators_])
            std = tree_preds.std()

            clearing_delay = CLEARING_DAYS[acc_cfg["payment_system"]]
            effective_net = pred_net if clearing_delay == 0 else pred_net * max(0, 1 - clearing_delay * 0.15)

            bal += effective_net
            bal = max(0.0, bal)

            row.update({
                "predicted_inflow": max(0.0, pred_in),
                "predicted_outflow": max(0.0, pred_out),
                "predicted_net": pred_net,
                "effective_net": effective_net,
                "predicted_balance": bal,
                "std": std,
                "lower_bound": bal - 1.5 * std,
                "upper_bound": bal + 1.5 * std,
                "clearing_delay": clearing_delay,
            })
            results.append(row)

            # Append to history for next iteration
            new_row = history.iloc[-1].copy()
            new_row["date"] = forecast_date
            new_row["net_flow"] = pred_net
            new_row["inflow"] = pred_in
            new_row["outflow"] = pred_out
            new_row["balance"] = bal
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(results)

    def forecast_all(self, days: int = 3, current_balances: dict | None = None) -> pd.DataFrame:
        frames = []
        for acc in ACCOUNTS:
            bal = None
            if current_balances:
                bal = current_balances.get(acc["id"])
            frames.append(self.forecast(acc["id"], days=days, current_balance=bal))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
