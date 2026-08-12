"""
Generate a lightweight ablation report from Master_Forecast_Tracking.

The current tracking file stores only a few prediction checkpoints, so this
report compares the available layers:
  - System_Predicted_Before_Brain: pre-brain/base system prediction
  - Final_Predicted_Guests: production prediction after correction layers

Output:
  outputs/ablation_report.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_system.agents.master_file_agent import MasterFileAgent
from forecast_system.config.settings import MASTER_FILE_NAME, PROJECT_ROOT


PREDICTION_COLUMNS = {
    "PRE_BRAIN_SYSTEM": "System_Predicted_Before_Brain",
    "FINAL_PRODUCTION": "Final_Predicted_Guests",
}


def _day_type(df: pd.DataFrame) -> pd.Series:
    is_holiday = df.get("Is_Holiday", False)
    if not isinstance(is_holiday, pd.Series):
        is_holiday = pd.Series(False, index=df.index)
    weekday = df["Weekday"].astype(str) if "Weekday" in df.columns else df["Date"].dt.day_name()
    return np.where(
        is_holiday.fillna(False).astype(bool),
        "HOLIDAY",
        np.where(weekday.isin(["Saturday", "Sunday"]), "WEEKEND", "WEEKDAY"),
    )


def _hit_rate(actual: pd.Series, pred: pd.Series) -> float:
    valid = actual > 0
    if not valid.any():
        return 0.0
    actual = actual[valid].astype(float)
    pred = pred[valid].astype(float)
    abs_error = (pred - actual).abs()
    pct_error = abs_error / actual
    hit = np.where(actual < 100, abs_error <= 10, pct_error <= 0.10)
    return round(float(hit.mean() * 100), 2)


def _metrics(df: pd.DataFrame, pred_col: str) -> dict:
    valid = df[(df["Actual_Guest"] > 0) & pd.notna(df[pred_col])].copy()
    if valid.empty:
        return {"samples": 0, "mae": np.nan, "mape": np.nan, "bias": np.nan, "hit_rate": 0.0}
    error = valid[pred_col].astype(float) - valid["Actual_Guest"].astype(float)
    return {
        "samples": int(len(valid)),
        "mae": round(float(error.abs().mean()), 2),
        "mape": round(float((error.abs() / valid["Actual_Guest"].astype(float)).mean() * 100), 2),
        "bias": round(float(error.mean()), 2),
        "hit_rate": _hit_rate(valid["Actual_Guest"], valid[pred_col]),
    }


def main() -> int:
    df = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
    if df.empty:
        print("Master file is empty")
        return 1

    required = {"Date", "Actual_Guest", "Final_Predicted_Guests"}
    missing = required - set(df.columns)
    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return 1

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Actual_Guest"])
    df = df[df["Actual_Guest"] >= 0]
    if df.empty:
        print("No evaluated rows")
        return 1

    df["day_type"] = _day_type(df)
    df["size_group"] = np.where(df["Actual_Guest"].astype(float) >= 100, "LARGE", "SMALL")
    df["segment"] = df.get("Shift", "DAILY").astype(str) + "|" + df["day_type"] + "|" + df["size_group"]

    max_date = df["Date"].max().normalize()
    rows = []
    for window in (7, 14, 30):
        start = max_date - pd.Timedelta(days=window - 1)
        dfw = df[df["Date"] >= start].copy()
        for layer, col in PREDICTION_COLUMNS.items():
            if col not in dfw.columns:
                continue
            for segment_name, dfs in [("ALL", dfw)] + list(dfw.groupby("segment")):
                m = _metrics(dfs, col)
                rows.append({
                    "window_days": window,
                    "max_actual_date": max_date.date().isoformat(),
                    "segment": segment_name,
                    "layer": layer,
                    **m,
                })

    report = pd.DataFrame(rows)
    if not report.empty and {"PRE_BRAIN_SYSTEM", "FINAL_PRODUCTION"} <= set(report["layer"]):
        pivot = report.pivot_table(
            index=["window_days", "segment"],
            columns="layer",
            values=["mae", "mape", "bias", "hit_rate"],
            aggfunc="first",
        )
        for metric in ("mae", "mape", "bias", "hit_rate"):
            pre = pivot[(metric, "PRE_BRAIN_SYSTEM")]
            final = pivot[(metric, "FINAL_PRODUCTION")]
            if metric in ("mae", "mape"):
                delta = final - pre
            else:
                delta = final - pre
            delta_name = f"{metric}_final_minus_pre"
            report = report.merge(
                delta.rename(delta_name).reset_index(),
                on=["window_days", "segment"],
                how="left",
            )

    out_dir = Path(PROJECT_ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")
    if not report.empty:
        top = report[
            (report["window_days"] == 7) &
            (report["layer"] == "FINAL_PRODUCTION") &
            (report["segment"] != "ALL")
        ].sort_values(["mae_final_minus_pre", "mae"], ascending=[False, False]).head(10)
        if not top.empty:
            print(top[["segment", "samples", "mae", "mape", "bias", "hit_rate", "mae_final_minus_pre"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
