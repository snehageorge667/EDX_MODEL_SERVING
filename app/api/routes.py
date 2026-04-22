from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
from datetime import datetime

from app.services.data_fetcher import DataFetcher
from app.services.preprocessor import DataPreprocessor
from app.services.feature_engineering import FeatureEngineer
from app.services.predictor import predict_xgboost, predict_sarimax

router = APIRouter()


# ---------------- CACHE ----------------
CACHE_DATA = None
CACHE_TIME = None
CACHE_TTL = 300  # seconds (5 min)


# ---------------- COMMON PIPELINE ----------------
def get_latest_features():
    global CACHE_DATA, CACHE_TIME

    now = datetime.now()

    if CACHE_DATA is not None and CACHE_TIME is not None:
        if (now - CACHE_TIME).seconds < CACHE_TTL:
            return CACHE_DATA

    fetcher = DataFetcher()
    pre = DataPreprocessor()
    fe = FeatureEngineer()

    macro = fetcher.fetch_macro_indicators()
    forex = fetcher.fetch_forex_data()

    df = pre.preprocess(macro_data=macro, forex_data=forex)
    df = fe.engineer_all_features(df)

    if df.empty:
        raise HTTPException(status_code=500, detail="Feature dataframe is empty")

    CACHE_DATA = df
    CACHE_TIME = now

    return df


# ---------------- SAFE VALUE ----------------
def safe_value(val):
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return float(val)


# ---------------- TODAY ----------------
@router.get("/xgboost/today")
def xgb_today():
    df = get_latest_features()
    pred = predict_xgboost(df.tail(1))
    return {"prediction": safe_value(pred.iloc[0])}


@router.get("/sarimax/today")
def sarimax_today():
    df = get_latest_features()
    pred = predict_sarimax(df.tail(1))
    return {"prediction": safe_value(pred.iloc[0])}


# ---------------- DATE → STEPS ----------------
def calculate_days(target_date: str):
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except:
        raise HTTPException(
            status_code=400,
            detail="Use format YYYY-MM-DD (example: 2026-04-24)"
        )

    today = datetime.today().date()

    delta = (target - today).days

    if delta < 0:
        raise HTTPException(
            status_code=400,
            detail="Target date must be in the future"
        )

    return delta


# ---------------- FUTURE LOGIC ----------------
def iterative_forecast(df, model_type, steps):
    results = []

    #  DO NOT recompute features every loop
    current_df = df.copy()

    # start from today
    current_date = pd.Timestamp.today().normalize()

    for i in range(steps + 1):

        latest = current_df.tail(1)

        try:
            if model_type == "xgb":
                pred = float(predict_xgboost(latest).iloc[0])
            else:
                pred = float(predict_sarimax(latest).iloc[0])

            if pd.isna(pred) or np.isinf(pred):
                pred = float(current_df["close"].iloc[-1])

        except:
            pred = float(current_df["close"].iloc[-1])

        # FIX DATE (start from today)
        if i > 0:
            current_date += pd.Timedelta(days=1)

        results.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted_close": pred
        })

        # ---------------- REAL FIX ----------------
        # Instead of re-engineering everything, just simulate minimal forward movement

        new_row = current_df.tail(1).copy()

        prev_close = float(current_df["close"].iloc[-1])
        new_row["close"] = pred

        # update simple dynamics
        if "returns" in new_row.columns and prev_close != 0:
            new_row["returns"] = np.log(pred / prev_close)

        # shift lag features manually (if exist)
        for col in current_df.columns:
            if "lag" in col:
                lag_num = int(col.split("_")[-1])
                if lag_num == 1:
                    new_row[col] = prev_close
                else:
                    prev_col = f"lag_{lag_num-1}"
                    if prev_col in current_df.columns:
                        new_row[col] = current_df[prev_col].iloc[-1]

        current_df = pd.concat([current_df, new_row], ignore_index=True)

    return results


# ---------------- FUTURE APIs ----------------
@router.get("/xgboost/future")
def xgb_future(
    target_date: str = Query(
        ...,
        description="Enter date in format YYYY-MM-DD (e.g., 2026-04-24)",
        example="2026-04-24"
    )
):
    df = get_latest_features()
    steps = calculate_days(target_date)

    preds = iterative_forecast(df, "xgb", steps)

    return {
        "target_date": target_date,
        "predictions": preds
    }


@router.get("/sarimax/future")
def sarimax_future(
    target_date: str = Query(
        ...,
        description="Enter date in format YYYY-MM-DD (e.g., 2026-04-24)",
        example="2026-04-24"
    )
):
    df = get_latest_features()
    steps = calculate_days(target_date)

    preds = iterative_forecast(df, "sarimax", steps)

    return {
        "target_date": target_date,
        "predictions": preds
    }