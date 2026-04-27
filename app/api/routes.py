from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

from app.services.data_fetcher import DataFetcher
from app.services.preprocessor import DataPreprocessor
from app.services.feature_engineering import FeatureEngineer
from app.services.predictor import predict_xgboost, predict_sarimax

router = APIRouter()


# ---------------- CACHE ----------------
CACHE_DATA = {}
CACHE_TIME = {}
CACHE_TTL = 21600  # 6 HOURS


# ---------------- PAIR FIX ----------------
def build_pair(base_currency: str, target_currency: str):
    base = base_currency.upper()
    target = target_currency.upper()

    if base == "USD":
        return f"{target}=X"
    else:
        return f"{base}{target}=X"


# ---------------- COMMON PIPELINE ----------------
def get_latest_features(pair: str = "INR=X"):
    global CACHE_DATA, CACHE_TIME

    if pair in CACHE_DATA:
        return CACHE_DATA[pair]

    cache_file = f"cache_{pair.replace('=', '')}.pkl"

    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))

        if (datetime.now() - file_time).seconds < CACHE_TTL:
            print(f"📦 Loading {pair} from file cache")

            with open(cache_file, "rb") as f:
                df = pickle.load(f)

            CACHE_DATA[pair] = df
            CACHE_TIME[pair] = datetime.now()

            return df

    raise HTTPException(
        status_code=503,
        detail="Data not ready yet. Please wait for scheduler."
    )


# ---------------- FUNCTION USED BY SCHEDULER ----------------
def load_and_store_features(pair: str):
    fetcher = DataFetcher()
    pre = DataPreprocessor()
    fe = FeatureEngineer()

    macro = fetcher.fetch_macro_indicators()
    forex = fetcher.fetch_forex_data(pair=pair)

    df = pre.preprocess(macro_data=macro, forex_data=forex)
    df = fe.engineer_all_features(df)

    if df.empty:
        raise Exception("Feature dataframe is empty")

    CACHE_DATA[pair] = df
    CACHE_TIME[pair] = datetime.now()

    cache_file = f"cache_{pair.replace('=', '')}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    print(f"💾 Saved {pair} to file cache")

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
    pred = predict_sarimax(df)
    return {"prediction": safe_value(pred.iloc[-1])}


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
    current_df = df.copy()
    current_date = pd.Timestamp.today().normalize()

    for i in range(steps + 1):

        try:
            if model_type == "xgb":
                latest = current_df.tail(1)
                pred = float(predict_xgboost(latest).iloc[0])
            else:
                pred_series = predict_sarimax(current_df)
                pred = float(pred_series.iloc[-1])

            if pd.isna(pred) or np.isinf(pred):
                pred = float(current_df["close"].iloc[-1])

        except:
            pred = float(current_df["close"].iloc[-1])

        if i > 0:
            current_date += pd.Timedelta(days=1)

        results.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted_close": pred
        })

        new_row = current_df.tail(1).copy()
        prev_close = float(current_df["close"].iloc[-1])
        new_row["close"] = pred

        if "returns" in new_row.columns and prev_close != 0:
            new_row["returns"] = np.log(pred / prev_close)

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
        description="Enter date in format YYYY-MM-DD",
        examples={"example": {"value": "2026-04-24"}}
    )
):
    df = get_latest_features()
    steps = calculate_days(target_date)
    preds = iterative_forecast(df, "xgb", steps)

    return {"target_date": target_date, "predictions": preds}


@router.get("/sarimax/future")
def sarimax_future(
    target_date: str = Query(
        ...,
        description="Enter date in format YYYY-MM-DD",
        examples={"example": {"value": "2026-04-24"}}
    )
):
    df = get_latest_features()
    steps = calculate_days(target_date)
    preds = iterative_forecast(df, "sarimax", steps)

    return {"target_date": target_date, "predictions": preds}


# ---------------- POST TODAY ----------------
@router.post("/prediction/today")
def prediction_today(base_currency: str, target_currency: str):
    pair = build_pair(base_currency, target_currency)
    df = get_latest_features(pair=pair)

    xgb_pred = float(predict_xgboost(df.tail(1)).iloc[0])
    sarimax_pred = float(predict_sarimax(df).iloc[-1])

    return {
        "pair": f"{base_currency.upper()}_{target_currency.upper()}",
        "date": datetime.today().strftime("%Y-%m-%d"),
        "xgboost_prediction": safe_value(xgb_pred),
        "sarimax_prediction": safe_value(sarimax_pred)
    }


# ---------------- POST FUTURE ----------------
@router.post("/prediction/future")
def prediction_future(
    base_currency: str,
    target_currency: str,
    target_date: str = Query(
        ...,
        description="Enter date in format YYYY-MM-DD",
        examples={"example": {"value": "2026-04-24"}}
    )
):
    pair = build_pair(base_currency, target_currency)
    df = get_latest_features(pair=pair)

    steps = calculate_days(target_date)

    xgb_preds = iterative_forecast(df.copy(), "xgb", steps)
    sarimax_preds = iterative_forecast(df.copy(), "sarimax", steps)

    return {
        "pair": f"{base_currency.upper()}_{target_currency.upper()}",
        "target_date": target_date,
        "xgboost_predictions": xgb_preds,
        "sarimax_predictions": sarimax_preds
    }