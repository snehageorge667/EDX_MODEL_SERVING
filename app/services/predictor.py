import pickle
import pandas as pd
import numpy as np


# ---------------- LOAD MODELS ----------------
with open("app/models/xgboost_model.pkl", "rb") as f:
    xgb_data = pickle.load(f)

with open("app/models/sarimax_model.pkl", "rb") as f:
    sarimax_data = pickle.load(f)

xgb_model = xgb_data["model"]
xgb_features = xgb_data["features"]

sarimax_model = sarimax_data["model"]
sarimax_features = sarimax_data["features"]


# ---------------- COMMON SAFE ----------------
def safe_number(x):
    if pd.isna(x) or np.isinf(x):
        return 0.0
    return float(x)


# ---------------- XGBOOST ----------------
def predict_xgboost(df: pd.DataFrame):
    df = df.copy()

    # STEP 1: get last close BEFORE feature selection
    if "close" in df.columns:
        last_close = df["close"].iloc[0]
    else:
        last_close = 1  # fallback

    # STEP 2: ensure all features exist
    for col in xgb_features:
        if col not in df.columns:
            df[col] = 0

    # STEP 3: select only model features
    df_model = df[xgb_features].copy()

    # STEP 4: clean data
    df_model = df_model.fillna(0)

    # STEP 5: predict return
    pred_return = xgb_model.predict(df_model)
    pred_return = pd.Series(pred_return, index=df_model.index)

    # STEP 6: convert return → price
    pred_price = last_close * np.exp(pred_return)

    # STEP 7: clean invalid values
    pred_price = pred_price.apply(safe_number)

    return pred_price


# ---------------- SARIMAX ----------------
def predict_sarimax(df: pd.DataFrame):
    df = df.copy()

    # STEP 1: get last close first
    if "close" in df.columns:
        last_close = df["close"].iloc[0]
    else:
        last_close = 1

    # STEP 2: ensure required features exist
    for col in sarimax_features:
        if col not in df.columns:
            df[col] = 0

    # STEP 3: prepare exogenous variables
    exog = df[sarimax_features].copy()
    exog = exog.ffill().bfill().fillna(0)

    # STEP 4: forecast
    preds = sarimax_model.forecast(
        steps=len(exog),
        exog=exog
    )

    preds = pd.Series(preds)

    # STEP 5: convert return → price
    preds = last_close * np.exp(preds)

    # STEP 6: clean invalid values
    preds = preds.apply(safe_number)

    return preds