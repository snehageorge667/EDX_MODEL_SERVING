"""Feature engineering module."""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create modeling features for XGBoost and SARIMAX."""

    def __init__(self):
        self.eps = 1e-12
        self.id_columns = [
            "date",
            "currency_pair",
            "base_currency",
            "target_currency",
            "base_iso",
            "target_iso",
        ]

    # ---------------- BASE FEATURES ----------------
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["currency_pair", "date"]).reset_index(drop=True)

        df["cc_log_return"] = df.groupby("currency_pair")["close"].transform(
            lambda series: np.log(series + self.eps).diff()
        )
        df["oc_log_return"] = np.log((df["close"] + self.eps) / (df["open"] + self.eps))
        df["overnight_gap"] = np.log(
            (df["open"] + self.eps) / (df.groupby("currency_pair")["close"].shift(1) + self.eps)
        )
        df["hl_spread_pct"] = (df["high"] - df["low"]) / (df["open"] + self.eps)
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + self.eps)
        df["upper_shadow_pct"] = (
            df["high"] - df[["open", "close"]].max(axis=1)
        ) / (df["open"] + self.eps)
        df["lower_shadow_pct"] = (
            df[["open", "close"]].min(axis=1) - df["low"]
        ) / (df["open"] + self.eps)

        return df

    # ---------------- LAG ----------------
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] | None = None) -> pd.DataFrame:
        if lags is None:
            lags = [1, 2, 3, 5, 7, 10, 14]

        cols = [
            "cc_log_return",
            "oc_log_return",
            "overnight_gap",
            "hl_spread_pct",
            "close_position",
            "upper_shadow_pct",
            "lower_shadow_pct",
        ]

        df = df.copy()

        for col in cols:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df.groupby("currency_pair")[col].shift(lag)

        return df

    # ---------------- ROLLING ----------------
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for w in [3, 5, 7, 14, 21]:
            df[f"return_mean_{w}"] = df.groupby("currency_pair")["cc_log_return"].transform(
                lambda s: s.rolling(w).mean()
            )
            df[f"return_std_{w}"] = df.groupby("currency_pair")["cc_log_return"].transform(
                lambda s: s.rolling(w).std()
            )

        return df

    # ---------------- MOMENTUM ----------------
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for lag in [1, 3, 5]:
            df[f"return_accel_{lag}"] = df.groupby("currency_pair")["cc_log_return"].transform(
                lambda s: s.diff(lag)
            )

        return df

    # ---------------- RSI ----------------
    def create_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        def compute_rsi(series: pd.Series):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            rs = gain / (loss + self.eps)
            return 100 - (100 / (1 + rs))

        df = df.copy()
        df["rsi_14"] = df.groupby("currency_pair")["close"].transform(compute_rsi)

        return df

    # ---------------- TARGET ----------------
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target_next_log_return"] = df.groupby("currency_pair")["cc_log_return"].shift(-1)
        return df

    # ---------------- MAIN PIPELINE ----------------
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_base_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_rsi(df)
        df = self.create_target(df)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)

        print(f"Final feature dataset shape: {df.shape}")

        return df

    # ---------------- SPLIT ----------------
    def split_by_date(self, df: pd.DataFrame, cutoff: str):
        cutoff = pd.Timestamp(cutoff)

        train = df[df["date"] < cutoff].copy()
        test = df[df["date"] >= cutoff].copy()

        if train.empty or test.empty:
            raise ValueError("Bad split")

        return train, test

    # ---------------- FEATURE SELECTION ----------------
    def _get_features(self, df: pd.DataFrame):
        excluded = set(
            self.id_columns + ["open", "high", "low", "close", "target_next_log_return"]
        )
        return [c for c in df.columns if c not in excluded]

    # ---------------- XGBOOST ----------------
    def get_xgboost_data(self, train, test):
        features = self._get_features(train)

        x_train = train[features]
        y_train = train["target_next_log_return"]

        x_test = test[features]
        y_test = test["target_next_log_return"]

        return x_train, y_train, x_test, y_test, features

    # ---------------- SARIMAX ----------------
    def get_sarimax_data(self, train, test):
        """Dynamically include macro features if available."""

        base_features = ["close"]

        macro_candidates = [
            "CPI",
            "INFLATION",
            "UNEMPLOYMENT",
            "FEDERAL_FUNDS_RATE",
            "TREASURY_YIELD",
            "REAL_GDP"
        ]

        # Keep only available macro columns
        available_macro = [col for col in macro_candidates if col in train.columns]

        selected_features = base_features + available_macro

        print("SARIMAX FEATURES:", selected_features)

        y_train = train["target_next_log_return"]
        y_test = test["target_next_log_return"]

        exog_train = train[selected_features]
        exog_test = test[selected_features]

        return y_train, y_test, exog_train, exog_test, selected_features