"""Data preprocessing module."""

from __future__ import annotations
from typing import Dict, Optional

import pandas as pd


class DataPreprocessor:
    """Preprocess, align, and merge macro and forex datasets."""

    def __init__(self):
        self.currency_to_iso = {
            "USD": "USA",
            "AUD": "AUS",
            "CAD": "CAN",
            "CHF": "CHE",
            "CNY": "CHN",
            "EUR": "EUR",
            "GBP": "GBR",
            "INR": "IND",
            "JPY": "JPN",
            "NPR": "NPL",
        }

    # ---------------- EXISTING METHODS ----------------

    def _normalize_macro_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        macro_df = df.copy()
        macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
        macro_df = macro_df.dropna(subset=["date"]).sort_values("date")
        macro_df = macro_df.drop_duplicates(subset=["date"], keep="last")
        return macro_df.reset_index(drop=True)

    def merge_macro_data(self, macro_indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        usable_frames = []

        for name, df in macro_indicators.items():
            normalized_df = self._normalize_macro_frame(df)

            if normalized_df.empty:
                print(f"Skipping empty macro frame: {name}")
                continue

            usable_frames.append(normalized_df)

        # allow empty macro
        if not usable_frames:
            print("No macro data available. Proceeding without macro features.")
            return pd.DataFrame(columns=["date"])

        merged = usable_frames[0]
        for frame in usable_frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged = merged.ffill()

        print(f"Merged macro data shape: {merged.shape}")
        return merged

    def _expand_macro_to_daily(self, macro_df: pd.DataFrame, forex_df: pd.DataFrame) -> pd.DataFrame:
        start_date = forex_df["date"].min()
        end_date = forex_df["date"].max()
        daily_index = pd.date_range(start=start_date, end=end_date, freq="D")

        expanded = macro_df.set_index("date").reindex(daily_index).ffill()
        expanded.index.name = "date"
        expanded = expanded.reset_index()

        print(f"Expanded macro data to daily frequency: {expanded.shape}")
        return expanded

    def merge_forex_macro(self, forex_df: pd.DataFrame, macro_df: pd.DataFrame, pair: str = "USDINR") -> pd.DataFrame:
        forex_data = forex_df.copy()

        forex_data["date"] = pd.to_datetime(forex_data["date"], errors="coerce")
        forex_data = forex_data.dropna(subset=["date", "open", "high", "low", "close"])
        forex_data = forex_data.sort_values("date").reset_index(drop=True)

        forex_data["currency_pair"] = pair
        forex_data["base_currency"] = pair[:3]
        forex_data["target_currency"] = pair[3:]
        forex_data["base_iso"] = forex_data["base_currency"].map(self.currency_to_iso)
        forex_data["target_iso"] = forex_data["target_currency"].map(self.currency_to_iso)

        # FIX: use macro ONLY if available
        if macro_df.empty:
            print("Macro unavailable → using forex-only dataset.")
            return forex_data

        daily_macro = self._expand_macro_to_daily(macro_df, forex_data)

        merged = pd.merge_asof(
            forex_data.sort_values("date"),
            daily_macro.sort_values("date"),
            on="date",
            direction="backward",
        )

        merged = merged.ffill()
        merged = merged.dropna(subset=["date", "open", "high", "low", "close"])

        print(f"Merged forex and macro data shape: {merged.shape}")
        return merged

    # ---------------- MAIN METHOD ----------------

    def preprocess(self,
                   macro_data: Optional[Dict[str, pd.DataFrame]] = None,
                   forex_data: Optional[pd.DataFrame] = None,
                   merged_data: Optional[pd.DataFrame] = None,
                   pair: str = "USDINR") -> pd.DataFrame:

        print("Starting preprocessing pipeline")

        # CASE 1: Already merged
        if merged_data is not None:
            df = merged_data.copy()

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").dropna()

            df["currency_pair"] = pair
            df["base_currency"] = pair[:3]
            df["target_currency"] = pair[3:]

            print(f"Preprocessed merged data shape: {df.shape}")
            return df

        # CASE 2: Raw inputs
        if macro_data is None or forex_data is None:
            raise ValueError("Provide either merged_data OR (macro_data + forex_data)")

        macro_df = self.merge_macro_data(macro_data)
        processed_df = self.merge_forex_macro(forex_data, macro_df, pair=pair)

        print(f"Preprocessing complete. Final shape: {processed_df.shape}")
        return processed_df