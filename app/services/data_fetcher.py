"""Data fetching module for economic indicators and forex data."""

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict
import time


class DataFetcher:
    """Fetch economic indicators and forex data from external APIs."""

    def __init__(self, api_key: str = None):
        
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set in environment")

        self.base_url = "https://www.alphavantage.co/query"

    # ---------------- FETCH MACRO ----------------
    def fetch_alpha_vantage(self, function: str, retry: int = 0, **params) -> pd.DataFrame:
        try:
            response = requests.get(
                self.base_url,
                params={"function": function, "apikey": self.api_key, **params},
                timeout=10
            )
            response.raise_for_status()
            resp = response.json()

            #  RATE LIMIT HANDLING (safe retry max 2 times)
            if "Note" in resp:
                if retry < 2:
                    print(f"[{function}] Rate limit hit → retrying...")
                    time.sleep(20)
                    return self.fetch_alpha_vantage(function, retry=retry + 1, **params)
                else:
                    print(f"[{function}] Rate limit exceeded (skipping)")
                    return pd.DataFrame(columns=["date", "value"])

            # API ERROR
            if "Error Message" in resp:
                print(f"[{function}] API error")
                return pd.DataFrame(columns=["date", "value"])

            if "data" not in resp:
                print(f"[{function}] Unexpected response")
                return pd.DataFrame(columns=["date", "value"])

            df = pd.DataFrame(resp["data"])

            if df.empty or "date" not in df.columns:
                return pd.DataFrame(columns=["date", "value"])

            return df

        except Exception as e:
            print(f"Error fetching {function}: {str(e)}")
            return pd.DataFrame(columns=["date", "value"])

    # ---------------- PARSE ----------------
    def parse_alpha_vantage(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.rename(columns={"value": col_name})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        return df.dropna().sort_values("date").reset_index(drop=True)

    # ---------------- FETCH ALL MACRO ----------------
    def fetch_macro_indicators(self) -> Dict[str, pd.DataFrame]:
        indicators = {
            "REAL_GDP": {"interval": "quarterly"},
            "REAL_GDP_PER_CAPITA": {"interval": "quarterly"},
            "TREASURY_YIELD": {"interval": "monthly", "maturity": "10year"},
            "FEDERAL_FUNDS_RATE": {"interval": "monthly"},
            "CPI": {"interval": "monthly"},
            "INFLATION": {"interval": "yearly"},
            "UNEMPLOYMENT": {"interval": "monthly"},
        }

        data = {}

        for indicator, params in indicators.items():
            print(f"Fetching {indicator}...")

            df = self.fetch_alpha_vantage(indicator, **params)
            df = self.parse_alpha_vantage(df, indicator)

            data[indicator] = df

            # Slight delay to avoid rate limit (optimized)
            time.sleep(12)

        return data

    # ---------------- FETCH FOREX ----------------
    def fetch_forex_data(self, pair: str = "INR=X", days: int = 180) -> pd.DataFrame:
        try:
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

            print(f"Fetching {pair} data...")

            df = yf.download(pair, start=start_date, end=end_date, progress=False)

            if df.empty:
                print("Forex data empty")
                return pd.DataFrame()

            # FIX multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[['Open', 'High', 'Low', 'Close']].copy()
            df.columns = ['open', 'high', 'low', 'close']

            df.index.name = 'date'
            df = df.reset_index()

            df['date'] = pd.to_datetime(df['date'], errors="coerce")

            return df.dropna().sort_values('date').reset_index(drop=True)

        except Exception as e:
            print(f"Error fetching forex data: {str(e)}")
            return pd.DataFrame()

    # ---------------- MERGE ----------------
    def merge_all_data(self, forex_df: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        if forex_df.empty:
            raise ValueError("Forex data is empty")

        df = forex_df.copy()

        for name, macro_df in macro_data.items():

            if macro_df.empty:
                continue

            macro_df = macro_df.set_index("date").resample("D").ffill().reset_index()

            df = pd.merge(df, macro_df, on="date", how="left")

        df = df.ffill().dropna()

        print(f"Final merged dataset shape: {df.shape}")

        return df

    # ---------------- FULL PIPELINE ----------------
    def fetch_all(self) -> pd.DataFrame:

        macro_data = self.fetch_macro_indicators()
        forex_df = self.fetch_forex_data()

        final_df = self.merge_all_data(forex_df, macro_data)

        return final_df