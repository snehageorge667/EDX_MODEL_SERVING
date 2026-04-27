from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from app.api.routes import build_pair, CACHE_DATA, CACHE_TIME, load_and_store_features
from app.services.data_fetcher import DataFetcher

scheduler = BackgroundScheduler()

PAIRS = [
    ("USD", "INR"),
]


def is_new_data_available(pair: str):
    """
    Check if latest forex data date is newer than cached data
    """
    try:
        fetcher = DataFetcher()

        latest_df = fetcher.fetch_forex_data(pair=pair, days=2)

        if latest_df.empty:
            return False

        latest_date = latest_df["date"].max()

        if pair in CACHE_DATA:
            cached_date = CACHE_DATA[pair]["date"].max()
            return latest_date > cached_date

        return True  # no cache → must load

    except Exception as e:
        print(f"⚠️ Data check failed for {pair}: {e}")
        return False


def refresh_cache():
    print("🔄 Running scheduled cache update:", datetime.now())

    for base, target in PAIRS:
        try:
            pair = build_pair(base, target)

            # condition 1 → 6 hour expiry
            expired = (
                pair not in CACHE_TIME or
                (datetime.now() - CACHE_TIME[pair]).seconds > 21600
            )

            # condition 2 → new market data
            new_data = is_new_data_available(pair)

            if expired or new_data:
                print(f"⬇️ Updating data for {pair} (expired={expired}, new_data={new_data})")

                df = load_and_store_features(pair)

                CACHE_DATA[pair] = df
                CACHE_TIME[pair] = datetime.now()

                print(f"✅ Cache updated for {pair}")

            else:
                print(f"⏩ Skipping {pair} (no update needed)")

        except Exception as e:
            print(f"❌ Failed for {base}_{target}: {str(e)}")


def start_scheduler():
    # run every 1 hour (not 6)
    # because we need to CHECK frequently
    scheduler.add_job(refresh_cache, "interval", hours=1)

    scheduler.start()