from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from app.api.routes import get_latest_features, build_pair, CACHE_DATA, CACHE_TIME

scheduler = BackgroundScheduler()

# choose important pairs
PAIRS = [
    ("USD", "INR"),
    ("EUR", "INR"),
]


def refresh_cache():
    print("🔄 Running scheduled cache update:", datetime.now())

    for base, target in PAIRS:
        try:
            pair = build_pair(base, target)
            df = get_latest_features(pair=pair)

            CACHE_DATA[pair] = df
            CACHE_TIME[pair] = datetime.now()

            print(f"✅ Cache updated for {pair}")

        except Exception as e:
            print(f"❌ Failed for {base}_{target}:", str(e))


def start_scheduler():
    # run every 6 hours
    scheduler.add_job(refresh_cache, "interval", hours=6)

    scheduler.start()