from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread

from app.api.routes import router
from app.scheduler import start_scheduler, refresh_cache


# ---------------- CREATE APP ----------------
app = FastAPI(
    title="Forex Prediction API",
    description="Predict Forex Close Prices using XGBoost & SARIMAX",
    version="1.0.0"
)


# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- HEALTH CHECK ----------------
@app.get("/")
def home():
    return {
        "message": "Forex Prediction API is running",
        "endpoints": [
            "/xgboost/today",
            "/sarimax/today",
            "/xgboost/future?target_date=YYYY-MM-DD",
            "/sarimax/future?target_date=YYYY-MM-DD"
        ]
    }


# Prevent duplicate scheduler start
scheduler_started = False


# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    global scheduler_started

    print("🚀 App starting...")

    # Run initial cache load
    Thread(target=refresh_cache).start()

    # Start scheduler ONLY once
    if not scheduler_started:
        start_scheduler()
        scheduler_started = True
        print(" Scheduler started")


# ---------------- INCLUDE ROUTES ----------------
app.include_router(router)