from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    print("🚀 App starting...")

    # preload cache once
    refresh_cache()

    # start scheduler
    start_scheduler()


# ---------------- INCLUDE ROUTES ----------------
app.include_router(router)