from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


# ---------------- CREATE APP ----------------
app = FastAPI(
    title="Forex Prediction API",
    description="Predict Forex Close Prices using XGBoost & SARIMAX",
    version="1.0.0"
)


# ---------------- CORS (IMPORTANT for frontend later) ----------------
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


# ---------------- INCLUDE ROUTES ----------------
app.include_router(router)