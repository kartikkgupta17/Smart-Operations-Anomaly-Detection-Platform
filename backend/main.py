import pandas as pd
import numpy as np


from fastapi import FastAPI

from src.analytics import (
    load_cleaned_data,
    calculate_kpis,
    gateway_performance,
)
from api.incidents import router as incidents_router
from api.anomalies import router as anomalies_router

app = FastAPI(title="Smart Operations Platform")
app.include_router(anomalies_router)
app.include_router(incidents_router)
@app.get("/")
def home():
    return {
        "message": "Smart Operations Platform is running!",
        "status": "success"
    }

@app.get("/health")
def health_check():
    return {
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "ml_model": "IsolationForest",
        "status": "healthy"
    }
@app.get("/analytics/kpis")
def get_analytics_kpis():
    df = load_cleaned_data()
    kpis = calculate_kpis(df)

    return kpis

@app.get("/analytics/gateways")
def get_gateway_performance():
    df = load_cleaned_data()

    gateway_df = gateway_performance(df)

    return gateway_df.to_dict(orient="records")

