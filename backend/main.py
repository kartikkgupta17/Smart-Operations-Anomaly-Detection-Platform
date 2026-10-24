import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI

from src.analytics import (
    load_cleaned_data,
    calculate_kpis,
    gateway_performance,
)

app = FastAPI(title="Smart Operations Platform")


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
