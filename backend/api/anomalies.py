from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_anomalies.csv"
)

@router.get("/anomalies")
def get_anomalies(
    limit: int = Query(default=50, ge=1, le=1000)
):
    if not INPUT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Anomaly results not found. Run anomaly_detector.py first.",
        ) 

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "transaction_id",
        "amount",
        "status",
        "anomaly_score",
        "is_anomaly",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Missing columns: {missing_columns}",
        )

    anomaly_mask = (
        df["is_anomaly"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    anomalies = df.loc[anomaly_mask]
    selected = anomalies.head(limit)

    records = (
        selected.astype(object)
        .where(pd.notna(selected), None)
        .to_dict(orient="records")
    )

    return {
        "total_anomalies": int(len(anomalies)),
        "returned_records": len(records),
        "limit": limit,
        "anomalies": records,
    }
