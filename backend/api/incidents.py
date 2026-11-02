from pathlib import Path
import pandas as pd

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "incidents.csv"
)


@router.get("/incidents")
def get_incidents(
    limit: int = Query(default=50, ge=1, le=1000)
):
    # Check whether incidents file exists
    if not INPUT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Incidents file not found. Run incident_manager.py first."
        )

    # Load incidents
    df = pd.read_csv(INPUT_FILE)

    # Return limited records
    limited_df = df.head(limit)

    records = (
        limited_df
        .astype(object)
        .where(pd.notna(limited_df), None)
        .to_dict(orient="records")
    )

    return {
        "total_incidents": int(len(df)),
        "returned_records": len(records),
        "limit": limit,
        "incidents": records
    }
