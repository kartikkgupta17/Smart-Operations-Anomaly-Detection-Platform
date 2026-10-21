from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "transactions_anomalies.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "incidents.csv"
)

def load_anomalies():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Anomaly results not found. "
            "Run anomaly_detector.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "timestamp",
        "transaction_id",
        "amount",
        "status",
        "gateway",
        "is_anomaly",
        "anomaly_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return df

def assign_severity(amount, status):
    if amount >= 25000:
        return "High"

    if status == "failed" and amount >= 5000:
        return "High"

    if amount >= 5000 or status == "failed":
        return "Medium"

    return "Low"


def create_incidents(df):
    anomaly_mask = (
        df["is_anomaly"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    anomalies = df.loc[anomaly_mask].copy()

    if anomalies.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "transaction_id",
                "timestamp",
                "gateway",
                "amount",
                "status",
                "anomaly_score",
                "severity",
                "reason",
                "incident_status",
            ]
        )

    anomalies["amount"] = pd.to_numeric(
        anomalies["amount"], errors="coerce"
    )

    anomalies["status"] = (
        anomalies["status"].astype(str).str.lower()
    )

    anomalies["severity"] = anomalies.apply(
        lambda row: assign_severity(
            row["amount"], row["status"]
        ),
        axis=1,
    )

    anomalies["incident_id"] = [
        f"INC-{number:04d}"
        for number in range(1, len(anomalies) + 1)
    ]

    anomalies["reason"] = (
        "Transaction flagged as unusual by Isolation Forest"
    )

    anomalies["incident_status"] = "Open"

    incident_columns = [
        "incident_id",
        "transaction_id",
        "timestamp",
        "gateway",
        "amount",
        "status",
        "anomaly_score",
        "severity",
        "reason",
        "incident_status",
    ]

    return anomalies[incident_columns]


def save_incidents(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nIncident records saved!")
    print("Output path:", OUTPUT_FILE)


def main():
    df = load_anomalies()

    incidents = create_incidents(df)

    print("\n--- Incident Summary ---")
    print("Total incidents:", len(incidents))

    if not incidents.empty:
        print("\nSeverity counts:")
        print(incidents["severity"].value_counts())

        print("\nSample incidents:")
        print(incidents.head(10).to_string(index=False))

    save_incidents(incidents)


if __name__ == "__main__":
    main()
