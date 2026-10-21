from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "transactions_cleaned.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "transactions_anomalies.csv"
)


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "transaction_id",
        "amount",
        "status",
        "refund_amount",
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

def prepare_features(df):
    features = pd.DataFrame(index=df.index)

    features["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    features["refund_amount"] = pd.to_numeric(
        df["refund_amount"], errors="coerce"
    )

    features["is_failed"] = (
        df["status"].eq("failed").astype(int)
    )

    features["is_refunded"] = (
        df["status"].eq("refunded").astype(int)
    )

    return features

def detect_anomalies(df):
    features = prepare_features(df)

    valid_rows = features.notna().all(axis=1)
    results = df.loc[valid_rows].copy()
    features = features.loc[valid_rows]

    if len(features) < 2:
        raise ValueError(
            "At least two valid transactions are required."
        )

    model = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        random_state=42
    )

    model.fit(features)

    predictions = model.predict(features)

    results["anomaly_label"] = predictions
    results["is_anomaly"] = (
        results["anomaly_label"] == -1
    )

    results["anomaly_score"] = model.decision_function(
        features
    )

    return results

def save_results(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nAnomaly results saved!")
    print("File:", OUTPUT_FILE)


def main():
    df = load_data()

    results = detect_anomalies(df)

    print("\n--- Anomaly Detection Summary ---")
    print("Transactions analysed:", len(results))
    print(
        "Anomalies detected:",
        int(results["is_anomaly"].sum())
    )

    print("\n--- Sample Anomalies ---")

    anomalies = results[results["is_anomaly"]]

    print(
        anomalies[
            [
                "transaction_id",
                "amount",
                "status",
                "anomaly_score",
            ]
        ].head(10).to_string(index=False)
    )

    save_results(results)

if __name__ == "__main__":
    main()
