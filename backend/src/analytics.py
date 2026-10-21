from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# Cleaned dataset ka path
PROCESSED_FILE = (
    BASE_DIR / "data" / "processed"
    / "transactions_cleaned.csv"
)

def load_cleaned_data():
    """Load the cleaned transaction dataset."""

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. "
            "Run data_processor.py first."
        )

    df = pd.read_csv(PROCESSED_FILE)

    required_columns = [
        "transaction_id",
        "amount",
        "status",
        "gateway",
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


def calculate_kpis(df):
    """Calculate overall transaction KPIs."""

    total_transactions = len(df)

    successful = df.query('status == "success"')
    failed = df[df["status"] == "failed"]
    refunded = df[df["status"] == "refunded"]

    successful_count = len(successful)
    failed_count = len(failed)
    refunded_count = len(refunded)

    success_rate = (
        successful_count / total_transactions * 100
        if total_transactions > 0 else 0
    )

    failure_rate = (
        failed_count / total_transactions * 100
        if total_transactions > 0 else 0
    )

    successful_amount = successful["amount"].sum()

    total_refunded_amount = refunded["refund_amount"].sum()

    average_successful_amount = (
        successful["amount"].mean()
        if successful_count > 0 else 0
    )

    kpis = {
        "total_transactions": total_transactions,
        "successful_transactions": successful_count,
        "failed_transactions": failed_count,
        "refunded_transactions": refunded_count,
        "success_rate_percent": round(success_rate, 2),
        "failure_rate_percent": round(failure_rate, 2),
        "successful_transaction_value": round(
            float(successful_amount), 2
        ),
        "total_refunded_amount": round(
            float(total_refunded_amount), 2
        ),
        "average_successful_amount": round(
            float(average_successful_amount), 2
        ),
    }

    return kpis


def gateway_performance(df):
    """Compare transaction performance across gateways."""

    performance = (
        df.groupby("gateway")
        .agg(
            total_transactions=("transaction_id", "count"),
            successful_transactions=(
                "status",
                lambda values: (values == "success").sum()
            ),
            failed_transactions=(
                "status",
                lambda values: (values == "failed").sum()
            ),
            total_transaction_value=("amount", "sum"),
        )
        .reset_index()
    )

    performance["failure_rate_percent"] = (
        performance["failed_transactions"]
        / performance["total_transactions"]
        * 100
    ).round(2)

    return performance


def main():
    df = load_cleaned_data()

    print("\n========== OVERALL KPIs ==========")

    kpis = calculate_kpis(df)

    for metric, value in kpis.items():
        print(f"{metric}: {value}")

    print("\n======= GATEWAY PERFORMANCE =======")

    gateway_df = gateway_performance(df)

    print(gateway_df.to_string(index=False))


if __name__ == "__main__":
    main()
