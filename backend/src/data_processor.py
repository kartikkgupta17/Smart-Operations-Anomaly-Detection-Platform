from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = BASE_DIR / "data" / "raw" / "transactions.csv"
PROCESSED_FILE = (
    BASE_DIR / "data" / "processed" / "transactions_cleaned.csv"
)

REQUIRED_COLUMNS = [
    "timestamp",
    "transaction_id",
    "amount",
    "status",
    "payment_method",
    "gateway",
    "region",
    "refund_amount",
]


def load_transactions():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_FILE}"
        )


    df = pd.read_csv(RAW_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df

def inspect_data(df):
    print("\n--- Dataset Preview ---")
    print(df.head())

    print("\n--- Dataset Shape ---")
    print("Rows and columns:", df.shape)

    print("\n--- Data Types ---")
    df.info()

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Transaction IDs ---")
    print(df["transaction_id"].duplicated().sum())

    print("\n--- Transaction Status Counts ---")
    print(df["status"].value_counts())


def clean_transactions(df):
    df = df.copy()
    original_rows = len(df)
    df.columns = df.columns.str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    for column in ["amount", "refund_amount"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    for column in [
        "transaction_id", "status",
        "payment_method", "gateway", "region"
    ]:
        df[column] = df[column].astype("string").str.strip()

    df["status"] = df["status"].str.lower()

    non_refunded = df["status"].isin(["success", "failed"])
    missing_refund = non_refunded & df["refund_amount"].isna()
    df.loc[missing_refund, "refund_amount"] = 0

    df["transaction_id"] = df["transaction_id"].replace("", pd.NA)

    valid_statuses = ["success", "failed", "refunded"]

    valid_rows = (
        df["transaction_id"].notna()
        & df["timestamp"].notna()
        & df["amount"].notna()
        & df["refund_amount"].notna()
        & df["amount"].ge(0)
        & df["refund_amount"].ge(0)
        & df["refund_amount"].le(df["amount"])
        & df["status"].isin(valid_statuses)
    )

    invalid_count = int((~valid_rows).sum())

    cleaned_df = df.loc[valid_rows].copy()

    duplicate_count = int(
        cleaned_df["transaction_id"].duplicated().sum()
    )

    cleaned_df = cleaned_df.drop_duplicates(
        subset=["transaction_id"], keep="first"
    ).copy()

    print("\n--- Cleaning Summary ---")
    print("Original rows:", original_rows)
    print("Invalid rows removed:", invalid_count)
    print("Duplicate IDs removed:", duplicate_count)
    print("Clean rows:", len(cleaned_df))

    return cleaned_df


def save_processed_data(df):
    PROCESSED_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(PROCESSED_FILE, index=False)

    print("\nCleaned dataset saved successfully!")
    print("Output path:", PROCESSED_FILE)


def main():
    df = load_transactions()

    inspect_data(df)

    cleaned_df = clean_transactions(df)

    save_processed_data(cleaned_df)

    print("\n--- Cleaned Dataset Preview ---")
    print(cleaned_df.head())


if __name__ == "__main__":
    main()
