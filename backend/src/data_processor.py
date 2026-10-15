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

    # Amount columns ko numeric banao
    for column in ["amount", "refund_amount"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Text fields ke extra spaces remove karo
    for column in [
        "transaction_id", "status",
        "payment_method", "gateway", "region"
    ]:
        df[column] = df[column].astype("string").str.strip()

    # Status ko consistent lowercase format mein rakho
    df["status"] = df["status"].str.lower()

    # Non-refunded transactions mein missing refund ko zero maano
    non_refunded = df["status"].isin(["success", "failed"])
    missing_refund = non_refunded & df["refund_amount"].isna()
    df.loc[missing_refund, "refund_amount"] = 0

    # Blank transaction IDs ko missing value treat karo
    df["transaction_id"] = df["transaction_id"].replace("", pd.NA)

    # Allowed transaction statuses
    valid_statuses = ["success", "failed", "refunded"]

    # Valid records ke rules
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

    # Invalid records ko exclude karo
    cleaned_df = df.loc[valid_rows].copy()

    # Duplicate transaction IDs count karo aur remove karo
    duplicate_count = int(
        cleaned_df["transaction_id"].duplicated().sum()
    )

    cleaned_df = cleaned_df.drop_duplicates(
        subset=["transaction_id"], keep="first"
    ).copy()

    # Cleaning summary
    print("\n--- Cleaning Summary ---")
    print("Original rows:", original_rows)
    print("Invalid rows removed:", invalid_count)
    print("Duplicate IDs removed:", duplicate_count)
    print("Clean rows:", len(cleaned_df))

    return cleaned_df


def save_processed_data(df):
    # Output folder nahi hai toh create karo
    PROCESSED_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    # DataFrame ko CSV mein save karo
    df.to_csv(PROCESSED_FILE, index=False)

    print("\nCleaned dataset saved successfully!")
    print("Output path:", PROCESSED_FILE)



def main():
    # Step 1: Raw dataset load karo
    df = load_transactions()

    # Step 2: Original data inspect karo
    inspect_data(df)

    # Step 3: Dataset clean aur validate karo
    cleaned_df = clean_transactions(df)

    # Step 4: Cleaned dataset save karo
    save_processed_data(cleaned_df)

    # Step 5: Cleaned data ki preview
    print("\n--- Cleaned Dataset Preview ---")
    print(cleaned_df.head())


if __name__ == "__main__":
    main()
