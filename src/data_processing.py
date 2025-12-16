import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw transaction data and parse datetime fields.
    """
    df = pd.read_csv(path)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from TransactionStartTime.
    """
    df = df.copy()
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    return df


def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate numerical and behavioral features at customer level.
    """
    agg_df = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        std_amount=("Amount", "std"),
        transaction_count=("TransactionId", "count"),
        avg_transaction_hour=("transaction_hour", "mean"),
        avg_transaction_day=("transaction_day", "mean"),
        avg_transaction_month=("transaction_month", "mean"),
        fraud_rate=("FraudResult", "mean"),
    ).reset_index()

    return agg_df


def aggregate_categorical_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Aggregate categorical features using most frequent category per customer.
    """
    cat_df = (
        df.groupby("CustomerId")[cat_cols]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    return cat_df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final customer-level feature dataset.
    """
    df = extract_time_features(df)

    
    df = df.drop(
        columns=["CountryCode", "CurrencyCode", "Value"],
        errors="ignore",
    )

    num_features = aggregate_customer_features(df)

    cat_features = aggregate_categorical_features(
        df,
        cat_cols=["ProviderId", "ProductCategory", "ChannelId", "PricingStrategy"],
    )

    final_df = num_features.merge(cat_features, on="CustomerId", how="left")

    return final_df


def build_preprocessing_pipeline(df: pd.DataFrame):
    """
    Build sklearn preprocessing pipeline for numerical and categorical features.
    """
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_features.remove("CustomerId")

    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_features.remove("CustomerId")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def process_and_save_data(raw_path: str, output_path: str) -> pd.DataFrame:
    """
    End-to-end processing: load raw data, engineer features, save processed data.
    """
    df = load_raw_data(raw_path)
    customer_df = build_customer_features(df)
    
    final_df = integrate_target_variable(customer_df, df)
    
    final_df.to_csv(output_path, index=False)
    return final_df

# ----------------------
# Task 4: Proxy Target Variable
# ----------------------

def calculate_rfm(df, snapshot_date=None):
    """Calculate Recency, Frequency, Monetary (RFM) metrics per customer"""
    df = df.copy()
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm_df = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
    ).reset_index()

    return rfm_df

def scale_rfm(rfm_df):
    """Standardize RFM metrics for clustering"""
    scaler = StandardScaler()
    rfm_scaled = rfm_df.copy()
    rfm_scaled[['recency', 'frequency', 'monetary']] = scaler.fit_transform(
        rfm_scaled[['recency', 'frequency', 'monetary']]
    )
    return rfm_scaled

def cluster_customers(rfm_scaled, n_clusters=3, random_state=42):
    """Cluster customers using KMeans"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_scaled['cluster'] = kmeans.fit_predict(rfm_scaled[['recency', 'frequency', 'monetary']])
    return rfm_scaled

def assign_high_risk_label(rfm_clustered):
    """Label high-risk cluster as 1, others as 0"""
    cluster_stats = rfm_clustered.groupby('cluster').agg(
        recency=('recency', 'mean'),
        frequency=('frequency', 'mean'),
        monetary=('monetary', 'mean')
    ).reset_index()

    # Heuristic: high recency & low frequency & low monetary -> high risk
    cluster_stats['risk_score'] = cluster_stats['recency'] - cluster_stats['frequency'] - cluster_stats['monetary']
    high_risk_cluster = cluster_stats.sort_values('risk_score', ascending=False).iloc[0]['cluster']

    rfm_clustered['is_high_risk'] = (rfm_clustered['cluster'] == high_risk_cluster).astype(int)
    return rfm_clustered[['CustomerId', 'is_high_risk']]

def integrate_target_variable(customer_df, transaction_df):
    """Add the is_high_risk column to the customer-level dataframe"""
    rfm_df = calculate_rfm(transaction_df)
    rfm_scaled = scale_rfm(rfm_df)
    rfm_clustered = cluster_customers(rfm_scaled)
    target_df = assign_high_risk_label(rfm_clustered)

    final_df = customer_df.merge(target_df, on='CustomerId', how='left')
    return final_df




if __name__ == "__main__":
    RAW_DATA_PATH = "./data/raw/data.csv"
    PROCESSED_DATA_PATH = "./data/processed/customer_features.csv"

    print("Starting data processing...")
    process_and_save_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("Data processing completed successfully.")
