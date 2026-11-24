from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["account_id", "txn_date", "amount", "status"]


def load_transactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("missing required columns")

    df["amount"] = pd.to_numeric(df["amount"],errors="coerce")
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    return df


def add_settled_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    status=df["status"].fillna("").astype(str).str.strip().str.lower()
    df["status"]=status

    df["is_settled"]=np.where(status.to_numpy() == "settled",True,False)
    return df


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    flagged = add_settled_flag(df)
    successful = flagged[flagged["is_settled"]].copy()
    successful["txn_date"]=pd.to_datetime(successful["txn_date"]).dt.normalize()
    summary = (
        successful.groupby(["account_id","txn_date"], as_index=False)
        .agg(total_amount=("amount", "sum"), txn_count=("amount", "size"))
    )
    return summary
