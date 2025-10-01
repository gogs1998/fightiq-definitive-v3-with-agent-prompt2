import os, pandas as pd
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def save_parquet(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path)); df.to_parquet(path, index=False)
def load_parquet(path: str) -> pd.DataFrame: return pd.read_parquet(path)
