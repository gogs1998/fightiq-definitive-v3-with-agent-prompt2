import pandas as pd, numpy as np
def time_grouped_folds(df: pd.DataFrame, time_col: str, group_col: str, n_folds: int = 5, embargo_days: int = 14, seed: int = 42):
    order = df.groupby(group_col)[time_col].min().sort_values().index.tolist()
    groups = np.array(order)
    fold_sizes = np.array_split(groups, n_folds)
    folds = []
    for i in range(n_folds):
        val_groups = set(fold_sizes[i])
        val_idx = df[group_col].isin(val_groups)
        val_min = df.loc[val_idx, time_col].min()
        val_max = df.loc[val_idx, time_col].max()
        emb_lo = val_min - pd.Timedelta(days=embargo_days)
        emb_hi = val_max + pd.Timedelta(days=embargo_days)
        train_idx = (~val_idx) & ~((df[time_col] >= emb_lo) & (df[time_col] <= emb_hi))
        folds.append((train_idx.values, val_idx.values))
    return folds
