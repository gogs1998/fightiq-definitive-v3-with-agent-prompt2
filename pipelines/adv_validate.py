from hydra import main
from omegaconf import OmegaConf
import pandas as pd, os
from src.utils.io import load_parquet
from src.cv.time_grouped import time_grouped_folds
from src.validation.adversarial import adversarial_validation, importance_weights
from src.utils.logging import get_logger
log = get_logger("adv_validate")
@main(config_path="../configs", config_name="cv.yaml")
def run(cfg):
    df = load_parquet("data/interim/features.parquet")
    with open("data/interim/feature_list.txt") as f: cols=[l.strip() for l in f if l.strip()]
    cv = cfg.cv
    folds = time_grouped_folds(df, time_col=cv['time_col'], group_col=cv['group_by'], n_folds=cv['n_folds'], embargo_days=cv['embargo_days'])
    tr_idx, va_idx = folds[-1]
    Xtr = df.loc[tr_idx, cols].values; Xva = df.loc[va_idx, cols].values
    res = adversarial_validation(Xtr, Xva)
    log.info(f"Adversarial AUC: {res['auc']:.4f}")
    w = importance_weights(res['probs'][:len(Xtr)])
    os.makedirs("data/interim/weights", exist_ok=True)
    import pandas as pd
    pd.DataFrame({"bout_id": df.loc[tr_idx,'bout_id'].values, "adv_weight": w}).to_parquet("data/interim/weights/adv_train_weights.parquet", index=False)
    log.info("Wrote adversarial weights for training set.")
if __name__ == "__main__": run()
