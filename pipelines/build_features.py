from hydra import main
from omegaconf import OmegaConf
import pandas as pd, os
from src.utils.logging import get_logger
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.features.groups import build_feature_matrix
log = get_logger("build_features")
@main(config_path="../configs", config_name="features/base.yaml")
def run(cfg):
    fights = load_parquet("data/processed/fights.parquet")
    X, cols = build_feature_matrix(fights, dict(cfg.features))
    ensure_dir("data/interim")
    save_parquet(X, "data/interim/features.parquet")
    with open("data/interim/feature_list.txt","w") as f:
        for c in cols: f.write(c+"\n")
    log.info(f"Wrote features with {len(cols)} columns and {len(X)} rows.")
if __name__ == "__main__": run()
