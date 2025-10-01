from hydra import main
from omegaconf import OmegaConf
import os, pandas as pd, numpy as np
from src.utils.logging import get_logger
from src.ensemble.weights import best_weights_logloss
from src.eval.metrics import summarize
from src.utils.io import ensure_dir
log = get_logger("ensemble")
@main(config_path="../configs", config_name="ensemble/convex.yaml")
def run(cfg):
    oof_dir = "data/interim/oof"; base=[]; names=[]; y=None; df_last=None
    for name in cfg.ensemble.base_models:
        path = os.path.join(oof_dir, f"{name}_oof.parquet")
        if not os.path.exists(path):
            log.warning(f"Missing OOF for {name}; skipping."); continue
        df = pd.read_parquet(path); df_last = df
        base.append(df["p"].values); names.append(name); y = df["y"].values
    if len(base)<2: log.warning("Need >=2 base models."); return
    w, best_ll = best_weights_logloss(base, y, trials=cfg.ensemble.trials, seed=42)
    log.info(f"Weights: {dict(zip(names, map(float,w)))} | OOF logloss={best_ll:.6f}")
    p_ens = np.vstack(base).T.dot(w); ensure_dir(oof_dir)
    out = pd.DataFrame({"bout_id": df_last["bout_id"], "y": y, "p": p_ens})
    out.to_parquet(os.path.join(oof_dir, "ensemble_oof.parquet"), index=False)
    log.info(f"Ensemble metrics: {summarize(y, p_ens)}")
if __name__ == "__main__": run()
