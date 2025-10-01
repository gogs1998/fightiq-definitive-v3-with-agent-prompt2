from hydra import main
from omegaconf import OmegaConf
import pandas as pd, os
from src.utils.logging import get_logger
from src.utils.io import ensure_dir, save_parquet
from src.data_build.synthetic import make_synthetic_fights
log = get_logger("build_data")
@main(config_path="../configs", config_name="dataset.yaml")
def run(cfg):
    cutoff = pd.to_datetime(cfg.dataset.cutoff)
    fights = make_synthetic_fights(n_events=160, fights_per_event=7, seed=cfg.dataset.seed)
    fights = fights[fights['bout_datetime']<=cutoff].copy()
    ensure_dir(cfg.dataset.processed_dir)
    save_parquet(fights, os.path.join(cfg.dataset.processed_dir, "fights.parquet"))
    log.info(f"Wrote processed fights: {len(fights)} rows")
if __name__ == "__main__": run()
