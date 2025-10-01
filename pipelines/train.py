from hydra import main
from omegaconf import OmegaConf
import pandas as pd, numpy as np, os, time
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from src.utils.logging import get_logger
from src.utils.io import load_parquet, ensure_dir
from src.cv.time_grouped import time_grouped_folds
from src.models.registry import make_model
from src.eval.metrics import summarize
import mlflow
log = get_logger("train")
@main(config_path="../configs", config_name="models/logreg.yaml")
def run(cfg):
    mlflow.set_experiment("FightIQ-Definitive")
    df = load_parquet("data/interim/features.parquet")
    y = df["label_a_wins"].values
    with open("data/interim/feature_list.txt") as f: cols = [l.strip() for l in f if l.strip()]
    X = df[cols].values
    from omegaconf import OmegaConf as _OC
    cv_cfg = _OC.load("configs/cv.yaml")['cv']
    folds = time_grouped_folds(df, time_col=cv_cfg['time_col'], group_col=cv_cfg['group_by'],
                               n_folds=cv_cfg['n_folds'], embargo_days=cv_cfg['embargo_days'], seed=cv_cfg['seed'])
    oof = np.zeros(len(df), dtype=float)
    run_name = f"{cfg.model.name}-" + time.strftime('%Y%m%d-%H%M%S')
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model": cfg.model.name, **dict(cfg.model.params)})
        mlflow.log_params({"cv_n_folds": cv_cfg['n_folds'], "embargo_days": cv_cfg['embargo_days']})
        for i,(tr_idx,va_idx) in enumerate(folds):
            Xtr,Xva = X[tr_idx], X[va_idx]; ytr,yva = y[tr_idx], y[va_idx]
            model = make_model(cfg.model.name, dict(cfg.model.params))
            Xtr_cal, Xcal, ytr_cal, ycal = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
            model.fit(Xtr_cal, ytr_cal)
            cal = getattr(cfg.model, "calibration", "none")
            if cal=="platt":
                calibrator = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv="prefit"); calibrator.fit(Xcal, ycal); p = calibrator.predict_proba(Xva)[:,1]
            elif cal=="isotonic":
                p_cal = model.predict_proba(Xcal)[:,1]
                iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(p_cal, ycal); p = iso.transform(model.predict_proba(Xva)[:,1])
            else:
                p = model.predict_proba(Xva)[:,1]
            oof[va_idx]=p
        ensure_dir("data/interim/oof")
        out = pd.DataFrame({"bout_id": df["bout_id"], "event_id": df["event_id"], "y": y, "p": oof, "bout_datetime": df["bout_datetime"]})
        path = os.path.join("data/interim/oof", f"{cfg.model.name}_oof.parquet")
        out.to_parquet(path, index=False)
        metrics_all = summarize(y, oof)
        mlflow.log_metrics(metrics_all); mlflow.log_artifact(path)
        log.info(f"OOF metrics: {metrics_all}")
if __name__ == "__main__": run()
