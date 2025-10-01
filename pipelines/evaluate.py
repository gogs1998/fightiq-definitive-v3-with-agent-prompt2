from hydra import main
from omegaconf import OmegaConf
import pandas as pd, os, time
from src.utils.logging import get_logger
from src.utils.io import load_parquet
from src.eval.metrics import summarize
import mlflow
log = get_logger("evaluate")
@main(config_path="../configs", config_name="cv.yaml")
def run(cfg):
    cv = cfg.cv; exp_id = getattr(cfg, "exp_id", None); note = getattr(cfg, "note", None)
    if os.path.exists("data/interim/oof/ensemble_oof.parquet"):
        df = load_parquet("data/interim/oof/ensemble_oof.parquet"); name="ensemble"
    else:
        import glob
        cands = glob.glob("data/interim/oof/*_oof.parquet")
        if not cands: log.error("No OOF files found."); return
        df = load_parquet(cands[0]); name=os.path.basename(cands[0])
    feats = load_parquet("data/interim/features.parquet")[["bout_id","bout_datetime"]]
    df = df.merge(feats, on="bout_id", how="left")
    lock_lo = pd.to_datetime(cv['test_lockbox']['start']); lock_hi = pd.to_datetime(cv['test_lockbox']['end'])
    m_lock = (df['bout_datetime']>=lock_lo) & (df['bout_datetime']<=lock_hi); m_oof = ~m_lock
    m_all = summarize(df['y'].values, df['p'].values)
    m_hist= summarize(df.loc[m_oof,'y'].values, df.loc[m_oof,'p'].values)
    m_lockm= summarize(df.loc[m_lock,'y'].values, df.loc[m_lock,'p'].values)
    log.info(f"Model: {name} | Overall {m_all} | Historic {m_hist} | Lockbox {m_lockm}")
    mlflow.set_experiment("FightIQ-Definitive")
    run_name = f"evaluate-{name}-" + time.strftime('%Y%m%d-%H%M%S')
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model_name": name, "lockbox_start": str(lock_lo.date()), "lockbox_end": str(lock_hi.date()), "rows_all": int(len(df)), "rows_lockbox": int(m_lock.sum())})
        mlflow.log_metrics({f"overall_{k}": v for k,v in m_all.items()})
        mlflow.log_metrics({f"historic_{k}": v for k,v in m_hist.items()})
        mlflow.log_metrics({f"lockbox_{k}": v for k,v in m_lockm.items()})
        out_csv = "reports/eval_predictions.csv"; os.makedirs("reports", exist_ok=True); df.to_csv(out_csv, index=False); mlflow.log_artifact(out_csv); run_id = run.info.run_id
    import csv, datetime as _dt
    row=[(exp_id or ""), _dt.datetime.utcnow().strftime("%Y-%m-%d"), f"Evaluation {name}", "DONE", f"logloss={m_all['logloss']:.5f} lockbox={m_lockm['logloss']:.5f}", run_id, (note or "")]
    with open("docs/experiments.csv","a",newline="",encoding="utf-8") as f: csv.writer(f).writerow(row)
if __name__ == "__main__": run()
