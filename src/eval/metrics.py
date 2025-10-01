import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
def safe_auc(y, p):
    try: return roc_auc_score(y, p)
    except Exception: return float("nan")
def ece_score(y_true, p_pred, n_bins=15):
    import numpy as np
    bins = np.linspace(0.0,1.0,n_bins+1); idx = np.digitize(p_pred, bins)-1
    ece=0.0; total=len(y_true)
    for b in range(n_bins):
        m = idx==b
        if m.sum()==0: continue
        conf = p_pred[m].mean(); acc = y_true[m].mean()
        ece += (m.sum()/total)*abs(acc-conf)
    return ece
def summarize(y_true, p_pred):
    return {"logloss": float(log_loss(y_true, p_pred, labels=[0,1])),
            "brier": float(brier_score_loss(y_true, p_pred)),
            "auc": float(safe_auc(y_true, p_pred)),
            "ece": float(ece_score(y_true, p_pred))}
