import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def adversarial_validation(X_train, X_val, random_state=42):
    X = np.vstack([X_train, X_val])
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_val))]
    clf = GradientBoostingClassifier(random_state=random_state)
    clf.fit(X, y); ps = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, ps); return {"auc": float(auc), "probs": ps, "model": clf}
def importance_weights(train_probs, clip=(0.05,0.95)):
    p_train = 1 - np.clip(train_probs, *clip); p_val = np.clip(train_probs, *clip); return p_val/p_train
