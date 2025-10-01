import numpy as np
from sklearn.metrics import log_loss
def random_simplex(n, rng):
    w = rng.random(n); return w/w.sum()
def best_weights_logloss(P, y, trials=2000, seed=42):
    rng = np.random.default_rng(seed)
    P_mat = np.vstack(P).T
    best_w = np.ones(P_mat.shape[1])/P_mat.shape[1]
    best_ll = log_loss(y, P_mat.dot(best_w), labels=[0,1])
    for _ in range(trials):
        w = random_simplex(P_mat.shape[1], rng)
        ll = log_loss(y, P_mat.dot(w), labels=[0,1])
        if ll < best_ll: best_ll, best_w = ll, w
    return best_w, best_ll
