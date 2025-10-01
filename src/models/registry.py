from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
def make_model(name: str, params: dict):
    n = name.lower()
    if n == "logreg": return LogisticRegression(**params)
    if n == "rf": return RandomForestClassifier(**params, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")
