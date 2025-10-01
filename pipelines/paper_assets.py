import os, pandas as pd, numpy as np
from src.utils.io import load_parquet, ensure_dir
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
def save_calibration(y, p, out_png):
    prob_true, prob_pred = calibration_curve(y, p, n_bins=15, strategy="uniform")
    plt.figure(); plt.plot(prob_pred, prob_true, marker="o", linestyle=""); plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.title("Reliability Diagram"); plt.tight_layout(); plt.savefig(out_png); plt.close()
def save_roc(y, p, out_png):
    fpr,tpr,_=roc_curve(y,p); auc=roc_auc_score(y,p)
    plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()
def main():
    path="data/interim/oof/ensemble_oof.parquet"
    if not os.path.exists(path):
        import glob; cands=glob.glob("data/interim/oof/*_oof.parquet")
        if not cands: print("No OOF files found."); return
        path=cands[0]
    df=load_parquet(path); y,p=df["y"].values, df["p"].values
    ensure_dir("paper/figures")
    save_calibration(y,p,"paper/figures/calibration.png"); save_roc(y,p,"paper/figures/roc.png")
    paper="paper/paper.md"
    if os.path.exists(paper):
        with open(paper,"r",encoding="utf-8") as f: s=f.read()
        if "figures/calibration.png" not in s:
            s=s.replace("## 6. Results","## 6. Results\n\n![Calibration](figures/calibration.png)\n\n![ROC](figures/roc.png)")
        with open(paper,"w",encoding="utf-8") as f: f.write(s)
    print("Saved figures to paper/figures and referenced in paper.md")
if __name__ == "__main__": main()
