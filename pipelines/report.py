import os, time, json, pandas as pd
from datetime import datetime
from src.utils.logging import get_logger
log = get_logger("report")
def write_md(path, content):
    with open(path, "w", encoding="utf-8") as f: f.write(content)
def main():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("reports","runs", ts); os.makedirs(outdir, exist_ok=True)
    lines=[]; lines.append("# FightIQ — Definitive MMA Prediction Study\n"); lines.append(f"_Auto-report generated: {ts} UTC_\n")
    ens_path="data/interim/oof/ensemble_oof.parquet"
    if os.path.exists(ens_path):
        df = pd.read_parquet(ens_path); lines.append(f"- Ensemble OOF rows: **{len(df)}**\n")
    else: lines.append("- Ensemble not present (single-model run).\n")
    # Journal tail
    if os.path.exists("docs/JOURNAL.md"):
        lines.append("\n## Journal (tail)\n")
        with open("docs/JOURNAL.md","r",encoding="utf-8") as f:
            tail = f.readlines()[-20:]
        lines.extend(tail); lines.append("\n")
    # Experiments registry tail
    reg="docs/experiments.csv"
    if os.path.exists(reg):
        lines.append("\n## Experiments Registry (tail)\n")
        try: lines.append(pd.read_csv(reg).tail(10).to_markdown(index=False) + "\n")
        except Exception as e: lines.append(f"Failed to parse experiments.csv: {e}\n")
    write_md(os.path.join(outdir,"index.md"), "\n".join(lines)); log.info(f"Wrote report to {outdir}/index.md")
if __name__ == "__main__": main()
