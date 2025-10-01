import argparse, datetime
PATH="docs/JOURNAL.md"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--note", required=True); ap.add_argument("--tag", default=None); a=ap.parse_args()
    ts=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    entry=f"\n### {ts}\n- {a.note}\n"; 
    if a.tag: entry += f"- Tag: {a.tag}\n"
    with open(PATH,"a",encoding="utf-8") as f: f.write(entry)
    print(f"Appended to {PATH}")
if __name__=="__main__": main()
