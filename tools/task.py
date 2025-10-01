import argparse, os, re, datetime
TASKS_DIR="docs/tasks"; INDEX="docs/TASKS_INDEX.md"
TEMPLATE = """
# {task_id} — {title}
- **Status**: {status}
- **Created**: {created}
- **Updated**: {updated}
## Description
{desc}
## Acceptance Criteria
- [ ] Criteria agreed
- [ ] Implemented
- [ ] Tested on synthetic
- [ ] Logged in MLflow
- [ ] Reported in report.md
## Links
- Related Experiments:
- PRs/Runs:
"""
def next_id():
    os.makedirs(TASKS_DIR, exist_ok=True); existing=[f for f in os.listdir(TASKS_DIR) if re.match(r"T-\d{4}\.md", f)]
    n=0
    for f in existing: n=max(n,int(f[2:6]))
    return f"T-{n+1:04d}"
def write_index_entry(task_id, title, status):
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    if not os.path.exists(INDEX):
        with open(INDEX,"w",encoding="utf-8") as f: f.write("# Tasks Index\n\n| ID | Title | Status |\n|----|-------|--------|\n")
    with open(INDEX,"a",encoding="utf-8") as f: f.write(f"| {task_id} | {title} | {status} |\n")
def create_task(title, desc):
    tid=next_id(); today=datetime.datetime.utcnow().strftime("%Y-%m-%d")
    content=TEMPLATE.format(task_id=tid, title=title, status="TODO", created=today, updated=today, desc=desc)
    path=os.path.join(TASKS_DIR, f"{tid}.md"); with open(path,"w",encoding="utf-8") as f: f.write(content)
    write_index_entry(tid, title, "TODO"); print(path)
def set_status(task_id, status):
    path=os.path.join(TASKS_DIR, f"{task_id}.md"); s=open(path,"r",encoding="utf-8").read()
    for st in ["TODO","DOING","DONE"]: s=s.replace(f"**Status**: {st}", f"**Status**: {status}")
    s=s.replace("**Updated**: ", f"**Updated**: {datetime.datetime.utcnow().strftime('%Y-%m-%d')}")
    open(path,"w",encoding="utf-8").write(s); print(f"{task_id} -> {status}")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd")
    cnew=sub.add_parser("new"); cnew.add_argument("--title", required=True); cnew.add_argument("--desc", default="")
    cdone=sub.add_parser("done"); cdone.add_argument("--id", required=True)
    cdoing=sub.add_parser("doing"); cdoing.add_argument("--id", required=True)
    a=ap.parse_args()
    if a.cmd=="new": create_task(a.title, a.desc)
    elif a.cmd=="done": set_status(a.id,"DONE")
    elif a.cmd=="doing": set_status(a.id,"DOING")
    else: ap.print_help()
