import numpy as np, pandas as pd
from datetime import datetime, timedelta
DIVISIONS = ["FLY","BW","FW","LW","WW","MW","LHW","HW"]
def make_synthetic_fights(n_events=160, fights_per_event=7, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2018,1,1)
    rows=[]; fighter_id=0; fighters={}
    def new_fighter():
        nonlocal fighter_id; fighter_id+=1
        return dict(
            fighter_id=fighter_id,
            reach=float(rng.normal(72,4)), height=float(rng.normal(70,3)),
            stance=str(rng.choice(["O","S","Switch"], p=[0.7,0.2,0.1])),
            dob=datetime(1988,1,1)+timedelta(days=int(rng.normal(0,2000)))
        )
    for e in range(n_events):
        event_date = start + timedelta(days=int(e*14 + rng.normal(0,2)))
        event_id = f"E{e:04d}"; division = DIVISIONS[e%len(DIVISIONS)]
        for f in range(fights_per_event):
            fa = fighters.get(rng.integers(1, max(2, fighter_id+1)))
            fb = fighters.get(rng.integers(1, max(2, fighter_id+1)))
            if fa is None: fa=new_fighter(); fighters[fa["fighter_id"]]=fa
            if fb is None: fb=new_fighter(); fighters[fb["fighter_id"]]=fb
            if fa["fighter_id"]==fb["fighter_id"]]: fb=new_fighter(); fighters[fb["fighter_id"]]=fb
            skill_a = rng.normal(0,1) + 0.02*(fa["height"]-70) + 0.02*(fa["reach"]-72)
            skill_b = rng.normal(0,1) + 0.02*(fb["height"]-70) + 0.02*(fb["reach"]-72)
            p_a = 1/(1+np.exp(-(skill_a - skill_b)))
            y = rng.binomial(1, p_a)
            rows.append(dict(
                event_id=event_id, bout_id=f"{event_id}_F{f:02d}", bout_datetime=event_date, division=division,
                a_id=fa["fighter_id"], b_id=fb["fighter_id"],
                a_height=fa["height"], a_reach=fa["reach"], a_stance=fa["stance"], a_dob=fa["dob"],
                b_height=fb["height"], b_reach=fb["reach"], b_stance=fb["stance"], b_dob=fb["dob"],
                label_a_wins=int(y)
            ))
    return pd.DataFrame(rows)
