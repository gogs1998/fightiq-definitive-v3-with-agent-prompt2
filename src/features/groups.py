import pandas as pd, numpy as np
def add_A_demographics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['a_age']=(out['bout_datetime']-pd.to_datetime(out['a_dob'])).dt.days/365.25
    out['b_age']=(out['bout_datetime']-pd.to_datetime(out['b_dob'])).dt.days/365.25
    out['age_diff']=out['a_age']-out['b_age']
    out['height_diff']=out['a_height']-out['b_height']
    out['reach_diff']=out['a_reach']-out['b_reach']
    stance_map={"O":0,"S":1,"Switch":0.5}
    out['a_stance_num']=out['a_stance'].map(stance_map).fillna(0.0)
    out['b_stance_num']=out['b_stance'].map(stance_map).fillna(0.0)
    out['stance_diff']=out['a_stance_num']-out['b_stance_num']
    return out
def add_dummy_groups(df: pd.DataFrame, cfg_feats: dict) -> pd.DataFrame:
    out=df.copy()
    out['layoff_days_a']=100.0; out['layoff_days_b']=100.0
    out['experience_a']=5.0; out['experience_b']=5.0
    out['eff_pace_diff']=np.tanh(out['height_diff']*0.02 + out['reach_diff']*0.02)
    rng=np.random.default_rng(0); out['durability_diff']=rng.normal(0,0.1,len(out))
    out['elo_diff']=out['height_diff']*0.1 + out['reach_diff']*0.1
    out['style_diff']=out['stance_diff']*0.2
    out['context_altitude']=0.0
    return out
def build_feature_matrix(df: pd.DataFrame, flags: dict):
    x = add_A_demographics(df) if flags.get('A_demographics', True) else df.copy()
    x = add_dummy_groups(x, flags)
    cols = ['age_diff','height_diff','reach_diff','stance_diff','layoff_days_a','layoff_days_b','experience_a','experience_b','eff_pace_diff','durability_diff','elo_diff','style_diff','context_altitude']
    cols = [c for c in cols if c in x.columns]
    return x, cols
