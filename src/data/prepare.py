import pandas as pd
from datetime import datetime

def to_maturity(exp_str, snapshot_date=None):
    exp = datetime.strptime(str(exp_str), '%Y%m%d')
    ref = snapshot_date or datetime.now()
    return (exp - ref).days / 365.0

def prepare_surface(df):
    df = df.copy()
    df['mid'] = (df['ask'] + df['bid']) / 2
    df['spread'] = df['ask'] - df['bid']

    new_df = df.pivot(index=['maturity','strike'], columns='right',
                      values=['mid','spread'])

    new_df.columns = [f"{'call' if r=='C' else 'put'}_{v}" for v, r in new_df.columns]
    new_df = new_df.reset_index().rename(columns={'strike': 'Strike'})

    new_df = new_df.dropna(subset=['call_mid', 'put_mid'])
    new_df['maturity'] = new_df['maturity'].apply(to_maturity)
    return new_df
