import os
from datetime import timezone
from datetime import timedelta
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def write_date(df_day,symbol,bar_size,what_to_show):
    bar_size = bar_size.replace(' ','')
    chemin_dossier = os.path.join("data", "bars", symbol, bar_size, what_to_show)
    os.makedirs(chemin_dossier,exist_ok=True)
    date_str = df_day['ts_utc'].iloc[0].strftime('%Y-%m-%d')
    chemin_fichier = os.path.join(chemin_dossier,f"{date_str}.parquet")
    if os.path.exists(chemin_fichier):
        return
    df_day.to_parquet(chemin_fichier)

def to_date_time(dt):
    dt = dt.astimezone(timezone.utc)
    return dt.strftime('%Y%m%d-%H:%M:%S')

def bar_per_day(bar_size):
    return int(390/bar_size)

def parsing(bar):
    char = bar.split(' ')
    if char[1] == 'mins' or char[1] == 'min':
        return int(char[0])
    elif char[1] == 'sec' or char[1] == 'secs':
        return int(char[0])/60
    elif char[1] == 'hour' or char[1] == 'hours':
        return int(char[0])*60
    else:
        raise ValueError(f"Unité inconnue : {bar}")

def plan_backfill(start, end, bar_time):
    chunk_days = int(0.8*2000/bar_per_day(parsing(bar_time)))
    tranches = []
    curseur = end

    while curseur > start:
        debut = max(curseur - timedelta(days=chunk_days),start)
        nb_jours = (curseur - debut).days

        end_time_str = to_date_time(curseur)
        duration_str = f"{nb_jours} D"

        tranches.append((end_time_str, duration_str))

        curseur = debut

    return tranches

def list_days(start, end):
    jours = []
    curseur = start

    while curseur <= end:
        jours.append(curseur.strftime('%Y-%m-%d'))
        curseur += timedelta(days=1)
    return jours


def path_for_day(symbol,bar_size,what_to_show,date):
    bar_size = bar_size.replace(' ','')
    chemin_dossier = os.path.join(PROJECT_ROOT,"data", "bars", symbol, bar_size, what_to_show)
    chemin_fichier = os.path.join(chemin_dossier,f"{date}.parquet")
    return chemin_fichier

def read_range(symbol,bar_size,what_to_show,start,end):
    dfs = []
    jours = list_days(start,end)
    for jour in jours:
        path = path_for_day(symbol,bar_size,what_to_show,jour)
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    if len(dfs) > 0:
        df = pd.concat(dfs).sort_values('ts_utc').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()