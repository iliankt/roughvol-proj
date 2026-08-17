import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.data.utils import read_range
from datetime import datetime
from datetime import timezone
from matplotlib import pyplot as plt

def realized_vol(barres):
    barres = barres.copy()
    barres['ret'] = np.log(barres['close']).diff()
    rv = barres.groupby(barres['ts_utc'].dt.date)['ret'].apply(lambda r: np.sqrt((r**2).sum()))
    return rv

def moment(log_vol, q, delta):
    diff = log_vol.shift(-delta) - log_vol
    diff = diff.dropna()
    return (np.abs(diff)**q).mean() 

def estimate_hurst(log_vol, q_values, delta_max):
    deltas = np.arange(1, delta_max + 1)
    zeta = []

    for q in q_values:
        m = np.array([moment(log_vol, q, d) for d in deltas])
        x = sm.add_constant(np.log(deltas))
        slope = sm.OLS(np.log(m), x).fit().params[1]
        zeta.append(slope)

    zeta = np.array(zeta)
    x = sm.add_constant(q_values)
    H = sm.OLS(zeta, x).fit().params[1]
    return H, zeta