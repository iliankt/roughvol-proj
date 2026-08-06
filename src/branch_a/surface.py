
import statsmodels.api as sm
import pandas as pd
import numpy as np
from src.utils.black_scholes import CallBS76, PutBS76

def ForwardExtraction(df, weighted=True):
    y = df['call_mid'] - df['put_mid']
    x = sm.add_constant(df['Strike'])

    if weighted:
        w = 1.0 / (df['call_spread'] + df['put_spread'])
        results = sm.WLS(y, x, weights=w).fit()
    else:
        results = sm.OLS(y, x).fit()

    intercept, slope = results.params
    D = -slope
    F = intercept / D
    return F, D

def svi_raw(k, a, b, rho, m, sigma):

    return a + b*(rho*(k - m) + np.sqrt((k - m)**2 + sigma**2))


def generate_synthetic_surface(maturities):

    lignes = []
    for m in maturities:
        T, F, D = m['T'], m['F'], m['D']
        a, b, rho, mm, sig = m['params']

        for K in m['strikes']:
            k = np.log(K / F)
            w = svi_raw(k, a, b, rho, mm, sig)
            sigma_imp = np.sqrt(w / T)

            call = CallBS76(F, K, T, 0.0, sigma_imp, D)
            put  = PutBS76(F, K, T, 0.0, sigma_imp, D)

            lignes.append({'maturity': T, 'Strike': K,
                           'call_mid': call, 'put_mid': put})

    return pd.DataFrame(lignes)