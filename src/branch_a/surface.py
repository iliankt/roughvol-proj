
import statsmodels.api as sm
import pandas as pd
import numpy as np
from src.utils.black_scholes import CallBS76, PutBS76, implied_vol_call76, implied_vol_put76
from scipy.optimize import least_squares
from scipy.integrate import quad
from src.data.fetch_surface import load_latest_surface
from src.data.prepare import prepare_surface
import matplotlib.pyplot as plt

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

def orchestrateur(df):
    lignes = []
    for T, bloc in df.groupby('maturity'):
        F, D = ForwardExtraction(bloc, weighted=False)

        for _, row in bloc.iterrows():
            K = row['Strike']
            if K >= F:
                sigma = implied_vol_call76(row['call_mid'], F, K, T, 0.0, D)
            else:
                sigma = implied_vol_put76(row['put_mid'], F, K, T, 0.0, D)

            if np.isnan(sigma):
                continue
            lignes.append({'maturity': T, 'k': np.log(K/F), 'w': sigma**2 * T})

    return pd.DataFrame(lignes)

def svi_residuals(params, k, w):
    a, b, rho, m, sigma = params
    return svi_raw(k, a, b, rho, m, sigma) - w

def calibrate_svi(k, w):
    x0 = [w.min(), 0.1, -0.5, k[np.argmin(w)], 0.1]
    lower = [-1e-6, 0.0, -0.999, k.min(), 1e-4]
    upper = [w.max(), 1.0, 0.999, k.max(), 1.0]
    res = least_squares(svi_residuals, x0, args=(k, w),
                        bounds=(lower, upper), method='trf')
    return res.x

def w_prime(k, a, b, rho, m, sigma):
    return b*(rho + (k-m)/np.sqrt((k-m)**2 + sigma**2))

def w_second(k, a, b, rho, m, sigma):
    return b*sigma**2/(np.sqrt((k-m)**2 + sigma**2)**3)

def g_func(k, a, b, rho, m, sigma):
    w = svi_raw(k, a, b, rho, m, sigma)
    w1 = w_prime(k, a, b, rho, m, sigma)
    w2 = w_second(k, a, b, rho, m, sigma)

    return (1 - k*w1/(2*w))**2 - w1**2/4 * (1/w + 1/4) + w2/2

def density(k,params):
    a, b, rho, m, sigma = params
    w = svi_raw(k, a, b, rho, m, sigma)
    g = g_func(k, a, b, rho, m, sigma)
    d_ = -k/np.sqrt(w) - np.sqrt(w)/2
    return g/np.sqrt(2*np.pi*w) * np.exp(-1/2 * d_ **2)

if __name__ == '__main__':
    df1_raw = load_latest_surface('AAPL')
    df1 = prepare_surface(df1_raw)
    grid1 = orchestrateur(df1)
    k = grid1['k'].values
    w = grid1['w'].values
    params1 = calibrate_svi(k, w)
    

    k_plot = np.linspace(k.min(), k.max(), 200)
    densite = density(k_plot,params1)
    w_fit = svi_raw(k_plot, *params1)
    F = ForwardExtraction(df1)[0]
    mass, _ = quad(lambda k: density(k, params1), k.min()-0.2, k.max()+0.2)
    print("masse :", mass)

    mean, _ = quad(lambda k: F*np.exp(k)*density(k, params1), k.min()-0.2, k.max()+0.2)
    print("E[S_T] :", mean, " vs forward :", F)

    plt.plot(k_plot, densite, 'r-', label='Densité')
    plt.xlabel('k (log-moneyness)'); plt.ylabel('g')
    plt.legend(); plt.show()