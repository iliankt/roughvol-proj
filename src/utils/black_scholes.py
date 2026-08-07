import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def CallBS(S,K,T,t,sigma,r,q):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return S*np.exp(-q*(T-t))*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)

def PutBS(S,K,T,t,sigma,r,q):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return -S*np.exp(-q*(T-t))*norm.cdf(-d1) + K*np.exp(-r*(T-t))*norm.cdf(-d2)

def implied_vol_call(C_obs, S, K, T, t, r, q,
                     sigma_low=1e-6, sigma_high=5.0):
    tau = T - t
    if tau <= 0 or C_obs is None or C_obs <= 0:
        return np.nan
    lower = max(S*np.exp(-q*tau) - K*np.exp(-r*tau), 0.0)
    upper = S*np.exp(-q*tau)
    if C_obs < lower or C_obs > upper:
        return np.nan
    g = lambda sigma: CallBS(S, K, T, t, sigma, r, q) - C_obs

    try:
        return brentq(g, sigma_low, sigma_high, xtol=1e-8, maxiter=100)
    except ValueError:
        return np.nan

def implied_vol_put(C_obs, S, K, T, t, r, q,
                     sigma_low=1e-6, sigma_high=5.0):
    tau = T - t
    if tau <= 0 or C_obs is None or C_obs <= 0:
        return np.nan
    lower = max(-S*np.exp(-q*tau) + K*np.exp(-r*tau), 0.0)
    upper = K*np.exp(-r*tau)
    if C_obs < lower or C_obs > upper:
        return np.nan
    g = lambda sigma: PutBS(S, K, T, t, sigma, r, q) - C_obs

    try:
        return brentq(g, sigma_low, sigma_high, xtol=1e-8, maxiter=100)
    except ValueError:
        return np.nan

def CallBS76(F,K,T,t,sigma,D):
    d1 = (np.log(F/K) + 0.5*sigma**2*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)

    return D*(F*norm.cdf(d1) - K*norm.cdf(d2))

def PutBS76(F,K,T,t,sigma,D):
    d1 = (np.log(F/K) + 0.5*sigma**2*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)

    return D*(-F*norm.cdf(-d1) + K*norm.cdf(-d2))

def implied_vol_call76(C_obs, F, K, T, t, D,
                     sigma_low=1e-6, sigma_high=5.0):
    tau = T - t
    if tau <= 0 or C_obs is None or C_obs <= 0:
        return np.nan
    lower = D*max(F-K, 0.0)
    upper = D*F
    if C_obs < lower or C_obs > upper:
        return np.nan
    g = lambda sigma: CallBS76(F,K,T,t,sigma,D) - C_obs

    try:
        return brentq(g, sigma_low, sigma_high, xtol=1e-8, maxiter=100)
    except ValueError:
        return np.nan

def implied_vol_put76(C_obs, F, K, T, t, D,
                     sigma_low=1e-6, sigma_high=5.0):
    tau = T - t
    if tau <= 0 or C_obs is None or C_obs <= 0:
        return np.nan
    lower = D*max(K-F, 0.0)
    upper = D*K
    if C_obs < lower or C_obs > upper:
        return np.nan
    g = lambda sigma: PutBS76(F,K,T,t,sigma,D) - C_obs

    try:
        return brentq(g, sigma_low, sigma_high, xtol=1e-8, maxiter=100)
    except ValueError:
        return np.nan

def vega76(F, K, T, t, sigma, D):
    tau = T - t
    d1 = (np.log(F/K) + 0.5*sigma**2*tau)/(sigma*np.sqrt(tau))
    return D * F * np.sqrt(tau) * norm.pdf(d1)