import numpy as np
from scipy.signal import fftconvolve
from scipy.integrate import quad
from matplotlib import pyplot as plt

def cov_cross(j, l, H, n):
    alpha = H - 0.5
    lo = max(j, l) - 1
    hi = max(j, l)
    integrand = lambda u: (u - (j-1))**alpha * (u - (l-1))**alpha
    val, _ = quad(integrand, lo, hi)
    return (1/n)**(2*H) * val

def build_cov(H, kappa, n):
    alpha = H - 0.5
    dt = 1/n
    size = kappa + 1
    C = np.zeros((size, size))

    C[0, 0] = dt

    for j in range(1, kappa+1):
        val, _ = quad(lambda u: (u - (j-1))**alpha, j-1, j)
        C[0, j] = C[j, 0] = dt**(H+0.5) * val

    for j in range(1, kappa+1):
        for l in range(1, kappa+1):
            C[j, l] = cov_cross(j, l, H, n)

    return C

def hybride_scheme(H,kappa,T,n):
    alpha = H - 1/2
    cov = build_cov(H,kappa,n)
    L = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((n,kappa+1))
    couples = Z @ L.T
    g = [(1/n)**alpha * (k**(alpha+1) - (k-1)**(alpha+1))/(alpha+1) for k in range(kappa + 1,n)]
    dW = couples[:, 0] 
    W = fftconvolve(dW, g)
    W = np.concatenate([np.zeros(kappa+1), W])[:n]
    I = couples[:,1:].sum(axis=1)
    return (np.sqrt(2*H) * (I + W), dW)

def variance_process(W_hat, xi0, eta, H, T, n):
    ti = np.arange(n) / n * T
    correction = 0.5 * eta**2 * ti**(2*H)
    V = xi0 * np.exp(eta * W_hat - correction)
    return V

def price_simu(S0,H,kappa,T,n,xi0,eta,rho):
    dt = 1/n
    W_hat, dW = hybride_scheme(H,kappa,T,n)
    V = variance_process(W_hat,xi0,eta,H,T,n)
    delta_W = np.random.standard_normal(size=n) * np.sqrt(dt)
    dZ = rho*dW + np.sqrt(1-rho**2)*delta_W
    increments = -0.5 * V[:-1] * dt + np.sqrt(V[:-1]) * dZ[1:]
    logS = np.concatenate([[0], np.cumsum(increments)])
    return S0 * np.exp(logS)

def price_forward_start(S0,H,kappa_bergomi,T1,T2,n,xi0,eta,rho,kappa_strike,M):
    i1 = int(T1/T2 * n)
    payoffs = np.empty(M)
    for m in range(M):
        S = price_simu(S0,H,kappa_bergomi,T2,n,xi0,eta,rho)
        payoffs[m] = max(S[-1]/S[i1] - kappa_strike,0) 
    return payoffs.mean()

prix = price_forward_start(100, 0.1, 1, 0.5, 1.0, 500, 0.04, 1.5, -0.7, 1.0, 5000)
print("prix forward-start :", prix)