from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad
from scipy.optimize import linprog
import numpy as np
from src.branch_a.surface import density

def discretize_quantile(params, F, N, k_min=-5, k_max=5, n_fin=2000):
    liste_k = np.linspace(k_min, k_max, n_fin)
    densite = density(liste_k, params)
    CDF = cumulative_trapezoid(densite, liste_k, initial=0)
    CDF /= CDF[-1]

    niveaux = np.linspace(0, 1, N+1)
    bords = np.interp(niveaux, CDF, liste_k)

    x = np.empty(N)
    p = np.empty(N)
    for i in range(N):
        ka, kb = bords[i], bords[i+1]
        num, _ = quad(lambda k: F*np.exp(k)*density(k, params), ka, kb)
        den, _ = quad(lambda k: density(k, params), ka, kb)
        x[i] = num/den
        p[i] = den

    p /= p.sum()
    return x, p

def call_hoped(K,p,x):
    return np.sum(p*(np.maximum(x-K,0.0)))

def is_convex(p1,p2,x1,x2, n_strikes=200,tol=10e-6):
    m1= np.sum(p1*x1)
    m2 = np.sum(p2*x2)
    mean_ok = np.abs(m1 - m2) < tol*m1
    Kmin = min(x1.min(), x2.min())
    Kmax = max(x1.max(), x2.max())
    strikes = np.linspace(Kmin, Kmax, n_strikes)
    diffs = np.array([call_hoped(K, p2, x2) - call_hoped(K, p1, x1) for K in strikes])
    calls_ok = np.all(diffs >= -tol * m1)
    return (mean_ok and calls_ok)


def build_payoff(x1,x2,kappa):
    ratio = x2[None, :] / x1[:, None]
    payoff = np.maximum(ratio - kappa,0)
    return payoff.flatten()

def build_constraints(x1, p1, x2, p2):
    N1 = len(x1)
    N2 = len(x2)

    A_rows = np.kron(np.eye(N1), np.ones(N2))

    A_cols = np.kron(np.ones(N1), np.eye(N2))

    A_mart = np.zeros((N1, N1 * N2))
    for i in range(N1):
        A_mart[i, i*N2:(i+1)*N2] = x2 - x1[i]

    A_eq = np.vstack([A_rows, A_cols, A_mart])
    b_eq = np.concatenate([p1, p2, np.zeros(N1)])

    return A_eq, b_eq

def mot_bounds(x1, p1, x2, p2, kappa):
    c = build_payoff(x1, x2, kappa)
    A_eq, b_eq = build_constraints(x1, p1, x2, p2)

    res_min = linprog(c, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method='highs')
    res_max = linprog(-c, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method='highs')

    if not res_min.success or not res_max.success:
        raise RuntimeError(f"LP echoue : min={res_min.message}, max={res_max.message}")

    P_min = res_min.fun
    P_max = -res_max.fun
    return P_min, P_max


