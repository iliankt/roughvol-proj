from src.branch_b.rough_bergomi import price_simu_vect,build_cov,build_weights
import numpy as np
from scipy.optimize import minimize
from src.data.fetch_surface import load_latest_surface
from src.data.prepare import prepare_surface

def obj(params, S0, Ks, H, kappa, T, n, M, L, g, prices_marche):
    rng = np.random.default_rng(42)
    xi0, eta, rho = params
    S = price_simu_vect(S0, H, kappa, T, n, xi0, eta, rho, M, L, g, rng=rng)
    prices_modele = np.array([np.maximum(S[:, -1] - K, 0).mean() for K in Ks])
    return np.sum((prices_modele - prices_marche)**2)
    
def optim(S0, Ks, H, kappa, T, n, M, prices_marche, params_init):
    n_steps = int(T*n)
    L = np.linalg.cholesky(build_cov(H, kappa, n))
    g = build_weights(H, kappa, n,n_steps)
    bounds = [(1e-4, 1.0), (1e-4, 5.0), (-0.999, 0.999)]
    res = minimize(obj, params_init,
                   args=(S0, Ks, H, kappa, T, n, M, L, g, prices_marche),
                   bounds=bounds, method='L-BFGS-B')
    return res.x, res.fun

def calibrate_rough_bergomi(H, kappa, n, M, params_init, maturite_idx=0):
    df_raw = load_latest_surface('AAPL')
    df = prepare_surface(df_raw)

    maturites = sorted(df['maturity'].unique())
    mat = maturites[maturite_idx]
    df_mat = df[df['maturity'] == mat]

    S0 = float(df_raw['spot'].iloc[0])
    T = float(df_mat['maturity'].iloc[0])
    Ks = df_mat['Strike'].values
    prices_marche = df_mat['call_mid'].values

    print(f"maturité {mat} : T={T:.4f}, S0={S0}, {len(Ks)} strikes")
    print(f"strikes : {Ks.min():.0f} -> {Ks.max():.0f}")

    params, err = optim(S0, Ks, H, kappa, T, n, M, prices_marche, params_init)
    print(f"params calibrés : xi0={params[0]:.4f}, eta={params[1]:.4f}, rho={params[2]:.4f}")
    print(f"erreur finale : {err:.6f}")
    return params, err
