from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timezone

from src.data.fetch_surface import load_surface_by_maturity

from src.branch_a.surface import prepare_surface, orchestrateur, ForwardExtraction, calibrate_svi, density

from src.branch_b.hurst import read_range, realized_vol, estimate_hurst
from src.branch_b.calibration import optim
from src.branch_b.rough_bergomi import price_forward_start

from src.branch_c.mot import discretize_quantile, is_convex, mot_bounds

if __name__ =='__main__':

#---------------------------- BRANCH A -----------------------------------

    df_raw_1 = load_surface_by_maturity('AAPL', '20260918')
    df_raw_2 = load_surface_by_maturity('AAPL', '20261120')
    df_1 = prepare_surface(df_raw_1)
    df_2 = prepare_surface(df_raw_2)
    grid_1 = orchestrateur(df_1)
    grid_2 = orchestrateur(df_2)
    F1,_ = ForwardExtraction(df_1)
    F2, _ = ForwardExtraction(df_2)
    params_sv1 = calibrate_svi(grid_1['k'].values, grid_1['w'].values)
    params_sv2 = calibrate_svi(grid_2['k'].values, grid_2['w'].values)
    density_1 = density(grid_1['k'].values,params_sv1)
    density_2 = density(grid_2['k'].values,params_sv2)

#---------------------------- BRANCH B -----------------------------------

    T1 = float(df_1['maturity'].iloc[0])
    T2 = float(df_2['maturity'].iloc[0])
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    barres = read_range('AAPL','5mins','TRADES',start,end)
    rv = realized_vol(barres)
    log_vol = np.log(rv)
    H_aapl, zeta = estimate_hurst(log_vol, np.array([0.5,1,1.5,2,3]), 30)

    Ks = df_2['Strike'].values
    prices_marche_2 = df_2['call_mid'].values
    S0 = float(df_raw_1['spot'].iloc[0])
    params, _ = optim(S0,Ks,H_aapl,1,T2,10000,2000,prices_marche_2,[0.04, 2.0, -0.7])
    xi0, eta, rho = params
    P_rb = price_forward_start(S0, H_aapl, 1, T1, T2, 10000, xi0, eta, rho, 1.0, 20000)

#---------------------------- BRANCH C -----------------------------------

    x1,p1 = discretize_quantile(params_sv1,F1,40)
    x2,p2 = discretize_quantile(params_sv2,F2,40)

    x1n = x1/F1
    x2n = x2/F2

    ok = is_convex(p1,p2,x1n,x2n)

    Pmin, Pmax = mot_bounds(x1n, p1, x2n, p2, kappa=1)
    print(f"Convex Order : {ok}, MOT : [{Pmin:.5f}, {Pmax:.5f}]")

#---------------------------- BRANCH D -----------------------------------