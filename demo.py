from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timezone
import argparse

from src.data.fetch_surface import load_surface_by_maturity

from src.branch_a.surface import prepare_surface, orchestrateur, ForwardExtraction, calibrate_svi, density, svi_raw

from src.branch_b.hurst import read_range, realized_vol, estimate_hurst
from src.branch_b.calibration import optim
from src.branch_b.rough_bergomi import price_forward_start

from src.branch_c.mot import discretize_quantile, is_convex, mot_bounds

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Demo rough Bergomi vs MOT")
    parser.add_argument('--sweep', action='store_true',
                        help="Activates the H sweeping (slower on calibration)")
    args = parser.parse_args()

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
    print(f"H estimated : {H_aapl:.4f}")

    Ks = df_2['Strike'].values
    prices_marche_2 = df_2['call_mid'].values
    S0 = float(df_raw_1['spot'].iloc[0])
    params, _ = optim(S0,Ks,H_aapl,1,T2,10000,2000,prices_marche_2,[0.04, 2.0, -0.7])
    xi0, eta, rho = params
    P_rb = price_forward_start(S0, H_aapl, 1, T1, T2, 10000, xi0, eta, rho, 1.0, 20000)
    print(f"Price rough Bergomi : {P_rb:.5f}")

#---------------------------- BRANCH C -----------------------------------

    x1,p1 = discretize_quantile(params_sv1,F1,40)
    x2,p2 = discretize_quantile(params_sv2,F2,40)

    x1n = x1/F1
    x2n = x2/F2

    ok = is_convex(p1,p2,x1n,x2n)

    Pmin, Pmax = mot_bounds(x1n, p1, x2n, p2, kappa=1)
    print(f"Convex Order : {ok}, MOT : [{Pmin:.5f}, {Pmax:.5f}]")

#---------------------------- BRANCH D -----------------------------------

    ok = Pmin <= P_rb <= Pmax
    pos = (P_rb - Pmin)/(Pmax - Pmin)
    print(f"Rough Bergomi price in MOT interval : {ok}, Position : {pos}")
    if args.sweep :
        H_values = np.linspace(0.05, 0.20, 8)
        prices_rb = []
        for H_test in H_values:
            params, _ = optim(S0,Ks,H_test,1,T2,10000,2000,prices_marche_2,[0.04, 2.0, -0.7])
            xi0, eta, rho = params
            prices_rb.append(price_forward_start(S0, H_test, 1, T1, T2, 10000, xi0, eta, rho, 1.0, 20000))
        prices_rb = np.array(prices_rb)

#------------------------------ PLOTS -------------------------------------

    k_plot_1 = np.linspace(grid_1['k'].min(), grid_1['k'].max(), 300)
    k_plot_2 = np.linspace(grid_2['k'].min(), grid_2['k'].max(), 300)

    if args.sweep:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0,0].scatter(grid_1['k'], grid_1['w'], s=12, color='steelblue', label='T1 observed')
        axes[0,0].plot(k_plot_1, svi_raw(k_plot_1, *params_sv1), color='steelblue', label='T1 SVI')
        axes[0,0].scatter(grid_2['k'], grid_2['w'], s=12, color='crimson', label='T2 observed')
        axes[0,0].plot(k_plot_2, svi_raw(k_plot_2, *params_sv2), color='crimson', label='T2 SVI')
        axes[0,0].set_xlabel('k (log-moneyness)'); axes[0,0].set_ylabel('w (total variance)')
        axes[0,0].set_title('SVI smiles'); axes[0,0].legend()

        axes[0,1].plot(k_plot_1, density(k_plot_1, params_sv1), color='steelblue', label='T1 (1 month)')
        axes[0,1].plot(k_plot_2, density(k_plot_2, params_sv2), color='crimson', label='T2 (3 months)')
        axes[0,1].set_xlabel('k (log-moneyness)'); axes[0,1].set_ylabel('density')
        axes[0,1].set_title('Risk-neutral densities'); axes[0,1].legend()

        q_values = np.array([0.5, 1, 1.5, 2, 3])
        axes[1,0].plot(q_values, zeta, 'o-', color='darkgreen', label='$\\zeta_q$ observed')
        axes[1,0].plot(q_values, H_aapl * q_values, '--', color='gray', label=f'slope H={H_aapl:.3f}')
        axes[1,0].set_xlabel('q'); axes[1,0].set_ylabel('$\\zeta_q$')
        axes[1,0].set_title('Roughness estimation (monoscaling)'); axes[1,0].legend()

        axes[1,1].axhspan(Pmin, Pmax, alpha=0.2, color='gray', label='MOT band')
        axes[1,1].axhline(Pmin, color='gray', ls='--', lw=0.8)
        axes[1,1].axhline(Pmax, color='gray', ls='--', lw=0.8)
        axes[1,1].plot(H_values, prices_rb, 'o-', color='crimson', label='rough Bergomi')
        axes[1,1].axvline(H_aapl, color='navy', ls=':', lw=1.2, label=f'H realized ={H_aapl:.3f}')
        axes[1,1].set_xlabel('H (roughness)'); axes[1,1].set_ylabel('forward-start price')
        axes[1,1].set_title('Rough Bergomi price vs H, within MOT band'); axes[1,1].legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(grid_1['k'], grid_1['w'], s=12, color='steelblue', label='T1 observed')
        axes[0].plot(k_plot_1, svi_raw(k_plot_1, *params_sv1), color='steelblue', label='T1 SVI')
        axes[0].scatter(grid_2['k'], grid_2['w'], s=12, color='crimson', label='T2 observed')
        axes[0].plot(k_plot_2, svi_raw(k_plot_2, *params_sv2), color='crimson', label='T2 SVI')
        axes[0].set_xlabel('k (log-moneyness)'); axes[0].set_ylabel('w (total variance)')
        axes[0].set_title('SVI smiles'); axes[0].legend()

        axes[1].plot(k_plot_1, density(k_plot_1, params_sv1), color='steelblue', label='T1 (1 month)')
        axes[1].plot(k_plot_2, density(k_plot_2, params_sv2), color='crimson', label='T2 (3 months)')
        axes[1].set_xlabel('k (log-moneyness)'); axes[1].set_ylabel('density')
        axes[1].set_title('Risk-neutral densities'); axes[1].legend()

    plt.tight_layout()
    plt.show()