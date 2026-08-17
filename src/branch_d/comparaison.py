import numpy as np
import matplotlib.pyplot as plt
from src.data.fetch_surface import load_surface_by_maturity
from src.data.prepare import prepare_surface
from src.branch_a.surface import orchestrateur, calibrate_svi, ForwardExtraction
from src.branch_c.mot import discretize_quantile, is_convex, mot_bounds
from src.branch_b.rough_bergomi import price_forward_start
from src.branch_b.calibration import optim

# ============ PARAMETRES GLOBAUX ============
kappa_bergomi = 1          # cellules exactes du schema hybride
kappa_strike = 1.0         # strike de la forward-start (ATM)
n = 10000                  # pas par an
M_cal = 2000               # trajectoires pour la CALIBRATION (petit, rapide)
M_price = 20000            # trajectoires pour le PRIX final (grand, precis)
N_disc = 40                # points de discretisation MOT

# ============ CHARGEMENT + BRANCHE A ============
df_raw_1 = load_surface_by_maturity('AAPL', '20260918')   # T1 = 1 mois
df_raw_2 = load_surface_by_maturity('AAPL', '20261120')   # T2 = 3 mois
df_1 = prepare_surface(df_raw_1)
df_2 = prepare_surface(df_raw_2)

S0 = float(df_raw_1['spot'].iloc[0])
T1 = float(df_1['maturity'].iloc[0])
T2 = float(df_2['maturity'].iloc[0])

grid_1 = orchestrateur(df_1)
grid_2 = orchestrateur(df_2)
params_svi_1 = calibrate_svi(grid_1['k'].values, grid_1['w'].values)
params_svi_2 = calibrate_svi(grid_2['k'].values, grid_2['w'].values)

F1,_ = ForwardExtraction(df_1)
F2,_ = ForwardExtraction(df_2)

# donnees marche de la maturite de calibration (T2 = 3 mois)
Ks2 = df_2['Strike'].values
prices_marche_2 = df_2['call_mid'].values

print(f"S0={S0:.2f}, T1={T1:.4f}, T2={T2:.4f}, F1={F1:.2f}, F2={F2:.2f}")

# ============ BANDE MOT (calculee UNE fois, invariante en H) ============
x1, p1 = discretize_quantile(params_svi_1, F1, N=N_disc)
x2, p2 = discretize_quantile(params_svi_2, F2, N=N_disc)
x1n, x2n = x1/F1, x2/F2
ok= is_convex(p1, p2, x1n, x2n)
Pmin, Pmax = mot_bounds(x1n, p1, x2n, p2, kappa=kappa_strike)
print(f"ordre convexe : {ok}, MOT : [{Pmin:.5f}, {Pmax:.5f}]")

# ============ BALAYAGE EN H ============
H_values = np.linspace(0.05, 0.20, 5)
prices_rb = []

for H_test in H_values:
    params_cal, _ = optim(S0, Ks2, H_test, kappa_bergomi, T2, n, M_cal,
                          prices_marche_2, params_init=[0.04, 2.0, -0.7])
    xi0_h, eta_h, rho_h = params_cal

    P = price_forward_start(S0, H_test, kappa_bergomi, T1, T2, n,
                            xi0_h, eta_h, rho_h, kappa_strike, M_price)
    prices_rb.append(P)
    print(f"H={H_test:.3f} : xi0={xi0_h:.4f}, eta={eta_h:.4f}, rho={rho_h:.4f}, P={P:.5f}")

prices_rb = np.array(prices_rb)

# ============ FIGURE FINALE ============
plt.figure(figsize=(9,5))
plt.axhspan(Pmin, Pmax, alpha=0.2, color='gray', label='bande MOT')
plt.axhline(Pmin, color='gray', ls='--', lw=0.8)
plt.axhline(Pmax, color='gray', ls='--', lw=0.8)
plt.plot(H_values, prices_rb, 'o-', color='crimson', label='rough Bergomi')
plt.xlabel('H (roughness)')
plt.ylabel('prix forward-start')
plt.legend()
plt.title('Prix rough Bergomi selon H, dans la bande MOT')
plt.show()