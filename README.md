# Rough Bergomi vs Martingale Optimal Transport

Model-based versus model-free pricing of a forward-start option under rough
volatility, and a study of the associated model risk on real AAPL data.

## Overview

A vanilla option surface fixes the marginal laws of the underlying at two
maturities, but not the dependence structure between them. Exotic payoffs that
depend on this joint law — such as the forward-start option
$(S_{T_2}/S_{T_1} - \kappa)^+$ — are therefore not uniquely priced by the
vanilla market.

This project compares two readings of the same surface:

- **Model-based (rough Bergomi):** a rough volatility model is calibrated to the
  surface and used to price the exotic, yielding a single price.
- **Model-free (MOT):** martingale optimal transport gives no-arbitrage bounds
  $[P_{\min}, P_{\max}]$ that hold across all models consistent with the two
  marginals.

The width $P_{\max} - P_{\min}$ measures the irreducible model risk. The project
also studies how the roughness exponent $H$ moves the rough Bergomi price within
the MOT band.

## Structure

The pipeline is organized in four branches:

- **Branch A — implied marginals:** extraction of the two marginals from the
  option surface via Breeden-Litzenberger, smoothed with an arbitrage-free SVI
  parametrization.
- **Branch B — rough Bergomi:** estimation of the roughness exponent $H$ from
  historical realized volatility, simulation via the Bennedsen-Lunde-Pakkanen
  hybrid scheme, calibration, and Monte Carlo pricing.
- **Branch C — MOT:** discretization of the martingale optimal transport problem
  into a linear program, solved to obtain the no-arbitrage bounds.
- **Branch D — confrontation:** comparison of the rough Bergomi price against the
  MOT interval, and study of the role of $H$.

## Data

Market data are snapshots extracted through the Interactive Brokers API
(delayed data). The repository ships with pre-captured AAPL surfaces and 5-minute
bars so that the full pipeline is reproducible without live market access. Data
are a fixed snapshot in time; running the demo reproduces the results on that
snapshot.

## Running the demo

`demo.py` runs the complete pipeline end to end: it loads the AAPL surfaces,
extracts the two marginals (branch A), estimates $H$ and calibrates rough
Bergomi (branch B), computes the MOT bounds (branch C), and confronts the
rough Bergomi price with the interval (branch D). It then displays the SVI
smiles and the risk-neutral densities.

```bash
python demo.py
```

Add the `--sweep` flag to also run the roughness sweep — recalibrating and
repricing across a range of $H$ values — and display the full 2x2 figure,
including the rough Bergomi price within the MOT band. This is slower, as it
performs several calibrations.

```bash
python demo.py --sweep
```

## Report

A full LaTeX report, covering the mathematical background (proofs) and the
implementation of each branch, is currently being written.

## Requirements

Python 3, with `numpy`, `scipy`, `pandas`, `matplotlib`, `statsmodels`, and
`pyarrow`. Live data capture additionally requires the Interactive Brokers API
(`ibapi`) and a running TWS/Gateway session.