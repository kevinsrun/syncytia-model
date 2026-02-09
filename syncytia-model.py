"""
syncytia_fit.py
Digitized-curve fitting + simple syncytia ODE model (Erlang 2-step fusion) with a DEATH parameter.

- ode_list(...) state ODEs
- SSR(...) objective
- scipy.optimize.minimize fit
- matplotlib plot overlay

NOTES
- This is an *approximate* model for a “cell index / normalized cell index” style readout.
- The "death rate" parameter (delta) is there specifically to let the curve fall / bend realistically.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize


# -----------------------------
# Model
# -----------------------------
# State vector:
#   D    = donors
#   A    = acceptors
#   F1   = fusion intermediate stage 1
#   F2   = fusion intermediate stage 2
#   S    = syncytia (fused cells)
#   X    = dead / lost cells (absorbing)
#
# Parameters:
#   gamma = mass-action fusion initiation
#   k     = Erlang stage transition rate (F1 -> F2 -> S)
#   delta = death / loss rate (applied to living compartments)
#
# Optional acceptor growth (can be left at 0 if you don't want it):
#   rA    = acceptor growth rate (logistic)
#   K     = carrying capacity


def ode_list(
    state: List[float],
    t: float,
    gamma: float,
    k: float,
    delta: float,
    rA: float,
    K: float,
    gamma2: float,
) -> List[float]:
    D, A, F1, F2, S, X = state

    # Fusion initiation terms:
    #   donor + acceptor -> fusion pipeline
    #   syncytia + acceptor -> fusion pipeline (optional secondary fusion)
    DA = gamma * D * A
    SA = gamma2 * S * A

    # Optional acceptor growth (logistic), helps match plateaus if your traces do that
    growth_A = rA * A * (1.0 - (A / K)) if (rA > 0 and K > 0) else 0.0

    dDdt = -DA - delta * D
    dAdt = -DA - SA + growth_A - delta * A

    # Erlang 2-step: F1 -> F2 -> S
    dF1dt = (2.0 * DA + SA) - (k * F1) - delta * F1
    dF2dt = (k * F1) - (k * F2) - delta * F2
    dSdt = (k * F2) - delta * S

    dXdt = delta * (D + A + F1 + F2 + S)

    return [dDdt, dAdt, dF1dt, dF2dt, dSdt, dXdt]


def simulate(
    t: np.ndarray,
    params: Dict[str, float],
    y0: Dict[str, float],
) -> np.ndarray:
    # pack initial conditions
    state0 = [
        y0["D0"],
        y0["A0"],
        y0["F10"],
        y0["F20"],
        y0["S0"],
        y0["X0"],
    ]

    sol = odeint(
        ode_list,
        state0,
        t,
        args=(
            params["gamma"],
            params["k"],
            params["delta"],
            params["rA"],
            params["K"],
            params["gamma2"],
        ),
        mxstep=5000,
    )

    # Living total (what a “cell index” often tracks, loosely)
    living = sol[:, 0] + sol[:, 1] + sol[:, 2] + sol[:, 3] + sol[:, 4]
    return living


def model_to_observed(living: np.ndarray, scale: float, offset: float) -> np.ndarray:
    # observed = scale * living + offset
    return scale * living + offset


# -----------------------------
# Data loading
# -----------------------------
def load_csv(path, tcol, ycol):
    df = pd.read_csv(path)

    # if requested columns exist, use them
    if tcol in df.columns and ycol in df.columns:
        t = df[tcol].astype(float).to_numpy()
        y = df[ycol].astype(float).to_numpy()
    else:
        # otherwise pick first two numeric columns
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            raise ValueError(
                f"{path}: need at least 2 numeric columns. Found: {df.columns.tolist()}"
            )
        t = num.iloc[:, 0].astype(float).to_numpy()
        y = num.iloc[:, 1].astype(float).to_numpy()

    idx = np.argsort(t)
    return t[idx], y[idx]


def normalize_y(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "y0":
        return y / (y[0] if y[0] != 0 else 1.0)
    if mode == "minmax":
        ymin, ymax = np.min(y), np.max(y)
        if ymax == ymin:
            return y * 0.0
        return (y - ymin) / (ymax - ymin)
    raise ValueError(f"Unknown normalize mode: {mode}")


# -----------------------------
# Fitting
# -----------------------------
@dataclass
class FitResult:
    params: Dict[str, float]
    curve_params: List[Dict[str, float]]
    ssr: float


def ssr_single(theta: np.ndarray, t: np.ndarray, y: np.ndarray, y0_cfg: Dict[str, float]) -> float:
    # theta = [gamma, k, delta, rA, logK, gamma2, scale, offset]
    gamma, k, delta, rA, logK, gamma2, scale, offset = theta

    # positivity guards (also handled by bounds, but keeps things stable)
    if gamma < 0 or k <= 0 or delta < 0 or scale == 0:
        return 1e18

    K = float(np.exp(logK))

    params = {"gamma": gamma, "k": k, "delta": delta, "rA": rA, "K": K, "gamma2": gamma2}
    living = simulate(t, params, y0_cfg)
    yhat = model_to_observed(living, scale=scale, offset=offset)

    resid = yhat - y
    return float(np.sum(resid * resid))


def ssr_multi(
    theta: np.ndarray,
    curves: List[Tuple[np.ndarray, np.ndarray]],
    y0_cfg: Dict[str, float],
) -> float:
    # shared model params + per-curve scale/offset
    # theta = [gamma, k, delta, rA, logK, gamma2, scale1, offset1, scale2, offset2, ...]
    gamma, k, delta, rA, logK, gamma2 = theta[:6]
    K = float(np.exp(logK))

    if gamma < 0 or k <= 0 or delta < 0:
        return 1e18

    params = {"gamma": gamma, "k": k, "delta": delta, "rA": rA, "K": K, "gamma2": gamma2}

    ssr = 0.0
    idx = 6
    for (t, y) in curves:
        scale = theta[idx]
        offset = theta[idx + 1]
        idx += 2

        living = simulate(t, params, y0_cfg)
        yhat = model_to_observed(living, scale=scale, offset=offset)
        resid = yhat - y
        ssr += float(np.sum(resid * resid))

    return ssr


def fit_single(
    t: np.ndarray,
    y: np.ndarray,
    y0_cfg: Dict[str, float],
    seed: int = 0,
) -> FitResult:
    rng = np.random.default_rng(seed)

    # Initial guesses (tweak if needed)
    gamma0 = 0.01
    k0 = 0.5
    delta0 = 0.02  # <-- death rate (this is what you asked for)
    rA0 = 0.00
    K0 = max(2.0 * y0_cfg["A0"], 10.0)
    gamma2_0 = 0.00

    # Scale/offset guess: map living(t0) ~ y0
    params0 = {"gamma": gamma0, "k": k0, "delta": delta0, "rA": rA0, "K": K0, "gamma2": gamma2_0}
    living0 = simulate(t, params0, y0_cfg)
    scale0 = (y[0] - 0.0) / (living0[0] if living0[0] != 0 else 1.0)
    offset0 = 0.0

    theta0 = np.array([gamma0, k0, delta0, rA0, np.log(K0), gamma2_0, scale0, offset0], dtype=float)

    # Bounds: keep it sane and stable
    bounds = [
        (0.0, 10.0),      # gamma
        (1e-6, 10.0),     # k
        (0.0, 2.0),       # delta  (death)
        (0.0, 2.0),       # rA
        (np.log(1e-3), np.log(1e6)),  # logK
        (0.0, 10.0),      # gamma2
        (-1e3, 1e3),      # scale
        (-1e3, 1e3),      # offset
    ]

    res = minimize(
        ssr_single,
        theta0,
        args=(t, y, y0_cfg),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000},
    )

    theta = res.x
    gamma, k, delta, rA, logK, gamma2, scale, offset = theta
    K = float(np.exp(logK))

    out_params = {"gamma": gamma, "k": k, "delta": delta, "rA": rA, "K": K, "gamma2": gamma2}
    curve_params = [{"scale": float(scale), "offset": float(offset)}]

    return FitResult(params=out_params, curve_params=curve_params, ssr=float(res.fun))


def fit_multi(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    y0_cfg: Dict[str, float],
    seed: int = 0,
) -> FitResult:
    # Shared model params
    gamma0, k0, delta0, rA0, K0, gamma2_0 = 0.01, 0.5, 0.02, 0.0, max(2.0 * y0_cfg["A0"], 10.0), 0.0

    theta0 = [gamma0, k0, delta0, rA0, np.log(K0), gamma2_0]

    bounds = [
        (0.0, 10.0),      # gamma
        (1e-6, 10.0),     # k
        (0.0, 2.0),       # delta
        (0.0, 2.0),       # rA
        (np.log(1e-3), np.log(1e6)),  # logK
        (0.0, 10.0),      # gamma2
    ]

    # Per-curve scale/offset
    for (t, y) in curves:
        params0 = {"gamma": gamma0, "k": k0, "delta": delta0, "rA": rA0, "K": K0, "gamma2": gamma2_0}
        living0 = simulate(t, params0, y0_cfg)
        scale0 = (y[0] - 0.0) / (living0[0] if living0[0] != 0 else 1.0)
        offset0 = 0.0
        theta0 += [scale0, offset0]
        bounds += [(-1e3, 1e3), (-1e3, 1e3)]

    theta0 = np.array(theta0, dtype=float)

    res = minimize(
        ssr_multi,
        theta0,
        args=(curves, y0_cfg),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 8000},
    )

    theta = res.x
    gamma, k, delta, rA, logK, gamma2 = theta[:6]
    K = float(np.exp(logK))
    out_params = {"gamma": gamma, "k": k, "delta": delta, "rA": rA, "K": K, "gamma2": gamma2}

    curve_params = []
    idx = 6
    for _ in curves:
        curve_params.append({"scale": float(theta[idx]), "offset": float(theta[idx + 1])})
        idx += 2

    return FitResult(params=out_params, curve_params=curve_params, ssr=float(res.fun))


# -----------------------------
# Plot + save
# -----------------------------
def plot_fit(
    paths: List[str],
    curves: List[Tuple[np.ndarray, np.ndarray]],
    fit: FitResult,
    y0_cfg: Dict[str, float],
    out_png: str,
) -> None:
    plt.figure()

    # Plot data + fitted model per curve
    for i, ((t, y), pth) in enumerate(zip(curves, paths)):
        cp = fit.curve_params[i] if i < len(fit.curve_params) else fit.curve_params[0]
        living = simulate(t, fit.params, y0_cfg)
        yhat = model_to_observed(living, cp["scale"], cp["offset"])

        plt.plot(t, y, marker="o", linestyle="None", label=f"data: {os.path.basename(pth)}")
        plt.plot(t, yhat, linestyle="-", label=f"fit:  {os.path.basename(pth)}")

    # Use the first file name (or a generic label) for the title stamp
    base = os.path.basename(paths[0]) if paths else "curve"

    gamma = float(fit.params.get("gamma", np.nan))
    k = float(fit.params.get("k", np.nan))
    delta = float(fit.params.get("delta", np.nan))

    plt.xlabel("Time (h)")
    plt.ylabel("Cell index / normalized signal")
    plt.title(f"{base} | gamma={gamma:.3g}, k={k:.3g}, delta={delta:.3g}")

    plt.legend()
    plt.text(0.02, 0.02, base, transform=plt.gca().transAxes)
    plt.tight_layout()

    # Save to the filename passed in from main()
    plt.savefig(out_png, dpi=300)

    # Non-blocking preview (batch-friendly)
    plt.show(block=False)
    plt.pause(1.0)
    plt.close()

def save_params_csv(fit: FitResult, paths: List[str], out_csv: str) -> None:
    rows = []

    # shared parameters
    for k, v in fit.params.items():
        rows.append({
            "group": "shared",
            "curve": "",
            "param": k,
            "value": float(v),
        })

    # per-curve parameters
    for i, cp in enumerate(fit.curve_params):
        curve_name = os.path.basename(paths[i]) if i < len(paths) else f"curve_{i+1}"
        for k, v in cp.items():
            rows.append({
                "group": "curve",
                "curve": curve_name,
                "param": k,
                "value": float(v),
            })

    rows.append({
        "group": "fit",
        "curve": "",
        "param": "SSR",
        "value": float(fit.ssr),
    })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--data", nargs="+", required=True, help="One or more CSV files (digitized curves).")
    ap.add_argument("--tcol", default="time_h", help="time column name")
    ap.add_argument("--ycol", default="value", help="y column name")


    ap.add_argument("--normalize", default="none", choices=["none", "y0", "minmax"])
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--out_prefix", default="fit_out")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--D0", type=float, default=1.0)
    ap.add_argument("--A0", type=float, default=9.0)
    ap.add_argument("--F10", type=float, default=0.0)
    ap.add_argument("--F20", type=float, default=0.0)
    ap.add_argument("--S0", type=float, default=0.0)
    ap.add_argument("--X0", type=float, default=0.0)

    args = ap.parse_args()

    y0_cfg = {
        "D0": float(args.D0),
        "A0": float(args.A0),
        "F10": float(args.F10),
        "F20": float(args.F20),
        "S0": float(args.S0),
        "X0": float(args.X0),
    }

    paths = args.data
    curves = []
    for p in paths:
        t, y = load_csv(p, args.tcol, args.ycol)
        y = normalize_y(y, args.normalize)
        curves.append((t, y))

    if args.multi and len(curves) > 1:
        fit = fit_multi(curves, y0_cfg, seed=args.seed)
    else:
        fit = fit_single(curves[0][0], curves[0][1], y0_cfg, seed=args.seed)
        curves = [curves[0]]
        paths = [paths[0]]

    # ✅ THIS is the “save outputs” logic you were missing:
    if len(paths) == 1:
        stem = os.path.splitext(os.path.basename(paths[0]))[0]
    else:
        stem = args.out_prefix

    out_png = f"{stem}_fit.png"
    out_csv = f"{stem}_params.csv"

    plot_fit(paths, curves, fit, y0_cfg, out_png)
    save_params_csv(fit, paths, out_csv)

    print("\n=== FIT DONE ===")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()