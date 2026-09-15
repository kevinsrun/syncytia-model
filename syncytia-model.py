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
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator


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

def draw_curve_through_points(t: np.ndarray, y: np.ndarray, mode: str, n: int = 300):
    if mode == "none":
        return None

    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    tu, unique_idx = np.unique(t, return_index=True)
    yu = y[unique_idx]

    if len(tu) < 2:
        return None

    tt = np.linspace(tu.min(), tu.max(), n)

    if mode == "linear":
        yy = np.interp(tt, tu, yu)
    elif mode == "pchip":
        yy = PchipInterpolator(tu, yu)(tt)
    else:
        return None

    return tt, yy

# -----------------------------
# Data loading
# -----------------------------
def _looks_numeric(s: str) -> bool:
    try:
        float(str(s).strip())
        return True
    except Exception:
        return False


def load_csv(path, tcol, ycol):
    """
    Robust CSV loader that supports:
      - Normal header CSVs with columns like time_h,value
      - Headerless 2-column CSVs (like WebPlotDigitizer exports)
      - Fallback to first two numeric columns
    """
    # First attempt: normal read (assume header row)
    df = pd.read_csv(path)

    # If requested columns exist, use them directly
    if (tcol in df.columns) and (ycol in df.columns):
        t = df[tcol].astype(float).to_numpy()
        y = df[ycol].astype(float).to_numpy()
        idx = np.argsort(t)
        return t[idx], y[idx]

    # Auto-detect: headerless 2-column file accidentally read with first row as header
    # Symptom: exactly 2 columns AND BOTH column names look like numbers.
    if df.shape[1] == 2 and all(_looks_numeric(c) for c in df.columns):
        df2 = pd.read_csv(path, header=None)
        # Use the first two columns as x,y
        t = df2.iloc[:, 0].astype(float).to_numpy()
        y = df2.iloc[:, 1].astype(float).to_numpy()
        idx = np.argsort(t)
        return t[idx], y[idx]

    # Otherwise: fallback to first two numeric columns in the file
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError(
            f"{path}: could not find columns '{tcol}' and '{ycol}', and "
            f"could not find 2 numeric columns. Found columns: {df.columns.tolist()}"
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



# Plot + save

def plot_fit(
    paths,
    curves,
    fit,
    y0_cfg,
    out_png,
    labels,
    data_curve,
    no_model,
    xlabel,
    ylabel,
):
    plt.figure()

    # plot each curve
    for i, ((t, y), pth) in enumerate(zip(curves, paths)):
        label = labels[i] if i < len(labels) else os.path.basename(pth)

        # points
        plt.plot(t, y, marker="o", linestyle="None", label=label)

        # smooth curve through points (paper-style)
        sm = draw_curve_through_points(t, y, data_curve)
        if sm is not None:
            tt, yy = sm
            plt.plot(tt, yy, linestyle="-", linewidth=2)

        # optional ODE model overlay (dashed)
        if not no_model:
            cp = fit.curve_params[i] if i < len(fit.curve_params) else fit.curve_params[0]
            living = simulate(t, fit.params, y0_cfg)
            yhat = model_to_observed(living, cp["scale"], cp["offset"])
            plt.plot(t, yhat, linestyle="--", linewidth=2)

    base = os.path.basename(paths[0]) if paths else "curve"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel} | {base}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
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

def run_batch_figures(
    base_dir: str,
    out_dir: str,
    xlabel: str,
    ylabel: str,
    data_curve: str = "pchip",
    no_model: bool = True,
):
    """
    Generates PNGs for: Fig2, Fig3A, Fig3B, Fig3C, Fig4A, Fig4B, Fig4C
    using your CSVs and combining the right sets.
    """
    os.makedirs(out_dir, exist_ok=True)

    groups = {
        "Fig2": [
            "Fig2_mock_TF.csv",
            "Fig2A_spike-TF.csv",
        ],
        "Fig3A": [
            "Fig3A_1_1.csv",
            "Fig3A_1_2.csv",
            "Fig3A_2_1.csv",
        ],
        "Fig3B": [
            "Fig3B_0.156ug.csv",
            "Fig3B_0.313ug.csv",
            "Fig3B_0.625ug.csv",
            "Fig3B_1.25ug.csv",
            "FIg3B_2.5ug.csv",   # note the capitalization in your zip
            "Fig3B_5ug.csv",
        ],
        "Fig3C": [
            "Fig3C_A549.ACE2+.csv",
            "Fig3C_A549.ACE2+.TMPRSS2+.csv",
        ],
        "Fig4A": [
            "Fig4A_untreated.csv",
            "Fig4A_1uM.csv",
            "Fig4A_10uM.csv",
        ],
        "Fig4B": [
            "Fig4B_untreated.csv",
            "Fig4B_1ug_mL.csv",
            "Fig4B_10ug_mL.csv",
        ],
        "Fig4C": [
            "Fig4C_untreated.csv",
            "Fig4C_0.4uM.csv",
            "Fig4C_2uM.csv",
        ],
    }

    # Simple plotting-only: use no_model=True and do NOT fit
    # We'll reuse your plot_fit by passing a dummy fit if needed.
    # Better: just plot data+smooth curve directly here.
    for fig_name, files in groups.items():
        paths = [os.path.join(base_dir, f) for f in files if os.path.exists(os.path.join(base_dir, f))]
        if not paths:
            print(f"[SKIP] {fig_name}: no files found")
            continue

        curves = []
        for p in paths:
            t, y = load_csv(p, "time_h", "value")  # your CSVs use these
            curves.append((t, y))

        labels = [label_from_filename(p) for p in paths]
        out_png = os.path.join(out_dir, f"{fig_name}.png")

        # Plot only (paper look)
        plt.figure()
        for (t, y), pth, lab in zip(curves, paths, labels):
            plt.plot(t, y, marker="o", linestyle="None", label=lab)

            sm = draw_curve_through_points(t, y, data_curve)
            if sm is not None:
                tt, yy = sm
                plt.plot(tt, yy, linestyle="-", linewidth=2)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{xlabel} vs {ylabel} | {fig_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

        print(f"[WROTE] {out_png}")

def label_from_filename(fname: str) -> str:
    """
    Convert filenames like:
      Fig4A_10uM.csv -> '10 uM'
      Fig4B_1ug_mL.csv -> '1 ug/mL'
      Fig4C_0.4uM.csv -> '0.4 uM'
      Fig3C_A549.ACE2+.TMPRSS2+.csv -> 'A549-ACE2+TMPRSS2+'
      Fig2_mock_TF.csv -> 'Mock'
      Fig2A_spike-TF.csv -> 'Spike-TF'
      *_untreated.csv -> 'Untreated'
      *_combined.csv -> 'Combined'
    """
    base = os.path.splitext(os.path.basename(fname))[0]

    if "untreated" in base.lower():
        return "Untreated"
    if "combined" in base.lower():
        return "Combined"
    if "mock" in base.lower():
        return "Mock"
    if "spike" in base.lower():
        return "Spike-TF"

    # strip leading panel prefix like Fig4A_
    base = re.sub(r"^Fig\d+[A-Z]?_", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^Fig\d+[A-Z]_", "", base, flags=re.IGNORECASE)
    base = base.replace("_", " ")

    # units normalization
    base = base.replace("uM", " uM")
    base = base.replace("ug mL", " ug/mL")
    base = base.replace("ug/mL", " ug/mL")
    base = base.replace("ug", " ug")

    # cell line labels
    if "A549" in base:
        base = base.replace(".", "-")
        base = base.replace("ACE2+", "ACE2+")
        base = base.replace("TMPRSS2+", "TMPRSS2+")
        base = base.replace(" ", "")

    return base.strip()

def _clean_label(stem: str) -> str:
    """
    Convert filename stem like:
      Fig4A_10uM -> "10 uM"
      Fig4B_1ug_mL -> "1 ug/mL"
      Fig3C_A549.ACE2+.TMPRSS2+ -> "A549.ACE2+.TMPRSS2+"
      Fig4C_untreated -> "Untreated"
    """
    s = stem

    # drop leading figure prefix "FigX_"
    s = re.sub(r"^Fig\d+[A-Z]?_", "", s)

    # untreated
    if s.lower() in {"untreated", "control"}:
        return "Untreated"

    # normalize unit formatting
    s = s.replace("_", "/")  # ug_mL -> ug/mL
    s = s.replace("uM", " uM")
    s = s.replace("ug/mL", " ug/mL")
    s = s.replace("ug", " ug")  # Fig3B_0.156ug -> "0.156 ug"

    # cleanup multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dose_sort_key(label: str):
    """
    Put Untreated first, then numeric doses ascending.
    Works for labels like "0.4 uM", "10 ug/mL", "0.156 ug", etc.
    """
    if label.lower().startswith("untreated"):
        return (-1.0, "")

    m = re.search(r"([-+]?\d*\.?\d+)", label)
    if not m:
        return (1e9, label)

    val = float(m.group(1))
    # unit bucket so uM and ug/mL don't intermix oddly (still rare)
    unit = label.replace(m.group(1), "").strip().lower()
    return (val, unit)


def run_batch_figures(
    base_dir: str,
    out_dir: str,
    xlabel: str,
    ylabel: str,
    data_curve: str = "pchip",
    no_model: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # All csv files in directory
    all_csv = sorted([f for f in os.listdir(base_dir) if f.lower().endswith(".csv")])
    if not all_csv:
        print(f"[WARN] No CSV files found in: {base_dir}")
        return

    # Group patterns you showed
    groups = {
        "Fig2":   r"^Fig2",
        "Fig3A":  r"^Fig3A",
        "Fig3B":  r"^Fig3B",
        "Fig3C":  r"^Fig3C",
        "Fig4A":  r"^Fig4A",
        "Fig4B":  r"^Fig4B",
        "Fig4C":  r"^Fig4C",
    }

    for fig, pat in groups.items():
        files = [f for f in all_csv if re.search(pat, os.path.splitext(f)[0], flags=re.I)]
        if not files:
            print(f"[SKIP] {fig}: no files found")
            continue

        paths = [os.path.join(base_dir, f) for f in files]

        # Build labels from filenames
        stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        labels = [_clean_label(st) for st in stems]

        # Sort by dose (Untreated first, then ascending)
        order = sorted(range(len(paths)), key=lambda i: _dose_sort_key(labels[i]))
        paths = [paths[i] for i in order]
        labels = [labels[i] for i in order]

        # Load curves
        curves = []
        for p in paths:
            t, y = load_csv(p, tcol="time_h", ycol="value")
            curves.append((t, y))

        # If you're in "paper look" mode: no model fit needed
        # We'll call plot_fit with fit=None safely by setting no_model=True
        out_png = os.path.join(out_dir, f"{fig}.png")

        # Minimal y0_cfg needed only if model is on
        y0_cfg = {"D0": 1.0, "A0": 9.0, "F10": 0.0, "F20": 0.0, "S0": 0.0, "X0": 0.0}

        # If your plot_fit currently requires a FitResult even when no_model=True,
        # create a dummy placeholder:
        dummy_fit = FitResult(
            params={"gamma": 0.0, "k": 1.0, "delta": 0.0, "rA": 0.0, "K": 1.0, "gamma2": 0.0},
            curve_params=[{"scale": 1.0, "offset": 0.0} for _ in curves],
            ssr=0.0,
        )

        plot_fit(
            paths=paths,
            curves=curves,
            fit=dummy_fit,
            y0_cfg=y0_cfg,
            out_png=out_png,
            labels=labels,
            data_curve=data_curve,
            no_model=no_model,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        print(f"[OK] Wrote {out_png}")

    # -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---- args ----
    ap.add_argument("--data", nargs="+", required=False, help="One or more CSV files")
    ap.add_argument("--tcol", default="time_h")
    ap.add_argument("--ycol", default="value")
    ap.add_argument("--xlabel", default="Time (h)")
    ap.add_argument("--ylabel", default="Cell index")
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

    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--data_curve", default="pchip", choices=["none", "linear", "pchip"])
    ap.add_argument("--no_model", action="store_true")

    ap.add_argument("--batch", action="store_true",
                    help="Generate Fig2/Fig3A/B/C/Fig4A/B/C PNGs from a folder of CSVs.")
    ap.add_argument("--csv_dir", default="figure_digitized_csvs",
                    help="Folder containing the CSVs (relative to script).")
    ap.add_argument("--out_dir", default="out_figures",
                    help="Output folder for PNGs.")

    args = ap.parse_args()

    # -----------------------------
    # Batch mode
    # -----------------------------
    if args.batch:
        run_batch_figures(
            base_dir=args.csv_dir,
            out_dir=args.out_dir,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            data_curve=args.data_curve,
            no_model=True,
        )
        return

    # -----------------------------
    # Normal mode (single/multi file)
    # -----------------------------
    if not args.data:
        raise ValueError("Either --data or --batch must be provided.")

    # ---- initial conditions ----
    y0_cfg = {
        "D0": args.D0,
        "A0": args.A0,
        "F10": args.F10,
        "F20": args.F20,
        "S0": args.S0,
        "X0": args.X0,
    }

    paths = args.data
    labels = args.labels if args.labels else [
        os.path.splitext(os.path.basename(p))[0] for p in paths
    ]

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
        labels = [labels[0]]

    stem = os.path.splitext(os.path.basename(paths[0]))[0]
    out_png = f"{stem}_fit.png"
    out_csv = f"{stem}_params.csv"

    plot_fit(
        paths,
        curves,
        fit,
        y0_cfg,
        out_png,
        labels,
        args.data_curve,
        args.no_model,
        args.xlabel,
        args.ylabel,
    )

    save_params_csv(fit, paths, out_csv)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
