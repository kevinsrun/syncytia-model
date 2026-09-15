"""Fit grouped syncytia curves with an Erlang two-step fusion/death model."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


PARAMETER_NAMES = ("donor_fraction", "gamma", "k_val", "delta")
PARAMETER_BOUNDS = (
    (0.001, 0.999),
    (1e-8, 500.0),
    (1e-8, 500.0),
    (0.0, 10.0),
)
INITIAL_GUESS = np.array([0.5, 0.001, 0.02, 0.02], dtype=float)
PENALTY = 1e18


def ode_list(state, time, gamma, k_val, delta):
    """Erlang two-step fusion model with death/loss from living compartments."""
    donor, acceptor, fusion_1, fusion_2, syncytia = state

    donor_acceptor_fusion = gamma * donor * acceptor
    syncytia_acceptor_fusion = gamma * syncytia * acceptor

    donor_rate = -donor_acceptor_fusion - delta * donor
    acceptor_rate = (
        -donor_acceptor_fusion
        - syncytia_acceptor_fusion
        - delta * acceptor
    )
    fusion_1_rate = (
        2.0 * donor_acceptor_fusion
        + syncytia_acceptor_fusion
        - k_val * fusion_1
        - delta * fusion_1
    )
    fusion_2_rate = k_val * fusion_1 - k_val * fusion_2 - delta * fusion_2
    syncytia_rate = k_val * fusion_2 - delta * syncytia

    return [
        donor_rate,
        acceptor_rate,
        fusion_1_rate,
        fusion_2_rate,
        syncytia_rate,
    ]


def read_csv_two_cols(path):
    """Read the first two numeric columns from a CSV file."""
    rows = []
    with open(path, newline="") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"{path}: no rows with two numeric columns")

    values = np.asarray(rows, dtype=float)
    return values[np.argsort(values[:, 0])]


def preprocess(values, max_time=None, baseline=True, normalize=True, invert=False):
    """Sort, optionally truncate, baseline, normalize, and invert one curve."""
    result = np.asarray(values, dtype=float).copy()

    if max_time is not None:
        result = result[result[:, 0] <= max_time]
    if not len(result):
        raise ValueError("no data points remain after preprocessing")

    if baseline:
        result[:, 1] -= result[0, 1]

    if normalize:
        y_min = np.min(result[:, 1])
        y_range = np.max(result[:, 1]) - y_min
        if y_range > 0:
            result[:, 1] = (result[:, 1] - y_min) / y_range

    if invert:
        result[:, 1] = 1.0 - result[:, 1]

    return result


def initial_state(donor_fraction, path):
    """Use the spike experiment's seeded fusion intermediate when applicable."""
    fusion_1 = 0.1 if "spike" in os.path.basename(path).lower() else 0.0
    return [donor_fraction, 1.0 - donor_fraction, fusion_1, 0.0, 0.0]


def predict(theta, times, path):
    donor_fraction, gamma, k_val, delta = theta
    solution = odeint(
        ode_list,
        initial_state(donor_fraction, path),
        times,
        args=(gamma, k_val, delta),
        mxstep=5000,
    )
    denominator = solution[:, 0] + solution[:, 1] + solution[:, 2] + solution[:, 3]
    if np.any(np.abs(denominator) < 1e-12):
        raise FloatingPointError("model produced a zero observation denominator")
    return solution[:, 4] / denominator


def objective(theta, times, observed, path):
    if not np.all(np.isfinite(theta)):
        return PENALTY
    try:
        predicted = predict(theta, times, path)
    except (FloatingPointError, RuntimeError, ValueError):
        return PENALTY
    if not np.all(np.isfinite(predicted)):
        return PENALTY
    residuals = predicted - observed
    return float(np.dot(residuals, residuals))


def fit_curve(final_arr, path):
    """Fit donor fraction, fusion rates, and the death parameter to one curve."""
    times = final_arr[:, 0]
    observed = final_arr[:, 1]
    result = minimize(
        objective,
        INITIAL_GUESS,
        args=(times, observed, path),
        method="Nelder-Mead",
        bounds=PARAMETER_BOUNDS,
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
    )
    predicted = predict(result.x, times, path)

    ssr = float(np.sum((predicted - observed) ** 2))
    sample_count = len(observed)
    parameter_count = len(PARAMETER_NAMES)
    residual_sigma = float(np.sqrt(ssr / max(sample_count - parameter_count, 1)))
    variance = max(ssr / max(sample_count, 1), 1e-12)
    aic = float(
        2 * parameter_count
        + sample_count * (1.0 + np.log(2.0 * np.pi * variance))
    )
    chi_squared = float(ssr / max(residual_sigma**2, 1e-12))
    return result, predicted, ssr, residual_sigma, chi_squared, aic


def panel_key(path):
    base = os.path.splitext(os.path.basename(path))[0].strip()
    if base.lower().startswith("figb_"):
        return None
    if base.lower().startswith("fig2_mock") or base.lower().startswith("fig2a_"):
        return "Fig 2A"
    match = re.match(r"^fig(\d+)([a-z])", base, re.IGNORECASE)
    if not match:
        return None
    return f"Fig {int(match.group(1))}{match.group(2).upper()}"


def condition_label(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_", 1)
    return parts[1].replace("_", " ") if len(parts) == 2 else "trace"


def group_paths(paths):
    groups = {}
    for path in paths:
        key = panel_key(path)
        if key is not None:
            groups.setdefault(key, []).append(path)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="christmas")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--max_time", type=float, default=None)
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    if args.mode.lower() != "christmas":
        raise SystemExit("Only --mode christmas is supported in this script.")

    paths = []
    for pattern in args.data:
        matches = glob.glob(pattern)
        paths.extend(matches or [pattern])
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("No CSV files matched --data.")

    groups = group_paths(paths)
    if not groups:
        raise SystemExit("No recognized figure CSV files were provided.")

    print(f"FILES FOUND: {len(paths)}")
    for figure, figure_paths in sorted(groups.items()):
        plt.figure(figsize=(7, 4))
        print(f"\n==== {figure} ====")

        for path in figure_paths:
            final_arr = preprocess(
                read_csv_two_cols(path),
                max_time=args.max_time,
                baseline=not args.no_baseline,
                normalize=not args.no_normalize,
                invert=args.invert,
            )
            result, predicted, ssr, sigma, chi_squared, aic = fit_curve(
                final_arr, path
            )

            fitted_parameters = dict(zip(PARAMETER_NAMES, result.x))
            print(os.path.basename(path))
            for name, value in fitted_parameters.items():
                print(f"{name}: {value:.8g}")
            print(f"SSR: {ssr:.8g}")
            print(f"residual sigma: {sigma:.8g}")
            print(f"chi^2: {chi_squared:.8g}")
            print(f"AIC: {aic:.8g}")

            plt.scatter(
                final_arr[:, 0],
                final_arr[:, 1],
                s=14,
                alpha=0.7,
                label=condition_label(path),
            )
            # A one-point "line" contains no curve information, so omit it.
            if len(np.unique(final_arr[:, 0])) >= 2:
                plt.plot(final_arr[:, 0], predicted, linewidth=2)

        plt.xlabel("Time (hr)")
        plt.ylabel("Cell Index")
        plt.ylim(-1, 2)
        plt.title(figure)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure.replace(" ", "") + ".png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
