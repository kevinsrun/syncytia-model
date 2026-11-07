print("SCRIPT STARTED")

import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import argparse
import os
import glob
import re

#Differential Equations. Erlang Two-Step Fusion 
def ode_list(li,t,gamma,k_val,death_parameter):
    D = li[0]
    A = li[1]
    F1 = li[2]
    F2 = li[3]
    S = li[4]

    dDdt = (-gamma*D*A) - (death_parameter*D)
    dAdt = (-gamma*D*A) - (gamma*S*A) - (death_parameter*A)
    dF1dt = (2*gamma*D*A)+(gamma*S*A)-(k_val*F1) - (death_parameter*F1)
    dF2dt = (k_val*F1)-(k_val*F2) - (death_parameter*F2)
    dSdt = (k_val*F2) - (death_parameter*S)

    return [dDdt, dAdt, dF1dt, dF2dt, dSdt]

#Shared model integration: unpack parameters, integrate the ODE, return the observed signal S/(D+A+F1+F2)
def simulate_y_values(li, t_vals):
    donor = li[0]
    gamma = li[1]
    k_val = li[2]
    death_parameter = li[3]

    vals = odeint(ode_list,[donor, 1-donor, 0, 0, 0], t_vals, args = (gamma, k_val, death_parameter), mxstep=5000)
    y_vals = vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])
    return y_vals

#Taking optimize values and getting y values 
def true_y_values(li, t_vals):
    return simulate_y_values(li.x, t_vals)

#Akaike's Information Criterion
def aic(li, SSR, m):
    n = len(li)
    sigma2 = SSR / max(n, 1)
    return 2*m + n*(1.0 + np.log(2*np.pi*sigma2))

def read_csv_two_cols(path):
    file = open(path, newline='')
    csvreader = csv.reader(file)
    header = next(csvreader, None)
    rows = []
    for row in csvreader:
        if row is None or len(row) < 2:
            continue
        try:
            rows.append([float(row[0]), float(row[1])])
        except:
            continue
    file.close()
    arr = np.array(rows, dtype=float)
    return arr, header

def preprocess(arr, max_time=None, baseline=True, normalize=True, invert=False):
    arr_sorted = sorted(arr, key = lambda x: x[0])
    arr2 = np.array(arr_sorted, dtype=float)

    if max_time is not None:
        cut = (arr2[:,0] <= float(max_time))
        arr2 = arr2[np.where(cut)[0]]

    if baseline and len(arr2) > 0:
        key = arr2[0,1]
        arr2[:,1] = arr2[:,1] - key
        arr2[0,1] = 0.0

    if normalize and len(arr2) > 0:
        y = arr2[:,1]
        y_min = np.min(y)
        y_max = np.max(y)
        rng = y_max - y_min
        if rng > 0:
            arr2[:,1] = (y - y_min) / rng


    if invert and len(arr2) > 0:
        arr2[:,1] = 1.0 - arr2[:,1]

    return arr2

def fit_one(final_arr):
    t_vals = final_arr[:,0]
    check = final_arr[:,1]

    #Getting SSR values from integration
    def SSR_code(li):
        donor_SSR = li[0]
        gamma_SSR = li[1]
        k_val_SSR = li[2]
        death_parameter_SSR = li[3]
        if donor_SSR <= 0 or donor_SSR >= 1:
            return 1e18
        if gamma_SSR <= 0 or k_val_SSR <= 0 or death_parameter_SSR < 0:
            return 1e18

        y_predicted_val = simulate_y_values(li, t_vals)
        p = np.mean((check - y_predicted_val)**2)
        return p

    ##### MAIN CODE #####
    initial_guess = [0.5, 0.001, 0.02, 0.02]
    opti = scipy.optimize.minimize(SSR_code, initial_guess, method = 'Nelder-Mead', options={'maxiter':5000})
    observed = true_y_values(opti, t_vals)

    SSR = float(opti.fun) * len(check)
    n = len(check)
    m = 4
    sigma = float(np.sqrt(SSR/max(n-m,1)))

    den = max(sigma**2, 1e-12)
    chi2 = float(SSR / den)
    AIC = float(aic(check, SSR, m))


    return opti, observed, SSR, sigma, chi2, AIC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="christmas")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max_time", type=float, default=None)
    ap.add_argument("--no_baseline", action="store_true")
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    if args.mode.lower() != "christmas":
        raise SystemExit("Only --mode christmas is supported in this strict script.")

    paths = []
    for p in args.data:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)

    paths = sorted(paths)
    print("FILES FOUND:", len(paths))
    if not paths:
        raise SystemExit("No CSV files matched your --data pattern.")

    # group paths by panel like "Fig 3A", "Fig 3B", ...
    groups = {}

    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0].strip()

        # skip malformed duplicate
        if base.lower().startswith("figb_"):
            continue

        # force mock into Fig 2A
        if base.lower().startswith("fig2_mock"):
            key = "Fig 2A"
        else:
            m = re.match(r"^(fig\d+[a-z])", base, re.IGNORECASE)
            if not m:
                continue
            raw = m.group(1)              # e.g. fig4c
            key = f"Fig {raw[3:-1]}{raw[-1].upper()}"  # "Fig 4C"

        groups.setdefault(key, []).append(path)



    # run each group -> one combined plot per panel
    for panel_key in sorted(groups.keys()):

        plt.figure(figsize=(7,4))
        print("\n====", panel_key, "====")

        for path in groups[panel_key]:

            arr, header = read_csv_two_cols(path)

            final_arr = preprocess(
                arr,
                max_time=args.max_time,
                baseline=(not args.no_baseline),
                normalize=(not args.no_normalize),
                invert=args.invert
            )

            opti, observed, SSR, sigma, chi2, AIC = fit_one(final_arr)

            print(os.path.basename(path))
            print("fun", opti.fun)
            print("death_parameter", opti.x[3])
            print("sigma", sigma)
            print("chi^2", chi2)
            print("AIC", AIC)

            label = os.path.splitext(os.path.basename(path))[0].split("_",1)[1]

            plt.scatter(final_arr[:,0], final_arr[:,1], s=14, alpha=0.7, label=label)
            plt.plot(final_arr[:,0], observed, linewidth=2)

        plt.xlabel("Time(hr)")
        plt.ylabel("Cell Index")
        plt.ylim(-1, 2)
        plt.title(panel_key)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(panel_key.replace(" ","") + ".png", dpi=200)
        plt.close()


if __name__ == "__main__":
        main()
