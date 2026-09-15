print("SCRIPT STARTED")

from tabnanny import check
import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
import argparse
import os
import glob
import re
import math

#Differential Equations. Erlang Two-Step Fusion 
def ode_list(li,t,gamma,k_val):
    D = li[0]
    A = li[1]
    F1 = li[2]
    F2 = li[3]
    S = li[4]

    dDdt = (-gamma*D*A)
    dAdt = (-gamma*D*A) - (gamma*S*A)
    dF1dt = (2*gamma*D*A)+(gamma*S*A)-(k_val*F1)
    dF2dt = (k_val*F1)-(k_val*F2)
    dSdt = (k_val*F2)

    return [dDdt, dAdt, dF1dt, dF2dt, dSdt]

#Taking optimize values and getting y values 
def true_y_values(li, t_vals):
    donor_y = li.x[0]
    gamma_y = li.x[1]
    k_val_y = li.x[2]

    true_vals = odeint(ode_list,[donor_y, 1-donor_y, 0, 0, 0], t_vals, args = (gamma_y, k_val_y), mxstep=5000)
    y_vals = true_vals[:,4] / (true_vals[:,0] + true_vals[:,1] + true_vals[:,2] + true_vals[:,3])
    return y_vals

#Chi Squared
def chi_squared(li_1,li_2):
    chi = sum((li_1 - li_2)**2/li_1)
    return chi

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
        if donor_SSR <= 0 or donor_SSR >= 1:
            return 1e18
        if gamma_SSR <= 0 or k_val_SSR <= 0:
            return 1e18

        var = [donor_SSR, 1-donor_SSR, 0, 0, 0]
        return_values = odeint(ode_list,var, t_vals, args = (gamma_SSR, k_val_SSR), mxstep=5000) 
        y_predicted_val = return_values[:,4]/(return_values[:,0]+return_values[:,1]+return_values[:,2]+return_values[:,3])
        p = np.mean((check - y_predicted_val)**2)
        return p

    ##### MAIN CODE #####
    initial_guess = [0.5, 0.001, 0.02]
    opti = scipy.optimize.minimize(SSR_code, initial_guess, method = 'Nelder-Mead', options={'maxiter':5000})
    observed = true_y_values(opti, t_vals)

    SSR = float(opti.fun) * len(check)
    n = len(check)
    m = 3
    sigma = float(np.sqrt(SSR/max(n-m,1)))

    check_chi = check.copy()
    if n > 3:
        check_chi[0] = 0.0001
        check_chi[1] = 0.0001
        check_chi[2] = 0.0001
        check_chi[3] = 0.0001

    den = max(sigma**2, 1e-12)
    chi2 = float(SSR / den)
    m = 3
    AIC = float(aic(check, SSR, m))


    return opti, observed, SSR, sigma, chi2, AIC

def plot_and_save(final_arr, observed, out_png, title):
    plt.figure(figsize=(6,4))
    plt.scatter(final_arr[:,0], final_arr[:,1])
    plt.plot(final_arr[:,0], observed)
    plt.xlabel("Time(hr)", fontsize=10)
    plt.ylabel("Cell Index", fontsize=10)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def write_params(out_csv, opti, SSR, sigma, chi2, AIC):
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["p","gamma","k_val","SSR","sigma","chi^2","AIC"])
        w.writerow([opti.x[0], opti.x[1], opti.x[2], SSR, sigma, chi2, AIC])


def panel_name_from_file(path):
    base = os.path.splitext(os.path.basename(path))[0]

    if base.startswith("FigB_"):
        return None

    if base.startswith("Fig2_mock"):
        return "Fig 2A"
    
    if base.lower().startswith("fig2_mock"):
        key = "Fig 2A"

    m = re.match(r"^(Fig\d+[A-Za-z])", base)
    if m:
        key = m.group(1)
        return f"{key[:3]} {key[3:]}"  # "Fig 4C"

    return None


def auto_segment_and_invert(final_arr):
    t = final_arr[:,0]
    y = final_arr[:,1]

    if len(y) < 5:
        return final_arr, False

    # detect U-shape-ish: minimum is not at ends
    i_min = int(np.argmin(y))
    u_shape = (2 <= i_min <= len(y)-3)

    if u_shape:
        # keep the increasing leg after the minimum
        t2 = t[i_min:]
        y2 = y[i_min:]

        # rebase like preprocess does: start at 0
        t2 = t2 - t2[0]
        y2 = y2 - y2[0]
        mx = np.max(y2) if len(y2) else 1.0
        if mx > 0:
            y2 = y2 / mx

        seg = np.column_stack([t2, y2])
    else:
        seg = final_arr.copy()

    # choose invert so segment trends upward overall
    yseg = seg[:,1]
    invert = (yseg[-1] < yseg[0])
    return seg, invert


def fig_sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(Fig\d+[A-Za-z])", base)
    if not m:
        return (10**9, "Z", base)
    num = int(m.group(2))
    letter = m.group(3).upper()
    rest = (m.group(4) or "").lower()
    return (num, letter, rest)
def fig_panel_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(Fig\d+[A-Za-z])", base)
    if not m:
        return base
    return f"{m.group(1)} {int(m.group(2))}{m.group(3).upper()}"  # e.g. "Fig 3A"

def fig_condition_label(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(Fig\d+[A-Za-z])", base)
    if not m:
        return base
    rest = m.group(4) or ""
    if not rest:
        return "trace"
    return rest.replace("_", " ")


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
