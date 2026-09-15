import os, glob
import numpy as np
import pandas as pd

DATA_DIR = "csv"
GROUPS = ["Fig2A", "Fig3A", "Fig3B", "Fig4A", "Fig4B", "Fig4C"]

def load_file(path):
    df = pd.read_csv(path, header=None, sep=",", engine="python")
    df = df.iloc[:, :2]
    df.columns = ["x", "y"]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("x")
    return df


def combine_group(prefix):
    files = glob.glob(os.path.join(DATA_DIR, f"{prefix}*.csv"))
    files = [f for f in files if "_combined" not in os.path.basename(f)]
    if not files:
        print(f"[none] {prefix}")
        return

    dfs = []
    for f in files:
        df = load_file(f)
        if len(df) >= 3:
            dfs.append(df)

    if not dfs:
        print(f"[skip] {prefix}: no usable inputs")
        return

    # Build a common x-grid = union of all x’s, sorted
    x_grid = np.unique(np.concatenate([d["x"].to_numpy() for d in dfs]))
    x_grid.sort()

    y_stack = []
    for d in dfs:
        # interpolate onto grid
        y_interp = np.interp(x_grid, d["x"].to_numpy(), d["y"].to_numpy(), left=np.nan, right=np.nan)
        y_stack.append(y_interp)

    y_mat = np.vstack(y_stack)
    y_mean = np.nanmean(y_mat, axis=0)

    out = pd.DataFrame({"x": x_grid, "y": y_mean}).dropna()
    if len(out) < 3:
        print(f"[skip] {prefix}: combined output too small after dropna")
        return

    out_path = os.path.join(DATA_DIR, f"{prefix}_combined.csv")
    out.to_csv(out_path, index=False, header=False)
    print(f"[saved] {out_path}  inputs={len(dfs)} rows={len(out)} bytes={os.path.getsize(out_path)}")

if __name__ == "__main__":
    for g in GROUPS:
        combine_group(g)