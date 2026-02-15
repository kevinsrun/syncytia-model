import os
import subprocess
import sys

CSV_DIR = "csv"
MODES = ["baseline", "stage1", "stage2", "ude", "full"]
GROUPS = ["Fig2A", "Fig3A", "Fig3B", "Fig4A", "Fig4B", "Fig4C"]

def main():
    py = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    script = os.path.join(os.getcwd(), "hybrid_rsv_models_min.py")

    for g in GROUPS:
        combined = os.path.join(CSV_DIR, f"{g}_combined.csv")
        if not os.path.exists(combined):
            print(f"[skip] missing {combined}")
            continue

        for mode in MODES:
            out = os.path.join(CSV_DIR, f"pred_{g}_{mode}.csv")
            cmd = [py, script, "--csv", combined, "--mode", mode]
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)

            # hybrid script writes predictions.csv; rename it to keep outputs
            if os.path.exists("predictions.csv"):
                os.replace("predictions.csv", out)
                print(f"[saved] {out}")

if __name__ == "__main__":
    main()