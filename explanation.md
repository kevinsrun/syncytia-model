# Syncytia model cleanup - change explanations

Conservative cleanup of `syncytia-model.py` (327 -> 222 lines). Physics, fit settings, printed numbers and plots are unchanged. Verified: all 22 fitted curves produce byte-identical `fun` / `death_parameter` / `sigma` / `chi^2` / `AIC` values and bit-identical panel PNGs against the original script in the same environment.

## What changed and why (every changed line)

1. **Added `simulate_y_values(li, t_vals)`.** The original integrated the ODE and computed the observed signal `S/(D+A+F1+F2)` in two separate places: inside `fit_one`'s `SSR_code` and inside `true_y_values`. The math was identical (same initial state `[donor, 1-donor, 0, 0, 0]`, same `mxstep=5000`, same signal formula). One shared function now does that integration, so a future fix only needs to happen in one place. This is Gerg's "one function that can be used multiple times" point.
2. **`true_y_values` now calls `simulate_y_values(li.x, t_vals)`.** Keeps its original optimizer-result interface (`li.x`); behavior identical.
3. **`SSR_code` now calls `simulate_y_values(li, t_vals)`** instead of its own inline copy of the same integration. Guard clauses and the mean-squared-error return are unchanged.
4. **Removed the `check_chi` block** (`check_chi = check.copy()` and the four `0.0001` assignments). It was written but never read or returned - dead code.
5. **Removed the duplicate `m = 4`.** `m` was set to 4 twice with no change in between.
6. **Removed 7 functions that were defined but never called anywhere:** `chi_squared`, `plot_and_save`, `panel_name_from_file`, `auto_segment_and_invert`, `fig_sort_key`, `fig_panel_key`, `fig_condition_label`. (Note: `main()` uses its own inline panel-grouping logic, not these helpers.)
7. **Removed 3 unused imports:** `from tabnanny import check` (shadowed by the local `check` variable in `fit_one`, never used at module level), `from scipy.optimize import minimize` (the code calls `scipy.optimize.minimize`, never bare `minimize`), `import math` (no `math.` usage anywhere).

## What was deliberately NOT changed

- The physics in `ode_list`, the fit initial guess `[0.5, 0.001, 0.02, 0.02]`, Nelder-Mead settings (`maxiter=5000`), and all SSR/sigma/chi^2/AIC math.
- All 9 print statements, including `SCRIPT STARTED` first - the numbers still print.
- CSV reading, preprocessing, panel grouping (including the intentional skip of `FigB_*` as a malformed duplicate), labels and plot output.
- Existing fit quality issues (some curves fit poorly) - out of scope for a cleanup.

## Function summary

| Function | What it does |
| --- | --- |
| `ode_list` | The Erlang two-step fusion ODE system: returns dD/dt, dA/dt, dF1/dt, dF2/dt, dS/dt with death/loss term. The physics. |
| `simulate_y_values` | Integrates the ODE for a parameter vector and returns the observable signal S/(D+A+F1+F2). Shared by the fitter and the curve generator. |
| `true_y_values` | Takes an optimizer result and returns the fitted curve (thin wrapper over `simulate_y_values`). |
| `aic` | Akaike Information Criterion from SSR and parameter count. |
| `read_csv_two_cols` | Reads a two-column (time, cell index) CSV into a float array, skipping bad rows. |
| `preprocess` | Sorts by time and applies optional max-time cut, baseline subtraction, 0-1 normalization and inversion. |
| `fit_one` | Fits one curve: Nelder-Mead minimization of mean squared error, then SSR, sigma, chi^2 and AIC. |
| `main` | CLI: groups CSVs into figure panels, fits each curve, prints the numbers, saves one PNG per panel. |

## Verification

- AST comparison vs the original: `ode_list`, `read_csv_two_cols`, `preprocess`, `aic` and `main` are structurally identical; only `fit_one` and `true_y_values` changed, exactly as listed above.
- Full rerun on `Fig.csvs`: 22 accepted curves, all printed metrics identical to the original; all 7 panel PNGs bit-identical (md5).
- `figure_digitized_csvs/` was not part of this regression (unchanged input path, untested here). Equivalence is shown for the tested inputs; this cleanup is not a scientific validation of the model.
