import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

# --- ODE: Erlang 2-step fusion (Christmas-style) ---
# state = [D, A, F1, F2, S]
def ode_list(state, t, gamma, k):
    D, A, F1, F2, S = state
    DA = gamma * D * A
    dDdt  = -DA
    dAdt  = -DA
    dF1dt = 2.0 * DA - k * F1
    dF2dt = k * F1 - k * F2
    dSdt  = k * F2
    return [dDdt, dAdt, dF1dt, dF2dt, dSdt]

def simulate_fraction_syncytia(t, p, gamma, k):
    # initial conditions (Christmas)
    y0 = [p, 1.0 - p, 0.0, 0.0, 0.0]
    sol = odeint(ode_list, y0, t, args=(gamma, k), mxstep=5000)
    D, A, F1, F2, S = sol[:,0], sol[:,1], sol[:,2], sol[:,3], sol[:,4]
    denom = (D + A + F1 + F2)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    return S / denom

# --- Objective: Negative log-likelihood with sigma ---
def nll(theta, t, y):
    # theta = [p, gamma, k, log_sigma]
    p, gamma, k, log_sigma = theta
    sigma = np.exp(log_sigma)

    # guards (keeps Nelder-Mead sane)
    if not (0.0 < p < 1.0): return 1e18
    if gamma <= 0.0 or k <= 0.0: return 1e18
    if sigma <= 0.0 or not np.isfinite(sigma): return 1e18

    yhat = simulate_fraction_syncytia(t, p, gamma, k)
    resid = (y - yhat)
    ssr = float(np.sum(resid * resid))

    n = len(y)
    return n * np.log(sigma) + 0.5 * ssr / (sigma * sigma)

def fit_curve(t, y):
    # good defaults for 1:1 overlay experiments:
    p0 = 0.5
    gamma0 = 0.01
    k0 = 0.5
    # sigma init: rough scale of residuals
    sigma0 = max(np.std(y) * 0.2, 1e-3)

    theta0 = np.array([p0, gamma0, k0, np.log(sigma0)], float)

    res = minimize(
        nll, theta0,
        args=(t, y),
        method="Nelder-Mead",
        options={"maxiter": 5000}
    )

    p, gamma, k, log_sigma = res.x
    return {
        "p": float(p),
        "gamma": float(gamma),
        "k": float(k),
        "sigma": float(np.exp(log_sigma)),
        "nll": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
    }
