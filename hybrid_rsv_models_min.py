import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus_pos(x):
    return F.softplus(x) + 1e-8

def read_csv(path):
    df = pd.read_csv(path, header=None, sep=",", engine="python")
    df = df.iloc[:, :2]
    df.columns = ["x", "y"]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()
    t = df["x"].to_numpy()
    y = df["y"].to_numpy()
    order = np.argsort(t)
    return t[order], y[order]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class HybridRSV(nn.Module):
    def __init__(self, mode="baseline", steps=6):
        super().__init__()
        self.mode = mode
        self.steps = steps

        self.raw_gamma  = nn.Parameter(torch.tensor(-2.0))
        self.raw_gamma2 = nn.Parameter(torch.tensor(-3.0))
        self.raw_k      = nn.Parameter(torch.tensor(-1.0))
        self.raw_delta  = nn.Parameter(torch.tensor(-3.0))
        self.raw_rA     = nn.Parameter(torch.tensor(-4.0))
        self.raw_K      = nn.Parameter(torch.tensor(3.0))

        self.raw_D0 = nn.Parameter(torch.tensor(2.0))
        self.raw_A0 = nn.Parameter(torch.tensor(2.0))
        self.raw_S0 = nn.Parameter(torch.tensor(-6.0))

        self.raw_aD = nn.Parameter(torch.tensor(0.0))
        self.raw_aA = nn.Parameter(torch.tensor(0.0))
        self.raw_aF = nn.Parameter(torch.tensor(0.0))
        self.raw_aS = nn.Parameter(torch.tensor(0.0))
        self.raw_offset = nn.Parameter(torch.tensor(0.0))

        self.corr_nn = MLP(7)
        self.obs_nn = MLP(6)

    def get_params(self):
        gamma  = softplus_pos(self.raw_gamma)
        gamma2 = softplus_pos(self.raw_gamma2)
        k      = softplus_pos(self.raw_k)
        delta  = softplus_pos(self.raw_delta)
        rA     = softplus_pos(self.raw_rA)
        K      = softplus_pos(self.raw_K)

        if self.mode == "stage1":
            gamma2 = torch.zeros_like(gamma2)
        if self.mode == "stage2":
            gamma = torch.zeros_like(gamma)

        return gamma, gamma2, k, delta, rA, K

    def initial_state(self):
        D0 = softplus_pos(self.raw_D0)
        A0 = softplus_pos(self.raw_A0)
        S0 = softplus_pos(self.raw_S0) if self.mode != "stage1" else torch.zeros_like(D0)
        if self.mode == "stage2":
            D0 = torch.zeros_like(D0)
        return torch.stack([D0, A0, torch.zeros_like(D0), torch.zeros_like(D0), S0, torch.zeros_like(D0)])

    def rhs(self, x, t):
        D, A, F1, F2, S, X = x
        gamma, gamma2, k, delta, rA, K = self.get_params()

        DA = gamma * D * A
        SA = gamma2 * S * A

        if self.mode in ["ude", "full"]:
            x_norm = x / (x.detach().abs().mean() + 1e-8)
            t_norm = t / (t.detach().abs() + 1.0)
            c = self.corr_nn(torch.cat([x_norm, t_norm.view(1)]).view(1, -1)).view(())
            c = torch.clamp(c, -2.0, 2.0)
            DA = (gamma * torch.exp(c)) * D * A

        growthA = rA * A * (1 - A / (K + 1e-8))

        dD  = -DA - delta * D
        dA  = -DA - SA + growthA - delta * A
        dF1 = 2*(DA + SA) - k*F1 - delta*F1
        dF2 = k*F1 - k*F2 - delta*F2
        dS  = k*F2 - delta*S
        dX  = delta*(D + A + F1 + F2 + S)

        return torch.stack([dD, dA, dF1, dF2, dS, dX])

    def integrate(self, t):
        x = self.initial_state()
        out = [x]
        for i in range(1, len(t)):
            dt = (t[i] - t[i-1]) / self.steps
            xi = out[-1]
            ti = t[i-1]
            for _ in range(self.steps):
                k1 = self.rhs(xi, ti)
                k2 = self.rhs(xi + 0.5*dt*k1, ti + 0.5*dt)
                k3 = self.rhs(xi + 0.5*dt*k2, ti + 0.5*dt)
                k4 = self.rhs(xi + dt*k3, ti + dt)
                xi = xi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                xi = torch.clamp(xi, min=0.0)
                ti = ti + dt
            out.append(xi)
        return torch.stack(out)

    def observe(self, states):
        D, A, F1, F2, S, X = states.T
        aD = softplus_pos(self.raw_aD)
        aA = softplus_pos(self.raw_aA)
        aF = softplus_pos(self.raw_aF)
        aS = softplus_pos(self.raw_aS)
        y = aD*D + aA*A + aF*(F1+F2) + aS*S + self.raw_offset
        if self.mode in ["obs", "full"]:
            y = y + 0.1*self.obs_nn(states).squeeze(-1)
        return y

def fit(model, t_np, y_np, epochs=2000, lr=3e-3):
    device = torch.device("cpu")
    model.to(device)

    t = torch.tensor(t_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_n = (y - y_mean)/y_std

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        states = model.integrate(t)
        yhat = model.observe(states)
        yhat_n = (yhat - y_mean)/y_std
        loss = torch.mean((yhat_n - y_n)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mode", default="baseline",
                        choices=["baseline","stage1","stage2","ude","obs","full"])
    parser.add_argument("--epochs", type=int, default=2500)
    args = parser.parse_args()

    t, y = read_csv(args.csv)
    model = HybridRSV(mode=args.mode)
    model = fit(model, t, y, epochs=args.epochs)

    tt = torch.tensor(t, dtype=torch.float32)
    with torch.no_grad():
        states = model.integrate(tt)
        yhat = model.observe(states).numpy()

    pd.DataFrame({"x": t, "y": y, "yhat": yhat}).to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()