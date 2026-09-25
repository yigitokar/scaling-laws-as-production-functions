"""Brute-force check of the binding data-cap case of Proposition A9 (family token budgets), part (vi).

A family of K members shares one token budget D, which is capped at Dbar (binding). Member i has value of loss
Pi_i and lifetime serving demand T_i; its objective is -Pi_i*L(N_i, D) - p*2*N_i*T_i (serving cost at price p per
serving FLOP, in training-FLOP units), and training costs 6*N_i*D. Sizes are chosen at interior optima given D = Dbar.
Checks:
  (a) sum_i pi_i (1 + m_N,i) = (1 + mu) * W_H >= W_H, with mu = xi_D / sum_j X_T,j the cap's normalized multiplier;
  (b) for members whose own marginal value of data exceeds its cost (rho_i >= 1), w_i - 1 <= m_N,i;
  (c) with a binding size cap on a member as well, the member bound can fail.
Technology: Chinchilla form with the parameters of Besiroglu et al. (2024). Deterministic; runs in < 1 s.
"""
import numpy as np
from scipy.optimize import minimize_scalar

E, A, B, al, be = 1.8172, 482.01, 2085.43, 0.3478, 0.3658
p = 1.0


def parts(N, D):
    u, v = A * N ** -al, B * D ** -be
    return u, v, E + u + v


def solve_member(Pi, T, D, nmax=None):
    """Optimal log N for one member given D (interior unless nmax binds)."""
    f = lambda n: Pi * parts(np.exp(n), D)[2] + 6 * np.exp(n) * D + p * 2 * np.exp(n) * T
    hi = 30.0 if nmax is None else nmax
    r = minimize_scalar(f, bounds=(5.0, hi), method="bounded", options={"xatol": 1e-12})
    return r.x


def check(members, Dbar, caps=None):
    caps = caps or [None] * len(members)
    rows = []
    for (Pi, T), cap in zip(members, caps):
        n = solve_member(Pi, T, Dbar, cap)
        N = np.exp(n)
        u, v, L = parts(N, Dbar)
        epsN, epsD = al * u / (u + v), be * v / (u + v)
        w = epsN / epsD
        XT = 6 * N * Dbar
        mN = p * 2 * N * T / XT            # = p*T/(3D)
        Lam = Pi * (u + v)                 # -(dV/dL)(L-E)
        rho = Lam * epsD / XT              # member's marginal value of data per unit of its training cost
        rows.append(dict(N=N, w=w, XT=XT, mN=mN, rho=rho, binding=(cap is not None and abs(n - cap) < 1e-6)))
    XT = np.array([r["XT"] for r in rows]); w = np.array([r["w"] for r in rows]); mN = np.array([r["mN"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    om = XT / XT.sum()
    WH = 1 / np.sum(om / w)
    pi = (om / w) / np.sum(om / w)
    # family marginal value of data per log point minus its cost, normalized: mu = sum om*rho - 1 (m_D = 0)
    mu = np.sum(om * rho) - 1
    lhs = np.sum(pi * (1 + mN))
    return rows, WH, pi, mu, lhs


def family_optimum(members):
    """Unconstrained family optimum of the common token budget (sizes re-optimized at each D)."""
    def total(d):
        D = np.exp(d)
        tot = 0.0
        for Pi, T in members:
            N = np.exp(solve_member(Pi, T, D))
            tot += Pi * parts(N, D)[2] + 6 * N * D + p * 2 * N * T
        return tot
    r = minimize_scalar(total, bounds=(20.0, 40.0), method="bounded", options={"xatol": 1e-10})
    return np.exp(r.x)


if __name__ == "__main__":
    members = [(2.0e21, 3.0e12), (6.0e21, 8.0e12), (4.0e22, 2.0e13)]
    Dstar = family_optimum(members)
    rows, WH, pi, mu, lhs = check(members, Dstar)
    print("Unconstrained family optimum D* = %.4e: mu = %.2e (family condition (iii): sum pi(1+mN) = W_H: %.6f vs %.6f)"
          % (Dstar, mu, lhs, WH))
    assert abs(mu) < 1e-5 and abs(lhs - WH) < 1e-5 * WH
    out = []
    for frac in [0.3, 0.5, 0.8]:
        Dbar = frac * Dstar
        rows, WH, pi, mu, lhs = check(members, Dbar)
        ok_a = abs(lhs - (1 + mu) * WH) < 1e-6 * WH and mu > 0 and lhs >= WH
        ok_b = all((r["w"] - 1 <= r["mN"] + 1e-9) for r in rows if r["rho"] >= 1)
        nb = sum(r["rho"] >= 1 for r in rows)
        out.append((frac, mu, lhs, (1 + mu) * WH, WH, ok_a, nb, ok_b,
                    [round(r["rho"], 3) for r in rows], [round(r["w"] - 1, 3) for r in rows],
                    [round(r["mN"], 3) for r in rows]))
    for o in out:
        print("Dbar=%.1f D*: mu=%.4f  sum pi(1+mN)=%.6f  (1+mu)W_H=%.6f  W_H=%.4f  (a) %s  members with rho>=1: %d  (b) %s"
              "  rho=%s  w-1=%s  mN=%s" % o)
    # (c) a binding size cap on the smallest member at Dbar = 0.5 D*
    Dbar = 0.5 * Dstar
    rows0, *_ = check(members, Dbar)
    cap_n = np.log(rows0[0]["N"]) - 0.7
    rows, WH, pi, mu, lhs = check(members, Dbar, caps=[cap_n, None, None])
    r0 = rows[0]
    print("(c) size cap on member 1 at 0.5 D*: rho=%.3f, w-1=%.3f, mN=%.3f; lower bound fails: %s"
          % (r0["rho"], r0["w"] - 1, r0["mN"], r0["w"] - 1 > r0["mN"]))
    assert all(o[5] and o[7] and o[6] > 0 for o in out)
    print("ALL CHECKS PASSED")
