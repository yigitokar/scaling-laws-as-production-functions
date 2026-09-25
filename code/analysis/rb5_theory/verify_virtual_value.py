"""Brute-force verification of the virtual value of compactness (round 3, decision D-1; fix list P1(g)-(i), P11(e)).

The value of compactness is now the virtual value v_N = m_N + nu_N: the objective's own marginal value of a smaller
model at given loss, m_N, plus the shadow value nu_N of any binding constraint on size alone (memory tier, latency
objective, menu of sizes; a minimum-size floor enters with a negative sign), both per unit of training expenditure.

Part A (Proposition A-wedge, one model). The developer maximizes
    Pi(n, d) = -Pv * L(n, d) - lam * rho_s * 2 N T - q_T(N) 6 N D - q_D D,     q_T(N) = (N / N0)^delta,
with the paper's reference technology (Chinchilla, kappa free; data/processed/ra2_wedge/headline.json), so the checks
also exercise kappa-invariance. For each constraint the optimum is found by direct maximization, the shadow value is
computed twice (as dPi/dn at the constrained point, and independently as the derivative of the constrained maximum
value with respect to the constraint level: the envelope theorem), and the wedge read off the technology is compared
with (1 + delta + v_N) / (1 + m_D [+ mu]).

Part B (Proposition A-family, a family on one token budget). The example of code/paper/verify_family_cap.py (three
members, Besiroglu et al. parameters) is extended with binding size caps and floors on members, a data cost, and a
binding cap on tokens. Checks:
    (iii)  sizes interior or at binding size constraints, D chosen:  sum_i pi_i (1 + v_N,i) = (1 + m_D) W_H;
    (iv)   binding floors have xi_i <= 0 (nu_N,i <= 0);
    (vi)   D at a binding cap:  sum_i pi_i (1 + v_N,i) = (1 + m_D + mu) W_H,  mu = xi_D / sum_j X_T,j  (xi_D from the
           envelope of the family's value in ln Dbar); with m_D = 0, member i's w_i - 1 <= v_N,i  iff  rho_i >= 1;
    remark numbers: member inversions T_hat_i / T_i at the family optimum, and the number of members that violate the
           member bound at caps of 30, 50 and 80 percent of the family optimum.
Deterministic; CPU only; runs in a few seconds. Usage: .venv/bin/python code/analysis/rb5_theory/verify_virtual_value.py
Writes code/analysis/rb5_theory/results/virtual_value.csv (and _detail.csv) and prints PASS/FAIL per check (exit code 1 on any FAIL).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = []


def record(cid, claim, ok, metric):
    RESULTS.append(dict(check=cid, claim=claim, passed=bool(ok), metric=metric))
    print(("[PASS] " if ok else "[FAIL] ") + f"{cid:<14} {claim}  ({metric})")


# ---------------------------------------------------------------------------------------------- technology
class Tech:
    def __init__(self, E, A, B, al, be, kappa=1.0):
        self.E, self.A, self.B, self.al, self.be, self.kappa = E, A, B, al, be, kappa

    def uv(self, n, d):
        return self.A * np.exp(-self.al * n), self.B * np.exp(-self.be * d)

    def L(self, n, d):
        u, v = self.uv(n, d)
        return self.E + (u + v) ** self.kappa

    def eps(self, n, d):
        u, v = self.uv(n, d)
        return self.kappa * self.al * u / (u + v), self.kappa * self.be * v / (u + v)

    def w(self, n, d):
        eN, eD = self.eps(n, d)
        return eN / eD


def ref_tech():
    p = json.load(open(os.path.join(ROOT, "data", "processed", "ra2_wedge", "headline.json")))["chin_q_point"]
    return Tech(float(p["E"]), float(np.exp(p["lnA"])), float(np.exp(p["lnB"])), float(p["alpha"]), float(p["beta"]),
                float(p["q"]))


BES = Tech(1.8172, 482.01, 2085.43, 0.3478, 0.3658, 1.0)


def fd(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------------------------------------- Part A
class Developer:
    """Pi(n, d) = -Pv L - lam*rho_s*2NT - (N/N0)^delta 6ND - qD D (all in training-FLOP units)."""

    def __init__(self, tech, Pv, T, lam=1.0, rho_s=1.0, delta=0.0, N0=1e9, qD=0.0):
        self.t, self.Pv, self.T, self.lam, self.rho_s, self.delta, self.N0, self.qD = tech, Pv, T, lam, rho_s, delta, N0, qD

    def XT(self, n, d):
        return (np.exp(n) / self.N0) ** self.delta * 6 * np.exp(n + d)

    def Pi(self, n, d):
        return (-self.Pv * self.t.L(n, d) - self.lam * self.rho_s * 2 * np.exp(n) * self.T - self.XT(n, d)
                - self.qD * np.exp(d))

    def mN(self, n, d):      # -(dV/dln N)_L / X_T with V = -Pv L - lam rho_s 2 N T
        return self.lam * self.rho_s * 2 * np.exp(n) * self.T / self.XT(n, d)

    def mD(self, n, d):
        return self.qD * np.exp(d) / self.XT(n, d)

    # maximizers ------------------------------------------------------------------------------------------------
    def best_d(self, n, dmax=None):
        hi = 45.0 if dmax is None else dmax
        r = minimize_scalar(lambda d: -self.Pi(n, d), bounds=(15.0, hi), method="bounded", options={"xatol": 1e-11})
        return r.x

    def best_n(self, d, nlo=None, nhi=None):
        lo = 10.0 if nlo is None else nlo
        hi = 35.0 if nhi is None else nhi
        r = minimize_scalar(lambda n: -self.Pi(n, d), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
        return r.x

    def optimum(self):
        # nested: for each n the best d; then the best n
        r = minimize_scalar(lambda n: -self.Pi(n, self.best_d(n)), bounds=(12.0, 32.0), method="bounded",
                            options={"xatol": 1e-11})
        n = r.x
        return n, self.best_d(n)


def part_a():
    ref = ref_tech()
    rows = []
    specs = [
        ("delta=mD=0", dict(delta=0.0, qD=0.0)),
        ("delta=0.08,qD>0", dict(delta=0.08, qD=2.0e9)),
    ]
    for label, kw in specs:
        dev = Developer(ref, Pv=4.0e22, T=3.0e12, lam=0.7, rho_s=1.5, **kw)
        n0, d0 = dev.optimum()
        w0 = ref.w(n0, d0)
        rhs0 = (1 + dev.delta + dev.mN(n0, d0)) / (1 + dev.mD(n0, d0))
        record("A.interior." + label, "interior optimum: w = (1+delta+m_N)/(1+m_D), v_N = m_N",
               abs(w0 / rhs0 - 1) < 1e-6, f"w={w0:.6f} formula={rhs0:.6f}")
        rows.append(dict(part="A", case="interior", spec=label, w=w0, formula=rhs0))

        # value of the problem with n fixed at nbar (d re-optimized)
        Vn = lambda nb: dev.Pi(nb, dev.best_d(nb))

        # (b) tier/memory cap N <= Nbar, (floor) N >= Nmin, and a menu of sizes
        for case, nb in (("size cap", n0 - 0.5), ("size floor", n0 + 0.5)):
            d1 = dev.best_d(nb)
            XT = dev.XT(nb, d1)
            xi_kkt = fd(lambda n: dev.Pi(n, d1), nb)             # multiplier on g = n - nbar (cap) / sign flip (floor)
            xi_env = fd(Vn, nb, 1e-5)                             # envelope: value of relaxing the constraint
            nuN = xi_kkt / XT                                    # constraints on size alone; a floor gives nu_N < 0
            vN = dev.mN(nb, d1) + nuN
            w1 = ref.w(nb, d1)
            rhs = (1 + dev.delta + vN) / (1 + dev.mD(nb, d1))
            ok = abs(w1 / rhs - 1) < 1e-6 and abs(xi_env / xi_kkt - 1) < 1e-4 and (nuN > 0 if case == "size cap" else nuN < 0)
            extra = ""
            if dev.delta == 0 and dev.qD == 0 and case == "size cap":
                ok = ok and (w1 - 1 >= dev.mN(nb, d1)) and abs((w1 - 1) - vN) < 1e-6 * vN
                extra = f"; w-1={w1 - 1:.4f} = v_N={vN:.4f} >= m_N={dev.mN(nb, d1):.4f}"
            record(f"A.{case.replace(' ', '')}." + label,
                   f"{case}: w = (1+delta+v_N)/(1+m_D), nu_N = xi/X_T (KKT = envelope), sign of nu_N",
                   ok, f"w={w1:.6f} formula={rhs:.6f} nu_N={nuN:.4f} xi_env/xi_kkt={xi_env / xi_kkt:.6f}{extra}")
            rows.append(dict(part="A", case=case, spec=label, w=w1, formula=rhs, nu_N=nuN, m_N=dev.mN(nb, d1),
                             env_over_kkt=xi_env / xi_kkt))

        # (c) latency objective Z(N) = N^ell <= Zbar, ell = 0.6: nu_N = ell * xi_Z / X_T, xi_Z from the envelope in ln Zbar
        ell = 0.6
        lnZbar = ell * (n0 - 0.4)
        VZ = lambda lz: Vn(lz / ell)
        nb = lnZbar / ell
        d1 = dev.best_d(nb)
        xi_Z = fd(VZ, lnZbar, 1e-5)
        nuN = ell * xi_Z / dev.XT(nb, d1)
        w1 = ref.w(nb, d1)
        rhs = (1 + dev.delta + dev.mN(nb, d1) + nuN) / (1 + dev.mD(nb, d1))
        record("A.latency." + label, "latency objective: case (b) with nu_N = ell*xi_Z/X_T", abs(w1 / rhs - 1) < 1e-5,
               f"w={w1:.6f} formula={rhs:.6f} nu_N={nuN:.4f}")
        rows.append(dict(part="A", case="latency", spec=label, w=w1, formula=rhs, nu_N=nuN))

        # size menu: the developer picks the best of a few sizes; the menu's shadow value is dPi/dn at the chosen size
        for mlabel, menu in (("menu binds as a cap", [n0 - 0.9, n0 - 0.3, n0 + 1.4]),
                             ("menu binds as a floor", [n0 - 1.6, n0 + 0.25, n0 + 1.2])):
            vals = [Vn(m) for m in menu]
            nb = menu[int(np.argmax(vals))]
            d1 = dev.best_d(nb)
            nuN = fd(lambda n: dev.Pi(n, d1), nb) / dev.XT(nb, d1)
            w1 = ref.w(nb, d1)
            rhs = (1 + dev.delta + dev.mN(nb, d1) + nuN) / (1 + dev.mD(nb, d1))
            ok = abs(w1 / rhs - 1) < 1e-6 and ((nuN > 0) if "cap" in mlabel else (nuN < 0))
            record(f"A.menu{'cap' if 'cap' in mlabel else 'floor'}." + label,
                   f"size menu ({mlabel}): w-1 identifies v_N whatever the sign of the menu's shadow value",
                   ok, f"w={w1:.6f} formula={rhs:.6f} nu_N={nuN:.4f}")
            rows.append(dict(part="A", case=mlabel, spec=label, w=w1, formula=rhs, nu_N=nuN))

        # (d) token cap D <= Dbar: mu = xi_D / X_T; with delta = m_D = 0, w-1 <= v_N (= m_N, no size constraint)
        db = d0 - 0.5
        n1 = dev.best_n(db)
        Vd = lambda dbb: dev.Pi(dev.best_n(dbb), dbb)
        mu_kkt = fd(lambda d: dev.Pi(n1, d), db) / dev.XT(n1, db)
        mu_env = fd(Vd, db, 1e-5) / dev.XT(n1, db)
        w1 = ref.w(n1, db)
        rhs = (1 + dev.delta + dev.mN(n1, db)) / (1 + dev.mD(n1, db) + mu_kkt)
        ok = abs(w1 / rhs - 1) < 1e-6 and abs(mu_env / mu_kkt - 1) < 1e-4 and mu_kkt > 0
        if dev.delta == 0 and dev.qD == 0:
            ok = ok and (w1 - 1 <= dev.mN(n1, db))
        record("A.tokencap." + label, "token cap: w = (1+delta+v_N)/(1+m_D+mu); w-1 bounds v_N from below", ok,
               f"w={w1:.6f} formula={rhs:.6f} mu={mu_kkt:.4f} env/kkt={mu_env / mu_kkt:.6f} v_N={dev.mN(n1, db):.4f}")
        rows.append(dict(part="A", case="token cap", spec=label, w=w1, formula=rhs, mu=mu_kkt))

        # size cap and token cap together
        nb, db = n0 - 0.3, d0 - 0.4
        nuN = fd(lambda n: dev.Pi(n, db), nb) / dev.XT(nb, db)
        mu = fd(lambda d: dev.Pi(nb, d), db) / dev.XT(nb, db)
        w1 = ref.w(nb, db)
        vN = dev.mN(nb, db) + nuN
        rhs = (1 + dev.delta + vN) / (1 + dev.mD(nb, db) + mu)
        ok = abs(w1 / rhs - 1) < 1e-6 and nuN > 0 and mu > 0
        if dev.delta == 0 and dev.qD == 0:
            ok = ok and (w1 - 1 <= vN)
        record("A.bothcaps." + label, "size and token caps both bind: w = (1+delta+v_N)/(1+m_D+mu); w-1 <= v_N", ok,
               f"w={w1:.6f} formula={rhs:.6f} nu_N={nuN:.4f} mu={mu:.4f}")
        rows.append(dict(part="A", case="size and token caps", spec=label, w=w1, formula=rhs, nu_N=nuN, mu=mu))
    return rows


# ---------------------------------------------------------------------------------------------- Part B
class Family:
    """Members (Pi_i, T_i) on one token budget D; objective sum_i[-Pi_i L(N_i, D) - p 2 N_i T_i] - sum_i 6 N_i D - cD D."""

    def __init__(self, tech, members, p=1.0, cD=0.0):
        self.t, self.m, self.p, self.cD = tech, members, p, cD

    def member_obj(self, i, n, d):
        Pi, T = self.m[i]
        return -Pi * self.t.L(n, d) - self.p * 2 * np.exp(n) * T - 6 * np.exp(n + d)

    def best_n(self, i, d, lo=None, hi=None):
        lo = 5.0 if lo is None else lo
        hi = 32.0 if hi is None else hi
        r = minimize_scalar(lambda n: -self.member_obj(i, n, d), bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-12})
        return r.x

    def sizes(self, d, caps=None, floors=None):
        caps = caps or [None] * len(self.m)
        floors = floors or [None] * len(self.m)
        return np.array([self.best_n(i, d, floors[i], caps[i]) for i in range(len(self.m))])

    def value(self, d, caps=None, floors=None):
        nn = self.sizes(d, caps, floors)
        return sum(self.member_obj(i, nn[i], d) for i in range(len(self.m))) - self.cD * np.exp(d)

    def optimum_d(self, caps=None, floors=None):
        r = minimize_scalar(lambda d: -self.value(d, caps, floors), bounds=(20.0, 40.0), method="bounded",
                            options={"xatol": 1e-11})
        return r.x

    def stats(self, d, caps=None, floors=None):
        nn = self.sizes(d, caps, floors)
        N, D = np.exp(nn), np.exp(d)
        XT = 6 * N * D
        w = np.array([self.t.w(nn[i], d) for i in range(len(nn))])
        eD = np.array([self.t.eps(nn[i], d)[1] for i in range(len(nn))])
        u_v = np.array([sum(self.t.uv(nn[i], d)) for i in range(len(nn))])
        Pi = np.array([m[0] for m in self.m])
        T = np.array([m[1] for m in self.m])
        # Lambda_i = -(dV_i/dL)(L_i - E) = Pi_i (L_i - E); L - E = (u+v)^kappa
        Lam = Pi * u_v ** self.t.kappa
        rho = Lam * eD / XT
        mN = self.p * 2 * N * T / XT
        xi = np.array([fd(lambda n: self.member_obj(i, n, d), nn[i]) for i in range(len(nn))])   # 0 if interior
        vN = mN + xi / XT
        om = XT / XT.sum()
        WH = 1 / np.sum(om / w)
        pi = (om / w) / np.sum(om / w)
        mD = self.cD * D / XT.sum()
        return dict(nn=nn, N=N, D=D, XT=XT, w=w, rho=rho, mN=mN, xi=xi, vN=vN, om=om, WH=WH, pi=pi, mD=mD, T=T)


def part_b():
    rows = []
    members = [(2.0e21, 3.0e12), (6.0e21, 8.0e12), (4.0e22, 2.0e13)]   # as in code/paper/verify_family_cap.py

    # ------------------------------------------------------------------ remark numbers (Besiroglu, no data cost)
    fam = Family(BES, members)
    d_star = fam.optimum_d()
    s = fam.stats(d_star)
    inv = (s["w"] - 1) / s["mN"]
    lhs, rhs = np.sum(s["pi"] * (1 + s["vN"])), (1 + s["mD"]) * s["WH"]
    record("B.remark.Dstar", "family optimum (sizes interior): sum pi(1+v_N) = W_H; member inversions T_hat/T",
           abs(lhs / rhs - 1) < 1e-6 and np.max(np.abs(s["xi"] / s["XT"])) < 1e-6,
           "inversions " + ", ".join(f"{x:.3f}" for x in inv) + f"; lhs={lhs:.6f} rhs={rhs:.6f}")
    rows.append(dict(part="B", case="remark: family optimum", inversions=";".join(f"{x:.4f}" for x in inv), lhs=lhs, rhs=rhs))
    viol = {}
    for frac in (0.3, 0.5, 0.8):
        db = d_star + np.log(frac)
        s = fam.stats(db)
        mu = fd(lambda d: fam.value(d), db, 1e-5) / s["XT"].sum()
        lhs, rhs = np.sum(s["pi"] * (1 + s["vN"])), (1 + s["mD"] + mu) * s["WH"]
        nviol = int(np.sum(s["w"] - 1 > s["vN"] + 1e-9))
        eq = np.all((s["w"] - 1 <= s["vN"] + 1e-9) == (s["rho"] >= 1 - 1e-9))
        viol[frac] = nviol
        record(f"B.cap{int(frac * 100)}", f"token cap at {int(frac * 100)}% of D*: sum pi(1+v_N) = (1+mu)W_H; "
               "member bound w_i-1 <= v_N,i iff rho_i >= 1", abs(lhs / rhs - 1) < 1e-5 and mu > 0 and eq,
               f"mu={mu:.4f} lhs={lhs:.6f} rhs={rhs:.6f} rho={np.round(s['rho'], 3).tolist()} "
               f"w-1={np.round(s['w'] - 1, 3).tolist()} v_N={np.round(s['vN'], 3).tolist()} violators={nviol}")
        rows.append(dict(part="B", case=f"token cap {int(frac * 100)}%", lhs=lhs, rhs=rhs, mu=mu, violators=nviol,
                         rho=";".join(f"{x:.4f}" for x in s["rho"])))
    record("B.remark.viol", "one of three members violates the member bound at 30 and 50 percent, two at 80 percent",
           viol == {0.3: 1, 0.5: 1, 0.8: 2}, str(viol))

    # ------------------------------------------------------------------ (iii)/(iv): size caps and floors, D chosen
    for tech_label, tech in (("Besiroglu", BES), ("reference (kappa free)", ref_tech())):
        for cD in (0.0, 3.0e9):
            fam = Family(tech, members, cD=cD)
            d0 = fam.optimum_d()
            n_free = fam.sizes(d0)
            caps = [n_free[0] - 0.6, None, None]            # member 1 capped (tier)
            floors = [None, None, n_free[2] + 0.4]          # member 3 at a floor (size target)
            dstar = fam.optimum_d(caps, floors)
            s = fam.stats(dstar, caps, floors)
            lhs, rhs = np.sum(s["pi"] * (1 + s["vN"])), (1 + s["mD"]) * s["WH"]
            # the D first-order condition: sum om rho = 1 + m_D
            foc = np.sum(s["om"] * s["rho"]) - (1 + s["mD"])
            ok = (abs(lhs / rhs - 1) < 1e-5 and abs(foc) < 1e-5 and s["xi"][0] > 0 and s["xi"][2] < 0
                  and abs(s["xi"][1] / s["XT"][1]) < 1e-6)
            # with m_D = 0 the family share identifies the pi-weighted v_N; the old m_N statement fails
            extra = ""
            if cD == 0:
                pv = np.sum(s["pi"] * s["vN"])
                pm = np.sum(s["pi"] * s["mN"])
                ok = ok and abs(pv - (s["WH"] - 1)) < 1e-5 * s["WH"]
                extra = f"; sum pi v_N={pv:.5f} = W_H-1={s['WH'] - 1:.5f} (sum pi m_N={pm:.5f})"
            record(f"B.iii.{tech_label[:3]}.cD{int(cD > 0)}",
                   f"{tech_label}, m_D {'> 0' if cD else '= 0'}, D chosen, member 1 capped and member 3 at a floor: "
                   "sum pi(1+v_N) = (1+m_D)W_H; xi_cap > 0 > xi_floor", ok,
                   f"lhs={lhs:.6f} rhs={rhs:.6f} D-FOC resid={foc:.1e} nu_N={np.round(s['xi'] / s['XT'], 4).tolist()}{extra}")
            rows.append(dict(part="B", case=f"(iii) caps+floor, {tech_label}, cD={cD:g}", lhs=lhs, rhs=rhs,
                             nu=";".join(f"{x:.5f}" for x in s["xi"] / s["XT"])))

            # (vi): token budget capped at 60% of the (constrained) optimum, member 1 still capped
            db = dstar + np.log(0.6)
            s = fam.stats(db, caps, floors)
            mu = fd(lambda d: fam.value(d, caps, floors), db, 1e-5) / s["XT"].sum()
            lhs, rhs = np.sum(s["pi"] * (1 + s["vN"])), (1 + s["mD"] + mu) * s["WH"]
            ok = abs(lhs / rhs - 1) < 1e-5 and mu > 0
            eqv = np.all((s["w"] - 1 <= (1 + s["vN"]) / (1 + s["mD"]) - 1 + 1e-9) == (s["rho"] >= 1 + s["mD"] - 1e-9))
            ok = ok and eqv
            record(f"B.vi.{tech_label[:3]}.cD{int(cD > 0)}",
                   f"{tech_label}, token cap binds with member 1 capped and member 3 at a floor: "
                   "sum pi(1+v_N) = (1+m_D+mu)W_H; member w_i <= (1+v_N,i)/(1+m_D) iff rho_i >= 1+m_D", ok,
                   f"mu={mu:.4f} lhs={lhs:.6f} rhs={rhs:.6f} rho={np.round(s['rho'], 3).tolist()}")
            rows.append(dict(part="B", case=f"(vi) token cap 60%, {tech_label}, cD={cD:g}", lhs=lhs, rhs=rhs, mu=mu))

    # ------------------------------------------------------------------ the old example (c) of verify_family_cap.py
    fam = Family(BES, members)
    d_star = fam.optimum_d()
    db = d_star + np.log(0.5)
    s0 = fam.stats(db)
    caps = [s0["nn"][0] - 0.7, None, None]
    s = fam.stats(db, caps)
    record("B.sizecap.member", "cap on tokens and a size cap on member 1: with rho_1 >= 1 its w-1 bounds v_N (not m_N)",
           (s["rho"][0] >= 1) and (s["w"][0] - 1 <= s["vN"][0]) and (s["w"][0] - 1 > s["mN"][0]),
           f"rho_1={s['rho'][0]:.3f} w-1={s['w'][0] - 1:.3f} v_N={s['vN'][0]:.3f} m_N={s['mN'][0]:.3f}")
    rows.append(dict(part="B", case="size cap on member 1 with token cap 50%", rho=f"{s['rho'][0]:.4f}"))
    return rows


if __name__ == "__main__":
    rows = part_a() + part_b()
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "virtual_value.csv")
    pd.DataFrame(RESULTS).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out.replace(".csv", "_detail.csv"), index=False)
    nfail = sum(not r["passed"] for r in RESULTS)
    print(f"{len(RESULTS) - nfail}/{len(RESULTS)} checks passed; wrote {out}")
    sys.exit(1 if nfail else 0)
