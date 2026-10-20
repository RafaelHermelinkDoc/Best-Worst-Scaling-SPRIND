"""
Simulation experiments for the design of a Case 1 best-worst scaling study.

Run from the repository root:
    python -m bws_sim.experiments <exp> [--reps R] [--workers W] [--out DIR]

Assumptions (placeholders, to be replaced by pilot estimates): sequential
best-worst logit, Gumbel errors, preference heterogeneity SD 0.5, true item
utilities evenly spaced, items randomly permuted to design positions per
respondent. Every replication r uses its own fixed seed, so results do not
depend on the number of workers.
"""
import os, sys, json, argparse, itertools
import numpy as np
from multiprocessing import Pool
from scipy import stats
from bws_sim.core import (bibd_13_4_1, bibd_7_4_2, gdd_2_7, simulate, Data, Spec, fit, counts)

BIBD13 = bibd_13_4_1()
B7 = bibd_7_4_2()
GDD14 = gdd_2_7()
ALPHA = 0.05


def welch_p(a, b):
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue


def wald_p(est, se):
    return 2 * stats.norm.sf(abs(est / se))


# ---------------------------------------------------------------- Exp 1
def exp1_rep(args):
    seed, lam1, delta, rand1, N = args
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    beta1 = beta.copy()
    J = 9                                  # item with a true group difference (beta 0.5)
    beta1[J] += delta
    d0 = simulate(BIBD13, np.arange(13), beta, N, rng, lam=1.0, group=0)
    d1 = simulate(BIBD13, np.arange(13), beta1, N, rng, lam=lam1, group=1, random_frac=rand1)
    D = Data.concat([d0, d1])
    _, p0 = counts(d0, 13)
    _, p1 = counts(d1, 13)
    pc = np.array([welch_p(p0[:, j], p1[:, j]) for j in range(13)])
    ref = 6
    # naive: separate beta per group, equal scale assumed
    spn = Spec(13, ref, n_groups=2, sep_beta=True)
    on = fit(D, spn)
    nf = len(spn.free)
    dn = on["theta"][nf:] - on["theta"][:nf]
    sen = np.sqrt(np.diag(on["cov"])[:nf] + np.diag(on["cov"])[nf:] - 2 * np.diag(on["cov"][:nf, nf:]))
    pn = np.full(13, np.nan)
    pn[spn.free] = wald_p(dn, sen)
    # scale-adjusted: common beta + free scale mu + delta_j, one item at a time
    sp0 = Spec(13, ref, n_groups=2, scale=True)
    o0 = fit(D, sp0, robust=False)
    ps = np.full(13, np.nan)
    for j in sp0.free:
        spj = Spec(13, ref, n_groups=2, scale=True, delta_items=[j])
        oj = fit(D, spj, x0=np.r_[o0["theta"], 0.0])
        ps[j] = wald_p(oj["theta"][-1], oj["se"][-1])
    mu = float(np.exp(o0["theta"][-1]))
    return dict(pc=pc.tolist(), pn=pn.tolist(), ps=ps.tolist(), mu=mu)


def exp1(reps):
    out = {}
    scen = [("equal scale", 1.0, 0.0, 0.0), ("lam 0.8", 0.8, 0.0, 0.0), ("lam 0.6", 0.6, 0.0, 0.0),
            ("15% random responders", 1.0, 0.0, 0.15),
            ("Delta 0.5, lam 1.0", 1.0, 0.5, 0.0), ("Delta 0.5, lam 0.8", 0.8, 0.5, 0.0),
            ("Delta 0.5, lam 0.6", 0.6, 0.5, 0.0)]
    N = 150
    with Pool(WORKERS) as pool:
        for name, lam1, delta, rand1 in scen:
            res = pool.map(exp1_rep, [(1000 + r, lam1, delta, rand1, N) for r in range(reps)])
            J = 9
            others = [j for j in range(13) if j not in (6, J)]
            agg = {}
            for key in ("pc", "pn", "ps"):
                P = np.array([x[key] for x in res], float)
                sig = P < ALPHA
                agg[key] = dict(
                    fp_rate_per_item=float(np.nanmean(sig[:, others])),
                    any_fp=float(np.mean(sig[:, others].any(1))),
                    power_J=float(np.mean(sig[:, J])))
            agg["mu_mean"] = float(np.mean([x["mu"] for x in res]))
            out[name] = agg
            print(name, json.dumps(agg), flush=True)
    return out


# ---------------------------------------------------------------- Exp 2
def exp2_rep(args):
    seed, N = args
    rng = np.random.default_rng(seed)
    bc = np.linspace(-0.9, 0.9, 10)
    beta = np.r_[bc, [1.2, 1.0, 0.8], [-0.6, -0.8, -1.0]]   # 0-9 common, 10-12 group-0 only, 13-15 group-1 only
    m0 = np.r_[np.arange(10), [10, 11, 12]]
    m1 = np.r_[np.arange(10), [13, 14, 15]]
    d0 = simulate(BIBD13, m0, beta, N, rng, group=0)
    d1 = simulate(BIBD13, m1, beta, N, rng, group=1)
    D = Data.concat([d0, d1])
    a0, p0 = counts(d0, 16)
    a1, p1 = counts(d1, 16)
    pc = np.array([welch_p(p0[:, j], p1[:, j]) for j in range(10)])
    diffc = (a0 - a1)[:10]
    ref = 4
    sp0 = Spec(16, ref, n_groups=2, scale=True)
    o0 = fit(D, sp0, robust=False)
    ps = np.full(10, np.nan)
    for j in range(10):
        if j == ref:
            continue
        spj = Spec(16, ref, n_groups=2, scale=True, delta_items=[j])
        oj = fit(D, spj, x0=np.r_[o0["theta"], 0.0])
        ps[j] = wald_p(oj["theta"][-1], oj["se"][-1])
    return dict(pc=pc.tolist(), ps=ps.tolist(), diffc=diffc.tolist())


def exp2(reps):
    with Pool(WORKERS) as pool:
        res = pool.map(exp2_rep, [(2000 + r, 150) for r in range(reps)])
    pc = np.array([x["pc"] for x in res])
    ps = np.array([x["ps"] for x in res], float)
    dc = np.array([x["diffc"] for x in res])
    idx = [j for j in range(10) if j != 4]
    out = dict(
        counts_fp_rate=float(np.mean(pc[:, idx] < ALPHA)),
        counts_fp_by_item=np.round(np.mean(pc < ALPHA, 0), 2).tolist(),
        counts_mean_diff_by_item=np.round(dc.mean(0), 3).tolist(),
        model_fp_rate=float(np.nanmean(ps[:, idx] < ALPHA)),
        model_fp_by_item=np.round(np.nanmean(ps < ALPHA, 0), 2).tolist())
    print(json.dumps(out), flush=True)
    return out


# ---------------------------------------------------------------- Exp 3
def exp3_rep(args):
    seed, N = args
    rng = np.random.default_rng(seed)
    beta = np.r_[np.linspace(-1, 1, 7), np.linspace(-0.9, 1.1, 7)]
    # one joint 14-item design
    dj = simulate(GDD14, np.arange(14), beta, N, rng)
    spj = Spec(14, 3)
    oj = fit(dj, spj, robust=False)
    bj = np.zeros(14)
    bj[spj.free] = oj["theta"]
    # two blocks answered by the same respondents (same eta in both blocks)
    rng2 = np.random.default_rng(seed + 77)
    eta = 0.5 * rng2.standard_normal((N, 14))
    ds, da = [], []
    # simulate() draws eta itself; pass person-level beta with eta_sd = 0 instead
    for i in range(N):
        bi = beta + eta[i]
        ds.append(simulate(B7, np.arange(7), bi, 1, rng2, eta_sd=0.0))
        da.append(simulate(B7, np.arange(7, 14), bi, 1, rng2, eta_sd=0.0))
    Ds = Data.concat(ds)
    Da = Data.concat(da)
    os_ = fit(Ds, Spec(14, 3, items_used=range(7)), robust=False)
    oa_ = fit(Da, Spec(14, 10, items_used=range(7, 14)), robust=False)
    bb = np.zeros(14)
    bb[Spec(14, 3, items_used=range(7)).free] = os_["theta"]
    bb[Spec(14, 10, items_used=range(7, 14)).free] = oa_["theta"]
    return dict(bj=bj.tolist(), bb=bb.tolist())


def exp3(reps):
    with Pool(WORKERS) as pool:
        res = pool.map(exp3_rep, [(3000 + r, 150) for r in range(reps)])
    beta = np.r_[np.linspace(-1, 1, 7), np.linspace(-0.9, 1.1, 7)]
    BJ = np.array([x["bj"] for x in res])
    BB = np.array([x["bb"] for x in res])
    within = [p for p in itertools.combinations(range(7), 2)] + \
             [p for p in itertools.combinations(range(7, 14), 2)]
    cross = [(i, j) for i in range(7) for j in range(7, 14)]
    sd = lambda B, prs: float(np.mean([np.std(B[:, i] - B[:, j]) for i, j in prs]))

    def top3(B, rng_):
        tr = set(np.argsort(-beta[rng_])[:3])
        return float(np.mean([set(np.argsort(-b[rng_])[:3]) == tr for b in B]))
    s_idx, a_idx = np.arange(7), np.arange(7, 14)
    out = dict(sd_within_joint=sd(BJ, within), sd_within_blocks=sd(BB, within),
               sd_cross_joint=sd(BJ, cross),
               top3_sensor_joint=top3(BJ, s_idx), top3_sensor_blocks=top3(BB, s_idx),
               top3_app_joint=top3(BJ, a_idx), top3_app_blocks=top3(BB, a_idx))
    print(json.dumps(out), flush=True)
    return out


def exp3b_rep(args):
    """Two blocks (7,4,2) plus n_bridge bridging sets with 2 items from each domain
    (random per respondent; each item at most once across the bridging sets)."""
    seed, N, n_bridge = args
    rng = np.random.default_rng(seed)
    beta = np.r_[np.linspace(-1, 1, 7), np.linspace(-0.9, 1.1, 7)]
    eta = 0.5 * rng.standard_normal((N, 14))
    parts = []
    for i in range(N):
        bi = beta + eta[i]
        parts.append(simulate(B7, np.arange(7), bi, 1, rng, eta_sd=0.0))
        parts.append(simulate(B7, np.arange(7, 14), bi, 1, rng, eta_sd=0.0))
        if n_bridge:
            s = rng.permutation(7)[:2 * n_bridge].reshape(n_bridge, 2)
            a = 7 + rng.permutation(7)[:2 * n_bridge].reshape(n_bridge, 2)
            br = np.c_[s, a]
            parts.append(simulate(np.arange(4 * n_bridge).reshape(n_bridge, 4), br.ravel(), bi, 1, rng,
                                  eta_sd=0.0, permute=False))
    for j, p in enumerate(parts):
        p.person = np.zeros_like(p.person) + 0  # set below
    # respondent ids: 2 or 3 parts per respondent
    per = 3 if n_bridge else 2
    for j, p in enumerate(parts):
        p.person[:] = j // per
    D = Data.concat(parts, person_offset=False)
    sp = Spec(14, 3)
    o = fit(D, sp, robust=False)
    b = np.zeros(14)
    b[sp.free] = o["theta"]
    return dict(b=b.tolist())


def exp3b(reps):
    beta = np.r_[np.linspace(-1, 1, 7), np.linspace(-0.9, 1.1, 7)]
    within = [p for p in itertools.combinations(range(7), 2)] + \
             [p for p in itertools.combinations(range(7, 14), 2)]
    cross = [(i, j) for i in range(7) for j in range(7, 14)]
    sd = lambda B, prs: float(np.mean([np.std(B[:, i] - B[:, j]) for i, j in prs]))
    out = {}
    with Pool(WORKERS) as pool:
        for nb in (1, 2, 3):
            res = pool.map(exp3b_rep, [(3500 + r, 150, nb) for r in range(reps)])
            B = np.array([x["b"] for x in res])
            out[f"bridge{nb}"] = dict(tasks=14 + nb, sd_within=sd(B, within), sd_cross=sd(B, cross))
            print(nb, json.dumps(out[f"bridge{nb}"]), flush=True)
    return out


# ---------------------------------------------------------------- Exp 4
def exp4_rep(args):
    seed, top, N = args
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    beta[12] = top
    d = simulate(BIBD13, np.arange(13), beta, N, rng)
    sp = Spec(13, 6)
    o = fit(d, sp, robust=False)
    b = np.zeros(13)
    b[sp.free] = o["theta"]
    r = np.arange(d.n)
    has = (d.items == 12).any(1)
    share = float(np.mean(d.items[r, d.best][has] == 12))
    agg, _ = counts(d, 13)
    return dict(b=b.tolist(), share=share, agg=agg.tolist())


def exp4(reps):
    out = {}
    beta0 = np.linspace(-1, 1, 13)
    with Pool(WORKERS) as pool:
        for top in (1.0, 2.0, 3.0):
            res = pool.map(exp4_rep, [(4000 + r, top, 150) for r in range(reps)])
            B = np.array([x["b"] for x in res])
            A = np.array([x["agg"] for x in res])
            prs = list(itertools.combinations(range(12), 2))
            sdp = float(np.mean([np.std(B[:, i] - B[:, j]) for i, j in prs]))
            tr = set(np.argsort(-beta0[:12])[:3])
            t3 = float(np.mean([set(np.argsort(-b[:12])[:3]) == tr for b in B]))
            t3c = float(np.mean([set(np.argsort(-a[:12])[:3]) == tr for a in A]))
            out[str(top)] = dict(best_share=float(np.mean([x["share"] for x in res])),
                                 sd_pairs_others=sdp, top3_others_model=t3, top3_others_counts=t3c)
            print(top, json.dumps(out[str(top)]), flush=True)
    return out


# ---------------------------------------------------------------- Exp 5
def exp5_rep(args):
    seed, N = args
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    d = simulate(BIBD13, np.arange(13), beta, N, rng, polar={6: (1.5, -1.5)})
    agg, per = counts(d, 13)
    sds = np.nanstd(per, 0)
    rank_mean = int(np.sum(agg > agg[6]))           # number of items ranked above the polarising item
    return dict(top_sd=int(np.argmax(sds)) == 6, rank=rank_mean,
                frac_extreme=float(np.mean(np.abs(per[:, 6]) >= 0.5)),
                frac_extreme_ref=float(np.mean(np.abs(per[:, 5]) >= 0.5)),
                per6=per[:, 6].tolist(), per5=per[:, 5].tolist(), agg6=float(agg[6]))


def exp5(reps):
    out = {}
    with Pool(WORKERS) as pool:
        for N in (30, 60, 150):
            res = pool.map(exp5_rep, [(5000 + r, N) for r in range(reps)])
            out[str(N)] = dict(p_highest_sd=float(np.mean([x["top_sd"] for x in res])),
                               items_ahead_mean=float(np.mean([x["rank"] for x in res])),
                               agg_mean=float(np.mean([x["agg6"] for x in res])),
                               frac_extreme=float(np.mean([x["frac_extreme"] for x in res])),
                               frac_extreme_ref=float(np.mean([x["frac_extreme_ref"] for x in res])))
            print(N, json.dumps(out[str(N)]), flush=True)
            if N == 150:
                out["example_per6"] = res[0]["per6"]
                out["example_per5"] = res[0]["per5"]
    return out


# ---------------------------------------------------------------- Exp 6
def exp6_rep(args):
    seed, drop, N, eta = args
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    d = simulate(BIBD13, np.arange(13), beta, N, rng, eta_sd=eta, drop_sets=drop)
    sp = Spec(13, 6)
    o = fit(d, sp, robust=False)
    b = np.zeros(13)
    b[sp.free] = o["theta"]
    return b.tolist()


def exp6(reps):
    out = {}
    prs = list(itertools.combinations(range(13), 2))
    with Pool(WORKERS) as pool:
        for eta in (0.5, 1.0):
            for N, drop in ((100, 0), (100, 3), (100, 6), (130, 3), (160, 5)):
                B = np.array(pool.map(exp6_rep, [(6000 + r, drop, N, eta) for r in range(reps)]))
                # relative SD (divided by mean estimated range) removes attenuation due to eta
                spread = np.mean(B[:, 12] - B[:, 0])
                sdp = float(np.mean([np.std(B[:, i] - B[:, j]) for i, j in prs]))
                key = f"eta{eta}_N{N}_tasks{13 - drop}"
                out[key] = dict(sd_pairs=sdp, spread=float(spread), rel=sdp / spread)
                print(key, json.dumps(out[key]), flush=True)
    return out


# ---------------------------------------------------------------- Exp 7
def exp7_rep(args):
    """Power for one item (J, beta 0.5) with a true shift delta in group 1:
    scale-adjusted Wald test (common beta, free mu, delta_J) vs. Welch test on counts."""
    seed, N, delta, lam1, rand1 = args
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    J = 9
    beta1 = beta.copy()
    beta1[J] += delta
    d0 = simulate(BIBD13, np.arange(13), beta, N, rng, group=0)
    d1 = simulate(BIBD13, np.arange(13), beta1, N, rng, lam=lam1, group=1, random_frac=rand1)
    D = Data.concat([d0, d1])
    _, p0 = counts(d0, 13)
    _, p1 = counts(d1, 13)
    pc = welch_p(p0[:, J], p1[:, J])
    sp = Spec(13, 6, n_groups=2, scale=True, delta_items=[J])
    o = fit(D, sp)
    ps = wald_p(o["theta"][-1], o["se"][-1])
    return (pc < ALPHA, ps < ALPHA)


def exp7(reps):
    out = {}
    with Pool(WORKERS) as pool:
        for delta in (0.3, 0.5):
            for lam1, rand1 in ((1.0, 0.0), (0.8, 0.0)):
                for N in (75, 100, 150, 200, 300):
                    r = np.array(pool.map(exp7_rep, [(7000 + i, N, delta, lam1, rand1) for i in range(reps)]))
                    key = f"d{delta}_lam{lam1}_rand{rand1}_N{N}"
                    out[key] = dict(power_counts=float(r[:, 0].mean()), power_scaleadj=float(r[:, 1].mean()))
                    print(key, json.dumps(out[key]), flush=True)
    return out


# ---------------------------------------------------------------- Exp 5b
def exp5b_rep(args):
    """Polarising item of strength +/-amp at small N: does it have the largest
    (or second largest) SD of individual count scores?"""
    seed, N, amp = args
    rng = np.random.default_rng(9000 + seed)
    beta = np.linspace(-1, 1, 13)
    d = simulate(BIBD13, np.arange(13), beta, N, rng, polar={6: (amp, -amp)})
    _, per = counts(d, 13)
    order = np.argsort(-np.nanstd(per, 0))
    return (bool(order[0] == 6), bool(6 in order[:2]))


def exp5b(reps):
    out = {}
    with Pool(WORKERS) as pool:
        for amp in (0.75, 1.0, 1.5):
            for N in (15, 20, 30, 50):
                r = np.array(pool.map(exp5b_rep, [(s, N, amp) for s in range(reps)]))
                key = f"amp{amp}_N{N}"
                out[key] = dict(p_highest_sd=float(r[:, 0].mean()), p_top2_sd=float(r[:, 1].mean()))
                print(key, json.dumps(out[key]), flush=True)
    return out


# ---------------------------------------------------------------- Exp 7b
def exp7b_rep(args):
    """Like exp7 (scale-adjusted single-item test), returning the p-value, with
    optional heterogeneity SD and random responders, to evaluate power with a
    Bonferroni correction over the 12 non-reference items (a lower bound for Holm)."""
    seed, N, delta, lam1, rand1, eta = args
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    J = 9
    beta1 = beta.copy()
    beta1[J] += delta
    d0 = simulate(BIBD13, np.arange(13), beta, N, rng, group=0, eta_sd=eta)
    d1 = simulate(BIBD13, np.arange(13), beta1, N, rng, lam=lam1, group=1, random_frac=rand1, eta_sd=eta)
    D = Data.concat([d0, d1])
    sp = Spec(13, 6, n_groups=2, scale=True, delta_items=[J])
    o = fit(D, sp)
    return float(wald_p(o["theta"][-1], o["se"][-1]))


def exp7b(reps):
    out = {}
    scen = [("base", 0.8, 0.0, 0.5), ("stress", 0.8, 0.15, 1.0)]
    with Pool(WORKERS) as pool:
        for name, lam1, rand1, eta in scen:
            for delta in (0.5, 0.3):
                for N in (100, 200, 300):
                    p = np.array(pool.map(exp7b_rep, [(7500 + i, N, delta, lam1, rand1, eta) for i in range(reps)]))
                    key = f"{name}_d{delta}_N{N}"
                    out[key] = dict(power_alpha05=float(np.mean(p < ALPHA)),
                                    power_bonferroni12=float(np.mean(p < ALPHA / 12)),
                                    lam1=lam1, random_frac=rand1, eta_sd=eta)
                    print(key, json.dumps(out[key]), flush=True)
    return out


EXPERIMENTS = {
    "exp1": "False positives / power of group comparisons under scale differences",
    "exp2": "Group-specific items: bias of count-based comparison of common items",
    "exp3": "One joint 14-item design vs. two separate 7-item blocks",
    "exp3b": "Two blocks plus 1-3 bridging sets",
    "exp4": "Cost of a dominant item for the precision of the other items",
    "exp5": "Detecting a polarising item from individual count scores",
    "exp5b": "Polarisation detection at soft-launch sizes for three strengths",
    "exp6": "Number of tasks per respondent vs. number of respondents",
    "exp7": "Power curve: scale-adjusted test vs. counts",
    "exp7b": "Power with Bonferroni correction and under a stress scenario",
}
DEFAULT_REPS = {"exp1": 200, "exp2": 200, "exp3": 300, "exp3b": 300, "exp4": 300,
                "exp5": 300, "exp5b": 300, "exp6": 300, "exp7": 200, "exp7b": 200}
WORKERS = 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=sorted(EXPERIMENTS))
    ap.add_argument("--reps", type=int, default=None, help="replications (default: as in the reported run)")
    ap.add_argument("--workers", type=int, default=2, help="parallel processes (does not change results)")
    ap.add_argument("--out", default="results/latest", help="output directory for res_<exp>.json")
    a = ap.parse_args()
    WORKERS = a.workers
    reps = a.reps or DEFAULT_REPS[a.exp]
    print(f"{a.exp}: {EXPERIMENTS[a.exp]} ({reps} replications, {WORKERS} workers)", flush=True)
    res = globals()[a.exp](reps)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(experiment=a.exp, description=EXPERIMENTS[a.exp], reps=reps, results=res),
              open(os.path.join(a.out, f"res_{a.exp}.json"), "w"), indent=1)
