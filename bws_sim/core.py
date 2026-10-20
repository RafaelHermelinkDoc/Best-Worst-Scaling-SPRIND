"""
Case 1 best-worst scaling (object case) - core functions for simulation studies.

Model: sequential best-worst logit (best chosen from all k items in a set, then
worst chosen from the remaining k-1 items with reversed sign), i.i.d. Gumbel
errors, preference heterogeneity beta_i = beta + eta_i with eta ~ N(0, sd^2),
and a scale parameter (lambda) per respondent group.

Contents:
  - Designs: BIBD(13,4,1), BIBD(7,4,2), 4-GDD of type 2^7 (14 items, 14 sets,
    each item 4 times, each pair at most once), design check
  - Response simulation (item-to-position permutation per respondent;
    optional polarising items and random responders)
  - Estimator: conditional sequential best-worst logit, optionally with a free
    scale parameter per group and a group-specific shift delta_j
    (Swait-Louviere logic); cluster-robust (sandwich) standard errors by respondent
  - Count scores (standardised best-minus-worst, per respondent and aggregate)
"""
import itertools
import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------- Designs
def bibd_13_4_1():
    return np.array([[(k + d) % 13 for d in (0, 1, 3, 9)] for k in range(13)])


def bibd_7_4_2():
    # complements of the Fano-plane lines {k, k+1, k+3} mod 7
    return np.array([[x for x in range(7) if x not in {k % 7, (k + 1) % 7, (k + 3) % 7}]
                     for k in range(7)])


def gdd_2_7():
    """4-GDD of type 2^7: 14 items in 7 groups {0,1},{2,3},...; 14 sets of size 4;
    every pair of items from different groups appears exactly once, pairs within
    a group never. Found by exact-cover backtracking (deterministic)."""
    groups = [(2 * g, 2 * g + 1) for g in range(7)]
    grp = {i: i // 2 for i in range(14)}
    pairs = {frozenset(p) for p in itertools.combinations(range(14), 2) if grp[p[0]] != grp[p[1]]}
    cands = []
    for gs in itertools.combinations(range(7), 4):
        for choice in itertools.product((0, 1), repeat=4):
            blk = tuple(groups[g][c] for g, c in zip(gs, choice))
            cands.append(blk)
    cand_pairs = [frozenset(frozenset(p) for p in itertools.combinations(b, 2)) for b in cands]
    by_pair = {p: [i for i, cp in enumerate(cand_pairs) if p in cp] for p in pairs}

    sol = []

    def search(uncovered):
        if not uncovered:
            return True
        # pair with the fewest remaining candidate blocks
        p = min(uncovered, key=lambda q: sum(1 for i in by_pair[q] if cand_pairs[i] <= uncovered))
        for i in by_pair[p]:
            cp = cand_pairs[i]
            if cp <= uncovered:
                sol.append(cands[i])
                if search(uncovered - cp):
                    return True
                sol.pop()
        return False

    assert search(frozenset(pairs))
    return np.array(sorted(sol))


def check_design(D, v):
    occ = np.bincount(D.ravel(), minlength=v)
    pc = np.zeros((v, v), int)
    for s in D:
        for a, b in itertools.combinations(s, 2):
            pc[a, b] += 1
            pc[b, a] += 1
    off = pc[~np.eye(v, dtype=bool)]
    return dict(sets=len(D), k=D.shape[1], occ_min=occ.min(), occ_max=occ.max(),
                pair_min=off.min(), pair_max=off.max())


# -------------------------------------------------------------- Simulation
class Data:
    def __init__(self, items, best, worst, group, person):
        self.items, self.best, self.worst = items, best, worst
        self.group, self.person = group, person
        self.n = len(best)

    @staticmethod
    def concat(ds, person_offset=True):
        off, parts = 0, []
        for d in ds:
            parts.append((d.items, d.best, d.worst, d.group, d.person + off))
            off += (d.person.max() + 1) if person_offset else 0
        return Data(*[np.concatenate(x) for x in zip(*parts)])


def simulate(design, item_map, beta, N, rng, lam=1.0, eta_sd=0.5, group=0,
             permute=True, polar=None, random_frac=0.0, drop_sets=0):
    """design: (b,k) array of local positions 0..v-1; item_map: local -> global item
    indices; beta: global true utilities (length V). polar: dict {global index:
    (value_a, value_b)} -> half of respondents hold value_a, half value_b (instead
    of beta + eta). random_frac: share of respondents answering at random
    (lambda = 0). drop_sets: number of sets dropped at random per respondent
    (fewer tasks)."""
    item_map = np.asarray(item_map)
    b, k = design.shape
    v = len(item_map)
    V = len(beta)
    perms = np.array([rng.permutation(v) if permute else np.arange(v) for _ in range(N)])
    sets = item_map[perms[:, design]]                        # (N,b,k) global items
    if drop_sets:
        keep = np.array([np.sort(rng.choice(b, b - drop_sets, replace=False)) for _ in range(N)])
        sets = np.take_along_axis(sets, keep[:, :, None], axis=1)
        b = b - drop_sets
    bi = beta[None, :] + eta_sd * rng.standard_normal((N, V))
    if polar:
        for j, (va, vb) in polar.items():
            half = rng.random(N) < 0.5
            bi[:, j] = np.where(half, va, vb)
    lam_i = np.full(N, lam)
    if random_frac > 0:
        lam_i[rng.random(N) < random_frac] = 0.0
    V_ = np.take_along_axis(bi[:, None, :].repeat(b, 1), sets, axis=2) * lam_i[:, None, None]
    ub = V_ + rng.gumbel(size=V_.shape)
    best = ub.argmax(2)
    uw = -V_ + rng.gumbel(size=V_.shape)
    np.put_along_axis(uw, best[:, :, None], -np.inf, axis=2)
    worst = uw.argmax(2)
    person = np.repeat(np.arange(N), b)
    return Data(sets.reshape(-1, k), best.ravel(), worst.ravel(),
                np.full(N * b, group), person)


# ----------------------------------------------------------------- Estimator
class Spec:
    """V: number of global items; ref: reference item (beta fixed at 0); items_used:
    items to estimate; n_groups; scale: free scale parameter for groups >= 1;
    delta_items: items with a shift delta_j in group 1; sep_beta: separate beta
    per group (no scale parameter)."""
    def __init__(self, V, ref, items_used=None, n_groups=1, scale=False, delta_items=(),
                 sep_beta=False):
        self.V, self.ref, self.G = V, ref, n_groups
        used = np.arange(V) if items_used is None else np.asarray(sorted(items_used))
        self.free = np.array([i for i in used if i != ref])
        self.scale = scale and n_groups > 1
        self.delta_items = np.asarray(delta_items, int)
        self.sep_beta = sep_beta
        nb = len(self.free) * (n_groups if sep_beta else 1)
        self.nb = nb
        self.ns = (n_groups - 1) if self.scale else 0
        self.nd = len(self.delta_items)
        self.P = nb + self.ns + self.nd

    def unpack(self, th):
        nf = len(self.free)
        betas = np.zeros((self.G, self.V))
        if self.sep_beta:
            for g in range(self.G):
                betas[g, self.free] = th[g * nf:(g + 1) * nf]
        else:
            betas[:, self.free] = th[:nf]
        logmu = np.zeros(self.G)
        if self.ns:
            logmu[1:] = th[self.nb:self.nb + self.ns]
        if self.nd:
            betas[1, self.delta_items] += th[self.nb + self.ns:]
        return betas, logmu


def _core(th, D, sp, need_rows=False):
    betas, logmu = sp.unpack(th)
    n, k = D.items.shape
    r = np.arange(n)
    mu = np.exp(logmu)[D.group]
    base = betas[D.group[:, None], D.items]
    u = mu[:, None] * base
    ub = u - u.max(1, keepdims=True)
    eb = np.exp(ub)
    sb = eb.sum(1, keepdims=True)
    pb = eb / sb
    ll = (ub[r, D.best] - np.log(sb[:, 0])).sum()
    gu = -pb
    gu[r, D.best] += 1
    v = -u
    v[r, D.best] = -np.inf
    vm = v - v.max(1, keepdims=True)
    ev = np.exp(vm)
    sv = ev.sum(1, keepdims=True)
    q = ev / sv
    ll += (vm[r, D.worst] - np.log(sv[:, 0])).sum()
    gw = -q
    gw[r, D.worst] += 1
    gu = gu - gw
    W = gu * mu[:, None]                      # d ll / d base
    # row-level scores (n x P), used for cluster-robust standard errors
    rows = np.zeros((n, sp.P)) if need_rows else None
    grad = np.zeros(sp.P)
    nf = len(sp.free)
    col_of = -np.ones(sp.V, int)
    col_of[sp.free] = np.arange(nf)
    for g in range(sp.G):
        mg = D.group == g
        if not mg.any():
            continue
        off = g * nf if sp.sep_beta else 0
        it = D.items[mg]
        cols = col_of[it]
        ok = cols >= 0
        w = W[mg]
        grad[off:off + nf] += np.bincount(cols[ok], weights=w[ok], minlength=nf)
        if need_rows:
            rr = np.repeat(np.where(mg)[0], k).reshape(-1, k)
            np.add.at(rows, (rr[ok], off + cols[ok]), w[ok])
        if sp.ns and g >= 1:
            sc = (gu[mg] * u[mg]).sum(1)
            grad[sp.nb + g - 1] += sc.sum()
            if need_rows:
                rows[np.where(mg)[0], sp.nb + g - 1] += sc
        if sp.nd and g == 1:
            dcol = -np.ones(sp.V, int)
            dcol[sp.delta_items] = np.arange(sp.nd)
            dc = dcol[it]
            okd = dc >= 0
            grad[sp.nb + sp.ns:] += np.bincount(dc[okd], weights=w[okd], minlength=sp.nd)
            if need_rows:
                np.add.at(rows, (rr[okd], sp.nb + sp.ns + dc[okd]), w[okd])
    return ll, grad, rows


def fit(D, sp, robust=True, x0=None):
    f = lambda th: tuple(-x for x in _core(th, D, sp)[:2])
    x0 = np.zeros(sp.P) if x0 is None else x0
    res = minimize(f, x0, jac=True, method="BFGS", options=dict(gtol=1e-6, maxiter=500))
    th = res.x
    out = dict(theta=th, ll=-res.fun, ok=res.success)
    if robust:
        h = 1e-5
        H = np.zeros((sp.P, sp.P))
        for j in range(sp.P):
            e = np.zeros(sp.P)
            e[j] = h
            H[:, j] = (f(th + e)[1] - f(th - e)[1]) / (2 * h)
        H = (H + H.T) / 2
        Binv = np.linalg.inv(H)
        _, _, rows = _core(th, D, sp, need_rows=True)
        npers = D.person.max() + 1
        S = np.zeros((npers, sp.P))
        np.add.at(S, D.person, rows)
        M = S.T @ S * npers / (npers - 1)
        out["cov"] = Binv @ M @ Binv
        out["se"] = np.sqrt(np.diag(out["cov"]))
    return out


# ----------------------------------------------------------------- Count scores
def counts(D, V):
    """Standardised count scores (B - W) / occurrences: aggregate (length V) and per
    respondent (N x V, NaN where an item was not shown)."""
    n, k = D.items.shape
    r = np.arange(n)
    bi = D.items[r, D.best]
    wi = D.items[r, D.worst]
    occ = np.bincount(D.items.ravel(), minlength=V).astype(float)
    B = np.bincount(bi, minlength=V)
    Wc = np.bincount(wi, minlength=V)
    agg = np.where(occ > 0, (B - Wc) / np.maximum(occ, 1), np.nan)
    npers = D.person.max() + 1
    po = np.zeros((npers, V))
    np.add.at(po, (np.repeat(D.person, k), D.items.ravel()), 1)
    pb = np.zeros((npers, V))
    np.add.at(pb, (D.person, bi), 1)
    np.add.at(pb, (D.person, wi), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(po > 0, pb / po, np.nan)
    return agg, per
