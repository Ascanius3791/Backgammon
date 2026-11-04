"""Fit an accuracy head on the frozen champion and compare it with cheap baselines.
   python3 sigma_fit.py data.txt [n_other]      (data from ./sigma; n_other = number of other nets, default 3)

Error e = v0 - v_search (the champion's static value minus the search value of the same position).
The head predicts s = log E[e^2] = w . h + b from the champion's hidden layer h, fitted by the Gaussian negative
log-likelihood  mean(e^2 exp(-s) + s) + lam |w|^2  (convex in w). Every baseline score x is given the same chance:
log sigma^2 = a + b log(x + eps), fitted on the training set. Split by game: game % 5 == 0 is the test set,
game % 5 == 1 the validation set (picks lam), the rest trains.
Metrics on the test set: Spearman rho(score, |e|); NLL (lower is better, comparable across rows); and the share of
the total e^2 found in the 10% of positions with the highest score (what selective deepening would catch).
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

d = np.loadtxt(sys.argv[1])
no = int(sys.argv[2]) if len(sys.argv) > 2 else 3
game, off, v0, v2, vol2, v3 = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
vo = d[:, 6:6 + no]
h = d[:, 6 + no:]
ens = np.column_stack([v0, vo])
feats = {
    "p(1-p) of v0": v0 * (1 - v0),
    "ensemble std (4 nets)": ens.std(1),
    "|v0 - mean(others)|": np.abs(v0 - vo.mean(1)),
    "volatility over throws (search)": vol2,
}
EPS = 1e-4
test, val = game % 5 == 0, game % 5 == 1
train = ~test & ~val


def fit_log_var(X, e2, lam):
    X1 = np.column_stack([X, np.ones(len(X))])
    reg = np.full(X1.shape[1], lam); reg[-1] = 0

    def f(w):
        s = X1 @ w
        r = e2 * np.exp(-s)
        g = X1.T @ (1 - r) / len(s) + 2 * reg * w
        return np.mean(r + s) + np.sum(reg * w * w), g
    w0 = np.zeros(X1.shape[1]); w0[-1] = np.log(e2.mean())
    w = minimize(f, w0, jac=True, method="L-BFGS-B").x
    return lambda Z: np.column_stack([Z, np.ones(len(Z))]) @ w


def nll(s, e2):
    return np.mean(e2 * np.exp(-s) + s)


def top_share(score, e2, q=0.1):
    k = max(1, int(len(score) * q))
    return e2[np.argsort(-score)[:k]].sum() / e2.sum()


def fit_mlp(X, e2, lam, k=32, seed=0):
    """log sigma^2 = v . tanh(W x + c) + b: a nonlinear head on the frozen features"""
    n, m = X.shape
    rng = np.random.default_rng(seed)
    shapes = [(m, k), (k,), (k,), ()]
    sizes = [int(np.prod(sh)) for sh in shapes]

    def unpack(w):
        out, i = [], 0
        for sh, sz in zip(shapes, sizes):
            out.append(w[i:i + sz].reshape(sh)); i += sz
        return out

    def f(w):
        W, c, v, b = unpack(w)
        A = np.tanh(X @ W + c)
        s = A @ v + b
        r = e2 * np.exp(-s)
        ds = (1 - r) / n
        dv = A.T @ ds; db = ds.sum()
        dA = np.outer(ds, v) * (1 - A * A)
        dW = X.T @ dA; dc = dA.sum(0)
        loss = np.mean(r + s) + lam * (np.sum(W * W) + np.sum(v * v))
        g = np.concatenate([(dW + 2 * lam * W).ravel(), dc, dv + 2 * lam * v, [db]])
        return loss, g
    w0 = np.concatenate([rng.normal(0, 1 / np.sqrt(m), m * k), np.zeros(k), rng.normal(0, 0.1, k), [np.log(e2.mean())]])
    w = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 300}).x
    W, c, v, b = unpack(w)
    return lambda Z: np.tanh(Z @ W + c) @ v + b


def mlp_head(X, e2, tr, va):
    # at most 20000 training positions for the MLP (keeps the fit to seconds)
    idx = np.flatnonzero(tr)
    if len(idx) > 20000:
        tr = np.zeros_like(tr); tr[np.random.default_rng(1).choice(idx, 20000, replace=False)] = True
    best = None
    for lam in [1e-4, 1e-3, 1e-2]:
        m = fit_mlp(X[tr], e2[tr], lam)
        v = nll(m(X[va]), e2[va])
        if best is None or v < best[0]:
            best = (v, lam, m)
    return best[2], best[1]


def head(X, e2, tr, va):
    best = None
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        m = fit_log_var(X[tr], e2[tr], lam)
        v = nll(m(X[va]), e2[va])
        if best is None or v < best[0]:
            best = (v, lam)
    return fit_log_var(X[tr | va], e2[tr | va], best[1]), best[1]


def report(name, e, have):
    e2 = e * e
    tr, va, te = train & have, val & have, test & have
    print(f"\n=== target {name}: {have.sum()} positions ({tr.sum()} train, {va.sum()} val, {te.sum()} test); "
          f"rms error {np.sqrt(e2[have].mean()):.4f}, mean |e| {np.abs(e[have]).mean():.4f}")
    print(f"  {'predictor':44s} {'rho':>6s} {'NLL':>7s} {'top10%':>7s}   rho on-policy / off-policy")
    rows = [("constant", np.full(len(e), np.log(e2[tr | va].mean())))]
    for k, x in feats.items():
        m = fit_log_var(np.log(x + EPS)[:, None][tr | va], e2[tr | va], 0)
        rows.append((k, m(np.log(x + EPS)[:, None])))
    m, lam = head(h, e2, tr, va); rows.append((f"head on h (lam {lam:g})", m(h)))
    m, lam = mlp_head(h, e2, tr, va); rows.append((f"MLP head (32 tanh) on h (lam {lam:g})", m(h)))
    X = np.column_stack([h, np.log(ens.std(1) + EPS)])
    m, lam = head(X, e2, tr, va); rows.append((f"head on h + log ens std (lam {lam:g})", m(X)))
    X = np.column_stack([h, np.log(ens.std(1) + EPS), np.log(vol2 + EPS)])
    m, lam = head(X, e2, tr, va); rows.append((f"head on h + ens + volatility (lam {lam:g})", m(X)))
    for k, s in rows:
        rho = spearmanr(s[te], np.abs(e[te]))[0] if k != "constant" else 0
        ron = spearmanr(s[te & (off == 0)], np.abs(e[te & (off == 0)]))[0] if k != "constant" else 0
        rof = spearmanr(s[te & (off == 1)], np.abs(e[te & (off == 1)]))[0] if k != "constant" else 0
        print(f"  {k:44s} {rho:6.3f} {nll(s[te], e2[te]):7.3f} {top_share(s[te], e2[te]) if k != 'constant' else 0.1:7.3f}   {ron:6.3f} / {rof:6.3f}")
    return rows


print(f"{len(d)} positions, {int((off == 1).sum())} from games with a random opening")
report("e2 = v0 - v(depth 2)", v0 - v2, np.ones(len(d), bool))
have3 = v3 >= 0
if have3.sum() > 200:
    report("e3 = v0 - v(depth 3)", v0 - v3, have3)
    # transfer: the head fitted on the plentiful depth-2 errors, scored against the depth-3 errors
    e2_ = (v0 - v2) ** 2
    m, lam = head(h, e2_, train, val)
    te = test & have3
    print(f"\n  head fitted on e2, tested on e3: rho {spearmanr(m(h)[te], np.abs(v0 - v3)[te])[0]:.3f}, "
          f"top10% {top_share(m(h)[te], (v0 - v3)[te] ** 2):.3f}")
    print(f"  corr(e2, e3) = {np.corrcoef((v0 - v2)[have3], (v0 - v3)[have3])[0, 1]:.3f}")
