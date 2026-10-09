"""
lib.py -- core components for the TP4 / paper experiments.

Contents
--------
* LogRegSGA      : logistic regression (binary sigmoid / multiclass softmax) trained
                   by mini-batch stochastic gradient ascent on the (weighted,
                   L2-penalised) log-likelihood  (Task 1.2).
* EDLLinear      : evidential linear classifier (Softplus evidence -> Dirichlet,
                   Sensoy et al. 2018 type-II loss + annealed KL), optional
                   random Fourier features for a non-linear variant.
* Selective-prediction utilities (entropy, risk-coverage, calibration, ...).
Everything is plain NumPy/SciPy so that no deep-learning stack is required.
"""
import numpy as np
from scipy.special import digamma, polygamma, gammaln, expit, softmax

EPS = 1e-12


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sqrt_inv_freq_weights(y, K):
    """Square-root inverse-frequency class weights (as in Aperstein et al.)."""
    counts = np.bincount(y, minlength=K).astype(float)
    w = np.sqrt(counts.max() / np.maximum(counts, 1))
    return w


def softplus(z):
    return np.logaddexp(0.0, z)


# --------------------------------------------------------------------------
# Logistic regression with stochastic gradient ASCENT  (Task 1.2)
# --------------------------------------------------------------------------
class LogRegSGA:
    """Binary (sigmoid) or multiclass (softmax) logistic regression.

    Update rule (per mini-batch B):  theta <- theta + lr * [ mean_i w_i (y_i - h_i) x_i - lam*theta ]
    which is stochastic gradient ascent on the weighted, L2-penalised log-likelihood.
    batch_size=1 recovers the per-sample rule of the conceptual template.
    """

    def __init__(self, lam=1e-4, lr=0.05, epochs=30, batch_size=32, decay=0.05,
                 class_weight=True, seed=0):
        self.lam, self.lr, self.epochs = lam, lr, epochs
        self.batch_size, self.decay = batch_size, decay
        self.class_weight, self.seed = class_weight, seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.K = int(y.max()) + 1
        cw = sqrt_inv_freq_weights(y, self.K) if self.class_weight else np.ones(self.K)
        sw = cw[y]
        if self.K == 2:
            self.theta, self.b = np.zeros(d), 0.0
        else:
            self.theta, self.b = np.zeros((d, self.K)), np.zeros(self.K)
            Y = np.eye(self.K)[y]
        for ep in range(self.epochs):
            lr = self.lr / (1.0 + self.decay * ep)
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                bi = idx[s:s + self.batch_size]
                Xb, wb = X[bi], sw[bi]
                if self.K == 2:
                    err = wb * (y[bi] - expit(Xb @ self.theta + self.b))
                    self.theta += lr * (Xb.T @ err / len(bi) - self.lam * self.theta)
                    self.b += lr * err.mean()
                else:
                    P = softmax(Xb @ self.theta + self.b, axis=1)
                    err = wb[:, None] * (Y[bi] - P)
                    self.theta += lr * (Xb.T @ err / len(bi) - self.lam * self.theta)
                    self.b += lr * err.mean(axis=0)
        return self

    def predict_proba(self, X):
        if self.K == 2:
            p1 = expit(X @ self.theta + self.b)
            return np.column_stack([1 - p1, p1])
        return softmax(X @ self.theta + self.b, axis=1)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# --------------------------------------------------------------------------
# Evidential classifier (EDL)
# --------------------------------------------------------------------------
def edl_loss_grad(Z, Y, W, lam_kl):
    """Sensoy et al. (2018) digamma loss + KL(Dir(alpha~) || Dir(1)).

    Z : (B,K) pre-activations, evidence e = softplus(Z), alpha = e + 1.
    Returns mean loss and dLoss/dZ  (B,K).  W: per-sample weights.
    """
    K = Z.shape[1]
    alpha = softplus(Z) + 1.0
    S = alpha.sum(1, keepdims=True)
    A = (Y * (digamma(S) - digamma(alpha))).sum(1)
    at = Y + (1 - Y) * alpha                        # alpha-tilde
    St = at.sum(1, keepdims=True)
    kl = (gammaln(St[:, 0]) - gammaln(at).sum(1) - gammaln(K)
          + ((at - 1) * (digamma(at) - digamma(St))).sum(1))
    loss = (W * (A + lam_kl * kl)).mean()
    # gradients wrt alpha
    dA = polygamma(1, S) - Y * polygamma(1, alpha)
    dKL = (1 - Y) * ((at - 1) * polygamma(1, at)
                     - polygamma(1, St) * (at - 1).sum(1, keepdims=True))
    dalpha = dA + lam_kl * dKL
    dZ = (W[:, None] * dalpha) * expit(Z) / Z.shape[0]
    return loss, dZ


class EDLLinear:
    """Evidential linear head.  rff>0 uses random Fourier features (RBF kernel)."""

    def __init__(self, rff=0, gamma=None, lam_l2=1e-3, lam_kl=0.1, kl_anneal=10,
                 lr=0.01, epochs=60, batch_size=64, class_weight=True, seed=0):
        self.rff, self.gamma, self.lam_l2 = rff, gamma, lam_l2
        self.lam_kl, self.kl_anneal = lam_kl, kl_anneal
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.class_weight, self.seed = class_weight, seed

    def _phi(self, X):
        if not self.rff:
            return X
        return np.cos(X @ self.Om + self.ph)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.K = int(y.max()) + 1
        if self.rff:
            g = self.gamma if self.gamma else 1.0 / d
            self.Om = rng.normal(0, np.sqrt(2 * g), size=(d, self.rff))
            self.ph = rng.uniform(0, 2 * np.pi, size=self.rff)
        F = self._phi(X)
        D = F.shape[1]
        cw = sqrt_inv_freq_weights(y, self.K) if self.class_weight else np.ones(self.K)
        sw = cw[y]
        Y = np.eye(self.K)[y]
        self.Wt = rng.normal(0, 0.01, size=(D, self.K))
        self.b = np.zeros(self.K)
        # Adam state
        m = [np.zeros_like(self.Wt), np.zeros_like(self.b)]
        v = [np.zeros_like(self.Wt), np.zeros_like(self.b)]
        t = 0
        for ep in range(self.epochs):
            lam = self.lam_kl * min(1.0, ep / self.kl_anneal)
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                bi = idx[s:s + self.batch_size]
                Z = F[bi] @ self.Wt + self.b
                _, dZ = edl_loss_grad(Z, Y[bi], sw[bi], lam)
                gW = F[bi].T @ dZ + self.lam_l2 * self.Wt
                gb = dZ.sum(0)
                t += 1
                for i, (p, g) in enumerate(((self.Wt, gW), (self.b, gb))):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    p -= self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def alpha(self, X):
        return softplus(self._phi(X) @ self.Wt + self.b) + 1.0

    def predict_proba(self, X):
        a = self.alpha(X)
        return a / a.sum(1, keepdims=True)

    def uncertainty(self, X):
        """Subjective-logic uncertainty mass u = K / S."""
        a = self.alpha(X)
        return self.K / a.sum(1)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# --------------------------------------------------------------------------
# uncertainty scores
# --------------------------------------------------------------------------
def entropy(P, base=2):
    P = np.clip(P, EPS, 1.0)
    return -(P * np.log(P)).sum(1) / np.log(base)


def ensemble_scores(P_list):
    """P_list: (B,n,K). returns predictive entropy, expected entropy, mutual information."""
    P_list = np.asarray(P_list)
    Pm = P_list.mean(0)
    H = entropy(Pm)
    EH = np.mean([entropy(P) for P in P_list], axis=0)
    return Pm, H, EH, H - EH


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def bal_acc(y, yhat):
    """Macro-averaged recall over the classes present in y (robust to rejection)."""
    cls = np.unique(y)
    if len(cls) == 0:
        return np.nan
    return float(np.mean([(yhat[y == c] == c).mean() for c in cls]))


def ece_score(P, y, bins=15):
    conf = P.max(1)
    pred = P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(P, y):
    Y = np.eye(P.shape[1])[y]
    return float(((P - Y) ** 2).sum(1).mean())


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).mean())


COV_GRID = np.round(np.arange(1.0, 0.499, -0.05), 2)


def selective_curve(y, yhat, score, cov_grid=COV_GRID, rng=None):
    """Balanced accuracy of the retained set for each coverage (reject highest score)."""
    n = len(y)
    jitter = (rng or np.random.RandomState(0)).uniform(0, 1e-9, size=n)
    order = np.argsort(score + jitter)              # most confident first
    out = []
    for c in cov_grid:
        k = max(int(round(c * n)), 2)
        keep = order[:k]
        out.append(bal_acc(y[keep], yhat[keep]))
    return np.array(out)


def random_curve(y, yhat, cov_grid=COV_GRID, reps=30, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    res = np.zeros((reps, len(cov_grid)))
    for r in range(reps):
        sc = rng.rand(n)
        res[r] = selective_curve(y, yhat, sc, cov_grid, rng)
    return res.mean(0)


def aurc_bal(curve, cov_grid=COV_GRID, lo=0.70):
    """Mean balanced risk (1-BA) over coverages in [lo, 1] (as in Aperstein et al.)."""
    m = cov_grid >= lo - 1e-9
    return float(np.mean(1 - curve[m]))


def quantile_threshold_eval(y_cal, yhat_cal, s_cal, y_te, yhat_te, s_te, q):
    """Aperstein-style calibration: tau = q-quantile of the score over *correctly
    classified* calibration samples; reject test samples with score > tau."""
    ok = yhat_cal == y_cal
    tau = np.quantile(s_cal[ok], q) if ok.sum() > 5 else np.quantile(s_cal, q)
    keep = s_te <= tau
    cov = keep.mean()
    ba = bal_acc(y_te[keep], yhat_te[keep]) if keep.sum() > 1 else np.nan
    return float(cov), ba
