"""
run_experiments.py -- reproduces every number in the paper.

usage:  python run_experiments.py {A|B|C}
A: Breast Cancer Wisconsin (real, made imbalanced)   B: synthetic imbalanced binary
C: synthetic 3-class (BUSI-like class proportions)
"""
import sys, time, pickle, warnings
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)
from lib import *

warnings.filterwarnings("ignore")
QS = [0.70, 0.80, 0.90, 0.95, 0.99]
NOISES = [0.0, 0.05, 0.20]


# ----------------------------------------------------------------- data
def get_data(ds, seed):
    rng = np.random.RandomState(seed)
    if ds == "A":
        d = load_breast_cancer()
        y = 1 - d.target                       # 1 = malignant (minority after subsampling)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        keep = np.concatenate([neg, rng.choice(pos, 63, replace=False)])   # 15 % minority
        # 10 'standard-error' features (cols 10-19): all 30 features give AUC~0.997 (saturated)
        return d.data[keep][:, 10:20], y[keep], 2
    if ds == "B":
        X, y = make_classification(10000, 40, n_informative=10, n_redundant=5,
                                   weights=[0.92, 0.08], class_sep=0.8, flip_y=0.02,
                                   n_clusters_per_class=1, random_state=1234)
        return X, y, 2
    if ds == "C":
        X, y = make_classification(6000, 30, n_informative=10, n_redundant=4, n_classes=3,
                                   weights=[0.17, 0.62, 0.21], class_sep=0.9, flip_y=0.02,
                                   n_clusters_per_class=1, random_state=1234)
        return X, y, 3


def split(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xca, ytr, yca = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), ytr, sc.transform(Xca), yca, sc.transform(Xte), yte


def flip_minority(y, K, frac, seed):
    """Flip `frac` of the minority-class labels (Task 3.2); other classes untouched."""
    if frac == 0:
        return y.copy()
    rng = np.random.RandomState(10_000 + seed)
    y2 = y.copy()
    mino = np.bincount(y, minlength=K).argmin()
    idx = np.where(y == mino)[0]
    f = rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False)
    for i in f:
        y2[i] = rng.choice([c for c in range(K) if c != mino])
    return y2


# ----------------------------------------------------------------- models
def cw_dict(y, K):
    w = sqrt_inv_freq_weights(y, K)
    return {c: w[c] for c in range(K)}


def platt(d_cal, y_cal, d_te, d_cal_for_proba):
    lr = LogisticRegression(C=1e6).fit(d_cal.reshape(-1, 1), y_cal)
    f = lambda d: lr.predict_proba(d.reshape(-1, 1))
    return f(d_cal_for_proba), f(d_te)


def run_model(name, Xtr, ytr, Xca, yca, Xte, K, seed, Xcal_threshold=None):
    """Return probabilities/scores on the rejection-calibration split and test.

    Xca/yca fit Platt scaling for SVMs; Xcal_threshold is disjoint and is used
    only to select rejection thresholds. Other models use the same disjoint
    threshold split to generate their calibration scores.
    """
    Xthreshold = Xca if Xcal_threshold is None else Xcal_threshold
    sw = sqrt_inv_freq_weights(ytr, K)[ytr]
    hp = {}
    if name.startswith("LogReg-SGA"):
        lam = 0.05 if name.endswith("-R") else 1e-4
        m = LogRegSGA(lam=lam, epochs=30, seed=seed).fit(Xtr, ytr); hp = dict(lam=lam)
    elif name == "NaiveBayes":
        m = GaussianNB().fit(Xtr, ytr)
    elif name == "KNN":
        m = KNeighborsClassifier(15).fit(Xtr, ytr); hp = dict(k=15)
    elif name == "DecisionTree":
        m = DecisionTreeClassifier(max_depth=5, class_weight=cw_dict(ytr, K), random_state=seed).fit(Xtr, ytr)
        hp = dict(max_depth=5)
    elif name == "AdaBoost":
        m = AdaBoostClassifier(n_estimators=100, random_state=seed).fit(Xtr, ytr, sample_weight=sw)
        hp = dict(n_estimators=100)
    elif name.startswith("GradBoost"):
        if name.endswith("-R"):
            hp = dict(max_depth=2, n_estimators=60, learning_rate=0.05, subsample=0.7, min_samples_leaf=10)
        else:
            hp = dict(max_depth=3, n_estimators=100, learning_rate=0.1, subsample=1.0, min_samples_leaf=1)
        m = GradientBoostingClassifier(random_state=seed, **hp).fit(Xtr, ytr, sample_weight=sw)
    elif name in ("LinearSVM", "RBF-SVM", "RBF-SVM-R"):
        if name == "LinearSVM":
            svm = LinearSVC(C=1.0, class_weight=cw_dict(ytr, K), max_iter=5000).fit(Xtr, ytr); hp = dict(C=1.0)
        else:
            C = 0.1 if name.endswith("-R") else 1.0
            svm = SVC(C=C, kernel="rbf", gamma="scale", class_weight=cw_dict(ytr, K)).fit(Xtr, ytr)
            hp = dict(C=C)
        dca, dte = svm.decision_function(Xca), svm.decision_function(Xte)
        dthreshold = svm.decision_function(Xthreshold)
        pca, pte = platt(dca, yca, dte, dthreshold)
        H = lambda P: entropy(P)
        return pca, pte, {"H": (H(pca), H(pte))}, hp, svm
    elif name in ("EDL-Linear", "EDL-RFF"):
        m = EDLLinear(rff=300 if name == "EDL-RFF" else 0, epochs=100, lam_kl=0.1, seed=seed).fit(Xtr, ytr)
        hp = dict(rff=m.rff, lam_kl=0.1, epochs=100)
        pca, pte = m.predict_proba(Xthreshold), m.predict_proba(Xte)
        return pca, pte, {"H": (entropy(pca), entropy(pte)),
                          "u": (m.uncertainty(Xthreshold), m.uncertainty(Xte))}, hp, m
    elif name == "Ens-LogReg":
        rng = np.random.RandomState(seed)
        Pc, Pt = [], []
        for b in range(20):
            bi = rng.choice(len(ytr), len(ytr), replace=True)
            mb = LogRegSGA(lam=1e-4, epochs=20, seed=seed + b).fit(Xtr[bi], ytr[bi])
            Pc.append(mb.predict_proba(Xthreshold)); Pt.append(mb.predict_proba(Xte))
        Pmc, Hc, _, MIc = ensemble_scores(Pc)
        Pmt, Ht, _, MIt = ensemble_scores(Pt)
        return Pmc, Pmt, {"H": (Hc, Ht), "MI": (MIc, MIt)}, dict(B=20), None
    pca, pte = m.predict_proba(Xthreshold), m.predict_proba(Xte)
    return pca, pte, {"H": (entropy(pca), entropy(pte))}, hp, m


# ----------------------------------------------------------------- evaluation
def evaluate(name, noise, seed, ds, K, yca, yte, Pca, Pte, scores, hp):
    rows, curves = [], {}
    yhat_te, yhat_ca = Pte.argmax(1), Pca.argmax(1)
    base = dict(dataset=ds, seed=seed, noise=noise, model=name,
                acc=accuracy_score(yte, yhat_te),
                f1_min=f1_score(yte, yhat_te, average="binary" if K == 2 else "macro", pos_label=1) if K == 2 else f1_score(yte, yhat_te, average="macro"),
                ba_full=bal_acc(yte, yhat_te), ece=ece_score(Pte, yte),
                brier=brier(Pte, yte), nll=nll(Pte, yte))
    if K == 2:
        base["roc_auc"] = roc_auc_score(yte, Pte[:, 1])
        base["pr_auc"] = average_precision_score(yte, Pte[:, 1])
    rnd = random_curve(yte, yhat_te, seed=seed)
    for sname, (s_ca, s_te) in scores.items():
        cur = selective_curve(yte, yhat_te, s_te)
        i70 = int(np.where(COV_GRID == 0.70)[0][0])
        be_full, be70 = 1 - cur[0], 1 - cur[i70]
        r = dict(base, score=sname, ba70=cur[i70], ba70_rand=rnd[i70],
                 rel_ber70=(be_full - be70) / max(be_full, 1e-9),
                 aurc=aurc_bal(cur), aurc_rand=aurc_bal(rnd))
        for q in QS:
            cov, ba = quantile_threshold_eval(yca, yhat_ca, s_ca, yte, yhat_te, s_te, q)
            r[f"cov_q{q}"], r[f"ba_q{q}"] = cov, ba
        rows.append(r)
        curves[(name, sname)] = cur
    return rows, curves, rnd


MAIN = ["LogReg-SGA", "LogReg-SGA-R", "RBF-SVM", "RBF-SVM-R", "GradBoost", "GradBoost-R",
        "EDL-Linear", "EDL-RFF", "Ens-LogReg"]
BASE_ONLY = ["NaiveBayes", "KNN", "DecisionTree", "AdaBoost", "LinearSVM"]
MAIN_C = ["LogReg-SGA", "LogReg-SGA-R", "GradBoost", "GradBoost-R", "EDL-Linear", "EDL-RFF", "Ens-LogReg"]


def main(ds, nseeds):
    all_rows, all_curves, rnd_curves = [], [], []
    t0 = time.time()
    for seed in range(nseeds):
        X, y, K = get_data(ds, seed)
        Xtr, ytr0, Xca0, yca0, Xte, yte = split(X, y, seed)
        Xca, Xct, yca_fit0, yct0 = train_test_split(
            Xca0, yca0, test_size=0.5, stratify=yca0, random_state=seed + 10_000)
        for noise in NOISES:
            ytr = flip_minority(ytr0, K, noise, seed)
            yca = flip_minority(yca_fit0, K, noise, seed + 500)
            yct = flip_minority(yct0, K, noise, seed + 1_000)  # separate threshold-calibration labels
            models = (MAIN_C if ds == "C" else MAIN) + (BASE_ONLY if (noise == 0 and ds != "C") else [])
            for name in models:
                Pca, Pte, scores, hp, _ = run_model(name, Xtr, ytr, Xca, yca, Xte, K, seed, Xcal_threshold=Xct)
                rows, curves, rnd = evaluate(name, noise, seed, ds, K, yct, yte, Pca, Pte, scores, hp)
                all_rows += rows
                for k, v in curves.items():
                    all_curves.append(dict(dataset=ds, seed=seed, noise=noise, model=k[0], score=k[1], curve=v))
                rnd_curves.append(dict(seed=seed, noise=noise, curve=rnd))
        print(f"[{ds}] seed {seed+1}/{nseeds} done  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(f"../results/results_{ds}.csv", index=False)
    pickle.dump(dict(curves=all_curves, rnd=rnd_curves), open(f"../results/curves_{ds}.pkl", "wb"))
    print("saved", df.shape)


if __name__ == "__main__":
    ds = sys.argv[1]
    main(ds, {"A": 20, "B": 10, "C": 10}[ds])
