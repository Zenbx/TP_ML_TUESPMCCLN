"""fig_boundaries.py -- Task 2.3 (decision boundaries, LogReg vs decision tree on a 2-D PCA
projection of dataset A) and Task 3.1 (precision-recall curves on dataset B)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from lib import LogRegSGA, EDLLinear, sqrt_inv_freq_weights
from run_experiments import get_data, split, cw_dict

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.grid": True, "grid.alpha": .25})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
axes = axes.ravel()

# ---- decision boundaries on 2-D PCA of dataset A (seed 0)
X, y, K = get_data("A", 0)
Xs = StandardScaler().fit_transform(X)
Z = PCA(2, random_state=0).fit_transform(Xs)
gx, gy = np.meshgrid(np.linspace(Z[:, 0].min() - .5, Z[:, 0].max() + .5, 300), np.linspace(Z[:, 1].min() - .5, Z[:, 1].max() + .5, 300))
G = np.c_[gx.ravel(), gy.ravel()]
models = [("LogReg (SGA)", LogRegSGA(lam=1e-4, epochs=60, seed=0).fit(Z, y)),
          ("Decision tree (d=5)", DecisionTreeClassifier(max_depth=5, class_weight=cw_dict(y, 2), random_state=0).fit(Z, y))]
for ax, (name, m) in zip(axes[:2], models):
    pr = m.predict_proba(G)[:, 1].reshape(gx.shape)
    ax.contourf(gx, gy, pr, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=.75)
    ax.contour(gx, gy, pr, levels=[.5], colors="k", linewidths=1)
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, c="#1f4e9c", edgecolor="none", label="benign")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=8, c="#b2182b", edgecolor="w", linewidth=.3, label="malignant")
    ax.set_title(("(a) " if "LogReg" in name else "(b) ") + name); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
axes[0].legend(frameon=True, loc="lower right")

# ---- ROC and PR curves on dataset B (seed 0, clean)
X, y, K = get_data("B", 0)
Xtr, ytr, Xca, yca, Xte, yte = split(X, y, 0)
sw = sqrt_inv_freq_weights(ytr, 2)[ytr]
svm = SVC(C=1, gamma="scale", class_weight=cw_dict(ytr, 2)).fit(Xtr, ytr)
gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr, sample_weight=sw)
sc = {"LogReg (SGA)": LogRegSGA(epochs=30, seed=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
      "RBF-SVM": svm.decision_function(Xte), "Gradient boosting": gb.predict_proba(Xte)[:, 1],
      "EDL-RFF": EDLLinear(rff=300, epochs=100, seed=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]}
for name, s in sc.items():
    f, t, _ = roc_curve(yte, s); axes[2].plot(f, t, label=f"{name} ({auc(f, t):.2f})")
    p, r, _ = precision_recall_curve(yte, s); axes[3].plot(r, p, label=f"{name} ({average_precision_score(yte, s):.2f})")
axes[2].plot([0, 1], [0, 1], "k:", lw=.8)
axes[3].axhline(yte.mean(), color="k", ls=":", lw=.8)
axes[2].set_title("(c) ROC, dataset B"); axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
axes[3].set_title("(d) PR curve, dataset B"); axes[3].set_xlabel("recall"); axes[3].set_ylabel("precision")
axes[2].legend(frameon=False, loc="lower right"); axes[3].legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("../paper/figs/boundaries_prc.pdf", bbox_inches="tight")
print("prevalence B test:", yte.mean())
