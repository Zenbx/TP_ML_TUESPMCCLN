"""run_ood.py -- OOD detection from uncertainty scores (cf. UIOS Sec. 2.3, BEDL Sec. 5.5).
ID = clean test split.  far-OOD: x ~ N(0, 3^2 I) in standardised space.
covariate shift: x' = 1.5 x + mu, mu_j = +-1 (random signs).  Score: higher => more uncertain.
AUROC of separating OOD (positive) from ID."""
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
from run_experiments import get_data, split, run_model
warnings.filterwarnings("ignore")

MODELS = ["LogReg-SGA", "RBF-SVM", "GradBoost", "Ens-LogReg", "EDL-Linear", "EDL-RFF"]
rows = []
for ds, nseeds in (("B", 10), ("A", 10)):
    for seed in range(nseeds):
        X, y, K = get_data(ds, seed)
        Xtr, ytr, Xca, yca, Xte, yte = split(X, y, seed)
        rng = np.random.RandomState(777 + seed)
        n, d = Xte.shape
        far = rng.normal(0, 3.0, size=(n, d))
        shift = 1.5 * Xte + rng.choice([-1.0, 1.0], size=d)
        XX = np.vstack([Xte, far, shift])
        for name in MODELS:
            _, _, scores, _, _ = run_model(name, Xtr, ytr, Xca, yca, XX, K, seed)
            for sname, (_, s) in scores.items():
                s_id, s_far, s_sh = s[:n], s[n:2 * n], s[2 * n:]
                for tag, s_ood in (("far", s_far), ("shift", s_sh)):
                    lab = np.r_[np.zeros(n), np.ones(n)]
                    rows.append(dict(dataset=ds, seed=seed, model=name, score=sname, ood=tag,
                                     auroc=roc_auc_score(lab, np.r_[s_id, s_ood])))
        print(ds, seed, flush=True)
pd.DataFrame(rows).to_csv("../results/results_ood.csv", index=False)
