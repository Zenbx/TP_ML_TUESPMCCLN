"""
mlflow_run.py -- Task 3.3.  Logs the Task-3.2 runs (baseline vs robust variants,
clean vs 5 % minority-label noise) of the three best models to a local MLflow store.

    python mlflow_run.py            # then:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import os, json, pickle
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from run_experiments import get_data, split, flip_minority, cw_dict
from lib import LogRegSGA, EDLLinear, sqrt_inv_freq_weights, entropy, selective_curve, COV_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri("sqlite:///" + os.path.abspath(os.path.join(HERE, "..", "mlflow.db")))
mlflow.set_experiment("TP4-robust-classification")

SEED, DS = 0, "B"
X, y, K = get_data(DS, SEED)
Xtr, ytr0, Xca, yca0, Xte, yte = split(X, y, SEED)

CONFIGS = {   # name -> (builder, hyper-parameters)
    "LogReg-SGA":   (lambda: LogRegSGA(lam=1e-4, epochs=30, seed=SEED), dict(lam=1e-4, epochs=30, lr=0.05, batch=32)),
    "LogReg-SGA-R": (lambda: LogRegSGA(lam=0.05, epochs=30, seed=SEED), dict(lam=0.05, epochs=30, lr=0.05, batch=32)),
    "RBF-SVM":      (None, dict(C=1.0, kernel="rbf", gamma="scale")),
    "RBF-SVM-R":    (None, dict(C=0.1, kernel="rbf", gamma="scale")),
    "GradBoost":    (None, dict(max_depth=3, n_estimators=100, learning_rate=0.1, subsample=1.0)),
    "GradBoost-R":  (None, dict(max_depth=2, n_estimators=60, learning_rate=0.05, subsample=0.7, min_samples_leaf=10)),
    "EDL-RFF":      (lambda: EDLLinear(rff=300, epochs=100, lam_kl=0.1, seed=SEED), dict(rff=300, lam_kl=0.1, epochs=100)),
}

for noise in (0.0, 0.05):
    ytr = flip_minority(ytr0, K, noise, SEED)
    sw = sqrt_inv_freq_weights(ytr, K)[ytr]
    for name, (builder, hp) in CONFIGS.items():
        with mlflow.start_run(run_name=f"{name}|noise={noise}"):
            mlflow.set_tags(dict(model=name, robust=name.endswith("-R"), dataset=DS))
            mlflow.log_params(dict(hp, label_noise=noise, seed=SEED))
            if name.startswith("RBF-SVM"):
                m = SVC(C=hp["C"], kernel="rbf", gamma="scale", class_weight=cw_dict(ytr, K)).fit(Xtr, ytr)
                yhat = m.predict(Xte); score = None
            elif name.startswith("GradBoost"):
                m = GradientBoostingClassifier(random_state=SEED, **{k: v for k, v in hp.items()}).fit(Xtr, ytr, sample_weight=sw)
                yhat = m.predict(Xte); score = entropy(m.predict_proba(Xte))
            else:
                m = builder().fit(Xtr, ytr)
                P = m.predict_proba(Xte); yhat = P.argmax(1); score = entropy(P)
            mlflow.log_metrics(dict(precision=precision_score(yte, yhat, zero_division=0),
                                    recall=recall_score(yte, yhat),
                                    f1=f1_score(yte, yhat),
                                    balanced_accuracy=balanced_accuracy_score(yte, yhat)))
            if score is not None:
                cur = selective_curve(yte, yhat, score)
                mlflow.log_metric("balanced_accuracy_at_cov70", float(cur[list(COV_GRID).index(0.70)]))
            # model artifact
            if hasattr(m, "get_params") and name.startswith(("RBF", "Grad")):
                mlflow.sklearn.log_model(m, "model", serialization_format="cloudpickle")
            else:                                     # NumPy models: parameters as artifact
                path = os.path.join(HERE, f"_{name}_params.json")
                params = {k: (v.tolist() if hasattr(v, "tolist") else v)
                          for k, v in vars(m).items() if k in ("theta", "b", "Wt", "Om", "ph")}
                json.dump(params, open(path, "w")); mlflow.log_artifact(path, "model"); os.remove(path)
print("MLflow runs logged to mlflow.db (ui: mlflow ui --backend-store-uri sqlite:///mlflow.db)")
