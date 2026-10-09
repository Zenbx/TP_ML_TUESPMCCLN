"""analyze.py -- turns results/*.csv|pkl into paper tables (LaTeX) and figures (PDF)."""
import os, pickle, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from lib import COV_GRID

R = "../results"; T = "../paper/tables"; F = "../paper/figs"
os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 6.8, "figure.dpi": 150, "axes.grid": True, "grid.alpha": .25})

A = pd.read_csv(f"{R}/results_A.csv"); B = pd.read_csv(f"{R}/results_B.csv"); C = pd.read_csv(f"{R}/results_C.csv")
D = {"A": A, "B": B, "C": C}
key = {}


def ms(x, d=3):
    return f"{np.mean(x):.{d}f}\\,{{\\scriptsize$\\pm${np.std(x):.{d}f}}}"


def pick(df, model, score="H", noise=0.0):
    return df[(df.model == model) & (df.score == score) & (df.noise == noise)].sort_values("seed")


# ---------------------------------------------------------------- Table 1: clean baselines
order = ["LogReg-SGA", "NaiveBayes", "KNN", "DecisionTree", "AdaBoost", "GradBoost", "LinearSVM", "RBF-SVM"]
names = {"LogReg-SGA": "LogReg (SGA)", "NaiveBayes": "Naive Bayes", "KNN": "KNN ($k$=15)", "DecisionTree": "Decision tree (d=5)",
         "AdaBoost": "AdaBoost", "GradBoost": "Gradient boosting", "LinearSVM": "Linear SVM", "RBF-SVM": "RBF-SVM"}
rows = []
for m in order:
    r = [names[m]]
    for ds in "AB":
        s = pick(D[ds], m)
        r += [ms(s.f1_min, 2), ms(s.ba_full, 3), ms(s.roc_auc, 3), ms(s.pr_auc, 3)]
    rows.append(r)
best = {}
tex = ("\\begin{tabular}{lcccccccc}\\toprule\n & \\multicolumn{4}{c}{Dataset A (Wisconsin, real)} & \\multicolumn{4}{c}{Dataset B (synthetic)}\\\\\n"
       "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nModel & F1 & BA & ROC-AUC & PR-AUC & F1 & BA & ROC-AUC & PR-AUC\\\\\\midrule\n")
tex += "\n".join(" & ".join(r) + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}"
open(f"{T}/clean_baselines.tex", "w").write(tex)
key["baseline_rank_B"] = {m: float(pick(B, m).ba_full.mean()) for m in order}
key["baseline_rank_A"] = {m: float(pick(A, m).ba_full.mean()) for m in order}
key["baseline_prauc_B"] = {m: float(pick(B, m).pr_auc.mean()) for m in order}
key["baseline_prauc_A"] = {m: float(pick(A, m).pr_auc.mean()) for m in order}

# ---------------------------------------------------------------- Table 2: selective prediction (clean + noisy)
sel_models = [("LogReg-SGA", "H", "LogReg-SGA + entropy"), ("RBF-SVM", "H", "RBF-SVM + entropy"),
              ("GradBoost", "H", "GradBoost + entropy"), ("Ens-LogReg", "H", "LogReg ensemble (H)"),
              ("Ens-LogReg", "MI", "LogReg ensemble (MI)"), ("EDL-Linear", "H", "EDL-Linear (H)"),
              ("EDL-Linear", "u", "EDL-Linear ($u$)"), ("EDL-RFF", "H", "EDL-RFF (H)"), ("EDL-RFF", "u", "EDL-RFF ($u$)")]


def sel_table(ds, noise, fname):
    rows = []
    for m, s, lab in sel_models:
        x = pick(D[ds], m, s, noise)
        rows.append([lab, ms(x.ba_full, 3), ms(x.ba70, 3), ms(x.ba70_rand, 3),
                     f"{100*np.mean(x.rel_ber70):.0f}\\%", ms(x.aurc, 3), ms(x.ece, 3)])
        key.setdefault(f"sel_{ds}_{noise}", {})[f"{m}|{s}"] = dict(
            ba_full=float(x.ba_full.mean()), ba70=float(x.ba70.mean()), rand=float(x.ba70_rand.mean()),
            rel=float(x.rel_ber70.mean()), aurc=float(x.aurc.mean()), ece=float(x.ece.mean()),
            ba70_sd=float(x.ba70.std()))
    tex = ("\\begin{tabular}{lcccccc}\\toprule\nModel + score & BA$_{1.0}$ & BA$_{0.7}$ & BA$_{0.7}^{\\text{rand}}$ & $\\Delta$BER & AURC$\\downarrow$ & ECE$\\downarrow$\\\\\\midrule\n")
    tex += "\n".join(" & ".join(r) + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}"
    open(f"{T}/{fname}.tex", "w").write(tex)


sel_table("B", 0.0, "sel_B_clean"); sel_table("A", 0.0, "sel_A_clean")
sel_table("B", 0.05, "sel_B_n05"); sel_table("B", 0.2, "sel_B_n20")
sel_table("A", 0.2, "sel_A_n20")

# ---------------------------------------------------------------- Table 3: noise robustness (Task 3.2)
fam = [("LogReg-SGA", "LogReg-SGA-R", "LogReg ($\\lambda$: $10^{-4}\\!\\to\\!0.05$)"),
       ("RBF-SVM", "RBF-SVM-R", "RBF-SVM ($C$: 1$\\to$0.1)"),
       ("GradBoost", "GradBoost-R", "Boosted trees (depth 3$\\to$2, shrinkage)")]
rows = []
for ds in "BA":
    for base_m, rob_m, lab in fam:
        for m, tag in ((base_m, "base"), (rob_m, "robust")):
            vals = {}
            for nz in (0.0, 0.05, 0.2):
                x = pick(D[ds], m, "H", nz)
                vals[nz] = x
            c0 = vals[0.0].ba_full.values
            d5 = c0 - vals[0.05].ba_full.values
            d20 = c0 - vals[0.2].ba_full.values
            rows.append((ds, lab if tag == "base" else "\\quad robust variant", ms(vals[0.0].ba_full, 3),
                         ms(vals[0.05].ba_full, 3), ms(d5, 3), ms(vals[0.2].ba_full, 3), ms(d20, 3)))
            key.setdefault("noise", {})[f"{ds}|{m}"] = dict(
                clean=float(c0.mean()), n05=float(vals[0.05].ba_full.mean()), n20=float(vals[0.2].ba_full.mean()),
                d5=float(d5.mean()), d20=float(d20.mean()), d5_sd=float(d5.std()), d20_sd=float(d20.std()),
                pr_clean=float(vals[0.0].pr_auc.mean()), pr20=float(vals[0.2].pr_auc.mean()),
                ece_clean=float(vals[0.0].ece.mean()), ece20=float(vals[0.2].ece.mean()))
for m in ("EDL-Linear", "EDL-RFF"):
    for ds in "BA":
        vals = {nz: pick(D[ds], m, "H", nz) for nz in (0.0, 0.05, 0.2)}
        c0 = vals[0.0].ba_full.values
        d5, d20 = c0 - vals[0.05].ba_full.values, c0 - vals[0.2].ba_full.values
        rows.append((ds, m.replace("-", "-"), ms(vals[0.0].ba_full, 3), ms(vals[0.05].ba_full, 3), ms(d5, 3),
                     ms(vals[0.2].ba_full, 3), ms(d20, 3)))
        key.setdefault("noise", {})[f"{ds}|{m}"] = dict(clean=float(c0.mean()), n05=float(vals[0.05].ba_full.mean()),
                                                      n20=float(vals[0.2].ba_full.mean()), d5=float(d5.mean()), d20=float(d20.mean()))
for ds, name in (("B", "Dataset B (synthetic)"), ("A", "Dataset A (Wisconsin)")):
    sub = [r for r in rows if r[0] == ds]
    tex = ("\\begin{tabular}{lccccc}\\toprule\nModel & BA clean & BA 5\\% & drop & BA 20\\% & drop\\\\\\midrule\n")
    tex += "\n".join(" & ".join(r[1:]) + "\\\\" for r in sub) + "\n\\bottomrule\\end{tabular}"
    open(f"{T}/noise_{ds}.tex", "w").write(tex)

# ---------------------------------------------------------------- Table 4: multiclass (C)
rows = []
for m, s, lab in [("LogReg-SGA", "H", "Softmax LogReg (SGA) + entropy"), ("LogReg-SGA-R", "H", "Softmax LogReg, $\\lambda$=0.05"),
                  ("GradBoost", "H", "Gradient boosting + entropy"), ("Ens-LogReg", "H", "LogReg ensemble (H)"),
                  ("Ens-LogReg", "MI", "LogReg ensemble (MI)"), ("EDL-Linear", "H", "EDL-Linear (H)"),
                  ("EDL-Linear", "u", "EDL-Linear ($u$)"), ("EDL-RFF", "H", "EDL-RFF (H)"), ("EDL-RFF", "u", "EDL-RFF ($u$)")]:
    x = pick(C, m, s, 0.0); xn = pick(C, m, s, 0.2)
    rows.append([lab, ms(x.acc, 3), ms(x.ba_full, 3), ms(x.ba70, 3), f"{100*x.rel_ber70.mean():.0f}\\%", ms(x.ece, 3), ms(xn.ba_full, 3)])
    key.setdefault("sel_C", {})[f"{m}|{s}"] = dict(acc=float(x.acc.mean()), ba=float(x.ba_full.mean()), ba70=float(x.ba70.mean()),
                                                  rel=float(x.rel_ber70.mean()), ece=float(x.ece.mean()), ba_n20=float(xn.ba_full.mean()),
                                                  rand=float(x.ba70_rand.mean()))
tex = ("\\begin{tabular}{lcccccc}\\toprule\nModel + score & Acc & BA$_{1.0}$ & BA$_{0.7}$ & $\\Delta$BER & ECE$\\downarrow$ & BA$_{1.0}$ (20\\% noise)\\\\\\midrule\n")
tex += "\n".join(" & ".join(r) + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}"
open(f"{T}/multiclass_C.tex", "w").write(tex)

# ---------------------------------------------------------------- paired tests (key claims)
def paired(ds, noise, a, b, metric):
    xa = pick(D[ds], *a, noise)[metric].values; xb = pick(D[ds], *b, noise)[metric].values
    d = xa - xb
    try:
        p = wilcoxon(xa, xb).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = np.nan
    return float(d.mean()), float(p)


tests = {}
cands = [("LogReg-SGA", "H"), ("RBF-SVM", "H"), ("GradBoost", "H"), ("Ens-LogReg", "H"), ("Ens-LogReg", "MI"),
         ("EDL-Linear", "H"), ("EDL-Linear", "u"), ("EDL-RFF", "H"), ("EDL-RFF", "u")]
for ds in "AB":
    for nz in (0.0, 0.2):
        # selective gain over random control, per method
        for (m, s) in cands:
            xa = pick(D[ds], m, s, nz); d = xa.ba70.values - xa.ba70_rand.values
            tests[f"{ds}|{nz}|{m}|{s}|gain_vs_random"] = (float(d.mean()), float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        for (a, b) in [(("EDL-RFF", "u"), ("EDL-RFF", "H")), (("EDL-Linear", "u"), ("EDL-Linear", "H")),
                       (("EDL-RFF", "H"), ("RBF-SVM", "H")), (("EDL-RFF", "u"), ("RBF-SVM", "H")),
                       (("EDL-Linear", "H"), ("LogReg-SGA", "H")), (("Ens-LogReg", "MI"), ("Ens-LogReg", "H")),
                       (("EDL-RFF", "H"), ("GradBoost", "H"))]:
            for met in ("ba70", "aurc", "ba_full", "ece"):
                tests[f"{ds}|{nz}|{a[0]}-{a[1]} vs {b[0]}-{b[1]}|{met}"] = paired(ds, nz, a, b, met)
key["tests"] = tests

# ---------------------------------------------------------------- published-gain comparison
pub = {"Aperstein_relBER_NIH": [0.25, 0.36], "Aperstein_gain_ba": [0.067, 0.087],
       "Aperstein_shift_relBER": [0.07, 0.53], "MedFormerUR_ECE": [0.1836, 0.0688]}
key["pub"] = pub
for ds in "AB":
    for nz in (0.0,):
        key[f"ours_relBER_{ds}"] = {f"{m}|{s}": float(pick(D[ds], m, s, nz).rel_ber70.mean()) for m, s in cands}
        key[f"ours_gain_{ds}"] = {f"{m}|{s}": float((pick(D[ds], m, s, nz).ba70 - pick(D[ds], m, s, nz).ba_full).mean()) for m, s in cands}

# ---------------------------------------------------------------- Figures
CC = {"LogReg-SGA|H": ("#1f77b4", "LogReg + entropy"), "RBF-SVM|H": ("#2ca02c", "RBF-SVM + entropy"),
      "GradBoost|H": ("#8c564b", "GradBoost + entropy"), "Ens-LogReg|MI": ("#9467bd", "LogReg ens. (MI)"),
      "EDL-RFF|u": ("#d62728", "EDL-RFF ($u$)"), "EDL-RFF|H": ("#ff7f0e", "EDL-RFF (H)")}


def mean_curve(ds, noise, model, score):
    P = pickle.load(open(f"{R}/curves_{ds}.pkl", "rb"))
    cs = [c["curve"] for c in P["curves"] if c["noise"] == noise and c["model"] == model and c["score"] == score]
    rs = [c["curve"] for c in P["rnd"] if c["noise"] == noise]
    return np.mean(cs, 0), np.mean(rs, 0)


fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (ds, nz, ttl) in zip(axes, [("B", 0.0, "(a) B, clean labels"), ("B", 0.2, "(b) B, 20% minority flips"),
                                    ("A", 0.0, "(c) A, clean labels"), ("A", 0.2, "(d) A, 20% minority flips")]):
    for k, (col, lab) in CC.items():
        m, s = k.split("|")
        cur, rnd = mean_curve(ds, nz, m, s)
        ax.plot(COV_GRID, cur, color=col, lw=1.2, label=lab)
    ax.plot(COV_GRID, rnd, "k--", lw=1, label="random rejection")
    ax.set_title(ttl, pad=5); ax.set_xlabel("coverage"); ax.invert_xaxis()
axes[0].set_ylabel("balanced accuracy (retained)")
axes[2].set_ylabel("balanced accuracy (retained)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.1, w_pad=0.8)
fig.savefig(f"{F}/risk_coverage.pdf", bbox_inches="tight"); plt.close(fig)

# threshold sensitivity (calibrated quantile thresholds), dataset B
fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
qs = [0.70, 0.80, 0.90, 0.95, 0.99]
for k in ["LogReg-SGA|H", "RBF-SVM|H", "EDL-RFF|u", "Ens-LogReg|MI"]:
    m, s = k.split("|"); x = pick(B, m, s, 0.0)
    cov = [x[f"cov_q{q}"].mean() for q in qs]; ba = [x[f"ba_q{q}"].mean() for q in qs]
    axes[0].plot(qs, cov, "o-", ms=3, color=CC[k][0], label=CC[k][1])
    axes[1].plot(cov, ba, "o-", ms=3, color=CC[k][0])
    key.setdefault("thr_B", {})[k] = dict(cov=cov, ba=ba)
axes[0].plot(qs, qs, "k:", lw=.8)
axes[0].set_xlabel("calibration quantile $q$"); axes[0].set_ylabel("test coverage"); axes[0].set_title("(a) coverage induced by $\\tau_q$")
axes[1].set_xlabel("test coverage"); axes[1].set_ylabel("BA (retained)"); axes[1].set_title("(b) accuracy--coverage trade-off")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{F}/threshold.pdf", bbox_inches="tight"); plt.close(fig)

json.dump(key, open(f"{R}/key_numbers.json", "w"), indent=1, default=float)
print("analysis done")

# ---------------------------------------------------------------- OOD table
O = pd.read_csv(f"{R}/results_ood.csv")
ood_rows = [("LogReg-SGA", "H", "LogReg + entropy"), ("EDL-Linear", "H", "EDL-Linear (H)"), ("EDL-Linear", "u", "EDL-Linear ($u$)"),
            ("Ens-LogReg", "H", "LogReg ensemble (H)"), ("Ens-LogReg", "MI", "LogReg ensemble (MI)"),
            ("GradBoost", "H", "GradBoost + entropy"), ("RBF-SVM", "H", "RBF-SVM + entropy"),
            ("EDL-RFF", "H", "EDL-RFF (H)"), ("EDL-RFF", "u", "EDL-RFF ($u$)")]
rows, okey = [], {}
for m, s, lab in ood_rows:
    r = [lab]
    for ds in "BA":
        for tag in ("far", "shift"):
            x = O[(O.dataset == ds) & (O.ood == tag) & (O.model == m) & (O.score == s)].sort_values("seed").auroc
            r.append(ms(x, 2)); okey[f"{ds}|{tag}|{m}|{s}"] = float(x.mean())
    rows.append(r)
tex = ("\\begin{tabular}{lcccc}\\toprule\n & \\multicolumn{2}{c}{Dataset B} & \\multicolumn{2}{c}{Dataset A}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nModel + score & far-OOD & shift & far-OOD & shift\\\\\\midrule\n")
tex += "\n".join(" & ".join(r) + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}"
open(f"{T}/ood.tex", "w").write(tex)
key["ood"] = okey
ot = {}
for ds in "BA":
    for tag in ("far", "shift"):
        g = lambda m, s: O[(O.dataset == ds) & (O.ood == tag) & (O.model == m) & (O.score == s)].sort_values("seed").auroc.values
        for (a, b) in [(("EDL-RFF", "u"), ("EDL-RFF", "H")), (("EDL-RFF", "u"), ("RBF-SVM", "H")), (("Ens-LogReg", "MI"), ("Ens-LogReg", "H"))]:
            d = g(*a) - g(*b)
            ot[f"{ds}|{tag}|{a[0]}-{a[1]} vs {b[0]}-{b[1]}"] = (float(d.mean()), float(wilcoxon(d).pvalue))
key["ood_tests"] = ot
json.dump(key, open(f"{R}/key_numbers.json", "w"), indent=1, default=float)
print("ood done")

# ---------------------------------------------------------------- compact combined tables
def mean(ds, m, s, noise, col):
    return float(pick(D[ds], m, s, noise)[col].mean())


def fmt_best(vals, i, d=3, hi=True):
    b = max(vals) if hi else min(vals)
    s = f"{vals[i]:.{d}f}"
    return f"\\textbf{{{s}}}" if abs(vals[i] - b) < 5e-4 else s


rows = []
cols = {}
for ds in "AB":
    cols[ds] = {c: [mean(ds, m, s, 0.0, c) for m, s, _ in sel_models] for c in ("ba_full", "ba70", "rel_ber70", "aurc", "ece")}
for i, (m, s, lab) in enumerate(sel_models):
    r = [lab]
    for ds in "AB":
        r += [f"{cols[ds]['ba_full'][i]:.3f}", fmt_best(cols[ds]["ba70"], i), f"{100*cols[ds]['rel_ber70'][i]:.0f}\\%",
              fmt_best(cols[ds]["aurc"], i, hi=False), f"{cols[ds]['ece'][i]:.3f}"]
    rows.append(r)
tex = ("\\begin{tabular}{lccccc|ccccc}\\toprule\n & \\multicolumn{5}{c}{Dataset A (Wisconsin, real)} & \\multicolumn{5}{c}{Dataset B (synthetic)}\\\\\n"
       "Model + score & BA$_{1.0}$ & BA$_{0.7}$ & $\\Delta$BER & AURC & ECE & BA$_{1.0}$ & BA$_{0.7}$ & $\\Delta$BER & AURC & ECE\\\\\\midrule\n")
tex += "\n".join(" & ".join(r) + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}"
open(f"{T}/sel_combined.tex", "w").write(tex)

# Calibration metrics used in medical-imaging uncertainty papers. These are
# evaluated on the untouched test labels; lower is better for every column.
cal_models = [(m, s, lab) for m, s, lab in sel_models]
rows = []
for m, s, lab in cal_models:
    row = [lab]
    for ds in "AB":
        x = pick(D[ds], m, s, 0.0)
        row += [ms(x.ece, 3), ms(x.brier, 3), ms(x.nll, 3)]
    rows.append(" & ".join(row) + "\\\\")
tex = ("\\begin{tabular}{lccc|ccc}\\toprule\n"
       " & \\multicolumn{3}{c}{Dataset A} & \\multicolumn{3}{c}{Dataset B}\\\\\n"
       "Model + score & ECE$\\downarrow$ & Brier$\\downarrow$ & NLL$\\downarrow$ & ECE$\\downarrow$ & Brier$\\downarrow$ & NLL$\\downarrow$\\\\\\midrule\n")
tex += "\n".join(rows) + "\n\\bottomrule\\end{tabular}"
open(f"{T}/calibration.tex", "w").write(tex)

fam_rows = [("LogReg-SGA", "LogReg (SGA), $\\lambda{=}10^{-4}$"), ("LogReg-SGA-R", "LogReg (SGA), $\\lambda{=}0.05$"),
            ("RBF-SVM", "RBF-SVM, $C{=}1$"), ("RBF-SVM-R", "RBF-SVM, $C{=}0.1$"),
            ("GradBoost", "Boosted trees, depth 3"), ("GradBoost-R", "Boosted trees, depth 2 + shrinkage"),
            ("EDL-Linear", "EDL-Linear"), ("EDL-RFF", "EDL-RFF")]
rows = []
for ds in "BA":
    vals = {c: [] for c in ("clean", "n5", "n20", "drop", "gain")}
    for m, _ in fam_rows:
        c0 = pick(D[ds], m, "H", 0.0).ba_full.values
        n20 = pick(D[ds], m, "H", 0.2)
        vals["clean"].append(c0.mean()); vals["n5"].append(pick(D[ds], m, "H", 0.05).ba_full.mean())
        vals["n20"].append(n20.ba_full.mean()); vals["drop"].append((c0 - n20.ba_full.values).mean())
        vals["gain"].append((n20.ba70.values - n20.ba_full.values).mean())
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{Dataset {ds}}}}}\\\\")
    for i, (m, lab) in enumerate(fam_rows):
        sd = np.std(pick(D[ds], m, "H", 0.0).ba_full.values - pick(D[ds], m, "H", 0.2).ba_full.values)
        rows.append(" & ".join([lab, f"{vals['clean'][i]:.3f}", f"{vals['n5'][i]:.3f}", fmt_best(vals["n20"], i),
                                f"{vals['drop'][i]:.3f}\\,{{\\scriptsize$\\pm${sd:.3f}}}", f"{vals['gain'][i]:+.3f}".replace('-', '$-$')]) + "\\\\")
    rows.append("\\midrule" if ds == "B" else "")
tex = ("\\begin{tabular}{lccccc}\\toprule\nModel & BA clean & BA 5\\% & BA 20\\% & drop (20\\%) & rejection gain (20\\%)\\\\\\midrule\n")
tex += "\n".join(rows) + "\n\\bottomrule\\end{tabular}"
open(f"{T}/noise_combined.tex", "w").write(tex)

# Paired across-seed uncertainty intervals and Holm correction for the
# prespecified clean-label entropy-rejection comparisons (three main model
# families on each binary dataset). These quantify run variability, not
# population uncertainty over new datasets or patients.
from scipy.stats import t as student_t

def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, ix in enumerate(order):
        running = max(running, (m - rank) * pvals[ix])
        adj[ix] = min(1.0, running)
    return adj

primary = []
for ds in "AB":
    for model in ("LogReg-SGA", "RBF-SVM", "GradBoost"):
        x = pick(D[ds], model, "H", 0.0).sort_values("seed")
        d = (x.ba70 - x.ba70_rand).to_numpy()
        n = len(d); se = d.std(ddof=1) / np.sqrt(n)
        half = student_t.ppf(0.975, n - 1) * se
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        primary.append(dict(dataset=ds, model=model, n=n, mean=float(d.mean()),
                            lo=float(d.mean() - half), hi=float(d.mean() + half), p=p))
p_adj = holm_adjust([r["p"] for r in primary])
for r, pa in zip(primary, p_adj): r["p_holm"] = float(pa)
rows = []
for r in primary:
    dsname = "A (real tabular)" if r["dataset"] == "A" else "B (synthetic)"
    modelname = {"LogReg-SGA": "Logistic regression", "RBF-SVM": "RBF-SVM", "GradBoost": "Gradient boosting"}[r["model"]]
    rows.append(f"{dsname} & {modelname} & {r['mean']:+.3f} & [{r['lo']:+.3f}, {r['hi']:+.3f}] & {r['p_holm']:.3g}\\\\")
tex = ("\\begin{tabular}{llccc}\\toprule\nDataset & Model & Mean $\\Delta$BA & 95\\% CI & Holm $p$\\\\\\midrule\n" +
       "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
open(f"{T}/primary_inference.tex", "w").write(tex)
key["primary_inference"] = primary

# Multiple operating points (50%, 70%, and 90% coverage) match the reporting
# used in the cited selective-prediction studies.
curve_data = {ds: pickle.load(open(f"{R}/curves_{ds}.pkl", "rb"))["curves"] for ds in "AB"}
cov_index = {float(c): i for i, c in enumerate(COV_GRID)}
rows = []
for ds in "AB":
    for model, label in (("LogReg-SGA", "Logistic regression"), ("RBF-SVM", "RBF-SVM"), ("GradBoost", "Gradient boosting")):
        curves = [np.asarray(x["curve"]) for x in curve_data[ds]
                  if x["noise"] == 0.0 and x["model"] == model and x["score"] == "H"]
        mat = np.stack(curves)
        vals = [f"{mat[:, cov_index[c]].mean():.3f}\\,{{\\scriptsize$\\pm${mat[:, cov_index[c]].std():.3f}}}" for c in (0.50, 0.70, 0.90)]
        aurc_vals = pick(D[ds], model, "H", 0.0).aurc.to_numpy()
        aurc = f"{aurc_vals.mean():.3f}\\,{{\\scriptsize$\\pm${aurc_vals.std():.3f}}}"
        rows.append(" & ".join([f"Dataset {ds}", label, *vals, aurc]) + "\\\\")
tex = ("\\begin{tabular}{llcccc}\\toprule\nDataset & Model & BA @50\\% & BA @70\\% & BA @90\\% & AURC$\\downarrow$\\\\\\midrule\n" +
       "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
open(f"{T}/coverage_metrics.tex", "w").write(tex)
json.dump(key, open(f"{R}/key_numbers.json", "w"), indent=1, default=float)
print("combined tables done")
