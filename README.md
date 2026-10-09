# TP4 – paper sources

paper/main.tex, refs.bib   -> compile: cd paper && latexmk -pdf main.tex  (tables/ and figs/ are inputs)
code/                      -> run order (from code/): run_experiments.py A|B|C ; run_ood.py ; analyze.py ; fig_boundaries.py ; mlflow_run.py
results/                   -> CSVs / curves produced by the code (analyze.py regenerates paper/tables and paper/figs)
mlflow.db                  -> MLflow store (view: mlflow ui --backend-store-uri sqlite:///mlflow.db)
