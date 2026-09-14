# Modelli serializzati BUILD_1

Questa cartella contiene i modelli scikit-learn addestrati e salvati con
joblib:

- `logistic_regression.joblib`: run iniziale Logistic Regression;
- `logistic_regression_max_iter_5000.joblib`: Logistic Regression con
  `max_iter=5000`;
- `decision_tree.joblib`: Decision Tree non limitato del primo confronto;
- `random_forest.joblib`: Random Forest compatto, limitato per evitare OOM.

Ogni modello richiede le stesse feature sparse e gli stessi artefatti salvati
in `../artifacts/`. I risultati comparativi sono nei file `../reports/`.
