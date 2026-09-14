# Report di baseline e training

Questa cartella contiene i report JSON degli esperimenti BUILD_1.

- `baseline.json`: Recall@6 della persistenza del routing
  `E_l(t) -> E_l(t+1)`;
- `logistic_regression*.json`: metriche e parametri dei run Logistic Regression;
- `decision_tree.json` e `random_forest.json`: metriche dei rispettivi modelli;
- `training_summary*.json`: riepilogo del run, inclusi seed, top-k e parametri
  globali.

La metrica di selezione è la Recall@6 su validation; train serve soprattutto a
identificare overfitting.
