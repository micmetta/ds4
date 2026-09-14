# Artifacts

Questa cartella contiene gli artefatti necessari per interpretare la
rappresentazione:

- `categorical_encoder.joblib`: OneHotEncoder del campo `layer`, fit soltanto sul train;
- `router_id_encoder.joblib`: codifica one-hot fissa di 256 expert ID per ognuno dei 16 slot dei router logits;
- `numeric_scaler.joblib`: StandardScaler fit soltanto sul train per i 16 valori dei router logits;
- `feature_names.json`: nomi ordinati delle 4.408 feature;
- `preprocessing_summary.json`: shape, nnz, split, target e configurazione dell'ablation.
