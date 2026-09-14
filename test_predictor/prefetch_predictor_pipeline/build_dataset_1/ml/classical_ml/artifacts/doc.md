# Trasformatori appresi sul train

Questa cartella conserva gli oggetti necessari per ricostruire lo stesso spazio
di feature sparse senza leakage.

- `categorical_encoder.joblib`: OneHotEncoder fit esclusivamente sul train;
- `numeric_scaler.joblib`: StandardScaler fit esclusivamente sul train;
- `feature_names.json`: nome e ordine delle colonne finali;
- `preprocessing_summary.json`: configurazione, shape, densità e conteggi.

Validation, test e runtime devono riutilizzare questi artefatti, non rifarli
sui propri dati.
