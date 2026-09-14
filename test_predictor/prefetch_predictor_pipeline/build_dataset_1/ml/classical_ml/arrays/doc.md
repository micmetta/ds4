# Matrici sparse per scikit-learn

Questa cartella contiene sei file:

- `X_classic_<split>.npz`: matrici sparse CSR con le feature categoriali
  one-hot e le feature numeriche standardizzate;
- `y_<split>.npy`: label multi-hot `[N, 256]` corrispondenti.

Gli split sono `train`, `validation` e `test`. Le matrici BUILD_1 hanno 6.168
colonne e sono l'input diretto di baseline e modelli classici.
