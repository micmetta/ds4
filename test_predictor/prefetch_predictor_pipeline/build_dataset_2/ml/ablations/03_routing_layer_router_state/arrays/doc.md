# Arrays

Questa cartella contiene le matrici sparse CSR per train, validation e test.
Ogni riga ha 4.408 feature e 39 valori non nulli:

```text
6 routing multi-hot + 1 layer one-hot + 16 router-ID one-hot + 16 logits = 39
```

Le label `y_<split>.npy` sono gli stessi target multi-hot di 256 expert di
BUILD_DATASET_2 e rappresentano `E_l(t+1)`.
