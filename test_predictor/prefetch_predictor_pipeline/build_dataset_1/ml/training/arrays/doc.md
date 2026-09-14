# Array NumPy del preprocessing base

Questa cartella contiene nove array, separati in train, validation e test:

- `X_<split>.npy`: feature dense float32 con shape `[N, 41]`;
- `y_<split>.npy`: target multi-hot uint8 con shape `[N, 256]` e sei expert
  positivi per riga;
- `target_expert_ids_<split>.npy`: i sei ID target originali, shape `[N, 6]`.

Sono gli array base BUILD_1; il preprocessing classico genera poi una diversa
rappresentazione sparse nella cartella `../../classical_ml/arrays/`.
