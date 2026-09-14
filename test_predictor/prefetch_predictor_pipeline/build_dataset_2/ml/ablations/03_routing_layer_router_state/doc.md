# Routing multi-hot + layer + router state

Questa ablation materializza la configurazione `03_routing_layer_router_state`
dello screening KNN. Per predire `E_l(t+1)`, usa le informazioni osservate in
`E_l(t)` per lo stesso layer `l`: routing multi-hot, layer e stato del router.

Il dataset BUILD_DATASET_2 risultante ha 4.408 feature:

```text
256 routing multi-hot + 40 layer one-hot + (16 slot x 256 expert ID) + 16 valori = 4.408
```

Gli ID dei 16 raw router logits hanno una codifica one-hot fissa: ogni slot
puo' rappresentare tutti gli expert ID da `0` a `255`, anche se un ID non e'
presente nel train. I 16 valori dei logits sono standardizzati colonna per
colonna con parametri calcolati soltanto sul train.

Le matrici e le label sono create da
`ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py`
con `--representation multihot_256_layer_router_state`.

- `arrays/`: matrici CSR `X_classic_<split>.npz` e label `y_<split>.npy` per train, validation e test.
- `artifacts/`: encoder, scaler, nomi delle feature e riepilogo del preprocessing.
