# Routing multi-hot + layer

Questa ablation materializza la configurazione `02_routing_multihot_layer` dello screening KNN. Le sei routing history ID di `E_l(t)` diventano un vettore multi-hot da 256 posizioni e il layer routed corrente diventa un vettore one-hot.

Il dataset risultante ha 296 feature: 256 per gli expert piu' 40 per il layer. Le matrici e le label vengono create da `ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py` con `--representation multihot_256_layer`.

- `arrays/`: matrici CSR `X_classic_<split>.npz` e label `y_<split>.npy` per train, validation e test.
- `artifacts/`: encoder del layer, nomi delle feature e riepilogo del preprocessing.
