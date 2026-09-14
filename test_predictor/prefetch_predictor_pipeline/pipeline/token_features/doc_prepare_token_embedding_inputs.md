# Input token per la FFN multi-input

## Scopo

`prepare_token_embedding_inputs.py` crea l'ablation `05` senza copiare o
modificare le matrici sparse dell'ablation `04`. Per ogni riga mantiene soltanto:

- l'indice compatto `int32` nel lookup GGUF congelato;
- `token_position`, standardizzata con media e deviazione standard calcolate
  esclusivamente sullo split train.

La FFN caricherà le feature sparse da `04` e questi due input aggiuntivi da
`05`. I 4.096 valori dell'embedding non vengono replicati per 1.025.400 righe.

## Output

```text
05_routing_layer_router_state_hidden_pca64_token_embedding/
├── arrays/
│   ├── token_embedding_index_train.npy
│   ├── token_embedding_index_validation.npy
│   ├── token_embedding_index_test.npy
│   ├── token_position_train.npy
│   ├── token_position_validation.npy
│   └── token_position_test.npy
└── artifacts/
    └── preprocessing_summary.json
```

L'indice punta alla riga di `token_embeddings_f16.npy`; il mapping è definito
da `token_ids.npy`. Il lookup di BUILD_DATASET_3 contiene gli ID osservati
nell'unione train/validation/test: questo non usa label né statistiche degli
split e non è leakage, perché l'embedding proviene dal modello GGUF congelato.

## Esecuzione per BUILD_DATASET_3

Da `~/ds4`:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/token_features/prepare_token_embedding_inputs.py \
  --training-parquet test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/training/prefetch_training.parquet \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/token_embedding_artifacts \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/04_routing_layer_router_state_hidden_pca64/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/05_routing_layer_router_state_hidden_pca64_token_embedding
```
