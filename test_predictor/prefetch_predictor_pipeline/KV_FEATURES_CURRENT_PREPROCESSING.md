# KV features current: trasformazione offline

Questo documento descrive il primo esperimento con `kv_features_current` per il `build_dataset_5`. Non usa PCA sui vettori KV: l'eventuale proiezione sarà valutata e documentata separatamente in futuro.

## Input e output

L'input e' il derived dataset costruito dalle trace DS4:

```text
build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv.jsonl
```

Ogni record possiede `kv_raw_ref`, cioe' un riferimento a una riga `kv` float32 di 512 dimensioni catturata dopo RMSNorm e RoPE. Lo script `pipeline/kv_features/create_kv_current_dataset.py` legge una riga alla volta e produce un nuovo JSONL. I file raw KV e il derived dataset di input non vengono modificati.

## Feature calcolate

Per ogni vettore corrente \(u_{t,l}\in\mathbb{R}^{512}\), vengono aggiunti al record:

- `l2_norm = ||u_{t,l}||_2`;
- `rms = sqrt(mean(u_{t,l}^2))`;
- `mean`;
- `std`, deviazione standard di popolazione (`ddof=0`);
- `max_abs = max(abs(u_{t,l}))`.

Il JSONL conserva `kv_raw_ref` per eventuali analisi o proiezioni future e aggiunge:

```json
"kv_features_current": {
  "source_tensor": "kv",
  "source_stage": "after_rmsnorm_rope",
  "input_dimension": 512,
  "stats_current": {
    "l2_norm": 0.0,
    "rms": 0.0,
    "mean": 0.0,
    "std": 0.0,
    "max_abs": 0.0
  }
}
```

## Assenza di leakage

Queste cinque feature sono statistiche della sola riga KV corrente. Non sono apprese e non richiedono fit su train, validation o test: il loro calcolo e' identico per ogni esempio e usa solo informazioni disponibili al token \(t\).

La normalizzazione necessaria per un modello ML sara' invece fit **solo sul train** nel successivo preprocessing/ablation; non e' svolta da questo script. Analogamente, questo step non costruisce una PCA.

## Esecuzione

Da `~/ds4`:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/kv_features/create_kv_current_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv.jsonl \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl
```

Lo script scrive anche un report JSON accanto all'output, con numero di record, definizione delle feature e intervalli min/max osservati. L'output deve essere nuovo: questo evita di sovrascrivere artefatti prodotti da run precedenti.

## Integrazione nel preprocessing ML

`pipeline/preprocess_dataset.py` riconosce automaticamente `kv_features_current` e aggiunge al Parquet le cinque colonne:

```text
kv_current_l2_norm
kv_current_rms
kv_current_mean
kv_current_std
kv_current_max_abs
```

L'ablation dedicata e' `multihot_256_layer_router_state_kv_current`. Mantiene tutte le feature dello stato router usate nell'ablation 03 e aggiunge le cinque statistiche KV. Le statistiche KV non vengono normalizzate nel JSONL: `StandardScaler` viene fit **esclusivamente sul train** nell'ablation e poi applicato invariato a validation e test. Lo scaler risultante viene salvato in `artifacts/kv_current_scaler.joblib`.

Esecuzione, dopo aver creato il JSONL con le feature KV:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/preprocess_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training

uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256_layer_router_state_kv_current \
  --training-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current
```

La PCA dei vettori KV raw 512D non fa parte di questo step.

## Vettore KV completo per la rete

Oltre alle cinque statistiche, la prima rete KV usa la riga compatta completa da 512 componenti. Lo script `pipeline/kv_features/materialize_normalized_kv_inputs.py` legge i riferimenti `kv_raw_ref` del JSONL, verifica l'allineamento con gli array state e produce un array contiguo per split. Questo evita di rileggere file binari con accessi casuali a ogni batch di training.

La normalizzazione e' esatta e train-only: per ogni layer e per ciascuna delle 512 componenti, lo script calcola media e deviazione standard su **tutte** le righe train. Applica poi gli stessi parametri a train, validation e test. Gli artefatti sono `kv_train_mean_by_layer.npy` e `kv_train_std_by_layer.npy`. Non c'e' leakage: validation e test non contribuiscono al fit delle statistiche.

L'output `kv_normalized_{split}.npy` e' `float16`, ha shape `(numero_esempi_split, 512)` ed e' nello stesso ordine delle righe state. Il vettore viene passato direttamente alla concatenazione della rete: non viene applicata PCA ne' un layer `Dense(512 -> 64)`.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/kv_features/materialize_normalized_kv_inputs.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --batch-size 4096
```
