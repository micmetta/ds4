# BUILD_2 — feature 4: hidden features PCA-64

BUILD_2 è **BUILD_1 + feature 4**. Prompt, feature 0–3, target, configurazione
deterministica e split per prompt restano invariati. La sola nuova informazione
è il tensor `ffn_norm` usato dal router:

```text
ffn_norm float32[4096] -> trace raw binario -> PCA train-only per layer -> 64 valori
```

## Raccolta

Compila il collector condiviso dalla root della repository:

```bash
make dataset_runner
```

La configurazione BUILD_2 usa:

```text
DS4_MODEL=<GGUF>
DS4_DATASET_OUTPUT_DIR=test_predictor/prefetch_predictor_pipeline/build_dataset_2/raw
DS4_DATASET_BUILD_ID=build_dataset_2
DS4_DATASET_CAPTURE_HIDDEN=1
DS4_DATASET_ROUTER_MAX_SCORES=16
DS4_DATASET_MAX_TOKENS=64
```

Il collector salva, per ogni prompt, routing JSONL, stdout/stderr e
`hidden/prompt_XXXX.ffn_norm.f32.bin`. Ogni vettore è `float32[4096]` (16 KiB);
il trace completo atteso è circa 3–4 GiB.

Un evento routed `decode` contiene il riferimento binario:

```json
"hidden_raw_ref": {
  "file": ".../prompt_0042.ffn_norm.f32.bin",
  "dtype": "float32",
  "shape": [4096],
  "offset_bytes": 4128768,
  "source_tensor": "ffn_norm"
}
```

I vettori di prefill non vengono salvati: BUILD_2 contiene solo esempi del
decoding autoregressivo.

## Pipeline offline

Gli script Python condivisi usano il progetto centrale `ML_pipeline/`. Una
sola volta, senza avviare raccolte o training:

```bash
cd test_predictor/prefetch_predictor_pipeline/ML_pipeline
uv sync
cd ../../..
```

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline test_predictor/prefetch_predictor_pipeline/pipeline/build_derived_dataset.py \
  --prompts test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl \
  --raw-dir test_predictor/prefetch_predictor_pipeline/build_dataset_2/raw \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_raw.jsonl \
  --history-length 1 --require-hidden
```

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/create_hidden_pca_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_raw.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_1/ml/training/split_by_prompt.json \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_hidden_pca64.jsonl \
  --artifacts-dir test_predictor/prefetch_predictor_pipeline/build_dataset_2/ml/hidden_pca_artifacts \
  --components 64
```

La PCA è separata per layer e viene fit-tata esclusivamente sui record train.
Gli artefatti `hidden_pca_layer_XX.npz` contengono media e componenti da
riutilizzare per validation, test e un futuro runtime. Sono salvati in
`ml/hidden_pca_artifacts/`, accanto agli altri artefatti ML della build: è
l'unica sotto-cartella aggiuntiva rispetto alla struttura comune di BUILD_1.

### 3. JSONL finale → Parquet e array

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline test_predictor/prefetch_predictor_pipeline/pipeline/preprocess_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_hidden_pca64.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_1/ml/training/split_by_prompt.json \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_2/ml/training
```

Questo produce un Parquet ispezionabile e array `[N, 105]`: le 41 feature di
BUILD_1 più le 64 coordinate PCA di `hidden_features`.

Nel JSONL finale ogni record contiene:

```json
"hidden_features": {
  "source_tensor": "ffn_norm",
  "projection": "pca",
  "projection_id": "pca_layerwise_64_train_v1",
  "input_dimension": 4096,
  "output_dimension": 64,
  "values": [0.184, -1.021, 0.337, ..., 0.912]
}
```

Nel file reale `values` contiene esattamente 64 numeri; l'ellissi è solo
illustrativa e non fa parte del JSON effettivamente scritto.
