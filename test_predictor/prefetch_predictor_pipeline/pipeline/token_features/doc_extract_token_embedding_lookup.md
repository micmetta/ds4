# Lookup compatto degli embedding token dal GGUF

## Scopo

`extract_token_embedding_lookup.py` prepara l'artefatto necessario per usare
gli embedding token pre-addestrati del modello DeepSeek nella futura FFN
multi-input. Non esegue inferenza DS4, non modifica il Parquet e non usa label
o statistiche fit sul dataset.

Il GGUF contiene il tensore congelato `token_embd.weight`. Nel modello usato
per BUILD_DATASET_3 la sua forma effettiva e':

```text
129280 token x 4096 dimensioni, float16
```

Il tensor completo occupa circa 1,06 GB. Lo script evita di copiarlo nella
pipeline ML: estrae soltanto le righe corrispondenti ai `token_id` distinti
presenti nel Parquet scelto.

## Rappresentazione futura

```text
token_id -> lookup GGUF congelato (4096D) -> Dense allenabile (4096 -> 64)
```

Il vettore 64D risultante verra' concatenato alle altre feature disponibili al
token e layer corrente. I pesi della proiezione lineare saranno supervisionati
dal task di prediction del routing; la tabella GGUF restera' congelata.

Gli embedding congelati esistono per tutti gli ID del vocabolario, compresi
quelli mai osservati nel train. Questo evita un embedding casuale o un OOV
unico per i token non visti.

## Input

| Argomento | Contenuto |
|---|---|
| `--training-parquet` | Parquet preprocessed che contiene `token_id`. Può includere train, validation e test: non vengono lette label. |
| `--gguf-model` | Stesso file GGUF usato dal collector DS4. |
| `--output-dir` | Nuova directory per gli artefatti. Lo script rifiuta directory esistenti. |
| `--tensor-name` | Nome del tensore GGUF; default `token_embd.weight`. |

Leggere embedding congelati di ID presenti in validation/test non e' leakage:
la tabella appartiene al modello sorgente ed e' disponibile anche a runtime;
non e' stimata da label, target o statistiche degli split.

## Output

```text
token_embedding_artifacts/
├── token_ids.npy
├── token_embeddings_f16.npy
└── token_embedding_lookup_summary.json
```

- `token_ids.npy`: ID raw distinti, `int32`, ordinati in modo crescente.
- `token_embeddings_f16.npy`: matrice `N x 4096` float16; la riga `i`
  corrisponde al token `token_ids[i]`.
- `token_embedding_lookup_summary.json`: sorgente GGUF, forma, dtype,
  vocabolario, intervallo degli ID estratti e contratto del mapping.

Il futuro loader FFN usera' `numpy.searchsorted(token_ids, raw_token_id)` per
ottenere l'indice compatto dell'embedding. Non verranno scritti 4096 valori per
ognuna delle oltre un milione di righe del dataset.

## Esecuzione per BUILD_DATASET_3

Da `~/ds4`:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/token_features/extract_token_embedding_lookup.py \
  --training-parquet test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/training/prefetch_training.parquet \
  --gguf-model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/token_embedding_artifacts
```

L'artefatto non sostituisce le ablation esistenti `03` e `04`: e' un input
aggiuntivo per una futura ablation FFN dedicata a `token_id` e
`token_position`.
