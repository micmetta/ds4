# Prefetch predictor pipeline

Questa cartella contiene l'intero flusso sperimentale del prefetch predictor:
raccolta delle trace DS4, costruzione del dataset, preprocessing, baseline e
addestramento dei modelli ML. Il codice riutilizzabile è separato dagli
artefatti di ciascuna versione del dataset.

```text
prefetch_predictor_pipeline/
├── dataset_definition/       # corpus V0, schema e documentazione delle feature
├── collector/                # C: una inferenza DS4 deterministica per prompt
├── pipeline/                 # JSONL derivato, PCA e preprocessing generico
├── ML_pipeline/              # progetto uv, baseline e modelli ML riutilizzabili
├── FUTURE_IMPROVEMENTS_BEFORE_FINAL_MODEL.md  # correzioni da chiudere prima del modello finale
├── build_dataset_1/          # artefatti della raccolta con feature 0–3
│   ├── raw/
│   ├── derived/
│   └── ml/
│       ├── training/
│       └── classical_ml/
└── build_dataset_2/          # BUILD_1 + feature 4 (ffn_norm/PCA-64)
```

## Ruolo delle cartelle

- `dataset_definition/` contiene il corpus `prefetch_prompts_v0.jsonl` e la
  documentazione che definisce semanticamente le feature; non è un output di
  una build.
- `collector/` è il codice C condiviso. L'output è scelto con
  `DS4_DATASET_OUTPUT_DIR`; il Makefile lo compila con `make dataset_runner`.
- `pipeline/` contiene script Python generici che ricevono sempre input e
  output espliciti da riga di comando. Include `token_features/`, che estrae
  lookup compatti degli embedding token congelati dal GGUF per ablation FFN
  future.
- `ML_pipeline/` contiene codice, documentazione, `pyproject.toml`, `uv.lock`
  e l'ambiente `.venv`. Non contiene risultati di una build.
- ogni `build_dataset_N/` conserva soltanto raw trace, dataset derivati,
  artefatti PCA e risultati ML che appartengono a quella versione.
- `FUTURE_IMPROVEMENTS_BEFORE_FINAL_MODEL.md` e' il registro delle correzioni
  o semplificazioni da valutare e chiudere prima dell'addestramento finale.

## Versioni disponibili

| Versione | Feature raccolte |
|---|---|
| `build_dataset_1` | Feature 0–3: metadati prompt, routing history, raw router logits e token ID/embedding lookup. |
| `build_dataset_2` | Feature 0–3 più feature 4: `ffn_norm` raw compresso offline con PCA layer-wise a 64 componenti. |
| `build_dataset_3` | Corpus combinato V0+V1 da 500 prompt, con feature 4 PCA-64 e split per prompt preservato per V0. |

I risultati BUILD_1 sono stati conservati integralmente nella nuova struttura:
la riorganizzazione ha spostato i file senza rigenerare dataset, array, modelli
o report.
