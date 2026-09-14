# `preprocess_dataset.py` — dal JSONL derivato a Parquet e array NumPy

## Scopo dello script

`preprocess_dataset.py` trasforma un dataset JSONL derivato in formati adatti
all'ispezione e al training:

1. una tabella Parquet, visualizzabile in formato tabellare in CLion;
2. matrici NumPy `.npy`, separate per train, validation e test;
3. un manifest che fissa l'ordine delle colonne feature.

È uno script generico per build che hanno già un JSONL completo. Gestisce sia
BUILD_DATASET_1, priva di hidden features, sia BUILD_DATASET_2 dopo la
proiezione PCA della feature 4.

## Input e output

| Argomento obbligatorio | Contenuto |
|---|---|
| `--input` | Dataset JSONL derivato da preprocessare. |
| `--split-by-prompt` | Mapping JSON `prompt_id -> train/validation/test`. |
| `--output-dir` | Directory in cui scrivere Parquet, array e manifest. |

L'output ha questa struttura:

```text
output-dir/
├── prefetch_training.parquet
├── feature_columns.json
└── arrays/
    ├── X_train.npy
    ├── X_validation.npy
    ├── X_test.npy
    ├── y_train.npy
    ├── y_validation.npy
    └── y_test.npy
```

## Flusso completo

```text
main()
  |
  +-- scan_dataset(input, splits)
  |     prima passata streaming: controlla il JSONL, conta gli esempi
  |     per split e determina la dimensione delle hidden feature
  |
  +-- load_splits(split_by_prompt)
  |     carica il mapping fisso prompt_id -> split
  |
  +-- seconda passata streaming
  |     appiattisce ogni record, scrive il Parquet in blocchi di 10.000 righe
  |     e popola direttamente gli array .npy per split su disco
  |
  `-- scrive feature_columns.json
```

## 1. Lettura streaming e conteggio iniziale

Il JSONL derivato viene letto in streaming: lo script non conserva mai tutti i
record in memoria. La prima passata verifica che il file non sia vuoto,
controlla Top-6, 16 logits e la dimensione coerente delle hidden feature, poi
conta gli esempi di train, validation e test. Questi conteggi permettono di
creare gli array `.npy` della dimensione esatta.

## 2. `load_splits()`

`load_splits(path)` legge `split_by_prompt.json` e restituisce il mapping
`prompt_id -> split`. Lo split è definito per prompt, non per riga: tutte le
righe originate dallo stesso prompt restano nello stesso split.

Questo evita leakage tra train e validation/test dovuto a token e layer molto
correlati appartenenti alla medesima sequenza generata.

## 3. Appiattimento e scrittura a blocchi

Durante la seconda passata, per ogni record lo script:

1. usa l'ultimo elemento di `routing_history.expert_ids` come routing corrente
   `E_l(t)`;
2. verifica Top-6 e la presenza di 16 raw logits;
3. legge eventuali `hidden_features.values` prodotti dalla PCA;
4. crea una riga tabellare leggibile, accumulata insieme a un massimo di
   9.999 altre righe prima di essere scritta nel Parquet;
5. scrive una riga di feature `X` e una label multi-hot `y` nella posizione
   corretta dello split, tramite array NumPy mappati su disco.

Tutti gli esempi devono avere la stessa dimensione di hidden feature. In
BUILD_DATASET_1 tale dimensione è `0`; dopo PCA può essere, ad esempio, `64`.
Questo approccio mantiene lo stesso formato di output, ma evita il picco di
memoria causato dal caricamento integrale di dataset molto grandi.

### Matrice `X`

Ogni riga di `X` contiene, in questo ordine:

```text
token_position, token_id, layer,
routing_history_expert_0 .. routing_history_expert_5,
router_logit_expert_id_0 .. router_logit_expert_id_15,
router_logit_value_0 .. router_logit_value_15,
hidden_pca_0 .. hidden_pca_(d-1)
```

La dimensione è quindi:

$$
3 + 6 + 16 + 16 + d = 41 + d
$$

dove `d` è il numero di componenti PCA della feature 4. BUILD_DATASET_1 ha
`d = 0` e dunque `X` con 41 colonne; con PCA-64, BUILD_DATASET_2 avrà 105
colonne prima degli encoding specifici del modello ML.

### Label `y`

`y` ha shape `[N, 256]` e tipo `uint8`. Per ciascuna riga gli ID presenti in
`target_experts_next_token` hanno valore `1`; tutti gli altri `0`. Ogni target
contiene sei `1`, uno per ciascun expert reale di $E_l(t+1)$.

Questa rappresentazione è adatta a un classificatore multi-label che assegna
uno score a ciascuno dei 256 expert e dal quale sarà poi estratta la Top-6.

## 4. Tabella Parquet e manifest

La tabella `prefetch_training.parquet` conserva anche metadati utili
all'ispezione: split, prompt, categoria, posizione token, layer, history,
`experts_t`, raw logits, target e — se presenti — valori `hidden_pca_*`.

`feature_columns.json` salva i nomi delle colonne che iniziano con:

```text
routing_
router_
hidden_
```

Il manifest permette agli step successivi di non ricostruire manualmente
l'ordine delle feature.

## Come eseguire lo script

Lo script richiede `numpy`, `pyarrow` e un motore Parquet, già dichiarati
nell'ambiente uv centrale. Dal root della repo:

```bash
cd ~/ds4
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/preprocess_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_pca64.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_1/ml/training/split_by_prompt.json \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_2/ml/training
```

Il mapping split di BUILD_DATASET_1 viene riutilizzato intenzionalmente anche
per BUILD_DATASET_2, perché i prompt sono gli stessi: questo rende confrontabili
le metriche tra le build. I file esistenti con lo stesso nome nella directory
di output vengono sovrascritti.
