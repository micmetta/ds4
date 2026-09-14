# `create_hidden_pca_dataset.py` — feature 4 con PCA layer-wise train-only

## Scopo dello script

`create_hidden_pca_dataset.py` trasforma i vettori raw della feature 4,
`ffn_norm`, in hidden feature PCA compatte da inserire nel dataset finale di
BUILD_DATASET_2.

Ogni vettore raw ha 4096 valori `float32` ed è il vettore normalizzato che il
router usa per calcolare i logits e selezionare gli expert del token corrente.
Salvarlo direttamente in ogni riga JSONL sarebbe molto costoso; lo script
impara quindi una proiezione PCA distinta per ogni layer routed e conserva,
per esempio, 64 valori:

$$
h^{\mathrm{router}}_{t,l} \in \mathbb{R}^{4096}
\quad\longrightarrow\quad
z_{t,l} \in \mathbb{R}^{64}
$$

La PCA viene calcolata esclusivamente sugli esempi train. Validation e test
sono solo trasformati con la stessa PCA già appresa, evitando data leakage.

## Input e output

| Argomento obbligatorio | Contenuto |
|---|---|
| `--input` | JSONL intermedio con `hidden_raw_ref`, prodotto da `build_derived_dataset.py --require-hidden`. |
| `--split-by-prompt` | Mapping fisso `prompt_id -> train/validation/test`. |
| `--output` | JSONL finale arricchito con `hidden_features`. |
| `--artifacts-dir` | Directory in cui salvare una PCA `.npz` per layer. |
| `--components` | Numero di componenti da conservare; default `64`. |
| `--batch-size` | Vettori raw elaborati insieme; default `4096`, deve essere almeno `--components`. |

Per evitare OOM sui corpus grandi, lo script non carica più tutto il JSONL in
RAM: indicizza prima le sole reference raw per layer, esegue fit e
trasformazione in batch e scrive il JSONL finale in streaming. Le directory
`--output` e `--artifacts-dir` devono essere nuove; in caso di run interrotta,
gli artefatti parziali vanno rimossi prima del rilancio.

## PCA incrementale: cosa significano i batch

I batch servono **solo** a limitare la memoria. Non viene creata una PCA
diversa per ogni gruppo di esempi e non viene conservato soltanto l'ultimo
batch. Per ogni layer routed, `IncrementalPCA` costruisce **un'unica**
proiezione usando tutti i vettori dello split train di quel layer.

Per esempio, se il layer 3 ha 41.325 vettori train e `--batch-size 4096`, il
fit procede su 11 batch. Dopo ogni batch, l'algoritmo aggiorna le statistiche
globali train (media e varianza per dimensione) e combina la decomposizione
già appresa con il nuovo batch tramite una SVD incrementale. Al termine
dell'undicesimo batch conserva una sola media globale e una sola matrice di 64
componenti per il layer 3. Il procedimento si ripete indipendentemente per
ogni layer da 3 a 42.

Questa è un'approssimazione memory-bounded della PCA che si otterrebbe
materializzando l'intera matrice train del layer in RAM. La proiezione finale
usa comunque tutti gli esempi train; la differenza rispetto a
`PCA(svd_solver="randomized")` è solo numerica e può dipendere dalla dimensione
del batch e dall'ordine dei dati. È il compromesso necessario per scalare al
corpus da 1.000 prompt senza OOM.

Solo **dopo** che tutti i batch train di un layer hanno aggiornato la sua PCA,
lo script usa quella proiezione finale per trasformare in batch tutte le righe
di quel layer: train, validation e test. Validation e test non chiamano mai
`partial_fit`, quindi non contribuiscono né alla media né alle componenti e non
introducono data leakage.

## L'output ha questa struttura

Con una proiezione PCA a 64 componenti, BUILD_DATASET_2 produrrà una struttura
simile a questa:

```text
build_dataset_2/
├── derived/
│   ├── prefetch_dataset_build_2_raw_hidden.jsonl
│   └── prefetch_dataset_build_2_pca64.jsonl
└── ml/
    └── hidden_pca_artifacts/
        ├── hidden_pca_layer_03.npz
        ├── hidden_pca_layer_04.npz
        ├── ...
        └── hidden_pca_layer_42.npz
```

Il JSONL `*_raw_hidden.jsonl` resta un artefatto intermedio: conserva
`hidden_raw_ref`, ovvero file binario, offset, tipo e shape del vettore 4096D.
Il JSONL `*_pca64.jsonl` è quello da passare a `preprocess_dataset.py`; in ogni
record sostituisce il riferimento raw con:

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

`values` contiene esattamente 64 numeri. Ogni file
`hidden_pca_layer_XX.npz` contiene media, componenti e varianza spiegata della
PCA necessaria a proiettare quel layer anche al runtime del predictor.

## Flusso completo

```text
main()
  |
  +-- load_records(input)
  |     verifica hidden_raw_ref float32[4096] in ogni record
  |
  +-- load_split_per_prompt(split_by_prompt)
  |     carica lo split fisso per prompt
  |
  +-- shard_references(...)
  |     scrive reference compatte separate per layer, senza trattenere i record
  |
  +-- fit_and_transform_layer(...)
  |     esegue una IncrementalPCA separata per ciascun layer, in batch
  |       |
  |       +-- HiddenReader.read(hidden_raw_ref)
  |       |     recupera un float32[4096] dal file binario e dall'offset indicato
  |       |
  |       +-- IncrementalPCA.partial_fit(train_batch)
  |       |     apprende media e 64 direzioni principali del solo train
  |       |
  |       +-- PCA.transform(all_layer_batch)
  |       |     proietta train, validation e test nella stessa base PCA
  |       |
  |       `-- np.savez_compressed(...)
  |             salva un artefatto PCA per il layer
  |
  `-- write_records(input, output)
        riscorre il JSONL e scrive il risultato finale senza hidden_raw_ref
```

## 1. `load_records()`

`load_records(path)` legge il JSONL intermedio e richiede per ogni record un
campo `hidden_raw_ref` valido:

```text
dtype = float32
shape = [4096]
```

Questa validazione protegge dalla lettura di offset incompatibili o da trace
raccolte senza la feature 4. Lo script rifiuta anche un JSONL vuoto.

## 2. `load_split_per_prompt()`

`load_split_per_prompt(path)` legge il mapping generato nel preprocessing di
BUILD_DATASET_1. Poiché BUILD_DATASET_2 usa gli stessi prompt V0, riusare il
medesimo split mantiene confrontabili i risultati tra le build.

La PCA riceve come train solo le righe il cui `prompt_id` è assegnato a
`"train"`. Validation e test non influenzano né la media né le componenti
principali.

## 3. `HiddenReader` e `record_matrix()`

`HiddenReader` apre in modo lazy i file binari referenziati da
`hidden_raw_ref` e li mantiene in cache finché servono. Il metodo `read()`:

1. cerca o apre il file indicato da `reference["file"]`;
2. si posiziona a `reference["offset_bytes"]`;
3. legge esattamente 4096 valori `float32`;
4. genera un errore se il vettore è incompleto.

I file sono chiusi da `HiddenReader.close()` anche in caso di errore, grazie al
blocco `finally`.

## 4. Fit e trasformazione memory-bounded

Per ogni layer lo script carica soltanto le sue reference compatte, seleziona
le righe train, quindi esegue `IncrementalPCA.partial_fit()` su batch di
vettori `float32[4096]`. Salva l'artefatto del layer e trasforma in batch tutte
le sue righe — train, validation e test — nella medesima base PCA. Infine una
passata streaming sul JSONL inserisce `hidden_features` e rimuove
`hidden_raw_ref`.

La PCA è quindi **layer-wise**: `W_17` e la media del layer 17 non vengono mai
usate per il layer 18. È anche **train-only**: i dati di validation e test non
contribuiscono all'apprendimento della proiezione.

### Artefatto `.npz` per layer

Ogni file `hidden_pca_layer_XX.npz` contiene:

| Chiave | Shape con PCA-64 | Significato |
|---|---:|---|
| `mean` | `[4096]` | Media $μ_l$ calcolata sul train del layer. |
| `components` | `[64, 4096]` | Le 64 righe della matrice PCA, equivalenti a $W_l^\top$. |
| `explained_variance_ratio` | `[64]` | Quanta varianza train spiega ciascuna componente. |

Per un nuovo hidden state raw, la trasformazione applicata è:

$$
z_{t,l} = W_l^\top\left(h^{\mathrm{router}}_{t,l} - \mu_l\right)
$$

Il valore `projection_id` identifica con chiarezza questa particolare famiglia
di proiezioni: numero di componenti, metodo layer-wise e train version.

## 5. `write_records()`

`write_records(records, output)` salva il dataset finale in JSONL compatto,
una riga per record. Crea la directory di output se manca e sostituisce il file
di destinazione se già esistente.

Il dataset finale non contiene più `hidden_raw_ref`: mantiene soltanto i valori
PCA compatti. I binari raw e gli artefatti PCA restano comunque su disco per
tracciabilità e per un eventuale utilizzo runtime.

## Come eseguire lo script

Lo script richiede `numpy` e `scikit-learn`, già definiti nell'ambiente uv di
`ML_pipeline`. Dal root della repo:

```bash
cd ~/ds4
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/create_hidden_pca_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_raw_hidden.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_1/ml/training/split_by_prompt.json \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_pca64.jsonl \
  --artifacts-dir test_predictor/prefetch_predictor_pipeline/build_dataset_2/ml/hidden_pca_artifacts \
  --components 64
```

Il comando non esegue inferenze DS4. Per sicurezza rifiuta output e artefatti
già esistenti: questo evita di mescolare una PCA parziale con una nuova run.
