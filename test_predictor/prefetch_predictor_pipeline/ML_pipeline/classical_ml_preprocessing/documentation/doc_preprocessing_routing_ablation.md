# Ablation controllata: routing history H=1 e router state

## Obiettivo

Questo esperimento verifica quanto i modelli classici riescano a predire

$$
E_l(t+1)
$$

usando esclusivamente il routing dello stesso layer osservato al token corrente:

$$
E_l(t)
$$

Il corpus, gli split per prompt, le label multi-label, i quattro modelli e la
metrica Recall@6 restano gli stessi di BUILD_DATASET_2. Cambia soltanto la
rappresentazione dell'input. La baseline deterministica copia direttamente
`E_l(t)` e ha ottenuto Recall@6 validation pari a `0.4036`.

## Le due rappresentazioni

### 1. `slots_onehot`

Usa soltanto:

```text
routing_history_expert_0 ... routing_history_expert_5
```

I sei ID sono passati a un `OneHotEncoder` fit soltanto sul train con la stessa
configurazione dell'esperimento classico completo:

```python
OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=41,
    sparse_output=True,
)
```

L'ordine Top-6 è conservato. Per esempio, `[96, 119, 202, 112, 90, 15]`
attiva sei feature poste in blocchi diversi: `slot_0_96`, `slot_1_119`, ...,
`slot_5_15`. Lo stesso expert in uno slot diverso è una feature differente.

### 2. `multihot_256`

Costruisce direttamente una matrice sparse di 256 colonne:

```text
routing_history_multihot_expert_0 ... routing_history_multihot_expert_255
```

Gli ID presenti in `E_l(t)` valgono `1`; tutti gli altri valgono `0`. Ogni riga
ha esattamente sei valori attivi. L'ordine Top-6 viene eliminato: routing con
gli stessi sei expert in un ordine diverso producono lo stesso vettore. Questa
è l'informazione esatta sfruttata dalla baseline di persistenza.

## Script e flusso

Lo script dedicato è:

```text
ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py
```

```text
main()
  |
  +-- leggi_argomenti()
  |     legge rappresentazione, training directory e output directory
  |
  +-- carica_tabella_base()
  |     legge split, routing history, layer e router state dal Parquet BUILD_DATASET_2
  |
  +-- carica_e_verifica_label()
  |     riusa e verifica le stesse y_[split].npy con shape [N, 256]
  |
  +-- slots_onehot:
  |     crea_encoder_slot() -> fit train -> transform di tutti gli split
  |
  +-- multihot_256:
  |     crea_multihot() -> una CSR [N, 256] con sei valori 1 per riga
  |
  +-- multihot_256_layer_router_state:
  |     fit train di OneHotEncoder + StandardScaler -> CSR con router state
  |
  +-- salva_output()
        salva X_classic_[split].npz, y_[split].npy e gli artefatti
```

Lo script non esegue alcun addestramento. Riutilizza direttamente le label
prodotte da `pipeline/preprocess_dataset.py`, quindi input e target restano
allineati riga per riga con BUILD_DATASET_2.

## Output

```text
build_dataset_2/ml/ablations/
├── routing_slots_onehot/
│   ├── arrays/X_classic_{train,validation,test}.npz
│   ├── arrays/y_{train,validation,test}.npy
│   ├── artifacts/categorical_encoder.joblib
│   ├── artifacts/feature_names.json
│   └── artifacts/preprocessing_summary.json
└── routing_multihot_256/
    ├── arrays/X_classic_{train,validation,test}.npz
    ├── arrays/y_{train,validation,test}.npy
    ├── artifacts/feature_names.json
    └── artifacts/preprocessing_summary.json
```

Entrambe le cartelle sono compatibili con `train_classical_models.py`: basta
passare la loro sottocartella `arrays/` tramite `--data-dir` e usare cartelle
`models/` e `reports/` interne alla stessa ablation. In questo modo non si
sovrascrivono i risultati dell'esperimento completo né quelli dell'altra
ablation.

## Comandi di preprocessing

Dalla cartella `ML_pipeline/`:

```bash
uv run classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation slots_onehot \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/ablations/routing_slots_onehot
```

```bash
uv run classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256 \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/ablations/routing_multihot_256
```

Prima di addestrare, controllare in `artifacts/preprocessing_summary.json`:

- le righe degli split devono restare `155560 / 20600 / 21280`;
- le label devono avere shape `[N, 256]`;
- nel multi-hot il numero di valori non nulli deve essere `6 × N_split`;
- nello slot one-hot ogni riga deve avere sei feature categoriche attive.

## Training controllato

Usare gli stessi modelli e iperparametri in entrambe le ablation. Esempio per
la prima rappresentazione:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models logistic_regression,decision_tree,random_forest,knn \
  --data-dir ../build_dataset_2/ml/ablations/routing_slots_onehot/arrays \
  --models-dir ../build_dataset_2/ml/ablations/routing_slots_onehot/models \
  --reports-dir ../build_dataset_2/ml/ablations/routing_slots_onehot/reports \
  --logistic-max-iter 5000 \
  --logistic-n-jobs -1
```

Per la seconda rappresentazione sostituire `routing_slots_onehot` con
`routing_multihot_256`. Confrontare inizialmente solo Recall@6 su train e
validation; il test resta separato fino alla scelta finale.

## Routing multi-hot + layer

La rappresentazione `multihot_256_layer` materializza la configurazione
`02_routing_multihot_layer` dello screening KNN. Mantiene il routing corrente
come vettore multi-hot e aggiunge il layer routed corrente come one-hot fit
soltanto sul train:

```text
256 feature routing multi-hot + 40 feature layer one-hot = 296 feature
```

Questo dataset persistente serve, in particolare, a confrontare una FFN con il
migliore sottoinsieme compatto identificato dal KNN. Il comando e':

```bash
uv run classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256_layer \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/ablations/02_routing_multihot_layer
```

Le matrici prodotte mantengono il nome classico `X_classic_<split>.npz` e sono
accettate anche da `neural_network/train_ffn.py` tramite `--data-dir`.

## Routing multi-hot + layer + router state

La rappresentazione `multihot_256_layer_router_state` materializza la
configurazione `03_routing_layer_router_state` dello screening KNN. Per
predire `E_l(t+1)`, usa sempre informazioni del token precedente `E_l(t)` e
dello stesso layer `l`:

- 256 feature multi-hot degli expert in `E_l(t)`;
- 40 feature one-hot del layer `l`;
- ID degli expert nei 16 raw router logits osservati in `E_l(t)`;
- 16 valori dei raw router logits, standardizzati con `StandardScaler` fit
  soltanto sul train.

Gli ID dei logits usano una codifica one-hot fissa: ogni slot ha sempre 256
colonne, una per ciascun expert ID da `0` a `255`. Questa codifica non dipende
dalle categorie presenti nel train e rappresenta quindi anche un expert ID che
compaia per la prima volta in validation o test. Il layer usa invece il proprio
`OneHotEncoder` fit soltanto sul train.

Di conseguenza, in `preprocessing_summary.json` la configurazione
`handle_unknown="infrequent_if_exist"` puo' riferirsi solo al campo
`layer_one_hot_configuration`: non si applica mai ai 16 ID router. Gli ID
router sono descritti separatamente da `router_state.router_id_encoding` come
`fixed_one_hot_per_slot`, con 4.096 feature fisse.

In BUILD_DATASET_2 la matrice ha 4.408 feature:

```text
256 routing + 40 layer + (16 slot x 256 expert ID) + 16 valori = 4.408
```

Le 4.096 feature degli ID derivano da `16 x 256`: per ognuno dei 16 slot dei
raw logits esistono 256 colonne, una per ogni possibile expert ID da `0` a
`255`. Le 40 feature del layer corrispondono ai 40 routed layer presenti in
BUILD_DATASET_2. Lo script salva `categorical_encoder.joblib` per il layer,
`router_id_encoder.joblib` per la codifica fissa degli ID e
`numeric_scaler.joblib` per i valori dei logits.

I 16 valori `router_logit_value_0` fino a `router_logit_value_15` sono
standardizzati separatamente colonna per colonna. `StandardScaler` calcola
media e deviazione standard di ogni posizione del logit usando soltanto il
train; gli stessi parametri sono poi applicati invariati a validation e test.
Gli ID degli expert non sono standardizzati: restano feature one-hot.

```bash
uv run classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256_layer_router_state \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/ablations/03_routing_layer_router_state
```

## Interpretazione

Ci aspettiamo che `multihot_256` sia la rappresentazione più vicina alla
baseline, perché entrambi usano soltanto l'appartenenza degli expert a
`E_l(t)`. Non è comunque una garanzia matematica: la baseline copia gli ID con
una regola deterministica, mentre i modelli devono stimare 256 probabilità
one-vs-rest e poi selezionare i sei score maggiori.

## Routing multi-hot + layer + router state + hidden PCA-64

La rappresentazione
`multihot_256_layer_router_state_hidden_pca64` aggiunge alla configurazione
`03` le 64 colonne `hidden_pca_0` fino a `hidden_pca_63` già presenti nel
Parquet BUILD_DATASET_3. Esse sono la PCA layer-wise del vettore `ffn_norm`
raw di 4.096 dimensioni raccolto al token corrente e nello stesso layer.

Le componenti PCA sono feature numeriche dense; lo script applica un secondo
`StandardScaler`, fit soltanto sullo split train, e riusa invariati media e
deviazione standard per validation e test. Il router state resta invariato:
gli ID dei suoi 16 raw logits mantengono il one-hot fisso da 4.096 feature.

```text
256 routing multi-hot
+ 40 layer one-hot
+ 4096 ID router logits: 16 slot x 256 ID fissi
+ 16 valori router logits standardizzati sul train
+ 64 hidden PCA standardizzate sul train
= 4472 feature
```

La nuova rappresentazione non richiede una nuova inferenza DS4 né un nuovo
fit PCA: richiede soltanto il Parquet derivato dalla build che contiene già le
colonne `hidden_pca_*`. Produce una nuova cartella di ablation, senza
sovrascrivere `02` o `03`, e salva anche
`artifacts/hidden_pca_scaler.joblib`.

Esempio per BUILD_DATASET_3, dalla root della repository:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256_layer_router_state_hidden_pca64 \
  --training-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/training \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/04_routing_layer_router_state_hidden_pca64
```
