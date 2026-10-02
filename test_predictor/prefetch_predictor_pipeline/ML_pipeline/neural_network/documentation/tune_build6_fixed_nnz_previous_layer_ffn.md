# Build 6 — selezione controllata della FFN con contesto del layer precedente

## Scopo

Lo script `neural_network/tune_build6_fixed_nnz_previous_layer_ffn.py` cerca
la migliore FFN Build 6 usando train e validation. Il test set non viene mai
usato durante la selezione.

Non usa Hyperband standard: applica una selezione controllata a tre fasi,
decisa esplicitamente per confrontare candidate iniziali e promozioni in modo
semplice. La Build 6 non contiene token embedding e conserva separati fino
alla fusione i rami del layer corrente `l` e del precedente `l-1`.

## Architettura generale

```text
state esteso 8.785D
  (state layer l + state layer l-1, con history, router e statistiche KV)
        |
        v
Dense(S) -> ReLU -> Dropout(0.3) ---------------------------+
                                                               |
raw hidden layer l 4.096D -> Dense(H_current) -> ReLU -------+
                                                               |
raw hidden layer l-1 4.096D -> Dense(H_previous) -> ReLU ----+
                                                               +--> Concatenate --> [Dense(F) -> ReLU] --> Dense(256) --> sigmoid
KV layer l 512D -> identita' oppure Dense(KV_current) ReLU --+       (expert_scores)
                                                               |
KV layer l-1 512D -> identita' oppure Dense(KV_previous) ReLU+
                                                               |
token_position standardizzata 1D -----------------------------+
```

`S`, `H_current`, `H_previous`, `KV_current`, `KV_previous` e `F` sono scelti
indipendentemente. Non esiste una concatenazione anticipata fra layer `l` e
`l-1`: la rete puo' apprendere un uso differente dei due contesti.

## Nota per la futura variante con `expert_transition_probabilities`

Lo script storico documentato qui **non** legge ancora la feature 7. La
variante futura per il fine tuning la aggiungera' come ramo separato,
preservando questo tuner e i suoi risultati immutati. Il nuovo input sara' il
vettore 256D gia' materializzato $\mathbf{q}_{t,l}$, non la matrice globale
$P_l(i\mid j)$: per ogni record contiene le probabilita' aggregate ottenute
dagli expert osservati al token corrente.

### Scala degli input

La FFN Build 6 non riceve tutte le feature sulla stessa scala, ma gli input
numerici sono gia' trattati in modo coerente durante il preprocessing:

- i raw hidden corrente e precedente sono normalizzati per componente con
  statistiche del solo train e poi quantizzati int8; il modello li dequantizza
  tramite il fattore di scala registrato;
- i KV completi sono i vettori `kv_normalized`, normalizzati nel preprocessing
  con statistiche train-only;
- `token_position` e' gia' standardizzata;
- lo state fixed-NNZ e le feature routing discrete sono rappresentazioni
  sparse/discrete trattate dagli embedding apprendibili, non vettori densi da
  standardizzare per colonna.

Per il ramo futuro `expert_transition_probabilities`, la scelta iniziale e'
fornire i valori originali float16, convertiti al dtype di calcolo del modello,
senza standardizzazione per componente. In questo modo le 256 componenti
restano interpretabili come probabilita' e la loro somma resta uno. Una prima
`Dense(64 o 128) -> ReLU` del ramo potra' apprendere la trasformazione utile
alla fusione. Questa scelta e' diversa dallo screening KNN: nel KNN la
standardizzazione temporanea e' necessaria per bilanciare le distanze
euclidee tra rami di scale differenti.

### Batch Normalization: ipotesi da isolare

Batch Normalization potrebbe stabilizzare l'ottimizzazione, in particolare
nel nuovo ramo probabilistico oppure dopo una proiezione dense dei raw hidden.
Non va pero' aggiunta automaticamente a tutti gli strati: cambierebbe insieme
architettura, regolarizzazione e comportamento rispetto al baseline, rendendo
ambigua l'interpretazione del risultato. La prima prova con la feature 7 deve
quindi usare il ramo `Dense -> ReLU` senza BatchNorm. Solo se emergono
convergenza instabile o scarso sfruttamento della feature, eseguire una
variante separata con BatchNorm immediatamente dopo la prima Dense del suo
ramo. I batch grandi previsti (`4096`) rendono questa prova tecnicamente
ragionevole, ma non sostituiscono un confronto controllato su validation.

## Spazio di ricerca

| Parte della rete | Scelte |
|---|---|
| State esteso | `8785D -> Dense(512, 1024 o 2048) -> ReLU -> Dropout(0.3)` |
| Raw hidden layer `l` | `4096D -> Dense(64, 512 o 1024) -> ReLU` |
| Raw hidden layer `l-1` | `4096D -> Dense(64, 512 o 1024) -> ReLU` |
| KV layer `l` | diretto `512D`, oppure `512D -> Dense(64, 128 o 256) -> ReLU` |
| KV layer `l-1` | diretto `512D`, oppure `512D -> Dense(64, 128 o 256) -> ReLU` |
| Dopo la concatenazione | nessun layer, oppure `Dense(512, 1024 o 2048) -> ReLU` |
| Learning rate Adam | `0.00003`, `0.00005`, `0.0001`, `0.00015` |

Lo spazio teorico contiene `3 x 3 x 3 x 4 x 4 x 4 x 4 = 6.912`
architetture. Lo script ne estrae soltanto 30, senza duplicati, con seed
riproducibile.

Parametri fissi:

```text
batch size             : 4096
shuffle block size     : 65536
dropout                : 0.3
L2                     : 0.0
policy TensorFlow      : float32
```

## Selezione in tre fasi

| Fase | Candidate | Budget massimo | Criterio di promozione |
|---|---:|---:|---|
| `phase_03` | 30 | 3 epoche | le migliori 15 passano a `phase_07` |
| `phase_07` | 15 | 7 epoche totali | le migliori 5 passano a `phase_20` |
| `phase_20` | 5 | 20 epoche totali | diventano le finaliste della conferma lunga |

Le candidate promosse non ripartono da zero: caricano il miglior checkpoint
della fase precedente, compresi pesi, iterazione e slot di Adam, e continuano
fino al budget della fase successiva. La classifica usa il massimo
`val_recall_at_6` osservato fino a quel punto, non solo l'ultima epoca.

L'early stopping con pazienza 3 puo' chiudere un training prima del budget
della fase. Non elimina automaticamente la candidata: il suo miglior valore
validation resta confrontabile e puo' comunque essere promosso. Nella prima
fase di sole tre epoche non interviene normalmente.

## Comando per lo screening

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/tune_build6_fixed_nnz_previous_layer_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/training/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/02_current_previous_fixed_nnz_state \
  --current-quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/05_current_hidden_int8 \
  --previous-quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/08_previous_hidden_int8 \
  --current-kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/09_current_kv_f16 \
  --previous-kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/10_previous_kv_f16 \
  --token-position-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/token_position_inputs \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/successive_halving_current_previous_hidden_kv_fusion_v1 \
  --dropout 0.3 \
  --batch-size 1024 \
  --shuffle-block-size 65536 \
  --early-stopping-patience 3
```

## Salvataggio e ripresa

```text
<output-dir>/
├── candidate_checkpoints/<candidate-id>/phase_03.model_and_adam.npz
├── candidate_checkpoints/<candidate-id>/phase_07.model_and_adam.npz
├── candidate_checkpoints/<candidate-id>/phase_20.model_and_adam.npz
├── candidate_progress/<candidate-id>/<phase>.jsonl
└── reports/
    ├── search_configuration.json
    ├── tuning_state.json
    ├── interruption_report.json
    └── tuning_report.json
```

Dopo ogni candidata conclusa viene aggiornato `tuning_state.json`; ogni epoca
scrive anche una riga JSONL. Se il processo viene interrotto, le fasi gia'
concluse restano valide. La candidata attiva viene ripetuta dalla checkpoint
stabile della fase precedente, senza ripetere tutte le altre candidate.

### Pulizia automatica dei checkpoint

I checkpoint possono essere grandi, specialmente per le architetture con
`Dense(2048)`. Ogni file `.model_and_adam.npz` contiene sia i pesi del modello
sia tutte le variabili dell'ottimizzatore Adam: iterazione, momenti del primo
ordine e momenti del secondo ordine. Questo permette a una candidata promossa
di continuare esattamente dal proprio migliore punto della fase precedente.

Per non saturare il disco, lo stato Adam e i pesi di una candidata scartata
vengono eliminati immediatamente dopo che la promozione e' stata decisa:

```text
dopo phase_03 : elimina i 15 checkpoint delle candidate non promosse
dopo phase_07 : elimina i 10 checkpoint delle candidate non promosse
                 e i phase_03 non piu' necessari delle 5 finaliste
dopo phase_20 : elimina i phase_03 e phase_07 residui delle finaliste
```

Alla fine restano quindi solo i cinque file
`candidate_checkpoints/<candidate-id>/phase_20.model_and_adam.npz`, uno per
finalista, oltre ai report leggeri JSON e JSONL. Durante `phase_03` esistono
temporaneamente i checkpoint completi delle 30 candidate, perche' la
promozione puo' essere decisa soltanto dopo averle misurate tutte. Lo spazio
liberato dagli intermedi float16 delle Build 4 e 5 viene riutilizzato dal
filesystem WSL per questo picco temporaneo; gli output non devono quindi essere
spostati fuori da `/home/michele/ds4`.

Durante una fase ancora in corso vengono conservati i checkpoint necessari per
poterla ripetere in sicurezza dopo un'interruzione. Le eliminazioni sono
annotate in `reports/tuning_state.json` sotto `checkpoint_cleanup`.

Per riprendere usare lo stesso comando e aggiungere:

```bash
--resume
```

La configurazione viene confrontata con `search_configuration.json` per non
mescolare risultati incompatibili.

### Rilascio delle risorse tra candidate

Alla conclusione, a errore o a interruzione di ogni singola candidata, il tuner
elimina il modello temporaneo, azzera la sessione Keras/TensorFlow e richiama il
garbage collector Python. I dataset Build 6 restano caricati e riutilizzabili;
vengono rilasciati soltanto grafo, cache e tensori della candidata ormai
conclusa.

Questa pulizia non altera checkpoint, stato Adam o possibilita' di ripresa. E'
necessaria per evitare che i grafi delle candidate precedenti si accumulino in
RAM e VRAM, facendo aumentare progressivamente il tempo per epoca durante uno
screening lungo come `30 -> 15 -> 5`.

## Dopo le 20 epoche

`reports/tuning_report.json` contiene le cinque finaliste. La fase successiva
non e' automatica: le candidate migliori saranno riaddestrate separatamente
fino a 1000 epoche, con early stopping di pazienza 5. Solo dopo tale scelta si
valutera' una configurazione una sola volta sul test set.

## Esito della selezione iniziale

La selezione `30 -> 15 -> 5`, completata senza interruzioni nella fase finale,
ha prodotto le seguenti cinque finaliste. I valori sono il massimo
`val_recall_at_6` osservato fino a 20 epoche, incluso il valore eventualmente
ereditato dalle fasi precedenti.

| Posizione | Candidata | Migliore Recall@6 validation | Epoche eseguite in `phase_20` |
|---:|---|---:|---:|
| 1 | `candidate_09` | **0.534194** | 13 |
| 2 | `candidate_17` | 0.530659 | 13 |
| 3 | `candidate_25` | 0.529976 | 11 |
| 4 | `candidate_20` | 0.528504 | 13 |
| 5 | `candidate_01` | 0.524595 | 9 |

La migliore configurazione iniziale e' `candidate_09`:

```text
state esteso:                         Dense(2048) -> ReLU -> Dropout(0.3)
raw hidden corrente 4096D:            Dense(512)  -> ReLU
raw hidden layer precedente 4096D:    Dense(64)   -> ReLU
KV corrente 512D:                     diretto
KV layer precedente 512D:             Dense(128)  -> ReLU
dopo la concatenazione:               Dense(2048) -> ReLU
learning rate Adam:                   0.0001
```

Questo valore validation supera il miglior Build 5 dopo retraining
(`0.5024`) di circa `0.0318` punti di Recall@6. Non e' ancora un risultato sul
test set: prima si riaddestra separatamente la candidata scelta e solo il
modello finale viene valutato una volta sul test set.
