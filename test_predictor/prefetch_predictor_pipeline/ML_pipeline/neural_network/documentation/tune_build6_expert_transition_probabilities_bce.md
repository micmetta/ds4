# Tuning Build 6 con expert-transition probabilities e BCE

## Scopo

Questo esperimento e' separato dai tuner storici. Usa la stessa FFN Build 6 a
rami separati, aggiungendo il vettore causale `expert_transition_probabilities`
di 256 componenti. Il vettore contiene i valori \(q_{t,l}(i)\) materializzati
dal train e allineati agli altri array di input.

La loss e' esclusivamente binary cross-entropy. Lo script forza
`--ranking-loss-weight 0.0`: la loss combinata BCE + ranking non entra in
questo esperimento.

## Spazio di ricerca

Le 30 candidate sono estratte in modo riproducibile con `RANDOM_STATE` e
passano per il successive halving 30 -> 15 -> 5, alle fasi 3, 7 e 20 epoche.

| Ramo | Scelte |
| --- | --- |
| State esteso | Dense(512, 1024, 2048) -> ReLU -> Dropout(0.3) |
| Raw hidden layer l | Dense(64, 512, 1024) -> ReLU |
| Raw hidden layer l-1 | Dense(64, 512, 1024) -> ReLU |
| KV layer l | diretto 512D, oppure Dense(64, 128, 256) -> ReLU |
| KV layer l-1 | diretto 512D, oppure Dense(64, 128, 256) -> ReLU |
| Expert transition probabilities 256D | diretto, Dense(128) -> ReLU, Dense(64) -> ReLU |
| Dopo la concatenazione | nessun layer, oppure Dense(512, 1024, 2048) -> ReLU |
| Adam learning rate | 0.00003, 0.00005, 0.0001, 0.00015 |

Il valore `diretto` del nuovo ramo concatena le 256 probabilita' senza pesi
aggiuntivi: e' il controllo che consente di misurare se una proiezione appresa
porta un vantaggio reale.

## Input richiesto

Il parametro `--expert-transition-probabilities-data-dir` deve indicare la root
creata dal materializzatore, per esempio:

```text
build_dataset_6/ml/transition_features/expert_transition_probabilities_v1
```

Lo script carica internamente gli array float16:

- `arrays/expert_transition_probabilities_train.npy`;
- `arrays/expert_transition_probabilities_validation.npy`.

I valori non sono standardizzati prima della FFN: restano probabilita' e il
ramo Dense, quando presente, apprende la trasformazione utile al task.

## Ripresa sicura

Il report `reports/search_configuration.json` registra anche il data directory
della feature e lo spazio del ramo nuovo. `--resume` e' consentito solo se il
comando ricostruisce esattamente la stessa configurazione. I checkpoint
includono pesi e stato Adam ai punti di promozione.

## Risultati dell'esperimento `v1`

Il tuning e' stato eseguito in Colab e il suo report e checkpoint sono stati
salvati in:

```text
DS4_Colab/build6_results/expert_transition_bce_v1/
```

Il riferimento puntuale della selezione e' il file:

```text
DS4_Colab/build6_results/expert_transition_bce_v1/reports/tuning_report.json
```

La finalista selezionata dal report e' `candidate_10`. La sua architettura
effettivamente ricostruita e' la seguente:

| Componente | Configurazione selezionata |
| --- | --- |
| State fixed-NNZ | 2048D -> ReLU -> Dropout(0.3) |
| Raw hidden layer `l` | 4096D -> Dense(64) -> ReLU |
| Raw hidden layer `l-1` | 4096D -> Dense(64) -> ReLU |
| KV layer `l` | 512D -> Dense(256) -> ReLU |
| KV layer `l-1` | 512D -> Dense(256) -> ReLU |
| `expert_transition_probabilities` | 256D diretto, senza proiezione Dense |
| Fusione | 2945D -> Dense(2048) -> ReLU -> Dropout(0.3) -> output sigmoid 256D |
| Adam | learning rate `0.00015` |
| Loss | Binary cross-entropy classica |

### Schema visuale della candidata `candidate_10`

```text
state fixed-NNZ 8.785D
    |
    v
Embedding pesata -> 2048D -> ReLU -> Dropout(0.3) -------------------+
                                                                        |
raw hidden layer l 4096D -> dequantizzazione -> Dense(64) -> ReLU ----+
                                                                        |
raw hidden layer l-1 4096D -> dequantizzazione -> Dense(64) -> ReLU --+
                                                                        |
KV layer l 512D -> Dense(256) -> ReLU --------------------------------+
                                                                        +--> Concatenate (2945D)
KV layer l-1 512D -> Dense(256) -> ReLU ------------------------------+          |
                                                                                   v
token_position standardizzata 1D -------------------------------------+   Dense(2048) -> ReLU
                                                                                   |
expert_transition_probabilities q_{t,l} 256D, diretto ----------------+   Dropout(0.3)
                                                                                   |
                                                                                   v
                                                                        Dense(256) -> sigmoid
                                                                             (expert_scores)
```

La configurazione registrata dal tuner per `previous_kv_projection` e'
`direct`; nella rete effettivamente costruita appare pero' una Dense(256),
perche' il builder storico applica alla proiezione precedente il valore della
proiezione KV corrente quando il valore precedente e' `None`. La tabella sopra
documenta quindi l'architettura realmente eseguita, verificabile dal
`model.summary()` della run.

| Misura | Validation Recall@6 |
| --- | ---: |
| Migliore valore fra le finaliste alla phase_20 | `0.523543` |
| `candidate_10`, continuata dal checkpoint phase_20 | `0.531629` |
| Modello BCE storico migliore, senza questa feature | circa `0.5369` |

La continuazione e' partita dall'epoch 20 ripristinando sia pesi sia stato
Adam, e si e' fermata all'epoch 46 con early stopping `patience=5`. Il relativo
modello e report sono in:

```text
DS4_Colab/build6_results/expert_transition_bce_v1_best_long/
DS4_Colab/build6_results/expert_transition_bce_v1_best_long/reports/training_report.json
```

Questa prima integrazione non batte quindi il baseline FFN BCE migliore, anche
se la nuova feature risulta predittiva se usata isolatamente.

## Ablazione post-addestramento dei rami

Per distinguere il potenziale della feature dal suo effettivo uso da parte della
FFN, il modello lungo di `candidate_10` e' stato valutato senza riaddestrarlo.
Il report locale e':

```text
build_dataset_6/ml/neural_network/expert_transition_bce_v1_best_long/
reports/neural_branch_ablation_validation_test.json
```

Per ogni riga viene alterato **un solo ramo alla volta**; tutti gli altri
restano identici al baseline. `zero` sostituisce il ramo con zeri, mentre
`permutation` ne permuta in modo riproducibile le righe e conserva quindi la
distribuzione marginale, ma distrugge l'associazione con l'esempio e il target
correnti. Il delta e' `Recall@6 completa - Recall@6 ablata`.

| Ramo alterato | Perdita validation, zero | Perdita validation, permutazione | Perdita test, zero | Perdita test, permutazione |
| --- | ---: | ---: | ---: | ---: |
| Baseline completa | `0.531629` | — | `0.535918` | — |
| State fixed-NNZ | `-0.380835` | `-0.478298` | `-0.380582` | `-0.482305` |
| Raw hidden `l` | `-0.039180` | `-0.059372` | `-0.040525` | `-0.059645` |
| Raw hidden `l-1` | `-0.022068` | `-0.035706` | `-0.023088` | `-0.036432` |
| KV `l` | `-0.013099` | `-0.019120` | `-0.013603` | `-0.019626` |
| KV `l-1` | `-0.012335` | `-0.018426` | `-0.012665` | `-0.018592` |
| Token position | `-0.000071` | `-0.000233` | `-0.000082` | `-0.000117` |
| `expert_transition_probabilities` 256D | `-0.000099` | `-0.000075` | `-0.000087` | `-0.000087` |

I valori negativi nella tabella sono riportati come perdita di Recall@6
rispetto al baseline, cioe' il valore assoluto di `drop_from_baseline` del
report. Lo state fixed-NNZ e' di gran lunga il ramo dominante; seguono raw
hidden corrente, raw hidden precedente e i due KV. Token position ha un effetto
quasi nullo in questa candidata.

Il risultato decisivo riguarda `expert_transition_probabilities`: sia
l'azzeramento sia la permutazione cambiano Recall@6 di meno di `0.0001` su
entrambi gli split. La FFN della candidata 10 sta quindi, in pratica,
**ignorando il ramo 256D**. Questo non contraddice il KNN o il baseline `q`
only: la feature contiene informazione predittiva, ma la semplice
concatenazione diretta non ha indotto questa architettura BCE a sfruttarla.
Il prossimo miglioramento deve quindi riguardare una fusione piu' mirata del
ramo `q`, non la sola aggiunta della feature ai dati.

## Baseline della sola feature di transizione

Lo script senza addestramento
`pipeline/transition_features/evaluate_expert_transition_probabilities_baseline.py`
prende, per ogni esempio, i sei indici con componenti maggiori di
`q_{t,l}` e li confronta direttamente con i sei expert target. Il report
locale e' in:

```text
build_dataset_6/ml/transition_features/expert_transition_probabilities_v1/
reports/top6_transition_only_baseline_train_validation_test.json
```

| Split | Recall@6 | Expert target recuperati in media |
| --- | ---: | ---: |
| Train out-of-fold | `0.363000` | `2.178 / 6` |
| Validation | `0.365505` | `2.193 / 6` |
| Test | `0.369350` | `2.216 / 6` |

La coerenza train out-of-fold / validation / test e' un controllo pratico
contro il leakage: i vettori del train sono stimati con matrici P costruite
senza il fold dell'esempio valutato. La feature da sola e' quindi informativa,
ma la sua fusione diretta nella candidata `candidate_10` non ha ancora
migliorato il modello completo.

## Ablation pianificata: Batch Normalization dopo la concatenazione

La prima integrazione della feature aggiunge direttamente 256 probabilita'
alla concatenazione. Queste componenti hanno scala e distribuzione diverse dai
rami hidden e KV: sono probabilita' non negative, mentre le rappresentazioni
Dense ReLU possono avere media e varianza molto differenti. Per verificare se
questa differenza di scala ostacola la fusione, si esegue una sola ablation
sulla migliore finalista `candidate_10`.

La prova conserva tutti gli iperparametri della candidata selezionata e usa
sempre BCE classica; aggiunge esclusivamente `BatchNormalization` dopo
`Concatenate` e prima della Dense di fusione. Durante il train normalizza ogni
componente con le statistiche del mini-batch; durante validation/inferenza usa
le statistiche mobili apprese sul train.

```text
state fixed-NNZ 8.785D -> Embedding pesata -> 2048D -> ReLU -> Dropout(0.3) --+
raw hidden l 4096D -> dequantizzazione -> Dense(64) -> ReLU ------------------+
raw hidden l-1 4096D -> dequantizzazione -> Dense(64) -> ReLU ----------------+
KV l 512D -> Dense(256) -> ReLU ----------------------------------------------+
KV l-1 512D -> Dense(256) -> ReLU --------------------------------------------+--> Concatenate (2945D)
token_position standardizzata 1D ----------------------------------------------+          |
q_{t,l} expert_transition_probabilities 256D diretto -------------------------+          v
                                                                     BatchNormalization
                                                                              |
                                                                              v
                                                                  Dense(2048) -> ReLU
                                                                              |
                                                                         Dropout(0.3)
                                                                              |
                                                                              v
                                                                  Dense(256) -> sigmoid
                                                                       (expert_scores)
```

Batch Normalization non rende i rami semanticamente equivalenti e non impone
uguale importanza: la Dense successiva resta libera di assegnare pesi diversi.
Evita solo che un vantaggio numerico di scala o varianza condizioni a priori la
prima fusione. Non e' possibile riutilizzare il checkpoint di `candidate_10`,
perche' la BN introduce parametri allenabili e statistiche mobili nuove;
l'ablation va quindi addestrata da zero con stesso seed, stessa candidata,
stessi dati, stessi limiti di epoche e stesso early stopping.
