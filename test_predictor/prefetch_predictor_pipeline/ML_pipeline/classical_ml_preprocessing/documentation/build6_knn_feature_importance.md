# Importanza rapida delle feature Build 6 con KNN

`run_build6_knn_feature_importance.py` applica alla Build 6 lo stesso
screening leave-one-group-out usato per Build 5. Misura il **segnale
predittivo sfruttabile da un KNN**, non l'importanza dei pesi della futura FFN
Build 6.

## Metodo

1. Estrae sottoinsiemi riproducibili di train e validation.
2. Costruisce tutti i rami disponibili nella Build 6, senza token embedding.
3. Addestra un KNN completo.
4. Riaddestra lo stesso KNN eliminando un gruppo alla volta.
5. Ordina i gruppi per perdita di Recall@6 validation:
   `Recall@6 completo - Recall@6 senza gruppo`.

Un delta positivo elevato indica che il gruppo contiene segnale utile per quel
KNN. Un valore vicino a zero o negativo indica ridondanza o una geometria
delle distanze sfavorevole; non prova che la futura FFN non possa sfruttarlo.

## Gruppi confrontati

Build 6 confronta separatamente le informazioni correnti del layer `l` e
quelle del layer MoE precedente `l-1`, sempre riferite allo stesso token `t`.

| Ramo | Gruppi |
|---|---|
| State corrente | routing history, layer, ID router, valori router, cinque statistiche KV |
| State precedente | routing history, ID router, valori router, cinque statistiche KV |
| Vettori densi correnti | raw hidden 4.096D, KV completo 512D |
| Vettori densi precedenti | raw hidden 4.096D, KV completo 512D |
| Transizione expert | `expert_transition_probabilities` 256D, cioe' il vettore precomputato $\mathbf{q}_{t,l}$ |
| Posizione | token position standardizzata |

Non esistono token ID, lookup compatti o embedding congelati nella Build 6.
Per il default veloce, i quattro vettori densi sono proiettati con matrici
gaussiane fisse e riproducibili a 32 dimensioni, poi standardizzati sul train.
Con `--projection-dimensions 0`, invece, ogni ramo conserva tutte le sue
dimensioni originali prima della standardizzazione; la variante richiede piu'
RAM e tempo.

## Esperimento con probabilita' di transizione degli expert

La variante aggiornata include come gruppo separato
`expert_transition_probabilities_256d`. Il gruppo contiene il vettore
$\mathbf{q}_{t,l}$ gia' materializzato per ogni record dalla feature 7:
ciascuna delle 256 componenti aggrega le probabilita' condizionate
$P_l(i\mid j)$ dei sei expert presenti al token corrente. Il KNN non calcola
le probabilita' al volo e non usa la matrice $P_l$ direttamente; legge gli
array float16 precomputati e li standardizza sul sottoinsieme train.

L'ablazione leave-one-group-out risponde alla domanda operativa: rimuovendo
solo questo vettore 256D, di quanto peggiora la Recall@6 del KNN? Un delta
positivo segnala informazione aggiuntiva utile per la geometria del KNN;
non misura direttamente il guadagno atteso della FFN.

Per non sovrascrivere il report senza la feature, l'esecuzione deve usare una
nuova directory di output, per esempio:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/run_build6_knn_feature_importance.py \
  --projection-dimensions 0 \
  --expert-transition-probabilities-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/transition_features/expert_transition_probabilities_v1/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/build6_knn_feature_importance_full_dim_with_expert_transition_probabilities_v1
```

### Risultato v1 a dimensione completa

L'esecuzione con `--projection-dimensions 0`, 5.000 esempi train e 5.000
validation ha prodotto Recall@6 validation **`0.153667`** per la
rappresentazione completa. Rimuovendo solo il gruppo
`expert_transition_probabilities_256d`, il valore scende a **`0.121533`**:

```text
perdita Recall@6 = 0.153667 - 0.121533 = +0.032133
```

Il valore senza la nuova feature coincide con il precedente screening Build 6
a dimensione completa; il confronto isola quindi il segnale addizionale del
vettore $\mathbf{q}_{t,l}$. Nel KNN questo e' il gruppo con la perdita di
Recall@6 piu' alta. I due rami KV restano il secondo e il terzo contributo
principale (`+0.012433` corrente e `+0.010767` precedente).

| Posizione | Gruppo rimosso | Recall@6 senza gruppo | Perdita Recall@6 |
|---:|---|---:|---:|
| 1 | Probabilita' di transizione expert 256D | 0.121533 | +0.032133 |
| 2 | KV completo corrente 512D | 0.141233 | +0.012433 |
| 3 | KV completo precedente 512D | 0.142900 | +0.010767 |
| 4 | Valori router precedenti | 0.152933 | +0.000733 |
| 5 | Statistiche KV correnti nello state | 0.153200 | +0.000467 |
| 6 | Statistiche KV precedenti nello state | 0.153200 | +0.000467 |
| 7 | Valori router correnti | 0.153267 | +0.000400 |
| 8 | Routing history precedente | 0.153333 | +0.000333 |
| 9 | Layer corrente | 0.153367 | +0.000300 |
| 10 | Routing history corrente | 0.153500 | +0.000167 |
| 11 | ID router corrente | 0.153500 | +0.000167 |
| 12 | Token position | 0.153500 | +0.000167 |
| 13 | ID router precedente | 0.153600 | +0.000067 |
| 14 | Raw hidden corrente 4.096D | 0.191533 | -0.037867 |
| 15 | Raw hidden precedente 4.096D | 0.197900 | -0.044233 |

Il report riproducibile e' in:

```text
build_dataset_6/ml/ablations/
build6_knn_feature_importance_full_dim_with_expert_transition_probabilities_v1/
reports/build6_knn_feature_importance.json
```

Questo risultato e' uno screening KNN: fornisce una forte motivazione per
aggiungere il ramo alla FFN, ma non sostituisce la validazione della rete
neurale.

## Comando consigliato

Il default usa 5.000 esempi train, 5.000 validation, cinque vicini e pesi
inversamente proporzionali alla distanza.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/run_build6_knn_feature_importance.py
```

Il report ordinato viene scritto in:

```text
build_dataset_6/ml/ablations/build6_knn_feature_importance/reports/
build6_knn_feature_importance.json
```

Il test set non viene usato: tutte le classifiche sono selezionate sul solo
validation set.

## Ripresa dopo un'interruzione

Il calcolo a dimensione completa puo' richiedere tempo. Lo script salva quindi
automaticamente un checkpoint dopo il KNN completo e dopo ciascuna ablazione:

```text
build_dataset_6/ml/ablations/<output-dir>/reports/
build6_knn_feature_importance_checkpoint.json
```

Se il processo viene interrotto, basta rieseguire **lo stesso comando**, con
lo stesso `--output-dir` e le stesse opzioni KNN. Il programma carica il
checkpoint, non ricalcola il modello completo ne' i gruppi gia' conclusi e
prosegue dal primo gruppo mancante. Al termine scrive il report finale e
rimuove il checkpoint.

Se si cambiano campionamento, dimensione della proiezione o numero di vicini,
occorre usare una nuova directory di output: i risultati parziali non sono
confrontabili con la nuova configurazione.

## Risultato della prima esecuzione Build 6

La prima esecuzione ha usato la proiezione fissa a 32D per hidden e KV. Il KNN
completo ha ottenuto Recall@6 validation **`0.144700`**. La graduatoria
seguente riporta il risultato lasciando fuori un gruppo alla volta.

| Posizione | Gruppo rimosso | Recall@6 senza gruppo | Perdita Recall@6 |
|---:|---|---:|---:|
| 1 | Routing history corrente | 0.135300 | +0.009400 |
| 2 | Statistiche KV correnti nello state | 0.135533 | +0.009167 |
| 3 | Statistiche KV precedenti nello state | 0.135567 | +0.009133 |
| 4 | KV completo corrente 512D | 0.137033 | +0.007667 |
| 5 | Routing history precedente | 0.137433 | +0.007267 |
| 6 | Layer corrente | 0.140433 | +0.004267 |
| 7 | KV completo precedente 512D | 0.140700 | +0.004000 |
| 8 | ID router precedente | 0.141633 | +0.003067 |
| 9 | ID router corrente | 0.142400 | +0.002300 |
| 10 | Token position | 0.145033 | -0.000333 |
| 11 | Valori router correnti | 0.145500 | -0.000800 |
| 12 | Raw hidden precedente proiettato | 0.145867 | -0.001167 |
| 13 | Raw hidden corrente proiettato | 0.147333 | -0.002633 |
| 14 | Valori router precedenti | 0.147533 | -0.002833 |

Il contesto `l-1` aggiunge quindi segnale KNN concreto: routing history,
statistiche KV, KV completo e ID router precedenti hanno tutti un delta
positivo. In particolare, il Recall@6 completo `0.144700` supera il valore
Build 5 con proiezione 32D (`0.095533`). Il confronto e' uno screening
indicativo: Build 6 usa i layer target `4...42`, mentre Build 5 usava
`3...42`, e non sostituisce la validazione FFN.

## Limiti dell'interpretazione

Il raw hidden e i valori router hanno delta negativi nel KNN proiettato, ma
non vanno rimossi dal dataset sulla sola base di questo risultato. La
proiezione casuale 32D puo' perdere informazione utile e un KNN non apprende
interazioni non lineari. In Build 5, ad esempio, il raw hidden ha mostrato
segnale positivo quando conservato a 4.096 dimensioni, pur non emergendo nella
variante a 32D.

La prova successiva consigliata e' quindi:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/run_build6_knn_feature_importance.py \
  --projection-dimensions 0 \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/build6_knn_feature_importance_full_dim
```

Successivamente sarà la FFN, con proiezioni Dense apprendibili per i quattro
rami densi, a stabilire se il nuovo contesto migliora il riferimento Build 5:
Recall@6 validation `0.502358287` e test `0.506056192`.
