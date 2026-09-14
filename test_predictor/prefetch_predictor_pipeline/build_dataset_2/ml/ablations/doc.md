# Ablation BUILD_DATASET_2

Questa cartella raccoglie esperimenti controllati derivati dallo stesso corpus
BUILD_DATASET_2. Ogni sotto-cartella cambia una sola rappresentazione delle
feature, mantenendo invariati prompt, split, target e Recall@6.

- `routing_slots_onehot/`: soltanto routing history H=1 come sei slot Top-6 ordinati;
- `routing_multihot_256/`: soltanto routing history H=1 come insieme multi-hot da 256 expert.
- `02_routing_multihot_layer/`: routing multi-hot da 256 expert piu' layer routed one-hot; corrisponde alla configurazione KNN `02`.
- `03_routing_layer_router_state/`: routing multi-hot, layer, ID e valori dei 16 raw router logits osservati in `E_l(t)`; gli ID usano 256 colonne fisse per slot e corrisponde alla configurazione KNN `03`.

La documentazione metodologica e i comandi sono in
`ML_pipeline/classical_ml_preprocessing/documentation/doc_preprocessing_routing_ablation.md`.

## Risultati: routing history H=1 come unica feature

Entrambe le ablation usano lo stesso corpus BUILD_DATASET_2, gli stessi split
per prompt, lo stesso target multi-label $E_l(t+1)$, gli stessi modelli e
Recall@6 come metrica. La sola differenza è la rappresentazione di $E_l(t)$.

| Modello | Sei slot one-hot ordinati | Multi-hot 256 | Baseline persistenza |
|---|---:|---:|---:|
| Logistic Regression | 0.1356 | 0.1623 | 0.4036 |
| Decision Tree | 0.2092 | 0.2153 | 0.4036 |
| Random Forest | 0.1260 | 0.2406 | 0.4036 |
| KNN | 0.1857 | 0.2578 | 0.4036 |

I valori sono Recall@6 sullo split **validation**. La baseline applica la
regola deterministica `predizione = E_l(t)`, cioè ripropone al token successivo
gli stessi sei expert osservati nel routing history H=1.

## Cosa abbiamo capito

Il multi-hot migliora tutti e quattro i modelli rispetto ai sei slot ordinati.
Per studiare la persistenza, quindi, l'insieme degli expert è una
rappresentazione più adatta dell'ordine Top-6: il ranking non è necessario alla
baseline e rende più difficile per gli algoritmi classici imparare la stessa
relazione.

Nessun modello raggiunge però Recall@6 `0.4036` della baseline. Questo non
indica un errore del multi-hot: gli algoritmi devono stimare 256 probabilità
one-vs-rest e poi selezionare i sei score maggiori, mentre la baseline copia
direttamente i sei ID già disponibili. Inoltre, la Logistic Regression non ha
raggiunto la convergenza con `max_iter=5000`, perciò il suo risultato non è un
limite definitivo di un modello lineare.

Il miglioramento del KNN da `0.1857` a `0.2578` è una conferma pratica che la
similarità tra due routing viene rappresentata più correttamente dal multi-hot
che dai sei slot indipendenti.

## Implicazione per il predictor finale

La persistenza dovrebbe rimanere una componente deterministica esplicita del
predictor runtime. Un componente appreso potrà concentrarsi nel prevedere quali
expert sostituire o aggiungere rispetto a $E_l(t)$, anziché dover reimparare da
zero a copiare l'intero routing corrente.
