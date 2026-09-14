# Ablation progressiva dei gruppi di feature con KNN

## Obiettivo

Questo script cerca le fonti di informazione più utili fra le feature già disponibili in BUILD_DATASET_2. Usa KNN come strumento di screening veloce: non è il predictor finale e non sostituisce il confronto successivo con Random Forest o Logistic Regression.

Il criterio di selezione è soltanto **Recall@6 sul validation set**. Il train serve a osservare l'overfitting; il test non viene caricato né valutato.

## Perché KNN

KNN non richiede un training iterativo e non emette warning di convergenza. Con un sottoinsieme train fisso e riproducibile di 20.000 esempi permette di confrontare molte rappresentazioni in poco tempo. La stessa selezione casuale, con `random_state = 42`, è riutilizzata per ogni configurazione.

## Risultati registrati

Il report [progressive_knn_summary.json](../../../build_dataset_2/ml/ablations/progressive_knn/reports/progressive_knn_summary.json) contiene i risultati dello screening eseguito su BUILD_DATASET_2. Ogni KNN usa gli stessi 20.000 esempi train, 5 vicini e pesi inversamente proporzionali alla distanza; le righe sono ordinate per **Recall@6 di validation**.

| Posizione | Configurazione | Gruppi di feature oltre al routing multi-hot | Feature totali | Recall@6 train | Recall@6 validation |
|---:|---|---|---:|---:|---:|
| 1 | `04_routing_layer_token_context` | layer, token context | 761 | 0,4105 | **0,2849** |
| 2 | `02_routing_multihot_layer` | layer | 296 | 0,4067 | 0,2847 |
| 3 | `01_routing_multihot` | nessuno | 256 | 0,3838 | 0,2578 |
| 4 | `03_routing_layer_router_state` | layer, router state | 4.408 | 0,3651 | 0,2395 |
| 5 | `07_all_available_with_multihot` | layer, token context, router state, hidden PCA-64 | 4.937 | 0,4179 | 0,1866 |
| 6 | `06_routing_layer_router_state_hidden_pca` | layer, router state, hidden PCA-64 | 4.472 | 0,4170 | 0,1862 |
| 7 | `05_routing_layer_hidden_pca` | layer, hidden PCA-64 | 360 | 0,4265 | 0,1810 |

Questa classifica riguarda i **gruppi** di feature, non una graduatoria delle singole colonne. In particolare, un risultato KNN debole per router state o hidden PCA non implica che quelle fonti siano inutili per la FFN: i due modelli sfruttano le feature in modo molto diverso.

## Feature comuni e significato dei gruppi

Ogni configurazione include sempre `routing_history`, rappresentata come vettore multi-hot da 256 posizioni. I gruppi aggiuntivi sono:

| Gruppo | Colonne | Significato e rappresentazione |
|---|---|---|
| Routing multi-hot | `routing_history_expert_0..5` | Gli sei expert di `E_l(t)`, cioe' gli expert scelti dal router al layer routed corrente per il token corrente. Diventano 256 feature: vale 1 nelle posizioni degli expert selezionati e 0 nelle altre. L'ordine Top-6 non e' conservato. |
| Layer | `layer` | Il layer routed corrente `l`, compreso fra 3 e 42. Nel KNN e' una variabile categorica codificata one-hot. |
| Router state: ID | `router_logit_expert_id_0..15` | Gli ID degli expert associati ai 16 raw router logits osservati per il layer corrente. Nel KNN sono 16 variabili categoriche one-hot. |
| Router state: valori | `router_logit_value_0..15` | I 16 valori raw dei router logits corrispondenti agli ID precedenti; esprimono la preferenza del router prima della selezione Top-6. Nel KNN sono standardizzati col train. |
| Token context | `token_id`, `token_position` | `token_id` identifica il token corrente ed e' codificato one-hot; `token_position` e' la sua posizione nella sequenza generata ed e' standardizzata col train. |
| Hidden PCA-64 | `hidden_pca_0..63` | Le 64 coordinate ottenute comprimendo con PCA l'hidden state del layer corrente. La PCA e' fit soltanto sui vettori train e le coordinate sono standardizzate col train per il KNN. |

`layer`, `token_id` e gli ID dei router logits sono categorici e sono codificati con `OneHotEncoder` fit soltanto sul train. `token_position`, i valori raw logits e le coordinate PCA sono standardizzati con `StandardScaler` fit soltanto sul train. Il routing base resta direttamente multi-hot, con 256 colonne e sei valori attivi per riga.

Questa ablation non usa il token embedding 4096D: BUILD_DATASET_2 lo descrive come lookup esterna, ma il pipeline classico non lo ha ancora materializzato.

## Flusso di esecuzione

```text
main()
  |
  +-- leggi_argomenti()
  |     legge cartelle, configurazioni richieste e limite train KNN
  |
  +-- trova_parquet_base() + valida_colonne()
  |     caricano il Parquet BUILD_DATASET_2 e verificano tutte le feature
  |
  +-- carica_label()
  |     riusa y_train.npy e y_validation.npy con target E_l(t+1)
  |
  +-- per ogni configurazione:
  |     |
  |     +-- trasforma_configurazione()
  |     |     crea routing multi-hot; fit encoder/scaler solo sul train;
  |     |     costruisce X train e validation in memoria
  |     |
  |     +-- seleziona_train_knn()
  |     |     estrae lo stesso sottoinsieme train riproducibile di 20.000 righe
  |     |
  |     +-- KNeighborsClassifier.fit()
  |     |     usa 5 vicini e pesi inversamente proporzionali alla distanza
  |     |
  |     +-- estrai_score_positivi() + calcola_recall_at_6()
  |           calcolano Recall@6 su train e validation
  |
  +-- salva_json()
        salva configurazione completa e risultati ordinati per validation
```

Le matrici e i modelli KNN non vengono salvati: sono artefatti temporanei di screening. Questo evita di creare sette copie persistenti degli stessi dati.

## Output

```text
build_dataset_2/ml/ablations/progressive_knn/
├── artifacts/
│   └── ablation_configurations.json
└── reports/
    └── progressive_knn_summary.json
```

`ablation_configurations.json` conserva le configurazioni, i gruppi di feature, i parametri KNN e il limite train. `progressive_knn_summary.json` contiene per ogni configurazione feature count, shape/nnz di X, Recall@6 train e validation, tempo impiegato e metadati delle trasformazioni. I risultati sono ordinati per Recall@6 validation decrescente.

## Esecuzione

Dalla cartella `ML_pipeline/`:

```bash
uv run classical_ml_preprocessing/run_progressive_knn_ablation.py \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/ablations/progressive_knn
```

Per rieseguire soltanto configurazioni specifiche:

```bash
uv run classical_ml_preprocessing/run_progressive_knn_ablation.py \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/ablations/progressive_knn \
  --configurations 03_routing_layer_router_state,05_routing_layer_hidden_pca
```

Il report può essere sovrascritto da un run successivo nella stessa cartella. Se vogliamo conservare risultati di parametri KNN differenti, dobbiamo passare una diversa `--output-dir`.

## Interpretazione

Una configurazione migliore mostra che il relativo gruppo di feature migliora la similarità fra esempi rispetto al target futuro. Non dimostra però che KNN sia il modello migliore: le configurazioni più promettenti andranno confermate con Random Forest e, se utile, Logistic Regression sullo stesso validation set.
