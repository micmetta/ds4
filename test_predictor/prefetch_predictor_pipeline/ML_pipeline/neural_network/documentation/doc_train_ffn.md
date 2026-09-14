# FFN Keras configurabile

## Obiettivo

`train_ffn.py` addestra una Feed-Forward Neural Network multi-label per predire i 256 expert di `E_l(t+1)`. Usa le matrici sparse CSR gia' prodotte da `classical_ml_preprocessing/preprocessing_classical.py`, quindi puo' usare le feature categoriche one-hot, le feature numeriche e le hidden PCA senza caricare una matrice densa completa in RAM.

Ogni output usa una sigmoid indipendente; la loss e' `binary_crossentropy`. La selezione finale continua a usare la stessa regola del progetto: i sei expert con score piu' alto e Recall@6.

## Metrica: Recall@6

Per ogni esempio il router reale seleziona esattamente sei expert per il layer
e il token successivi, cioe' il target `E_l(t+1)`. Anche il predictor deve
prefetchare esattamente sei expert: ordina quindi i 256 score prodotti dalla
rete e seleziona la propria Top-6. `Recall@6` misura quanti dei sei expert
reali sono presenti in questa Top-6:

$$
\mathrm{Recall@6} = \frac{|\mathrm{Top6}_{pred} \cap E_l(t+1)|}{6}.
$$

La metrica e' scelta perche' corrisponde direttamente al budget operativo del
router e del prefetch: non valuta una soglia arbitraria sulle sigmoid, ma la
qualita' dei sei expert che sarebbero effettivamente prefetched. Poiche'
predizione e target contengono entrambi sei expert, Precision@6 e Recall@6
coincidono in questo progetto.

Un valore `Recall@6 = 0,4364` equivale a `0,4364 x 6 = 2,62`: in media circa
2,62 dei sei expert reali vengono predetti correttamente per ogni
layer/token. Il valore e' una media: un singolo esempio puo' avere da zero a
sei expert corretti.

## Storico sintetico degli esperimenti FFN

La tabella riporta la Recall@6 di **validation**. Il test set non entra nella
selezione di architettura o iperparametri; non confrontare direttamente le
metriche assolute dei corpus da 100 e 500 prompt, che hanno distribuzioni e
split diversi.

| Corpus | Configurazione | Validation Recall@6 | Risultato |
|---|---|---:|---|
| 100 prompt (BUILD_DATASET_2) | Baseline persistenza `E_l(t)` | 0,4036 | Riferimento senza modello ML. |
| 100 prompt (BUILD_DATASET_2) | FFN: routing multi-hot + layer | 0,4098 | Primo miglioramento sulla baseline. |
| 100 prompt (BUILD_DATASET_2) | FFN: routing multi-hot + layer + router state, tuning e retraining 512/ReLU/dropout 0,3/LR 0,00006997208 | 0,4169 | Migliore FFN del corpus storico; superiore alla baseline di +0,0133. |
| 500 prompt (BUILD_DATASET_3) | Baseline persistenza `E_l(t)` | 0,4040 | Riferimento sul corpus combinato V0+V1. |
| 500 prompt (BUILD_DATASET_3) | FFN: routing multi-hot + layer + router state, 512/ReLU/dropout 0,3/LR 0,00006997208 | 0,4364 | Migliore rete corrente prima del tuning sul corpus da 500 prompt; superiore alla baseline di +0,0324. |
| 500 prompt (BUILD_DATASET_3) | FFN: routing + layer + router state + hidden PCA-64, batch 4.096 | 0,4432 | Migliore ablation validation corrente; il confronto pulito con `03` richiede la stessa batch size. |
| 1.000 prompt (BUILD_DATASET_4) | Baseline persistenza `E_l(t)` | 0,4020 | Riferimento sul corpus V0+V1+V2. |
| 1.000 prompt (BUILD_DATASET_4) | FFN: routing + layer + router state + hidden PCA-64, 512/ReLU/dropout 0,3/LR 0,00006997208, batch 4.096 | 0,4552 | Migliore validation corrente; +0,0532 sulla baseline. |
| 1.000 prompt (BUILD_DATASET_4) | FFN multi-input: `04` + embedding GGUF token congelato/proiezione 4.096→64 + token position, stessi iperparametri | 0,4576 | +0,0024 su `04`; miglioramento piccolo da confermare sul test finale. |
| BUILD_DATASET_5 | FFN fixed-NNZ multi-input: state 512 ReLU, raw hidden 4096→512 ReLU, token 4096→64 ReLU, KV diretto 512D, fusione 512 ReLU; warm-start del trial Hyperband 0024 fino a early stopping | **0,5024** | **Migliore risultato ottenuto finora:** validation 0,502358287 e test 0,506056192, valutato una sola volta. |

Il router state include gli ID e i valori dei 16 raw router logits del token
corrente, oltre al routing corrente e al layer. Lo storico mostra che questa
informazione aggiuntiva e la FFN estraggono segnale oltre alla sola
persistenza. Non isola tuttavia il contributo individuale di ciascun gruppo di
feature: per quello servono ablation controllate sullo stesso corpus.

L'ablation successiva `05` aggiunge `token_id` tramite lookup GGUF congelato
e proiezione lineare allenabile 4.096→64, senza salvare 4.096 feature dense per
ogni riga. La relativa procedura è descritta in
`doc_token_embedding_ffn.md`.

## Input

La cartella indicata con `--data-dir` deve contenere:

```text
arrays/
├── X_train.npz
├── X_validation.npz
├── y_train.npy
└── y_validation.npy
```

Le feature restano sparse fino ai mini-batch. Lo script converte soltanto il batch corrente in `tf.SparseTensor`; non converte in una densa matrice `[N, feature]` l'intero dataset.

XLA viene disabilitato esplicitamente: alcune operazioni TensorFlow necessarie
per il primo layer Dense con input sparse non sono supportate da XLA GPU. La
rete continua comunque a usare la GPU CUDA normalmente.

Il training usa dataset ripetibili e un numero di step esplicito per epoca,
cosi' Keras non interrompe il run con l'avviso fuorviante `input ran out of data`.

## Iperparametri

| Opzione | Significato | Default |
|---|---|---|
| `--data-dir` | Cartella `arrays/` con feature `X_train.npz` oppure `X_classic_train.npz`, feature validation e label `y_*.npy` | `../build_dataset_2/ml/classical_ml/arrays` |
| `--output-dir` | Cartella dell'esperimento in cui salvare modello Keras e report JSON; non deve puntare alla cartella input | `../build_dataset_2/ml/neural_network/ffn_default` |
| `--hidden-layers` | Nodi dei layer nascosti, separati da virgole | `512,256` |
| `--activation` | Attivazione Keras dei layer nascosti | `relu` |
| `--dropout` | Dropout dopo ogni layer nascosto | `0.2` |
| `--learning-rate` | Learning rate di Adam | `0.001` |
| `--l2-regularization` | Coefficiente L2 opzionale | `0.0` |
| `--batch-size` | Esempi per mini-batch | `512` |
| `--epochs` | Limite massimo di epoche | `30` |
| `--early-stopping-patience` | Epoche senza miglioramento di validation Recall@6 | `5` |

L'early stopping osserva `val_recall_at_6` e ripristina i pesi dell'epoca migliore.

Ad esempio, `--hidden-layers 64` crea un solo layer hidden con 64 neuroni. Invece `--hidden-layers 512,256` crea due layer hidden, rispettivamente da 512 e 256 neuroni.

## Batch size e GPU

`--batch-size` indica quanti esempi vengono elaborati insieme prima di un
aggiornamento dei pesi. I batch sono **sequenziali**: il batch successivo usa
i pesi aggiornati dal precedente. All'interno di un singolo batch, invece, la
GPU esegue in parallelo le operazioni sui molti esempi e le moltiplicazioni tra
matrici.

Un batch piu' grande puo' rendere l'addestramento piu' veloce per epoca:

- la GPU riceve matrici piu' grandi e tende a essere utilizzata meglio;
- diminuiscono gli step per epoca e l'overhead di lancio/sincronizzazione;
- piu' esempi e attivazioni devono pero' stare contemporaneamente in VRAM.

Per un training set di dimensione `N`, gli step per epoca sono circa
`ceil(N / batch_size)`. Per esempio, con 820.440 esempi train:

| Batch size | Step per epoca circa |
|---:|---:|
| 512 | 1.603 |
| 1.024 | 802 |
| 2.048 | 401 |

Un batch piu' grande non garantisce un dimezzamento esatto del tempo: ogni
singolo step costa di piu' e il guadagno dipende da GPU, VRAM e dimensione del
modello. Se la VRAM non basta, TensorFlow termina con un errore out-of-memory.
Inoltre batch molto grandi rendono il gradiente meno rumoroso e possono
richiedere di rivalutare il learning rate o la qualita' di generalizzazione.
Per questo e' ragionevole provare prima `512`, poi `1024`, mantenendo fissi gli
altri iperparametri e confrontando validation Recall@6 e tempo totale.

## Output

```text
<output-dir>/
├── models/
│   └── ffn.keras
└── reports/
    └── training_report.json
```

Il report conserva percorsi, iperparametri, numero di epoche effettive, Recall@6 validation e la history Keras.

Per proteggere gli esperimenti, lo script rifiuta una `--output-dir` gia'
esistente con `FileExistsError`. Scegliere quindi un nome nuovo per ogni run.

## Esecuzione

Dalla cartella `ML_pipeline/`, dopo `uv sync`:

```bash
uv run neural_network/train_ffn.py \
  --data-dir ../build_dataset_2/ml/classical_ml/arrays \
  --output-dir ../build_dataset_2/ml/neural_network/ffn_512_256 \
  --hidden-layers 512,256 \
  --activation relu \
  --dropout 0.2 \
  --learning-rate 0.001
```

Ogni esperimento deve usare una diversa `--output-dir`, cosi' modello e report restano confrontabili.
