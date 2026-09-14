# BUILD_DATASET_4 — da 500 nuovi prompt al training FFN

## Scopo e risultato attuale

Questo documento registra il flusso riproducibile che parte dal nuovo corpus
inglese V2 di 500 prompt e porta agli input pronti per addestrare il predictor
su un corpus totale di 1.000 prompt:

```text
V0 (100) + V1 (400) + V2 (500) = 1.000 prompt
```

`BUILD_DATASET_4` non sovrascrive `BUILD_DATASET_3`: conserva una nuova
raccolta DS4, un nuovo split per prompt, una nuova PCA train-only e nuovi
artefatti ML. Tutti i comandi sotto sono eseguiti da `~/ds4`.

## Convenzioni mantenute costanti

- Modello collector: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
- Generazione massima: 64 token per prompt.
- Per ogni token e layer routed si osservano `E_l(t)`, i 16 raw router logits,
  `token_id`, `token_position` e `ffn_norm` hidden state.
- Target: i sei expert del router al token successivo, `E_l(t+1)`.
- Split per prompt, mai per singola riga/token: 80% train, 10% validation,
  10% test.
- I 500 prompt V0+V1 mantengono il proprio split di BUILD_DATASET_3; solo V2
  viene assegnato dal nuovo split con seed 42.

## 0. Verifica del corpus V2

Input:

```text
dataset_definition/prefetch_prompts_v2_500_english.jsonl
```

Il corpus è stato controllato prima della raccolta:

- 500 JSONL validi, schema `id`, `category`, `prompt`;
- ID continui 501–1000 e nessuna sovrapposizione con V0/V1;
- 10 categorie, 50 prompt per categoria;
- nessun duplicato esatto interno o rispetto a V0/V1;
- prompt in inglese; nessun prompt vuoto.

Alcuni template linguistici sono intenzionalmente simili tra prompt della
stessa categoria (specialmente SQL, programmazione e language tasks), ma gli
argomenti e le operazioni richieste sono diversi.

## 1. Raccolta DS4 delle trace raw V2

### Obiettivo

Eseguire la stessa inferenza deterministica usata per V0 e V1, catturando le
trace necessarie a costruire tutte le feature fin qui testate.

### Preparazione una tantum

```bash
make dataset_runner
```

### Comando di raccolta

```bash
DS4_MODEL="gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" \
DS4_DATASET_PROMPTS_PATH="test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v2_500_english.jsonl" \
DS4_DATASET_CORPUS_ID="prefetch_prompts_v2_500_english" \
DS4_DATASET_OUTPUT_DIR="test_predictor/prefetch_predictor_pipeline/build_dataset_4/raw" \
DS4_DATASET_BUILD_ID="build_dataset_4_v2" \
DS4_DATASET_CAPTURE_HIDDEN=1 \
DS4_DATASET_ROUTER_MAX_SCORES=16 \
DS4_DATASET_MAX_TOKENS=64 \
./test_predictor/prefetch_predictor_pipeline/collector/dataset_runner
```

### Esito ottenuto

```text
processed=500 success=500 failed=0
```

Non è stato necessario recuperare prompt singoli. Se in una raccolta futura
un prompt fallisse, si può rilanciare soltanto quello con
`DS4_DATASET_FORCE_PROMPT_ID=<id>`.

## 2. Costruzione del dataset derivato V2

### Obiettivo

Convertire le trace raw di ogni prompt in record supervisionati per
token/layer, conservando la reference al hidden raw. `--history-length 1`
mantiene un solo passo osservato di routing, cioè il routing corrente
`E_l(t)` usato per predire `E_l(t+1)`; `--require-hidden` assicura che nessun
record privo di hidden state entri nel dataset.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/build_derived_dataset.py \
  --prompts test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v2_500_english.jsonl \
  --raw-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/raw \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v2_raw_hidden.jsonl \
  --history-length 1 \
  --require-hidden \
  --source-dataset prefetch_prompts_v2_500_english
```

Esito ottenuto: **1.037.320 esempi da 500 sequenze**.

## 3. Merge V0, V1 e V2

### Obiettivo

Unire record derivati compatibili, preservando la provenienza di ciascun
corpus. Il primo input è già il merge V0+V1 da 500 prompt prodotto in
BUILD_DATASET_3; il secondo è V2.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/merge_derived_datasets.py \
  --inputs \
    test_predictor/prefetch_predictor_pipeline/build_dataset_3/derived/prefetch_dataset_build_3_v0_v1_raw_hidden.jsonl \
    test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v2_raw_hidden.jsonl \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v0_v1_v2_raw_hidden.jsonl
```

Esito ottenuto:

```text
2.062.720 esempi da 1.000 prompt
V0=197.440, V1=827.960, V2=1.037.320 esempi
```

## 4. Split train/validation/test per prompt

### Obiettivo

Creare uno split senza leakage tra token dello stesso prompt. Il flag
`--preserve-split` mantiene invariati i 500 prompt storici; il seed determina
in modo riproducibile l'assegnazione dei 500 prompt V2.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/create_prompt_split.py \
  --prompts \
    test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl \
    test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v1_400_english.jsonl \
    test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v2_500_english.jsonl \
  --preserve-split test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/training/split_by_prompt.json \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/split_by_prompt.json \
  --seed 42
```

Esito ottenuto: **train=800, validation=100, test=100 prompt**.

## 5. PCA layer-wise degli hidden state

### Obiettivo

Comprimere `ffn_norm` raw da 4.096 dimensioni a 64 componenti PCA per layer,
senza usare validation o test durante il fit. La PCA viene riaddestrata per
BUILD_DATASET_4 perché il train set è passato da 400 a 800 prompt.

### Problema incontrato e correzione

L'implementazione originaria caricava l'intero JSONL da 2 milioni di record e
matrici raw intere in RAM. Il kernel ha terminato Python per OOM a circa 10,3
GB di RAM residente.

`create_hidden_pca_dataset.py` è stato quindi reso memory-bounded:

1. indicizza su disco le reference raw per layer;
2. per ogni layer esegue una sola `IncrementalPCA` sui **tutti** i vettori
   train, in batch da 4.096;
3. trasforma, solo dopo il fit completo di quel layer, train/validation/test;
4. scrive il JSONL finale in streaming.

`IncrementalPCA` è un'approssimazione numerica della PCA interamente in RAM,
non una PCA separata per batch. I batch aggiornano progressivamente un'unica
media e un'unica matrice di componenti per layer. Vedere
`pipeline/hidden_features/doc_create_hidden_pca_dataset.md` per i dettagli.

Prima del rilancio sono stati rimossi soltanto gli output parziali:

```bash
rm -rf test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/hidden_pca_artifacts
rm -f test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v0_v1_v2_hidden_pca64.jsonl
```

### Comando completato

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/create_hidden_pca_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v0_v1_v2_raw_hidden.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/split_by_prompt.json \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v0_v1_v2_hidden_pca64.jsonl \
  --artifacts-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/hidden_pca_artifacts \
  --components 64 \
  --batch-size 4096
```

Esito ottenuto: PCA train-only completata per tutti i layer 3–42 e JSONL PCA
con **2.062.720 esempi**.

## 6. Preprocessing generico

### Obiettivo

Leggere il JSONL PCA in streaming e produrre il Parquet canonico, le label
multi-hot da 256 expert e i metadati per ciascuno split.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/preprocess_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v0_v1_v2_hidden_pca64.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/split_by_prompt.json \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training
```

Esito ottenuto: **2.062.720 esempi** preprocessati in
`build_dataset_4/ml/training/`.

## 7. Ablation sparse 04: routing, layer, router state, hidden PCA

### Obiettivo

Costruire le matrici sparse della migliore configurazione non-token fin qui
testata. Le 4.472 feature sono:

| Gruppo | Dimensioni | Rappresentazione |
|---|---:|---|
| Routing corrente `E_l(t)` | 256 | Multi-hot dei sei expert scelti al layer corrente. |
| Layer routed corrente | 40 | One-hot. |
| ID dei 16 raw router logits | 4.096 | 16 blocchi fissi da 256 posizioni: ogni slot attiva l'ID expert osservato. |
| Valori dei 16 raw router logits | 16 | Valori standardizzati con statistiche train-only. |
| Hidden `ffn_norm` PCA | 64 | Componenti PCA layer-wise train-only. |
| **Totale** | **4.472** | Input sparse della FFN `04`. |

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256_layer_router_state_hidden_pca64 \
  --training-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/04_routing_layer_router_state_hidden_pca64
```

Esito ottenuto:

| Split | Shape X | Esempi |
|---|---:|---:|
| Train | `(1.653.000, 4.472)` | 1.653.000 |
| Validation | `(200.520, 4.472)` | 200.520 |
| Test | `(209.200, 4.472)` | 209.200 |

## 8. Lookup embedding token GGUF

### Obiettivo

Preparare il lookup congelato per l'ablation `05`, senza scrivere 4.096 valori
per ogni record. Il file GGUF contiene l'embedding per ogni token del
vocabolario; qui estraiamo solo gli ID presenti nel corpus offline.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/token_features/extract_token_embedding_lookup.py \
  --training-parquet test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/prefetch_training.parquet \
  --gguf-model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/token_embedding_artifacts
```

Esito ottenuto: lookup con **7.136 token ID** distinti, matrice
`(7.136, 4.096)` `float16`. Il GGUF sorgente ha vocabolario 129.280; per il
runtime finale resta da definire la gestione degli ID non presenti nel lookup
compatto. Questo punto è registrato in `FUTURE_IMPROVEMENTS_BEFORE_FINAL_MODEL.md`.

## 9. Ablation 05: input token per FFN multi-input

### Obiettivo

Preparare gli input addizionali dell'ablation `05` senza duplicare le matrici
sparse `04`:

- indice `int32` nel lookup embedding congelato;
- `token_position` scalare standardizzata con media/deviazione train-only.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/token_features/prepare_token_embedding_inputs.py \
  --training-parquet test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/prefetch_training.parquet \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/token_embedding_artifacts \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/04_routing_layer_router_state_hidden_pca64/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/05_routing_layer_router_state_hidden_pca64_token_embedding
```

Esito ottenuto: sei array token per train/validation/test, coerenti con le
shape di `04`. Il summary registra media train `53,4792` e deviazione standard
train `18,4003` per `token_position`.

## 10. Baseline di persistenza

### Obiettivo

Calcolare il riferimento senza rete: predire il routing corrente `E_l(t)` come
stima del routing successivo `E_l(t+1)`.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/baseline.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/prefetch_training.parquet \
  --report test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/baseline/routing_persistence_validation.json
```

Esito ottenuto:

```text
train Recall@6:      0,4004
validation Recall@6: 0,4020
```

## 11. Training controllato della FFN 04

Prima di aggiungere gli embedding token è necessario addestrare `04` sul
corpus da 1.000 prompt. Questo fornisce il confronto controllato per `05`:
stesso corpus, stesso split, stessa rete sparse, stessi iperparametri e stessa
batch size; `05` differirà solo per `token_id` e `token_position`.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_ffn.py \
  --data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/04_routing_layer_router_state_hidden_pca64/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/neural_network/ffn_routing_layer_router_state_hidden_pca64_HL512_DO03_LR000069972_EP500_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --batch-size 4096 \
  --epochs 500 \
  --early-stopping-patience 5
```

L'early stopping seleziona l'epoca con migliore `val_recall_at_6` e ne
ripristina i pesi. Il run ha completato 302 epoche e ha ottenuto:

```text
validation Recall@6: 0,4552
```

Rispetto alla baseline validation `0,4020`, il miglioramento è `+0,0532`.
`0,4552 × 6 = 2,73`: in media la rete include circa 2,73 dei sei expert reali
nella propria Top-6. L'ultima Recall@6 train era circa `0,4578`, molto vicina
al valore validation, quindi non emerge un divario di generalizzazione ampio.

Il test set non va usato in questa fase: verrà valutato una sola volta dopo
aver scelto la configurazione finale su validation.

## 12. Training FFN 05: token embedding e token position

Se `04` completa correttamente, `05` userà:

```text
state sparse 4472D -> Dense hidden 512D
token_id -> embedding GGUF congelato 4096D -> Dense lineare allenabile 64D
token_position standardizzata train-only
             -> concatenazione -> 256 sigmoid
```

Il run ha mantenuto gli stessi dati, split, dimensione hidden, attivazione,
dropout, learning rate e batch size di `04`; ha richiesto un massimo di 1.000
epoche, ma l'early stopping ha concluso il training dopo 265 epoche. Esito:

```text
05 validation Recall@6: 0,4576
04 validation Recall@6: 0,4552
differenza:              +0,0024
```

Il guadagno corrisponde a `0,0024 × 6 = 0,0144` expert corretti aggiuntivi in
media nella Top-6. È un miglioramento misurato sullo stesso validation set, ma
di piccola entità; il test finale servirà a verificare se si mantiene su prompt
mai usati. L'ultima Recall@6 train era `0,4640`, contro `0,4576` validation:
il gap è moderato e non segnala overfitting marcato.

Keras emette l'avviso che `shuffle=True` viene ignorato per un `tf.data.Dataset`.
Non altera questo run: il generatore del progetto rimescola esplicitamente gli
indici train a ogni nuova ripetizione del dataset. L'avviso riguarda soltanto
il flag generico di Keras, non l'ordine effettivo degli esempi del generatore.

I dettagli dell'architettura e dei comandi sono in
`ML_pipeline/neural_network/documentation/doc_token_embedding_ffn.md`.

## 13. Prossimo step consigliato: test finale predefinito

Le configurazioni `04` e `05` sono ora definite senza usare il test set. Se non
si intendono fare tuning o nuove modifiche guidate da test, il passo corretto è
valutarle entrambe una sola volta sullo split test, con batch size 4.096. Il
confronto misurerà se il piccolo vantaggio validation di `05` si conferma.

## 14. Ablation successiva: proiezione supervisionata del raw hidden

Prima del test finale è stata scelta una nuova ablation, separata da `04` e
`05`: sostituire la PCA-64 con una proiezione lineare trainable del vettore
raw `ffn_norm` 4096D e conservarne insieme la proiezione trainable del token
embedding GGUF. La rete userà quindi lo state sparse senza PCA (`03`, 4408D),
il raw hidden normalizzato train-only per layer, `token_id` e
`token_position`.

Le due dimensioni di compressione sono parametri indipendenti:
`--hidden-projection-dimension` e `--token-projection-dimension`. Il primo
confronto usa 64/64; le run successive possono cambiare una dimensione alla
volta sul validation set. La guida, i comandi e la politica di normalizzazione
sono in `ML_pipeline/neural_network/documentation/doc_raw_hidden_token_embedding_ffn.md`.

Per evitare che ogni epoch rilegga i raw hidden tramite accessi frammentati ai
file collector, la pipeline materializza inoltre una sola volta gli hidden già
normalizzati in array contigui float16 (`07`). Il training rimescola con seed
fisso macro-blocchi contigui da 65.536 esempi, mantenendo sequenziali i batch
interni; non cambia split, esempi o label.

Il ramo raw hidden mantiene inoltre gli array e il trasferimento GPU in
float16 e usa una Dense con policy `mixed_float16`; i pesi master della
proiezione restano float32 e trainable end-to-end.
