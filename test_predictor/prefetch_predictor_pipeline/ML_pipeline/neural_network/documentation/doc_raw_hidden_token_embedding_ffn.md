# FFN con proiezioni supervisionate di hidden state e token embedding

## Obiettivo

Questa ablation sostituisce le 64 componenti PCA dell'hidden state con una
proiezione lineare appresa direttamente dalla FFN. Combina quindi:

```text
state sparse 4408D -> Dense state (es. 512D) -------------------+
hidden raw 4096D, normalizzato per layer -> Dense lineare H_D ---+-> concatenazione -> sigmoid 256D
token_id -> lookup GGUF congelato 4096D -> Dense lineare T_D ----+
token_position standardizzata train-only ------------------------+
```

`H_D` e `T_D` sono parametri della run, rispettivamente
`--hidden-projection-dimension` e `--token-projection-dimension`.
Entrambi valgono 64 per il primo confronto controllato.

La normalizzazione del raw hidden è calcolata solo sul train e separatamente
per ogni layer routed: `(h - mean_train[layer]) / std_train[layer]`. La
proiezione `4096 -> H_D` resta invece trainable e ottimizzata end-to-end sulla
stessa loss multi-label degli expert. Il lookup GGUF rimane congelato; la sua
proiezione `4096 -> T_D` è trainable.

In concreto, per ciascuna delle 4.096 componenti dell'hidden state vengono
calcolate, sul solo split train, una media e una deviazione standard per ogni
layer routed. La trasformazione sottrae la media e divide per la deviazione
standard del layer dell'esempio. Serve a rendere comparabili le scale
numeriche delle componenti e quelle dei layer: senza questo passaggio la Dense
`4096 -> H_D` dovrebbe prima imparare a compensare medie e scale diverse,
rendendo l'ottimizzazione meno stabile. Le stesse statistiche train vengono
applicate inalterate anche a validation e test; questi split non contribuiscono
mai al loro calcolo, quindi non viene introdotto data leakage.

## 1. Preparare i riferimenti raw hidden

Questo step non copia i vettori hidden completi: salva riferimenti ai file
binari già raccolti e le statistiche train-only. Il calcolo delle statistiche
legge i vettori una volta in batch, quindi può richiedere tempo ma usa memoria
limitata.

Da `~/ds4`:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/prepare_raw_hidden_inputs.py \
  --raw-hidden-input test_predictor/prefetch_predictor_pipeline/build_dataset_4/derived/prefetch_dataset_build_4_v0_v1_v2_raw_hidden.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/split_by_prompt.json \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/03_routing_layer_router_state/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/06_routing_layer_router_state_raw_hidden_token_embedding \
  --batch-size 4096
```

L'output `06` contiene soltanto gli input aggiuntivi raw hidden. Gli array
sparse state e gli input token non vengono duplicati.

## 2. Materializzare gli hidden normalizzati per training veloce

La prima implementazione leggeva i raw hidden direttamente dai file collector
a ogni epoch. È sicura per la RAM ma troppo lenta, perché lo shuffle delle
righe produceva molti accessi piccoli e non contigui su disco. Il seguente
step usa gli indici e le statistiche di `06` una sola volta, legge i vettori
in ordine fisico di file e offset, applica la normalizzazione e salva tre
array contigui `float16`.

Lo spazio aggiuntivo atteso per BUILD_DATASET_4 è circa 15,8 GiB. Non è una
nuova inferenza DS4 e non modifica `06`, i raw collector o le precedenti
ablation. La conversione in float16 dimezza lo spazio; durante il training il
batch resta float16 anche durante il trasferimento alla GPU. La Dense
`4096 -> H_D` usa la policy `mixed_float16`: input e calcolo sfruttano i Tensor
Core, mentre i pesi master trainable restano float32. Dopo la compressione,
solo le `H_D` componenti risultanti vengono fuse con gli altri rami. Questa
scelta mantiene la proiezione completamente end-to-end ma evita di creare e
trasferire una copia float32 da circa 25 GiB per epoch.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/materialize_normalized_hidden_inputs.py \
  --raw-hidden-input-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/06_routing_layer_router_state_raw_hidden_token_embedding \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/07_routing_layer_materialized_raw_hidden_f16 \
  --batch-size 4096
```

L'ordine delle righe negli array materializzati resta identico agli array
state, token e label. Durante ogni epoch la FFN non rimescola le singole
righe: rimescola con seed riproducibile l'ordine dei macro-blocchi contigui.
Con il valore consigliato di 65.536 esempi, ogni macro-blocco contiene sedici
batch da 4.096 e viene letto sequenzialmente prima di passare al successivo.
Tutti gli esempi compaiono esattamente una volta per epoch, ma il disco compie
molti meno salti. Le run che usano lo stesso seed, batch size, macro-blocco e
iperparametri sono direttamente confrontabili tra loro.

Il training passa inoltre `shuffle=False` a Keras perché lo shuffle viene già
eseguito esplicitamente dal generatore a livello di macro-blocchi. Questo
elimina il warning generico di Keras senza cambiare l'ordine effettivo degli
esempi.

## 3. Addestrare la prima configurazione 64D + 64D

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_materialized_hidden_token_embedding_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/03_routing_layer_router_state/arrays \
  --materialized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/07_routing_layer_materialized_raw_hidden_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/05_routing_layer_router_state_hidden_pca64_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/neural_network/ffn_routing_layer_raw_hidden64_token_embedding64_HL512_DO03_LR000069972_EP500_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --epochs 500 \
  --early-stopping-patience 5
```

Per confrontare dimensioni diverse, si cambiano soltanto i due parametri di
proiezione e si usa ogni volta una nuova `--output-dir`. Esempi utili, da
provare uno alla volta sul validation set: `32/64`, `64/64`, `128/64`, poi
eventualmente `64/32` e `64/128`. Il test resta intatto finché non si sceglie
una configurazione finale.

## Test finale

Solo dopo avere scelto la configurazione su validation:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/evaluate_materialized_hidden_token_embedding_ffn.py \
  --state-data-dir <ablation_03_arrays> \
  --materialized-hidden-data-dir <ablation_07> \
  --token-data-dir <ablation_05_arrays> \
  --token-embedding-dir <token_embedding_artifacts> \
  --model-path <output-dir>/models/ffn.keras \
  --report-path <output-dir>/reports/test_report.json \
  --batch-size 4096
```

## 4. Variante INT8 per ridurre il collo di bottiglia I/O

Su una macchina con circa 11 GiB disponibili in WSL, gli array hidden
`float16` di BUILD_DATASET_4 occupano circa 15,8 GiB e il solo train circa
12,6 GiB. Non possono quindi restare interamente nella cache RAM: durante ogni
epoch molte pagine devono essere rilette dal disco. La GPU resta spesso in
attesa dei dati, anche se la moltiplicazione `4096 -> H_D` viene eseguita sulla
GPU.

La variante INT8 usa una quantizzazione simmetrica globale calibrata
esclusivamente su un campione deterministico del train:

```text
limit = percentile_99.99(abs(hidden_normalizzato_train))
scale = limit / 127
q = clip(round(hidden_normalizzato / scale), -127, 127)
hidden_approssimato = q * scale
```

Validation e test non contribuiscono alla calibrazione. Lo script registra per
ogni split la frazione di valori saturati e l'RMSE di ricostruzione. Gli array
INT8 occupano circa la metà dei corrispondenti array float16; per BUILD4 sono
attesi circa 7,9 GiB complessivi. Questo aumenta la probabilità che train e
validation rimangano nella cache del sistema operativo.

La **saturazione** misura la percentuale di valori originali che superano
l'intervallo rappresentabile scelto dalla calibrazione. Con il percentile
99,99 di BUILD_DATASET_4, l'intervallo è circa `[-3,875, +3,875]`: un valore
come `4,20` viene quindi rappresentato come `3,875`, mentre `-5,00` viene
rappresentato come `-3,875`. Si perde l'ampiezza esatta di questi valori
estremi, ma non il loro segno.

I risultati osservati sono:

| Split | Saturazione | Interpretazione approssimativa | RMSE di ricostruzione |
|---|---:|---:|---:|
| Train | 0,01087% | circa 1 valore ogni 9.200 | 0,00949 |
| Validation | 0,01059% | circa 1 valore ogni 9.400 | 0,00947 |
| Test | 0,01123% | circa 1 valore ogni 8.900 | 0,00952 |

La saturazione è quindi molto bassa e quasi identica nei tre split. Questo è
un segnale positivo: la scala ricavata esclusivamente dal train descrive bene
anche validation e test. I valori non saturati subiscono comunque il normale
arrotondamento ai livelli discreti INT8. L'RMSE di circa `0,0095` misura
l'errore complessivo prodotto da arrotondamento e clipping; è piccolo rispetto
alla deviazione standard degli hidden normalizzati, che è circa `1`. La prova
decisiva resta tuttavia il confronto della Recall@6 sul validation set.

La quantizzazione non elimina l'apprendimento end-to-end della proiezione:
durante il training il batch INT8 viene trasferito alla GPU, dequantizzato in
float16 mediante la scala salvata e passato alla Dense trainable
`4096 -> H_D`. È però un'approssimazione lossy dell'hidden normalizzato, quindi
velocità e Recall@6 devono entrambe essere confrontate con la variante float16.

### Creazione degli array INT8

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/quantize_materialized_hidden_inputs.py \
  --input-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/07_routing_layer_materialized_raw_hidden_f16 \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/08_routing_layer_materialized_raw_hidden_int8 \
  --batch-size 4096 \
  --calibration-sample-rows 4096 \
  --calibration-percentile 99.99
```

Il report viene scritto in
`08_routing_layer_materialized_raw_hidden_int8/artifacts/quantization_summary.json`.

### Training INT8, con la stessa architettura logica

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_quantized_hidden_token_embedding_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/03_routing_layer_router_state/arrays \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/08_routing_layer_materialized_raw_hidden_int8 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/ablations/05_routing_layer_router_state_hidden_pca64_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/neural_network/ffn_raw_hidden64_token64_quantized_int8_HL512_DO03_LR000069972_EP1000_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --epochs 1000 \
  --early-stopping-patience 5
```

Per prima cosa conviene osservare il tempo dei primi due epoch. Se rimane molto
più alto delle run PCA/token, la quantizzazione non ha risolto il collo di
bottiglia su questa macchina e non conviene completare 1.000 epoch.
