# Build dataset 5: workflow completo fino all'addestramento

Questa guida documenta la pipeline usata per costruire `build_dataset_5` e preparare l'addestramento della rete neurale con le nuove `kv_features_current`.

Il build usa 1.000 prompt, mantiene lo stesso split per prompt del `build_dataset_4` e conserva i build precedenti senza sovrascriverli. La PCA dei vettori KV da 512 dimensioni **non** e' inclusa: in questo esperimento usiamo le cinque statistiche dirette e il vettore KV completo a 512 dimensioni.

## Struttura della configurazione finale

La prima rete documentata in questa guida riceve:

- stato di routing: history Top-6, layer, ID e score dei 16 router logits;
- cinque feature `kv_features_current`: norma L2, RMS, media, deviazione standard e massimo assoluto;
- vettore KV compatto corrente completo a 512D, normalizzato per layer e passato direttamente alla fusione;
- hidden state `ffn_norm` raw 4096D, normalizzato per layer, quantizzato int8 e proiettato dalla rete a 64D;
- embedding congelato del token corrente 4096D, proiettato dalla rete a 64D;
- posizione del token standardizzata.

La label resta l'insieme multi-label degli esperti del token successivo nello stesso layer, valutato con Recall@6.

## Prima configurazione: vettore KV completo senza proiezione

La prima run fornisce alla rete anche il vettore `kv` completo di 512 dimensioni. Il vettore e' normalizzato per layer con statistiche calcolate solo sul train e materializzato in un array contiguo float16 per evitare letture casuali dai file raw. Non viene applicata PCA e non viene aggiunto un layer `Dense(512 -> 64)` sul ramo KV.

Le cinque statistiche KV resteranno comunque nello state branch. Sono funzioni deterministiche del vettore completo, ma rendono immediatamente disponibili al classificatore grandezze come norma, RMS e massimo assoluto; il confronto con la prima run misurera' il contributo informativo aggiuntivo delle 512 componenti.

La rete multi-input pianificata sara':

```text
state 4.413D
  (routing history, layer, router ID/score, 5 statistiche KV)
        |
        v
Dense(512) -> ReLU -> Dropout(0.3) ------------------+
                                                       |
hidden raw 4.096D -> Dense(64) -----------------------+
                                                       |
token embedding congelato 4.096D -> Dense(64) --------+
                                                       +--> Concatenate(1.153D) --> Dense(256) --> sigmoid
KV completo normalizzato 512D -> identita' -----------+       (expert_scores)
                                                       |
token_position standardizzata 1D ---------------------+
```

`Dense(512)` e' il layer `state_hidden_1` gia' presente nella rete: riceve **solo** le feature state 4.413D, non il vettore KV completo. Il ramo KV entra direttamente nella concatenazione con tutte le 512 componenti. Le dimensioni della fusione sono quindi `512 + 64 + 64 + 512 + 1 = 1.153`.

Questa e' la configurazione effettiva del primo training KV. Le cinque statistiche restano nello state branch e il KV completo entra in parallelo nella fusione.

## 0. Corpus completo da 1.000 prompt

### Obiettivo

Unire i tre corpus originali senza modificarli, mantenendo la provenienza di ciascun prompt. Il file combinato e' l'input della raccolta DS4.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/create_combined_prompt_corpus.py \
  --input prefetch_prompts_v0=test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl \
  --input prefetch_prompts_v1_400_english=test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v1_400_english.jsonl \
  --input prefetch_prompts_v2_500_english=test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v2_500_english.jsonl \
  --output test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl
```

### Output

`dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl`, composto da 1.000 prompt con ID univoci da 1 a 1.000.

## 1. Compilazione del collector DS4

### Obiettivo

Compilare il collector con il supporto alle trace esistenti e alla nuova trace `kv` compatta.

### Comando

```bash
make dataset_runner && make ds4
```

### Output

Il binario `test_predictor/prefetch_predictor_pipeline/collector/dataset_runner` e l'eseguibile DS4 aggiornati.

## 2. Raccolta DS4 di routing, hidden e KV

### Obiettivo

Eseguire l'inferenza sui 1.000 prompt e salvare, per ogni coppia *(token, layer)*:

- routing e router logits;
- hidden state `ffn_norm` raw;
- riga `kv` compatta 512D, dopo RMSNorm e RoPE.

La riga KV e' la rappresentazione interna che DS4 aggiunge alla cache del layer e riusa nei token successivi. Non rappresenta due vettori Key e Value classici separati.

### Comando

```bash
DS4_MODEL="gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" \
DS4_DATASET_PROMPTS_PATH="test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl" \
DS4_DATASET_CORPUS_ID="prefetch_prompts_v0_v1_v2_1000" \
DS4_DATASET_OUTPUT_DIR="test_predictor/prefetch_predictor_pipeline/build_dataset_5/raw" \
DS4_DATASET_BUILD_ID="build_dataset_5_kv" \
DS4_DATASET_CAPTURE_HIDDEN=1 \
DS4_DATASET_CAPTURE_KV=1 \
DS4_DATASET_ROUTER_MAX_SCORES=16 \
DS4_DATASET_MAX_TOKENS=64 \
./test_predictor/prefetch_predictor_pipeline/collector/dataset_runner
```

### Output e ripresa

Le trace sono in `build_dataset_5/raw/{routing,hidden,kv}` e il manifest e' `build_dataset_5/raw/manifest.jsonl`.

Lo stesso comando puo' essere rilanciato dopo un'interruzione: il collector salta i prompt con stato `success` nel manifest e riprende da quelli non completati. Non cancellare log vuoti: non indicano un errore.

La raccolta completata contiene 1.000 successi, senza fallimenti, con circa 2.76 GiB di routing, 32.08 GiB di hidden e 4.01 GiB di KV raw.

## 3. Costruzione del derived dataset

### Obiettivo

Trasformare le trace raw in esempi supervisionati causali: feature al token `t`, target di routing al token `t+1` nello stesso layer e riferimenti agli hidden/KV raw.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/build_derived_dataset.py \
  --prompts test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl \
  --raw-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/raw \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv.jsonl \
  --history-length 1 \
  --require-hidden \
  --require-kv
```

### Output

`build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv.jsonl` con 2.062.680 esempi e un riferimento `kv_raw_ref` valido in ogni record.

## 4. Split train/validation/test per prompt

### Obiettivo

Riutilizzare esattamente lo split del build 4: 800 prompt train, 100 validation e 100 test. Questo rende confrontabili le metriche tra build 4 e build 5.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/create_prompt_split.py \
  --prompts test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl \
  --preserve-split test_predictor/prefetch_predictor_pipeline/build_dataset_4/ml/training/split_by_prompt.json \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --seed 42
```

### Output

`build_dataset_5/ml/training/split_by_prompt.json` con split `800 / 100 / 100`.

## 5. Calcolo di `kv_features_current`

### Obiettivo

Leggere una riga KV raw 512D alla volta e aggiungere cinque statistiche economiche al derived dataset:

- `l2_norm`;
- `rms`;
- `mean`;
- `std` (deviazione standard di popolazione);
- `max_abs`.

Questo step non apprende parametri e non normalizza globalmente le feature: ogni valore dipende solo dalla riga KV del token/layer corrente. Non produce PCA KV.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/kv_features/create_kv_current_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv.jsonl \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl
```

### Output

Il nuovo derived dataset contiene `kv_features_current` in tutti i 2.062.680 record. Il file `kv_raw_ref` resta presente per eventuali PCA KV o feature storiche future. Il report associato e':

```text
build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current_summary.json
```

## 6. Preprocessing generico in Parquet e array label

### Obiettivo

Serializzare i record in Parquet e creare le label multi-hot per split. Le cinque statistiche KV diventano colonne del Parquet; i riferimenti raw non vengono inseriti negli array dense del training generico.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/preprocess_dataset.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training
```

### Output

`build_dataset_5/ml/training/prefetch_training.parquet` e le label `y_{train,validation,test}.npy`.

## 7. Ablation state + KV current

### Obiettivo

Creare la matrice sparse usata dalla rete per lo stato di routing e le cinque feature KV. Il `StandardScaler` delle statistiche KV viene fit **solo sul train** e poi riusato invariato per validation e test: non c'e' leakage.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/preprocessing_routing_ablation.py \
  --representation multihot_256_layer_router_state_kv_current \
  --training-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current
```

### Output

Le matrici hanno 4.413 feature: 4.408 dello stato router e 5 KV. Le dimensioni effettive sono:

| Split | Esempi | Matrice X |
|---|---:|---:|
| Train | 1.652.640 | `(1652640, 4413)` |
| Validation | 200.640 | `(200640, 4413)` |
| Test | 209.400 | `(209400, 4413)` |

Lo scaler train-only e' salvato in `artifacts/kv_current_scaler.joblib`.

## 8. Lookup degli embedding token dal GGUF

### Obiettivo

Estrarre dal GGUF soltanto gli embedding dei token ID effettivamente presenti nel build 5, invece dell'intero vocabolario.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/token_features/extract_token_embedding_lookup.py \
  --training-parquet test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/prefetch_training.parquet \
  --gguf-model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts
```

### Output

Lookup di 7.157 token distinti, con embedding congelati float16 da 4.096 dimensioni.

## 9. Input token per la rete

### Obiettivo

Allineare a ogni riga state un indice compatto nel lookup token e una `token_position` standardizzata usando statistiche train-only.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/token_features/prepare_token_embedding_inputs.py \
  --training-parquet test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/prefetch_training.parquet \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding
```

## 10. Riferimenti raw hidden e normalizzazione per layer

### Obiettivo

Allineare le righe `ffn_norm` raw agli array state e calcolare media/deviazione standard per ciascun layer usando esclusivamente train. Gli hidden non vengono ancora caricati tutti in RAM.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/prepare_raw_hidden_inputs.py \
  --raw-hidden-input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/11_routing_layer_router_state_kv_current_raw_hidden_token_embedding \
  --batch-size 4096
```

## 11. Materializzazione degli hidden normalizzati float16

### Obiettivo

Convertire le letture sparse dai file raw in array contigui e normalizzati. Questo elimina il collo di bottiglia delle letture casuali durante l'addestramento.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/materialize_normalized_hidden_inputs.py \
  --raw-hidden-input-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/11_routing_layer_router_state_kv_current_raw_hidden_token_embedding \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/12_routing_layer_router_state_kv_current_materialized_raw_hidden_f16 \
  --batch-size 4096
```

## 12. Quantizzazione hidden int8

### Obiettivo

Ridurre di quattro volte l'I/O rispetto agli array float16. La scala viene calibrata sul solo train al percentile 99.99; durante il training la rete dequantizza sulla GPU prima della proiezione hidden 4096D -> 64D.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/hidden_features/quantize_materialized_hidden_inputs.py \
  --input-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/12_routing_layer_router_state_kv_current_materialized_raw_hidden_f16 \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --batch-size 4096 \
  --calibration-sample-rows 4096 \
  --calibration-percentile 99.99
```

### Output

Per questo build: limite `+-3.89453` e scala `0.0306656`.

## 13. Materializzazione del vettore KV completo 512D

### Obiettivo

Creare un array KV contiguo e allineato alle righe state per ogni split. Lo script calcola media e deviazione standard di ogni componente KV separatamente per layer, usando esclusivamente i record train; applica poi la stessa normalizzazione a train, validation e test. Il risultato resta 512D: non usa PCA e non applica una proiezione appresa.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/kv_features/materialize_normalized_kv_inputs.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --batch-size 4096
```

### Output

`arrays/kv_normalized_{train,validation,test}.npy`, array contigui `float16` con shape `(esempi, 512)`, e gli artefatti train-only `kv_train_mean_by_layer.npy` e `kv_train_std_by_layer.npy`.

## 14. Addestramento della rete neurale con KV completo (prossimo comando)

### Obiettivo

Addestrare la FFN multi-input comparabile alla migliore run precedente, aggiungendo sia le cinque feature KV allo state branch sia il vettore KV completo normalizzato da 512D alla fusione. Il lookup token e' congelato; le proiezioni 4096D -> 64D per hidden e token sono apprese durante il training. Il ramo KV non contiene una proiezione appresa.

Il test set non deve essere usato per tuning. Il riferimento validation da battere e' Recall@6 = `0.4634` della configurazione build 4 con hidden raw int8 e token embedding, ma senza le nuove feature KV.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/ffn_router_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_HL512_DO03_LR000069972_EP1000_BS4096 \
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

L'early stopping monitora il validation set e ripristina il miglior checkpoint. Il test set verra' valutato solo dopo avere scelto la configurazione definitiva.

### Risultato della prima run

La prima run con tutte le feature elencate sopra ha completato 167 epoche. Il miglior `val_recall_at_6` e' stato `0.4724` all'epoca 162; dopo cinque epoche senza un miglioramento strettamente superiore, l'early stopping ha ripristinato quei pesi. La valutazione finale sul validation set e' `Recall@6 = 0.4724`.

Come confronto descrittivo, la migliore run build 4 senza feature KV aveva `Recall@6 = 0.4634`; la differenza osservata e' `+0.0090` (0,90 punti percentuali). Non attribuiamo ancora causalmente tutto il guadagno al vettore KV completo: build 5 e build 4 derivano da raccolte DS4 separate. Per isolare il contributo delle 512 componenti KV rispetto alle sole cinque statistiche, serve una run abbinata build 5 senza il ramo `direct_kv512`.

## 15. Test di mixed precision globale

### Obiettivo

Ripetere la stessa configurazione della sezione 14 con batch, dati, split, seed, macro-block shuffle e iperparametri invariati, ma chiedendo a TensorFlow di usare la policy globale `mixed_float16` per i layer compatibili con la GPU. I pesi restano memorizzati in modo stabile e l'output sigmoid viene mantenuto `float32`, quindi loss e Recall@6 non perdono precisione. Questo test misura soprattutto il tempo di addestramento; piccole differenze numeriche rispetto alla run `float32` sono normali.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/ffn_router_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_MP16_HL512_DO03_LR000069972_EP1000_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --mixed-precision \
  --epochs 1000 \
  --early-stopping-patience 5
```

## 16. Test batch size 8192

### Obiettivo

Misurare l'effetto del raddoppio del batch mantenendo `float32`: questa scelta isola il batch size dal test di mixed precision. Dati, feature, seed, learning rate e macro-block shuffle restano uguali; gli update dell'ottimizzatore per epoca passano pero' da 404 a circa 202, quindi Recall@6 e tempo devono essere confrontati come una nuova configurazione, non come una replica identica della run con batch 4096.

La cartella di output e' diversa da quella del test MP16 interrotto, che viene conservato senza essere sovrascritto.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/ffn_router_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_HL512_DO03_LR000069972_EP1000_BS8192 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 8192 \
  --shuffle-block-size 65536 \
  --epochs 1000 \
  --early-stopping-patience 5
```

## 17. Profiling della pipeline di training

### Obiettivo

Individuare il collo di bottiglia prima di cambiare nuovamente architettura o batch size. La modalita' `--profile-only-batches` costruisce la stessa rete e usa gli stessi input della run completa, effettua alcuni aggiornamenti train reali e salva un trace TensorBoard. Non completa un'epoca, non esegue validation/test e non salva un modello: il risultato serve solo a capire quanto tempo va in preparazione batch, input sparse, I/O, trasferimento host→GPU e calcolo GPU.

### Comando

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/profile_router_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --profile-only-batches 25 \
  --profile-warmup-batches 3
```

### Output e lettura del trace

Il report sintetico e' `reports/profile_report.json`; il trace e' nella directory `tensorboard_profile` dell'output. Per aprirlo:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  tensorboard --logdir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/profile_router_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_BS4096
```

In TensorBoard aprire la scheda **Profile**. Se la maggior parte del tempo e' in input/Python/host, si interviene sulla pipeline dati; se e' nelle operazioni sparse o Dense GPU, si valuta una branca state strutturata.

### Risultato del profiling

Il profiling a batch 4096 ha mostrato un device step medio di circa 9 ms e ha classificato la pipeline come potenzialmente input-bound: 35,5% del tempo campionato era in I/O o esecuzione Python. L'analisi host ha attribuito il 100% del tempo di input al preprocessing; l'operazione piu' costosa era il `Generator` Python che, nel trainer sparse, estrae una slice CSR e costruisce una `tf.SparseTensor` per batch.

## 18. Percorso alternativo: state fixed-NNZ contiguo

### Obiettivo

Eliminare la conversione CSR -> `tf.SparseTensor` eseguita a ogni batch, senza densificare lo state a 4.413 dimensioni e senza cancellare alcun artefatto o trainer precedente.

Nel build 5 ogni riga state ha esattamente 44 feature non nulle. Il nuovo formato salva quegli indici e valori in array contigui e aggiunge uno slot sentinella per rappresentare il bias. Una lookup trainabile e una somma pesata sostituiscono quindi il primo `Dense(512)` sparse, mantenendo numero di parametri, feature e semantica del ramo state. La documentazione dettagliata e' [FIXED_NNZ_STATE_INPUTS.md](../pipeline/state_features/FIXED_NNZ_STATE_INPUTS.md).

Questo e' un percorso parallelo: le matrici CSR in `ablations/09` e il trainer `train_quantized_hidden_token_embedding_kv_ffn.py` restano intatti e utilizzabili per ogni confronto o rollback.

### 18.1 Materializzare gli slot state

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/state_features/materialize_fixed_nnz_state_inputs.py \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --chunk-rows 65536
```

Output nuovo e separato: `ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state`. Per ogni split contiene `state_feature_indices_<split>.npy` (`uint16`) e `state_feature_values_<split>.npy` (`float32`), entrambi con 45 slot: 44 feature originali piu' un bias sentinella.

### 18.2 Profilare la nuova pipeline prima dell'addestramento completo

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/profile_fixed_nnz_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --profile-only-batches 8 \
  --profile-warmup-batches 2
```

Solo dopo il confronto del profiling verra' lanciata una run completa con un output directory distinto.

### Risultato del profiling fixed-NNZ

Il confronto a batch 4096 conferma il beneficio del nuovo formato:

| Misura TensorBoard | Trainer CSR sparse | Trainer fixed-NNZ | Variazione |
|---|---:|---:|---:|
| Tempo del `Generator` Python | 261 ms / 5 chiamate = circa 52,2 ms/batch | 29 ms / 8 chiamate = circa 3,6 ms/batch | circa 14,4 volte minore |
| Quota step in I/O/Python (`All Others`) | 35,5% | 9,4% | input pipeline molto meno dominante |
| Device step medio campionato | 9,0 ms | 19,4 ms | maggiore lavoro GPU per lookup e somma pesata |

Il nuovo ramo state sposta deliberatamente una piccola parte del lavoro dall'host alla GPU: l'embedding di 45 slot e la somma pesata sono piu' costosi sulla GPU del precedente kernel sparse, ma eliminano la costruzione Python della `SparseTensor`. I numeri XProf non sono una stima diretta e completa del tempo di un'epoca, poiche' prefetch e GPU possono sovrapporsi; mostrano pero' che il collo di bottiglia identificato nel `Generator` e' stato rimosso. Il prossimo test deve quindi misurare tempi e metriche per alcune epoche reali.

### Risultato del benchmark di 30 epoche

Il benchmark `benchmark_fixed_nnz_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_HL512_DO03_LR000069972_EP30_BS4096` ha completato le 30 epoche.

| Misura, epoche 1-30 | Trainer CSR sparse precedente | Trainer fixed-NNZ | Differenza |
|---|---:|---:|---:|
| Tempo totale training+validation | 5.717 s | 1.676 s | 3,41 volte piu' veloce |
| Tempo medio per epoca | 190,6 s | 55,9 s | -70,7% |
| Mediana tempo per epoca | 199 s | 52 s | -73,9% |
| `val_recall_at_6` all'epoca 30 | 0,4583 | 0,4594 | +0,0011 |

Il trainer fixed-NNZ non mostra il degrado progressivo dei tempi che caratterizzava la pipeline CSR. La run di 30 epoche non e' una valutazione finale: serve a validare velocita' e coerenza della curva. La configurazione definitiva deve essere lanciata in una nuova output directory con fino a 1000 epoche ed early stopping, mantenendo batch 4096, split e iperparametri invariati.

## 19. Test mixed precision sul trainer fixed-NNZ

### Obiettivo

Nel profiling fixed-NNZ il 99,3% del compute GPU era float32. A differenza del precedente test MP16 sul trainer CSR, il collo di bottiglia Python e' stato quasi eliminato; ha quindi senso misurare nuovamente `mixed_float16` sulla nuova pipeline.

Il flag `--mixed-precision` cambia soltanto la policy TensorFlow dei layer compatibili. I pesi restano memorizzati in float32 e `expert_scores` resta esplicitamente float32, quindi probabilita', loss e Recall@6 mantengono precisione numerica stabile. Dati, seed, batch, architettura e iperparametri restano identici al benchmark fixed-NNZ float32.

### Comando benchmark

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/benchmark_fixed_nnz_state_raw_hidden_int8_64_token_embedding64_token_position_kv_current_stats_direct_kv512_MP16_HL512_DO03_LR000069972_EP30_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --mixed-precision \
  --epochs 30 \
  --early-stopping-patience 5
```

Confrontare il tempo totale e il `val_recall_at_6` con il benchmark float32 della sezione 18. Se MP16 non migliora sensibilmente il tempo o peggiora la metrica, il trainer fixed-NNZ float32 resta la configurazione di riferimento.

### Risultato iniziale del test MP16

Il test MP16 e' stato interrotto dopo tre epoche, perche' il peggioramento dei tempi era gia' netto e le metriche risultavano praticamente sovrapposte al float32:

| Epoca | fixed-NNZ float32 | fixed-NNZ MP16 | `val_recall_at_6` float32 / MP16 |
|---|---:|---:|---:|
| 1 | 44 s | 70 s | 0,0644 / 0,0644 |
| 2 | 93 s | 135 s | 0,1722 / 0,1721 |
| 3 | 92 s | 168 s | 0,2851 / 0,2849 |

Sulle prime tre epoche MP16 richiede 373 s contro 229 s in float32: circa 1,63 volte piu' lento. Il ramo state fixed-NNZ e' dominato da embedding gather, moltiplicazioni e riduzioni su tensori `(batch, 45, 512)`, operazioni per le quali i cast e la gestione mixed precision non compensano con Tensor Core come avviene per grandi matmul dense. Il trainer fixed-NNZ **float32** resta quindi il riferimento e non va usato `--mixed-precision` nelle run successive.

## 20. Manutenzione dello spazio: hidden float16 intermedi rimossi

Gli hidden `float16` materializzati sono un passaggio intermedio tra i
riferimenti raw normalizzati e gli input `int8`. Una volta completata con
successo la quantizzazione, training ed evaluation della FFN usano gli array
`int8`, la rispettiva scala di dequantizzazione e le statistiche train-only;
gli array `float16` non vengono piu' letti.

Per recuperare spazio sul disco fisico del PC sono quindi state rimosse queste
sole directory rigenerabili:

```text
build_dataset_5/ml/ablations/12_routing_layer_router_state_kv_current_materialized_raw_hidden_f16
build_dataset_4/ml/ablations/07_routing_layer_materialized_raw_hidden_f16
```

Ognuna occupava circa 16 GiB, per un recupero complessivo di circa 32 GiB nel
filesystem WSL. Restano invece conservati gli input quantizzati necessari:

```text
build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8
```

La directory Build 4 rimossa appartiene a una pipeline storica e non e'
richiesta da Build 5 o Build 6. Entrambi gli intermedi float16 possono essere
ricostruiti in modo deterministico rieseguendo la corrispondente fase di
materializzazione hidden sui rispettivi riferimenti raw e statistiche per
layer. Non vanno invece rimossi gli array int8 o le loro scale se si vuole
addestrare, valutare o riprodurre i modelli correnti.
