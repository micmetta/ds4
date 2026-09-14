# FFN con embedding token GGUF congelati

## Architettura dell'ablation 05

Questa è una nuova FFN separata: non modifica `train_ffn.py`, le matrici
esistenti né i modelli delle ablation `03` e `04`.

```text
state sparse 4472D -> Dense hidden (es. 512, ReLU) --+
token_id -> lookup GGUF congelato 4096D -> Dense lineare 64D -+-> concatenazione -> output sigmoid 256D
token_position standardizzata train-only --------------------+
```

La tabella `token_embd.weight` non è allenabile. La proiezione `4096 -> 64`
è invece allenata insieme alla FFN e permette di usare anche token assenti nel
train, purché il loro embedding GGUF sia disponibile. La sigmoid finale resta
corretta perché l'obiettivo è multi-label: il router target contiene sei
expert, non una sola classe mutuamente esclusiva.

## Training

Da `~/ds4`, dopo aver creato gli input `05`:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_token_embedding_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/04_routing_layer_router_state_hidden_pca64/arrays \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/05_routing_layer_router_state_hidden_pca64_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/neural_network/ffn_routing_layer_router_state_hidden_pca64_token_embedding_HL512_DO03_LR000069972_EP500_BS4096 \
  --hidden-layers 512 \
  --activation relu \
  --dropout 0.3 \
  --learning-rate 0.00006997208 \
  --batch-size 4096 \
  --epochs 500 \
  --early-stopping-patience 5
```

Con una sola dimensione in `--hidden-layers`, quella è la branca sparse
iniziale. Eventuali dimensioni successive aggiungono layer dense dopo la
concatenazione. L'output directory deve essere nuova.

Il warning Keras relativo a `shuffle=True` con `tf.data.Dataset` non indica che
gli esempi train restino ordinati: il generatore del progetto rimescola gli
indici train a ogni ripetizione. Il warning riguarda soltanto il flag generico
di Keras, che non viene usato quando l'input è un dataset già costruito.

## Test finale

Il test va eseguito soltanto dopo avere scelto la configurazione su validation:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/evaluate_token_embedding_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/04_routing_layer_router_state_hidden_pca64/arrays \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/ablations/05_routing_layer_router_state_hidden_pca64_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/token_embedding_artifacts \
  --model-path <output-dir>/models/ffn.keras \
  --report-path <output-dir>/reports/test_report.json \
  --batch-size 4096
```
