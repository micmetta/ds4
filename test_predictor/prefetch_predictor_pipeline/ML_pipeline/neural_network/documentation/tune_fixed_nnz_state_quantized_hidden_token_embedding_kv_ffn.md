# Hyperband Build 5 — tuning della FFN fixed-NNZ

## Scopo

Lo script `neural_network/tune_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py`
cerca configurazioni della rete Build 5 senza modificare dataset, artefatti o
run esistenti. Usa train e validation; il test set resta riservato alla
valutazione finale.

Usa **Hyperband** con `val_recall_at_6` come obiettivo da massimizzare. Le
activation dei rami raw hidden e token sono fisse a ReLU.

## Spazio di ricerca

| Parte della rete | Scelte |
|---|---|
| Raw hidden | `4096D -> Dense(64 | 512 | 1024) -> ReLU` |
| Token embedding congelato | `4096D -> Dense(64 | 512 | 1024) -> ReLU` |
| KV corrente | diretto `512D`, oppure `512D -> Dense(64 | 128 | 512) -> ReLU` |
| Dopo la concatenazione | nessun layer, oppure `Dense(512) -> ReLU` |
| Learning rate Adam | `0.00003`, `0.00005`, `0.00006997208`, `0.0001`, `0.00015` |

Il valore `0.00006997208` e' incluso esattamente perché è la baseline.
Lo spazio teorico contiene `3 × 3 × 4 × 2 × 5 = 360` combinazioni. Hyperband
non le addestra tutte fino in fondo: elimina presto quelle con Recall@6
validation meno promettente.

Parametri fissi nello screening:

```text
state fixed-NNZ        : Dense(512) -> ReLU -> Dropout(0.3)
batch size             : 4096
shuffle block size     : 65536
L2                     : 0.0
policy TensorFlow      : float32
```

La mixed precision globale resta esclusa: nelle prove Build 5 è risultata più
lenta sul sistema corrente.

## Comando per lo screening

Il primo screening usa al massimo 20 epoche nel budget maggiore e pazienza 3.
Con `max_epochs=20`, `factor=3` e una iterazione Hyperband, KerasTuner avvia
22 configurazioni iniziali distinte. Le configurazioni migliori vengono
promosse in round successivi, per un totale di 29 esecuzioni di trial: alcune
sono quindi la continuazione della stessa configurazione, non nuovi candidati.
Non vengono esplorate tutte le 360 combinazioni.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/tune_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1 \
  --state-hidden-units 512 \
  --dropout 0.3 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --max-epochs 20 \
  --early-stopping-patience 3 \
  --factor 3 \
  --hyperband-iterations 1
```

Un trial 1024D può richiedere molta più VRAM. Se il backend non lo esegue, il
tuner registra il trial fallito e continua con gli altri, fino al limite di
dieci fallimenti consecutivi.

## Salvataggio, Ctrl+C e ripresa

```text
<output-dir>/
├── tuner/build5_fixed_nnz_hyperband/   stato interno KerasTuner e trial
├── trial_progress/<trial-id>.jsonl     metriche e durata di ogni epoca terminata
├── trial_progress/<trial-id>_timing.json durata totale del trial
├── trial_checkpoints/<trial-id>/best.weights.h5
└── reports/
    ├── search_configuration.json
    ├── trials_snapshot.json
    ├── tuning_report.json
    └── interruption_report.json        creato dopo Ctrl+C
```

Alla fine di ogni trial viene scritto `trials_snapshot.json`; dopo ogni epoca
terminata viene aggiunta una riga JSONL. `Ctrl+C` viene intercettato: lo script
scrive un ultimo snapshot e il report di interruzione. I trial completati
restano quindi analizzabili integralmente con iperparametri, score, tempi,
metriche per epoca e checkpoint migliore. Del trial attivo restano le epoche concluse,
ma non viene considerato un confronto finale.

Per riprendere usare lo stesso comando e aggiungere `--resume`:

```bash
... --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1 --resume
```

La configurazione viene confrontata con `reports/search_configuration.json`;
se dati o spazio di ricerca differiscono, la ripresa viene rifiutata per non
mescolare trial incompatibili.

## Fase 2: conferma lunga

`reports/tuning_report.json` elenca fino a cinque configurazioni migliori.
Ognuna sarà rieseguita con il trainer ordinario, in una nuova output directory,
usando:

```text
--epochs 1000
--early-stopping-patience 5
```

La fase lunga non è automatica: separarla conserva confronti chiari e non
spende risorse sulle configurazioni scartate.

La conferma lunga del migliore trial `0024` e' stata eseguita separatamente,
partendo dai suoi pesi dell'epoca 20. Con `--epochs 1000` e pazienza 5 si e'
fermata all'epoca 56, con il massimo all'epoca 51:

```text
validation Recall@6 iniziale (trial 0024) : 0.490852605
validation Recall@6 migliore dopo ripresa : 0.502358287
```

Questo e' il **migliore risultato di validation Build 5 ottenuto finora**.
La sua valutazione finale, eseguita una sola volta sul test set da 209.400
esempi, ha ottenuto Recall@6 **`0.506056192`**: e' il migliore risultato Build
5 ottenuto finora anche sul test. Il modello risultante e' in
`retrain_best_hyperband_trial_0024_EP1000_PAT5/models/ffn.keras`; il
checkpoint dei soli pesi migliori e' `models/best.weights.h5`. Il checkpoint
Hyperband originale `trial_checkpoints/0024/best.weights.h5` e' stato
conservato invariato.

## Valutazione una sola volta del checkpoint migliore sul test set

Il test set non viene usato durante Hyperband. Dopo avere scelto il miglior
trial tramite validation, si può valutare il suo checkpoint una sola volta sul
test set con `evaluate_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn_checkpoint.py`.
Lo script ricostruisce esattamente la rete, carica soltanto i suoi pesi salvati
e produce predizioni: non esegue `model.fit`, non modifica pesi e non usa train
o validation.

Nel tuning `hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1`, il migliore era
il trial KerasTuner `0024` (la numerazione interna parte da zero), con
validation Recall@6 `0.49085259437561035`. Il checkpoint
`trial_checkpoints/0024/best.weights.h5`, non il numero progressivo `#25`
mostrato nel log, e' stato valutato una sola volta sul test e ha ottenuto
Recall@6 `0.494631`.

Successivamente quel checkpoint e' stato riaddestrato nella fase lunga e ha
raggiunto il nuovo record di validation `0.502358287`. La sua valutazione
finale una sola volta sul test ha ottenuto Recall@6 `0.506056192`, superiore
al precedente `0.494631`. Questo e' il checkpoint da considerare migliore per
le future integrazioni runtime.

Il report JSON della valutazione conserva il checkpoint, la configurazione e
il Recall@6 sul test. Una volta letto quel valore, non va scelto un altro trial
in base al test: il test resta una verifica finale imparziale.
