# Neural-network experiments

Questa cartella contiene gli output degli esperimenti neurali eseguiti sul corpus BUILD_DATASET_2. Gli script vivono in `ML_pipeline/neural_network/`; qui vengono salvati soltanto modelli, report e stato del tuner per ogni esperimento.

Ogni run deve usare una sotto-cartella con un nome descrittivo, per esempio:

```text
ml/neural_network/
├── ffn_512_256/
│   ├── models/ffn.keras
│   └── reports/training_report.json
└── ffn_random_search_v1/
    ├── tuner/
    ├── models/best_ffn.keras
    └── reports/tuning_report.json
```

I file prodotti dall'addestramento sono ignorati da Git: il repository conserva la struttura e la documentazione, non gli artefatti pesanti.

Gli script neurali rifiutano una cartella di esperimento gia' esistente, per
evitare sovrascritture accidentali di modelli, report o risultati del tuning.
