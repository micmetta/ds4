# Routing multi-hot 256

Questa ablation usa esclusivamente i sei expert della routing history H=1 come
un vettore sparse di 256 posizioni. Sei posizioni valgono `1`; l'ordine Top-6
non viene conservato.

Le sottocartelle contengono matrici/label (`arrays/`), metadati (`artifacts/`),
modelli addestrati (`models/`) e metriche (`reports/`).
