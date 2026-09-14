# Routing slots one-hot

Questa ablation usa esclusivamente i sei campi `routing_history_expert_0..5`.
Ogni slot Top-6 viene codificato separatamente con OneHotEncoder, quindi
l'ordine del routing è preservato.

Le sottocartelle contengono matrici/label (`arrays/`), encoder e metadati
(`artifacts/`), modelli addestrati (`models/`) e metriche (`reports/`).
