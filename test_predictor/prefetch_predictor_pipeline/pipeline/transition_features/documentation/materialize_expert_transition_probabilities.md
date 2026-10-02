# Materializzazione delle probabilita' di transizione degli expert - Build 6

## Obiettivo

Lo script `../materialize_expert_transition_probabilities.py` costruisce la
feature `expert_transition_probabilities` definita in
[`dataset_definition/features.md`](../../../dataset_definition/features.md).
Per ogni layer routed produce:

\[
P_l(i\mid j),
\]

la probabilita' che l'expert target $i$ sia attivo al token successivo, sapendo
che l'expert corrente $j$ e' attivo nel token presente. Per ciascun record il
predictor riceve il vettore $Q_{t,l}$ 256D, ottenuto facendo la media delle sei
colonne associate agli expert in `experts_t`.

## Perche' gli artefatti sono separati

Le matrici $P$ sono la statistica di riferimento e vengono salvate in
`float32`: `[40, 256, 256]` pesa esattamente 10 MiB. I vettori 256D sono invece
input della rete, vengono salvati in `float16` e rimangono allineati alle righe
degli array state esistenti. Per l'intera Build 6 occupano circa 0,959 GiB,
anziche' 1,918 GiB in `float32`.

## Anti-leakage

- Validation e test usano una sola matrice stimata da tutte e sole le
  transizioni train.
- Il train usa out-of-fold raggruppato per `prompt_id`. Con il default di 5
  fold, un prompt e' assegnato stabilmente a `prompt_id % 5`; le righe di quel
  prompt consultano una matrice che esclude tutti i conteggi del loro fold.
- Le cinque matrici OOF e la matrice globale vengono conservate in `float32`
  per audit e riproducibilita'. Occupano 60 MiB complessivi con 5 fold.

## Output

```text
<output-dir>/
  arrays/
    expert_transition_probabilities_train.npy       # float16 [N_train, 256]
    expert_transition_probabilities_validation.npy  # float16 [N_validation, 256]
    expert_transition_probabilities_test.npy        # float16 [N_test, 256]
  artifacts/
    transition_probabilities_train_global_float32.npy
    transition_probabilities_train_oof_fold_00_float32.npy
    ...
    materialization_summary.json
```

L'asse degli array $P$ e' `[layer_index, target_expert_i, current_expert_j]`:
la colonna `j` contiene quindi $P_l(i\mid j)$ per tutti i target $i$.

## Report HTML di consultazione

Lo script `../create_expert_transition_probabilities_html_report.py` crea un
file HTML autosufficiente, pensato esclusivamente per ispezione umana. Include
tutti i valori della matrice globale `float32` e permette di selezionare un
layer routed, osservare la heatmap 256 x 256 e leggere il valore esatto della
cella puntata. Non viene usato durante preprocessing o addestramento.

Il report incorpora la matrice binaria nel file tramite Base64: e' quindi piu'
grande della matrice `.npy` (circa 13,4 MiB anziche' 10 MiB), ma si puo' aprire
direttamente con un browser senza server locale, JSON aggiuntivo o accesso ai
file della Build 6.

Dopo la materializzazione, generarlo con:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/transition_features/create_expert_transition_probabilities_html_report.py \
  --probabilities test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/transition_features/expert_transition_probabilities_v1/artifacts/transition_probabilities_train_global_float32.npy \
  --output-html test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/transition_features/expert_transition_probabilities_v1/reports/transition_probabilities_train_global.html \
  --first-layer 3
```

## Esecuzione

Da `/home/michele/ds4`:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/transition_features/materialize_expert_transition_probabilities.py \
  --input test_predictor/prefetch_predictor_pipeline/build_dataset_6/derived/prefetch_dataset_build_6_previous_layer.jsonl \
  --split-by-prompt test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/01_current_previous_state/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/transition_features/expert_transition_probabilities_v1 \
  --alpha 0.1 \
  --train-oof-folds 5
```

Non rieseguire il comando sulla stessa directory di output: lo script si ferma
volutamente se esiste gia', per non sovrascrivere un artefatto statisticamente
diverso.
