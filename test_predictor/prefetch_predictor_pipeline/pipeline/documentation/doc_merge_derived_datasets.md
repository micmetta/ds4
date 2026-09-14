# `merge_derived_datasets.py` — fusione di JSONL derivati

## Scopo

Lo script unisce più JSONL già prodotti da `build_derived_dataset.py`, senza
rieseguire DS4 e senza alterare record, target o riferimenti agli hidden state.
Serve per comporre il corpus storico V0 (100 prompt) con V1 (400 prompt) prima
della PCA e del preprocessing ML.

## Controlli

La fusione viene interrotta se:

- manca un campo necessario alla pipeline BUILD_2;
- gli input usano versioni di schema differenti;
- uno stesso `prompt_id` appare in più input;
- uno stesso prompt ha valori `source_dataset` incoerenti;
- l'output richiesto esiste già.

L'output è scritto prima in un file temporaneo e pubblicato solo se tutti i
controlli hanno successo.

## Esempio V0 + V1

Dal root della repository:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/merge_derived_datasets.py \
  --inputs \
    test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_raw.jsonl \
    test_predictor/prefetch_predictor_pipeline/build_dataset_3/derived/prefetch_dataset_build_3_v1_raw_hidden.jsonl \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_3/derived/prefetch_dataset_build_3_v0_v1_raw_hidden.jsonl
```

Il JSONL unito conserva `source_dataset` per ciascun record: `prefetch_prompts_v0`
per il corpus storico e `prefetch_prompts_v1_400_english` per il nuovo corpus.
