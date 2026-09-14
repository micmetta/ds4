# `create_prompt_split.py` — split stratificato per prompt

## Scopo

Lo script crea un mapping `prompt_id -> train/validation/test` stratificato per
categoria. Può preservare integralmente uno split precedente e assegnare solo
i prompt nuovi: questa modalità mantiene confrontabili gli esperimenti V0 da
100 prompt e il corpus combinato V0+V1 da 500 prompt.

## Strategia V0 + V1

V0 contiene 10 prompt per categoria, già assegnati con rapporto 8/1/1. V1 ne
aggiunge 40 per categoria. Conservando V0, lo script assegna V1 per ottenere,
in ogni categoria del corpus combinato, 40 prompt train, 5 validation e 5 test.
Il totale è quindi 400/50/50 prompt.

## Esempio

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/create_prompt_split.py \
  --prompts \
    test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl \
    test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v1_400_english.jsonl \
  --preserve-split test_predictor/prefetch_predictor_pipeline/build_dataset_1/ml/training/split_by_prompt.json \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_3/ml/training/split_by_prompt.json \
  --seed 42
```

Il file di output non viene sovrascritto: per rigenerarlo bisogna scegliere una
nuova destinazione oppure rimuovere esplicitamente quella esistente.
