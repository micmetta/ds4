# Preprocessing base per il training

Questa cartella contiene la rappresentazione tabellare e gli array NumPy
generati dal JSONL derivato BUILD_1.

- `prefetch_training_build_1.parquet`: tabella completa, ispezionabile in
  formato colonnare;
- `feature_columns.json`: ordine e descrizione delle 41 feature numeriche di
  base;
- `split_by_prompt.json`: mapping riproducibile prompt -> train/validation/test;
- `preprocessing_summary.json`: conteggi, shape e percorso del Parquet;
- `arrays/`: matrici dense `X`, label multi-hot `y` e target Top-6 per split.

Lo split è per prompt, quindi nessuna riga dello stesso prompt attraversa split
diversi.
