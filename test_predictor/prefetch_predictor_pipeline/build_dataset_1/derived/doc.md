# Dataset derivato BUILD_1

Questa cartella contiene i file ottenuti dai trace raw dopo il join con il
corpus V0 e la costruzione degli esempi supervisionati
`E_l(t) -> E_l(t+1)`.

- `prefetch_dataset_build_1.jsonl`: 199.000 record JSONL con feature 0–3 e
  `target_experts_next_token` multi-label.
- `embedding_table_manifest.json`: metadati della tabella di embedding GGUF;
  gli embedding non sono duplicati dentro ogni record JSONL.

Questo JSONL è l'input storico del preprocessing tabellare BUILD_1.
