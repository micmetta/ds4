# Collector

Questa cartella contiene il componente C riutilizzabile che esegue le inferenze
sui prompt e raccoglie le trace necessarie per costruire i dataset del
prefetch predictor.

## File contenuti

- `dataset_runner.c`: legge il dataset di prompt, avvia un'inferenza DS4 per
  ciascun prompt e salva le trace nella directory di output indicata dalla
  configurazione della build. Può raccogliere routing, raw router logits,
  `token_id`, gli hidden state `ffn_norm` della feature 4 e le righe KV
  correnti della feature 5.
- `.gitignore`: evita di versionare l'eseguibile compilato `dataset_runner`.
- `doc.md`: questa documentazione.

Il collector non appartiene a una build specifica: `build_dataset_1` e
`build_dataset_2` lo riutilizzano cambiando variabili d'ambiente e directory di
output. L'eseguibile viene generato dal target Makefile `dataset_runner`.

## Configurazione del corpus

Il corpus letto dal runner e la sua provenienza nei `sequence_id` sono
configurabili separatamente:

- `DS4_DATASET_PROMPTS_PATH`: percorso del JSONL dei prompt; se omesso usa
  `dataset_definition/prefetch_prompts_v0.jsonl`;
- `DS4_DATASET_CORPUS_ID`: identificatore testuale inserito nei `sequence_id`;
  se omesso usa `prefetch_prompts_v0`.

Per il corpus inglese da 400 prompt si devono impostare entrambi, ad esempio
`DS4_DATASET_PROMPTS_PATH=.../prefetch_prompts_v1_400_english.jsonl` e
`DS4_DATASET_CORPUS_ID=prefetch_prompts_v1_400_english`. In questo modo la
provenienza delle trace resta distinguibile quando verranno unite a quelle V0.

## Cattura opzionale degli hidden e delle KV

Il runner abilita la cattura delle feature dense tramite due flag separati:

- `DS4_DATASET_CAPTURE_HIDDEN=1`: salva `ffn_norm` float32 da 4.096 valori
  per evento routed in `hidden/prompt_XXXX.ffn_norm.f32.bin`;
- `DS4_DATASET_CAPTURE_KV=1`: salva la riga `kv` float32 da 512 valori del
  token e layer correnti in `kv/prompt_XXXX.kv.f32.bin`.

Per ogni evento di routing, DS4 aggiunge al JSONL `hidden_raw_ref` e/o
`kv_raw_ref`. Il riferimento contiene il path, il dtype, la shape e l'offset
binario del vettore. `kv_raw_ref.source_stage` vale `after_rmsnorm_rope`: è la
riga KV già preparata per la cache e disponibile prima del router dello stesso
token/layer.

Le due catture sono indipendenti, ma per BUILD_DATASET_5 è consigliato abilitarle
entrambe: il build rimane autosufficiente per le ablation hidden e KV future.
La dimensione KV è circa 2 KiB per evento, contro 16 KiB per `ffn_norm`.
