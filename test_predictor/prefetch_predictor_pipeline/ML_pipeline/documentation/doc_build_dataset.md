# `build_dataset.py` — dalle trace raw al dataset supervisionato JSONL

## Scopo dello script

`build_dataset.py` trasforma le trace di routing raccolte da DS4 in un dataset
JSONL supervisionato. Non addestra alcun modello e non esegue inferenze: legge
esclusivamente file già prodotti dal collector.

Ogni record rappresenta un layer routed `l` durante il decoding del token `t`:

$$
E_l(t) \longrightarrow E_l(t+1)
$$

Le feature del token corrente diventano l'input del futuro predictor; gli
expert selezionati nello stesso layer al token successivo diventano il target.
Nella configurazione BUILD_DATASET_1 la history predefinita è `H = 1`.

## Input e output

Con i valori di default, lo script legge:

```text
dataset_definition/prefetch_prompts_v0.jsonl
build_dataset_1/raw/manifest.jsonl
build_dataset_1/raw/routing/prompt_XXXX.routing.jsonl
```

e genera:

```text
build_dataset_1/derived/
├── prefetch_dataset_build_1.jsonl
└── embedding_table_manifest.json
```

| File | Utilità |
|---|---|
| `prefetch_dataset_build_1.jsonl` | Un record supervisionato per ogni coppia valida `(token t, layer l)`. |
| `embedding_table_manifest.json` | Descrive dove recuperare la tabella `token_embd.weight` dal GGUF senza duplicare l'embedding 4096D in ogni record. |

## Flusso completo

```text
main()
  |
  +-- load_prompts(path)
  |     legge i prompt V0 e costruisce il mapping prompt_id -> metadati
  |
  +-- successful_manifest_entries(path)
  |     seleziona l'ultima raccolta con status="success" per ogni prompt_id
  |
  +-- build_dataset(prompts, manifests, history_length, output_path)
  |     percorre le trace valide e scrive il JSONL supervisionato
  |       |
  |       +-- load_routing_trace(path)
  |       |     conserva soltanto gli eventi decode non-hash, ordinati per layer e token
  |       |
  |       `-- make_record(prompt, manifest, layer, current, history, target)
  |             costruisce un record JSONL con feature a t e target a t+1
  |
  `-- write_embedding_manifest(path, model_path)
        salva il riferimento stabile alla tabella embedding nel GGUF
```

`main()` orchestra queste funzioni, controlla gli argomenti, crea la directory
di output se manca e stampa il numero finale di esempi e sequenze raccolte.

## 1. `load_prompts()`

`load_prompts(path: Path) -> dict[int, dict]` legge
`prefetch_prompts_v0.jsonl`, una riga JSON per prompt. Il risultato è un
mapping in memoria:

```text
prompt_id -> {id, category, prompt, ...}
```

Lo script blocca l'esecuzione se trova due volte lo stesso `id`: senza un ID
univoco non sarebbe possibile associare in modo sicuro una trace al prompt che
l'ha prodotta.

## 2. `successful_manifest_entries()`

`successful_manifest_entries(path: Path) -> list[dict]` legge
`raw/manifest.jsonl`. Un prompt può essere stato raccolto più volte; la
funzione conserva soltanto l'ultima entry con `status = "success"` per ciascun
`prompt_id`.

In questo modo non entrano nel dataset trace incomplete o fallite, né copie
obsolete dello stesso prompt. Le entry finali vengono ordinate per `prompt_id`
per rendere l'output riproducibile.

## 3. `load_routing_trace()`

`load_routing_trace(path: Path) -> dict[int, list[dict]]` carica una trace
`prompt_XXXX.routing.jsonl` e la raggruppa per layer routed.

La funzione:

1. mantiene soltanto gli eventi con `phase = "decode"`;
2. scarta gli eventi `is_hash = true`, che seguono il mapping deterministico e
   non sono esempi del routing MoE da predire;
3. richiede `token_index`, `token_id`, `layer` ed `experts`;
4. ordina gli eventi di ogni layer per `token_index`;
5. verifica che un layer non abbia due eventi con la stessa posizione token.

Il risultato è concettualmente:

```text
layer 17 -> [evento token 3, evento token 4, evento token 5, ...]
layer 18 -> [evento token 3, evento token 4, evento token 5, ...]
...
```

## 4. `build_dataset()` e costruzione degli esempi

`build_dataset()` attraversa ogni prompt riuscito e ogni layer della sua
trace. Per ogni posizione valida seleziona:

```text
history = [E_l(t-H+1), ..., E_l(t)]
current = E_l(t)
target  = E_l(t+1)
```

Prima di generare il record verifica che `target.token_index` sia esattamente
`current.token_index + 1`. Questo impedisce di associare accidentalmente due
token non consecutivi.

Con `--history-length 1`, il primo esempio disponibile per un layer usa:

```text
feature history: E_l(3)
target:          E_l(4)
```

Il token 3 è il primo token routed utile dopo la fase hash deterministica. Il
record viene creato da `make_record()` e scritto immediatamente nel JSONL,
senza accumulare l'intero dataset in RAM.

## 5. `make_record()` e formato del record

`make_record()` combina metadati del prompt, evento corrente, history e target
in un record con schema `prefetch_dataset_v1`.

I campi principali sono:

| Gruppo | Campi | Significato |
|---|---|---|
| Provenienza | `source_dataset`, `prompt_id`, `prompt_category`, `sequence_id` | Ricollegano ogni esempio al prompt e alla run DS4 originale. |
| Stato corrente | `token_position`, `token_id`, `layer`, `experts_t` | Descrivono il token/layer osservato al tempo `t`. |
| Feature 1 | `routing_history` | History degli expert dello stesso layer, lunga `H`. |
| Feature 2 | `router_logits` | Raw logits e ID expert già registrati nel trace corrente. |
| Feature 3 | `token_embedding` | Riferimento esterno alla tabella embedding del GGUF tramite `token_id`. |
| Label | `target_experts_next_token` | I sei expert realmente scelti in `E_l(t+1)`. |

`experts_t` e l'ultimo elemento di `routing_history` coincidono quando `H=1`:
il primo migliora la leggibilità del record, mentre la history rende il formato
estensibile a finestre più lunghe.

## 6. `write_embedding_manifest()`

`write_embedding_manifest()` scrive una sola volta il metadato della tabella
embedding:

```text
source tensor: token_embd.weight
dimension:     4096
storage:       external_lookup
```

Ogni record JSONL conserva quindi solo `token_id` e il riferimento logico alla
tabella. La matrice degli embedding resta nel GGUF e potrà essere estratta o
consultata in un preprocessing dedicato, evitando di salvare 4096 float per
ognuno dei circa 199 mila esempi BUILD_DATASET_1.

## Argomenti disponibili

| Argomento | Default | Utilità |
|---|---|---|
| `--prompts` | dataset V0 | File JSONL dei prompt e delle categorie. |
| `--raw-dir` | `build_dataset_1/raw` | Directory contenente manifest, log e trace raw. |
| `--output` | `build_dataset_1/derived/prefetch_dataset_build_1.jsonl` | Dataset JSONL supervisionato da creare. |
| `--history-length` | `1` | Lunghezza `H` della routing history; deve essere almeno 1. |
| `--model` | GGUF DeepSeek V4 Flash | Percorso registrato nel manifest dell'embedding. Non carica il modello. |

Per BUILD_DATASET_1 manteniamo `--history-length 1`. Usare `H > 1` è possibile
solo quando i record disponibili contengono abbastanza token precedenti; in
quel caso anche gli script di preprocessing successivi devono essere aggiornati
per leggere la nuova shape della history.

## Come eseguire lo script

Lo script usa esclusivamente la libreria standard di Python: può essere avviato
dal root della repo senza dipendenze ML e senza `.venv`.

```bash
cd ~/ds4
python3 test_predictor/prefetch_predictor_pipeline/ML_pipeline/build_dataset.py
```

Esempio con percorsi espliciti:

```bash
python3 test_predictor/prefetch_predictor_pipeline/ML_pipeline/build_dataset.py \
  --prompts test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl \
  --raw-dir test_predictor/prefetch_predictor_pipeline/build_dataset_1/raw \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_1/derived/prefetch_dataset_build_1.jsonl \
  --history-length 1
```

Al termine, il terminale stampa un riepilogo del tipo:

```text
BUILD_1 preprocessing complete: 199000 examples from 100 sequences -> ...
```

Questo comando **sovrascrive** il file indicato da `--output`; eseguilo solo
quando vuoi rigenerare il dataset derivato a partire dalle trace raw.
