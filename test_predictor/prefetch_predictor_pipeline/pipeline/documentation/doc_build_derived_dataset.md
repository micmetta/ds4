# `build_derived_dataset.py` — dalle trace del collector al JSONL supervisionato

## Scopo dello script

`build_derived_dataset.py` è la versione generica dello step che costruisce il
dataset supervisionato del prefetch predictor a partire dalle trace raw del
collector. Non avvia DS4, non raccoglie nuovi dati e non addestra modelli:
legge soltanto artefatti già presenti su disco.

Ogni record prodotto descrive il problema:

$$
E_l(t) \longrightarrow E_l(t+1)
$$

cioè: osservare le feature del routing nel layer routed `l` al token `t`, per
predire i sei expert dello stesso layer al token successivo.

Lo script è indipendente da una build specifica. BUILD_DATASET_1 può usarlo
senza hidden feature; BUILD_DATASET_2 lo usa con `--require-hidden` per
preservare i riferimenti raw necessari alla feature 4.

## Input e output

I percorsi sono tutti obbligatori, così lo stesso script può essere riusato
senza ambiguità:

| Argomento | Contenuto |
|---|---|
| `--prompts` | Corpus JSONL V0 con `id`, `category` e testo del prompt. |
| `--raw-dir` | Directory raw della build, contenente `manifest.jsonl` e le trace routing. |
| `--output` | Destinazione del JSONL supervisionato derivato. |
| `--history-length` | Lunghezza `H` della routing history; default `1`. |
| `--require-hidden` | Richiede `hidden_raw_ref` in ogni evento; va usato per BUILD_DATASET_2. |
| `--source-dataset` | Identificatore del corpus scritto in ogni record; default `prefetch_prompts_v0`. |

Esempio BUILD_DATASET_2:

```text
build_dataset_2/raw/manifest.jsonl
build_dataset_2/raw/routing/prompt_XXXX.routing.jsonl
        |
        v
build_dataset_2/derived/prefetch_dataset_build_2_raw_hidden.jsonl
```

## L'output ha questa struttura

Per BUILD_DATASET_2, dopo l'esecuzione del comando documentato più sotto, la
directory derivata contiene almeno:

```text
build_dataset_2/derived/
└── prefetch_dataset_build_2_raw_hidden.jsonl
```

`prefetch_dataset_build_2_raw_hidden.jsonl` contiene un record per ogni coppia
valida `(token t, layer l)`. Include le feature 1–3, il target
`target_experts_next_token` e `hidden_raw_ref`: quest'ultimo è il riferimento
al vettore `ffn_norm` raw della feature 4, non ancora la sua proiezione PCA.

Il successivo script della cartella `pipeline/hidden_features/` leggerà questo
file, applicherà la PCA train-only e creerà il dataset con
`hidden_features.values` pronto per `preprocess_dataset.py`.

## Flusso completo

```text
main()
  |
  +-- load_successful_manifests(raw/manifest.jsonl)
  |     seleziona l'ultima entry riuscita per ogni prompt
  |
  +-- load_prompts(prompts.jsonl)
  |     costruisce il mapping prompt_id -> prompt originale
  |
  +-- build_dataset(prompts, manifests, H, output, require_hidden)
  |     percorre prompt, layer e token consecutivi
  |       |
  |       +-- load_decode_events(trace, require_hidden)
  |       |     filtra decode non-hash e raggruppa gli eventi per layer
  |       |
  |       `-- make_record(prompt, manifest, layer, current, history, target)
  |             serializza un esempio E_l(t) -> E_l(t+1)
  |
  `-- stampa conteggio di esempi e sequenze scritti
```

## 1. `load_prompts()`

`load_prompts(path)` legge il corpus V0 riga per riga e restituisce un mapping
`prompt_id -> record JSON`. Blocca il processo se un ID compare due volte:
l'associazione tra trace e prompt deve essere univoca.

## 2. `load_successful_manifests()`

`load_successful_manifests(path)` esamina `manifest.jsonl` e conserva solo
l'ultima raccolta con `status = "success"` per ogni prompt. Le raccolte fallite
o superate da una run successiva non entrano quindi nel dataset.

Le entry finali sono ordinate per `prompt_id`, rendendo il JSONL generato
riproducibile a parità di file raw.

## 3. `load_decode_events()`

`load_decode_events(path, require_hidden)` legge la trace routing di un prompt
e:

1. conserva soltanto eventi `phase = "decode"`;
2. elimina eventi `is_hash = true`, perché appartengono al mapping
   deterministico iniziale e non al routing MoE da predire;
3. verifica la presenza di `token_index`, `token_id`, `layer` ed `experts`;
4. con `--require-hidden`, verifica anche `hidden_raw_ref`;
5. ordina gli eventi per `token_index` all'interno di ogni layer e rifiuta
   duplicati.

L'output è una struttura concettuale `layer -> [eventi ordinati per token]`.

## 4. `build_dataset()`

`build_dataset()` percorre ciascun layer e costruisce gli esempi usando solo
posizioni di decoding consecutive. Per ogni indice valido seleziona:

```text
history = [E_l(t-H+1), ..., E_l(t)]
current = E_l(t)
target  = E_l(t+1)
```

Se `target.token_index` non è `current.token_index + 1`, lo script interrompe
il processo: predire un token non adiacente cambierebbe il problema studiato.

Il JSONL viene scritto progressivamente, senza tenere l'intero output in RAM.
La funzione restituisce il numero di esempi e di sequenze che hanno prodotto
almeno un record.

## 5. `make_record()`

`make_record()` crea il record `prefetch_dataset_v1` unendo:

| Gruppo | Campi principali |
|---|---|
| Provenienza | `prompt_id`, `prompt_category`, `sequence_id` |
| Stato a `t` | `token_position`, `token_id`, `layer`, `experts_t` |
| Feature 1 | `routing_history` |
| Feature 2 | `router_logits` |
| Feature 3 | `token_embedding` come riferimento esterno al GGUF |
| Target | `target_experts_next_token` = $E_l(t+1)$ |
| Feature 4 raw | `hidden_raw_ref`, solo se presente nella trace |

`hidden_raw_ref` non contiene i 4096 float nel JSONL. È un riferimento al
file binario raw, con offset, tipo e shape, che lo step PCA userà in seguito.

## Come eseguire lo script

Dal root della repo:

```bash
cd ~/ds4
python3 test_predictor/prefetch_predictor_pipeline/pipeline/build_derived_dataset.py \
  --prompts test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl \
  --raw-dir test_predictor/prefetch_predictor_pipeline/build_dataset_2/raw \
  --output test_predictor/prefetch_predictor_pipeline/build_dataset_2/derived/prefetch_dataset_build_2_raw_hidden.jsonl \
  --history-length 1 \
  --require-hidden \
  --source-dataset prefetch_prompts_v0
```

Senza `--require-hidden` può essere riutilizzato per una build priva della
feature 4. Il file indicato da `--output` viene sovrascritto: esegui il comando
solo quando vuoi rigenerare il dataset derivato.
