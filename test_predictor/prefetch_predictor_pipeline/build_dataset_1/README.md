# BUILD_1 — Raccolta iniziale del dataset per il prefetch predictor

> **Nota sul collector:** il sorgente condiviso è ora
> `test_predictor/prefetch_predictor_pipeline/collector/dataset_runner.c`; si compila con
> `make dataset_runner`. BUILD_1 conserva i suoi trace e risultati storici,
> mentre BUILD_2 aggiunge la feature 4 senza modificarli.

## Obiettivo

BUILD_1 è la prima implementazione della pipeline di raccolta del dataset per
il `prefetch_predictor`. Usa i prompt di
`../dataset_definition/prefetch_prompts_v0.jsonl` ed esegue una inferenza DS4 separata
per ogni prompt, con CUDA e SSD streaming.

In questa build raccoglieremo soltanto le feature 0--3 descritte in
`../dataset_definition/features.md`:

| Feature | Cosa raccogliamo | Quando viene ottenuta |
|---|---|---|
| 0. `prompt_metadata` | `id`, `category` e identificativo univoco della sequenza | Direttamente dal corpus V0. |
| 1. `routing_history` | Expert Top-6 del router per token e layer | Il trace grezzo viene registrato durante il decode; la history viene costruita offline. |
| 2. `router_logits` | I migliori logit grezzi del router, fino alla soglia configurata | Durante il decode, immediatamente prima della selezione Top-6. |
| 3. `token_embedding` | Riferimento al vettore di embedding del `token_id` corrente | Il `token_id` è nel trace; la tabella degli embedding viene esportata una sola volta. |

La prima build **non** raccoglie ancora hidden state, KV cache, feature
dell'indexer o attention. Non è neppure un test A/B/C: non usa oracle,
prefetch worker o replay del routing. Il router reale di DS4 resta sempre la
fonte della label osservata.

## Risultato atteso

Al termine della raccolta avremo due livelli di dati:

1. trace grezzi, leggibili e riprendibili, uno per prompt;
2. dataset JSONL derivato, con un record supervisionato per
   `(sequenza, token t, layer l)` e target `E_l(t+1)`.

Il record derivato contiene il routing osservato a `t`, mentre il target è il
routing realmente osservato allo stesso layer nel token successivo. Con
`history_length = 1`, per esempio, il record costruito dopo aver osservato
`E_17(3)` serve a predire `E_17(4)`.

Il corpus V0 ha 100 prompt. Con circa 63 token di decode per prompt e 43
layer, la stima è di circa 270.900 eventi grezzi e 266.600 esempi con target
al token successivo. Il dettaglio delle stime è in `../dataset_definition/README.md`.

## Struttura prevista

Tutto ciò che verrà aggiunto per questa prima build resterà sotto questa
cartella, senza mescolarsi con i test di prefetch.

```text
build_dataset_1/
├── README.md                         questa documentazione
├── ../collector/dataset_runner.c     runner condiviso: un processo DS4 per prompt
├── build_dataset.py                  preprocessing: trace -> JSONL derivato
├── preprocessing.py                  preprocessing: JSONL derivato -> Parquet + .npy
├── requirements.txt                  dipendenze Python del preprocessing
├── .gitignore                        esclude binario, trace e dataset generati
├── raw/                              output locale, escluso dal versionamento
│   ├── manifest.jsonl                stato di ogni prompt e dati di ripresa
│   ├── routing/                      trace JSONL per prompt
│   └── embeddings/                   tabella embedding esterna
└── derived/                          dataset JSONL e artefatti di preprocessing
└── training/                         Parquet e tensori per ML, creati localmente
```

`raw/` e `build_dataset_1/derived/` sono creati automaticamente dal runner e dal
preprocessing. Contengono dati generati localmente e non sono codice sorgente.
Anche `training/` e' un output locale, escluso dal versionamento.

## Preprocessing per il training ML

`ML_pipeline/build_dataset.py` ha gia' costruito il dataset supervisionato JSONL. Il passo
successivo e' `ML_pipeline/preprocessing.py`: non addestra ancora un algoritmo, ma prepara
un formato comodo sia per l'ispezione sia per un futuro script di training.

Prima installa le sole dipendenze necessarie. In un ambiente virtuale attivo:

```bash
python3 -m pip install -r test_predictor/prefetch_predictor_pipeline/build_dataset_1/requirements.txt
```

Poi, dalla root della repo:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/preprocessing.py
```

L'output e' la cartella `training/`:

```text
training/
├── prefetch_training_build_1.parquet  tabella completa, apribile in PyCharm
├── feature_columns.json               ordine e significato delle colonne di X
├── split_by_prompt.json               split riproducibile, con seed usato
├── preprocessing_summary.json         conteggi e shape prodotti
└── arrays/
    ├── X_train.npy                    feature train, float32, shape [N, 41]
    ├── X_validation.npy
    ├── X_test.npy
    ├── y_train.npy                    multi-label, uint8, shape [N, 256]
    ├── y_validation.npy
    ├── y_test.npy
    ├── target_expert_ids_train.npy    target Top-6 originale, shape [N, 6]
    ├── target_expert_ids_validation.npy
    └── target_expert_ids_test.npy
```

Il Parquet contiene una riga per esempio e colonne appiattite, quindi e'
leggibile in forma tabellare: metadati del prompt, token/layer, i sei expert
del routing corrente, history H=1, i 16 ID/logit del router e i sei expert
target. La colonna `split` indica a quale partizione appartiene ogni riga.

Lo split e' effettuato **per prompt**, non per singola riga: nessun token di
uno stesso prompt puo' quindi finire sia in training sia in validation/test.
Poiche' V0 contiene dieci prompt per ciascuna categoria, ogni categoria viene
divisa in 8 prompt train, 1 validation e 1 test (80/10/10), con seed 42.

La matrice `X` contiene, in quest'ordine: `token_position`, `token_id`,
`layer`, sei ID della `routing_history` H=1, 16 ID dei raw logits e i loro 16
valori. `experts_t` non viene duplicato in `X` perche' con H=1 coincide con la
routing history; resta comunque nel Parquet per chiarezza. La label `y` e' un
vettore multi-hot da 256 elementi: `y[e] = 1` se l'expert `e` appartiene a
`target_experts_next_token`.

In questa prima versione `token_embedding` 4096D non e' ancora incluso nella
matrice `X`: BUILD_1 conserva correttamente il solo `token_id` e il riferimento
alla tabella del GGUF, ma non ha ancora materializzato la tabella embedding.
Inoltre ID di token ed expert sono indici discreti, non misure continue: un
futuro modello dovra' trattarli come input categorici/embedding, senza fare una
standardizzazione globale o una PCA in questo preprocessing.

## Flusso di raccolta

```text
prefetch_prompts_v0.jsonl
        |
        v
../collector/dataset_runner
        |
        +-- per ogni prompt: nuovo processo ds4
        |       |
        |       `-- trace routing + logit + token_id in raw/routing/
        |
        v
manifest.jsonl  (successo, errore, EOS, configurazione, resume)
        |
        v
build_dataset
        |
        +-- verifica prompt_id e sequence_id
        +-- collega E_l(t) a E_l(t+1)
        +-- costruisce routing_history con H configurabile
        `-- associa il riferimento alla tabella embedding
        |
        v
derived/prefetch_dataset_build_1.jsonl
```

La history e il target non verranno calcolati mentre DS4 genera: derivarli dai
trace consecutivi evita disallineamenti e permette di passare da `H=1` a una
history più lunga senza rieseguire le inferenze. Il trace può anche contenere
eventi con `phase: "prefill"` quando DS4 usa il prefill SSD token-per-token;
il preprocessing li esclude esplicitamente e usa solo `phase: "decode"`.

## Schema dei trace grezzi in `raw/routing/`

Ogni file `raw/routing/prompt_XXXX.routing.jsonl` è JSONL UTF-8: ogni riga è
un evento del router per una coppia **(token, layer)** della singola
inferenza. Il file non ripete i metadati del prompt in ogni riga: il mapping
al prompt V0 avviene attraverso il nome del file e, in modo autoritativo,
attraverso l'entry corrispondente in `raw/manifest.jsonl`.

Un record routed di esempio è:

```json
{
  "phase": "decode",
  "token_index": 20,
  "token_id": 35,
  "layer": 17,
  "is_hash": false,
  "experts": [96, 119, 202, 112, 15, 90],
  "top_candidates": [
    [96, 1.42, 27.47, 28.89],
    [119, 1.18, 27.46, 28.64]
  ],
  "cutoff_margin": 0.19,
  "router_logits": {
    "kind": "raw_logits",
    "max_scores_configured": 16,
    "stored_count": 16,
    "expert_ids_ranked": [96, 119, 202, 112, 15, 90, 155, 247, 101, 118, 59, 150, 110, 71, 243, 68],
    "values_ranked": [-0.54, -1.10, -2.36, -2.41, -2.57, -2.62, -2.83, -2.94, -2.94, -2.95, -3.15, -3.26, -3.29, -3.36, -3.38, -3.39]
  },
  "router_weights": null,
  "original_experts": null,
  "original_router_weights": null
}
```

I numeri dell'esempio sono solo illustrativi. Le liste complete prodotte dal
trace non vengono abbreviate.

### Campi di identificazione e fase

- `phase`: fase dell'inferenza che ha prodotto l'evento. Vale
  `"decode"` per il decoding autoregressivo, cioè l'unica fase usata per il
  dataset supervisionato del predictor. Può valere `"prefill"` quando DS4
  usa il prefill SSD token-per-token: tali righe sono diagnostiche e
  `ML_pipeline/build_dataset.py` le esclude. Questa distinzione evita di usare token del
  prompt come se fossero token generati.
- `token_index`: posizione assoluta del token nella sequenza DS4. Non è un
  contatore che riparte da zero nel decode: se il prompt renderizzato occupa
  20 token, il primo evento `decode` ha `token_index: 20`.
- `token_id`: ID nel vocabolario del token elaborato a quella posizione. È
  il dato raw necessario alla feature 3 `token_embedding`: il dataset
  derivato non duplica il vettore da 4096 dimensioni, ma potrà recuperarlo
  dalla tabella `token_embd.weight` del modello tramite questo ID.
- `layer`: indice del layer DS4, da 0 a 42. Ogni token genera quindi 43
  righe di trace, una per layer.
- `is_hash`: `true` nei layer hash 0--2, in cui la selezione degli expert
  è una mappa deterministica token→expert; `false` nei layer routed 3--42,
  dove la scelta è prodotta dal router MoE. Il predictor sarà addestrato sui
  soli layer routed, ma mantenere le righe hash rende il trace completo e
  verificabile.

### Routing osservato: base della feature 1

- `experts`: lista ordinata dei sei expert realmente selezionati dal modello
  per questo token e layer. È il trace canonico $E_{t,l}$ da cui il
  preprocessing ricostruisce `experts_t`, la `routing_history` di
  lunghezza (H) e la label `target_experts_next_token = E_{t+1,l}`.
  La lista ha sempre sei ID; per i layer hash è il risultato della mappa
  deterministica, per quelli routed è il Top-6 del router reale.

Il raw trace non contiene un oggetto `routing_history`: salvarla qui
replicherebbe gli stessi dati in molte righe e bloccherebbe la scelta di
(H). È costruita solo nel JSONL derivato, raggruppando per
`sequence_id` e `layer` e ordinando per `token_index`.

### Diagnostica del router e feature 2

- `router_logits`: oggetto con i raw logits del router nei soli layer
  routed; vale `null` per i layer hash. È la feature 2 raccolta da BUILD_1.
  `expert_ids_ranked[i]` e `values_ranked[i]` sono allineati e ordinati
  dal logit raw più alto al più basso. `stored_count` è la lunghezza delle
  due liste e `max_scores_configured` riporta il valore effettivo richiesto
  con `DS4_DATASET_ROUTER_MAX_SCORES` (16 nel run iniziale).
- `top_candidates`: diagnostica aggiuntiva, separata dai raw logits. Per
  ogni candidato conserva una lista di quattro valori, senza sotto-campi con
  nome: `[ID expert, valore router pre-bias, bias, somma]`. Il secondo valore
  è il valore interno del tensor DS4 chiamato `router_probs`; il quarto è la
  somma del secondo e del terzo valore. Il secondo valore non va interpretato
  necessariamente come una probabilità normalizzata. Le otto liste sono
  ordinate dal quarto valore più alto al più basso. Non deve essere confuso con
  `router_logits.values_ranked`: le due quantità hanno semantica numerica
  diversa e possono avere ordine diverso.

Non esiste quindi una chiave JSONL separata chiamata `bias`. Per una riga
`top_candidates[i]`, il bias è esattamente `top_candidates[i][2]`; il valore
prima del bias è `top_candidates[i][1]` e la somma che determina l'ordine è
`top_candidates[i][3]`.

#### Perché l'ordine dei raw logits può differire dal routing reale

Gli expert in `experts` seguono l'ordine del quarto valore dentro ogni lista
di `top_candidates`, quindi coincidono con i suoi primi sei ID.
`router_logits`, invece, è
ordinato esclusivamente per raw logit: non è la graduatoria che DS4 usa per la
decisione finale. Un bias può quindi invertire due expert.

Nel primo evento routed osservato nel corpus, per esempio:

```text
raw logits:       expert 15 = -2.57362795  > expert 90 = -2.61992455
                  quindi router_logits ordina 15 prima di 90

top_candidates:   expert 90 -> [90, 0.265103459, 9.00365162, 9.26875496]
                  expert 15 -> [15, 0.271091402, 8.98697281, 9.25806427]

calcolo:          top_candidates[4][3]
                    = top_candidates[4][1] + top_candidates[4][2]
                    = 0.265103459 + 9.00365162
                    = 9.268755079  (salvato come 9.26875496)

                  top_candidates[5][3]
                    = top_candidates[5][1] + top_candidates[5][2]
                    = 0.271091402 + 8.98697281
                    = 9.258064212  (salvato come 9.25806427)

                  9.26875 > 9.25806: il quarto valore mette 90 prima di 15;
                  lo stesso ordine compare in experts
```

Non è un'incoerenza: `router_logits` descrive lo stato raw del router, mentre
`experts` descrive la scelta effettiva dopo l'applicazione del bias. Per lo
stesso motivo non è garantito che tutti e sei gli expert selezionati siano
sempre compresi nella Top-16 dei raw logits. Un expert oltre la sedicesima
posizione raw può ricevere un bias sufficiente a entrare nelle prime sei
posizioni di `top_candidates` e quindi nel campo `experts`.

Questo si osserva realmente nel decode del corpus BUILD_1: in 15 eventi
routed su 203.000 la Top-16 dei raw logits non contiene tutti i sei expert
selezionati; in 6 di tali eventi ne contiene soltanto uno, quindi gli altri
cinque sono necessariamente oltre la sedicesima posizione raw. Il trace non
può indicarne la posizione raw esatta perché conserva soltanto i primi 16
logits. Il routing reale non è però perso: è sempre completo in `experts`, e
`top_candidates` conserva per tutti i sei selezionati il valore pre-bias, il
bias e il quarto valore che determina il loro ordine finale.

Per una raccolta futura, se vorremo fornire al predictor anche lo stato che
ha determinato direttamente la decisione finale, potremo aggiungere una
feature distinta basata sui valori di `top_candidates`, senza confonderla con
i soli `router_logits`.

Un'alternativa più semplice è aumentare
`DS4_DATASET_ROUTER_MAX_SCORES` da `16` a `32`: ogni evento conserverebbe i
raw logits dei primi 32 expert invece dei primi 16. Questo riduce la
probabilità che un expert inizialmente lontano nella graduatoria raw, ma
promosso dal bias nel Top-6 finale, non abbia il proprio raw logit nel trace.
Non è però una garanzia matematica: un bias abbastanza grande può promuovere
anche un expert oltre la posizione 32. Il costo è circa il raddoppio della
sola parte `router_logits` del trace e del dataset derivato; `experts` e
`top_candidates` restano invariati.

- `cutoff_margin`: differenza tra lo score con bias del sesto expert e
  quello del settimo nella graduatoria `top_candidates`. Misura quanto il
  confine del Top-6 sia netto o incerto. È un campo diagnostico utile per
  analisi future, ma non è ancora incluso nelle feature 0--3 del dataset
  derivato.
- `router_weights`: pesi di gating degli expert selezionati, se il percorso
  DS4 che ha prodotto il trace li rende disponibili. In BUILD_1 può essere
  `null`, quindi non è usato come feature del predictor. Se sarà raccolto
  in modo affidabile in tutti i percorsi CUDA, potrà diventare una feature
  distinta.
- `original_experts` e `original_router_weights`: campi riservati ai test
  di oracle/replay. In BUILD_1 non facciamo replay, perciò valgono sempre
  `null`; sono presenti perché il trace DS4 usa uno schema comune con i
  test di prefetch.

### Collegamento con il JSONL derivato

| Elemento nel dataset derivato | Origine |
|---|---|
| Feature 0 `prompt_metadata` | `prefetch_prompts_v0.jsonl` + `manifest.jsonl` |
| Feature 1 `experts_t` / `routing_history` | `phase: "decode"`, `experts`, `layer`, `token_index` |
| Feature 2 `router_logits` | Campo `router_logits` della stessa riga routed |
| Feature 3 `token_embedding` | `token_id` + lookup esterna `token_embd.weight` |
| `target_experts_next_token` | Campo `experts` della riga decode successiva, stesso layer e stessa sequenza |

In sintesi, `raw/routing` conserva il fatto osservato durante l'inferenza;
`build_dataset_1/derived/prefetch_dataset_build_1.jsonl` unisce quel trace ai metadati del
prompt e trasforma righe consecutive in un esempio supervisionato.


## Configurazione DS4 scelta per BUILD_1

La raccolta iniziale replica il percorso di inferenza stabile dei TEST_5/6:

```text
DS4_CUDA_MMQ=0
./ds4
  --cuda
  --ssd-streaming
  --ssd-streaming-cache-experts 8GB
  -c 1024
  -n 64
  --nothink
  --temp 0
```

| Impostazione | Valore BUILD_1 | Motivo |
|---|---:|---|
| Modello | `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | È il modello usato nei test precedenti. |
| GPU | CUDA, `DS4_CUDA_MMQ=0` | Mantiene il percorso CUDA già validato. |
| Expert cache | 8 GB | Baseline stabile già documentata in `command_inference.md`. |
| Contesto | 1024 token | Sufficiente per V0 e coerente con i test. |
| Decode massimo | 64 token | Permette una prima raccolta riproducibile e circa 62 transizioni con label per prompt. |
| Sampling | `--temp 0` e `--nothink` | Decode greedy e meno variabilità fra esecuzioni. |
| Router logit salvati | 16 per evento (proposta iniziale) | Include sempre i Top-6 e conserva un margine diagnostico con volume contenuto. |

La soglia di 16 sarà esposta dalla futura variabile
`DS4_DATASET_ROUTER_MAX_SCORES`. Il suo intervallo ammesso sarà `6..256`:
`256` salverà tutti i logit degli expert, mentre `16` è il default iniziale.

## Configurazione CLion

La configurazione CLion è una **Native Application** con questi valori:

| Campo | Valore previsto |
|---|---|
| Nome | `BUILD_1 Dataset Collection` |
| Target | `dataset_runner` (`build_1_runner` resta un alias compatibile) |
| Executable | `/home/michele/ds4/test_predictor/prefetch_predictor_pipeline/collector/dataset_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | vuoto nella prima versione; input e output saranno definiti dal runner. |

Per il primo collaudo su cinque prompt, inserisci nelle variabili d'ambiente
una sola riga separata da `;`:

```text
DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_CUDA_MMQ=0;DS4_DATASET_ROUTER_MAX_SCORES=16;DS4_DATASET_LIMIT=5
```

`DS4_MODEL` e `DS4_CUDA_MMQ` sono già convenzioni usate dai test. Invece
`DS4_DATASET_ROUTER_MAX_SCORES` è la nuova variabile BUILD_1. Il runner
imposta autonomamente, per ogni prompt, il percorso del trace: non bisogna
impostare a mano `DS4_ROUTING_TRACE_PATH` in CLion.

Se il collaudo ha successo, rimuovi `DS4_DATASET_LIMIT` e riesegui la stessa
configurazione: il manifest salta automaticamente i prompt già conclusi con
successo e prosegue sui rimanenti. Opzioni utili sono
`DS4_DATASET_START_ID`, `DS4_DATASET_MAX_TOKENS` (default `64`),
`DS4_DATASET_PROMPTS_PATH` e `DS4_DATASET_OUTPUT_DIR`. Per rigenerare un
prompt già riuscito, imposta anche `DS4_DATASET_FORCE_PROMPT_ID=<id>`: il
nuovo trace sostituisce il precedente e il preprocessing usa l'ultima entry
riuscita del manifest per quel prompt.

Al termine del runner esegui, dalla root della repository:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/build_dataset.py
```

Questo crea `build_dataset_1/derived/prefetch_dataset_build_1.jsonl` e
`build_dataset_1/derived/embedding_table_manifest.json`. Non devono essere lanciate inferenze
DS4 concorrenti con questo modello.

## Build CUDA necessaria

Sì: poiché BUILD_1 aggiunge il readback opzionale dei raw router logits in
`ds4.c`, prima del primo run devi ricompilare DS4 CUDA. Dalla root:

```bash
make cuda-generic
make dataset_runner
```

`make cuda-generic` ricostruisce anche `./ds4`; `make dataset_runner` compila
soltanto il piccolo orchestratore C. Su questa macchina GCC 12 ha mostrato un
internal compiler error nel codice preesistente durante l'ottimizzazione
`-O3`. In questa repository il workaround `-O2` ha completato la build CUDA e
il linking di `./ds4`; se l'ICE si ripresenta, usa quindi:

```bash
make cuda-generic CFLAGS='-O2 -ffast-math -g -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only'
make dataset_runner
```

## Formato e tracciabilità

Ogni inferenza riceverà un `sequence_id` derivato in modo deterministico dal
corpus e dal prompt, ad esempio `prefetch_prompts_v0_42_run_001`. Il manifest
legherà tale sequenza a `prompt_id`, `prompt_category`, file trace, stato e
configurazione effettiva.

### Significato di `token_index`

`token_index` è la posizione assoluta del token nella sequenza elaborata da
DS4, non un contatore relativo ai soli token generati. Le posizioni iniziali
sono occupate dai token del prompt renderizzato nel template chat e processati
nel prefill; il primo token di decode parte quindi dalla lunghezza tokenizzata
di quel prompt.

Per esempio, se il prompt occupa 20 token, il trace di decode inizia così:

```text
token_index 0..19  -> prompt / prefill
token_index 20     -> primo token generato
token_index 21     -> secondo token generato
...
token_index 82     -> sessantatreesimo token generato
```

La posizione iniziale varia fra prompt, perché variano il testo e il numero di
token introdotti dal template chat. Non è un problema per il predictor: la
sequenza temporale viene sempre ricostruita all'interno della stessa
`sequence_id`; `token_index` preserva la posizione reale nel contesto.

Nel dataset derivato saranno mantenuti sia `prompt_id` sia `prompt_category`.
Per la feature 3 il JSONL salva `token_id` e un riferimento a
`token_embd.weight` nel GGUF: la matrice statica non viene duplicata per ogni
evento. `embedding_table_manifest.json` documenta questo lookup. Prima
dell'addestramento un preprocessing separerà metadata, history, feature e
target, effettuerà lo split per prompt/sequenza e impedirà leakage temporale.
La specifica completa del record JSONL cumulativo è in `../dataset_definition/features.md`.

## Piano di implementazione

1. Eseguire il runner e il manifest, inizialmente su 5 prompt.
2. Verificare nei trace i logit grezzi configurabili del router.
3. Creare il JSONL derivato e verificare mapping, token successivo e H=1.
4. Misurare tempi e volume su 5 prompt; solo dopo lanciare V0 completo.

## Decisioni già fissate e decisioni da confermare

Sono già fissati: corpus V0, inferenza separata per prompt, CUDA + SSD
streaming, decode greedy, `-n 64`, feature 0--3 e dataset JSONL derivato.

Le due scelte che possiamo ancora modificare senza cambiare l'architettura
sono il default iniziale dei logit salvati (`16` qui proposto) e la lunghezza
massima del decode. Entrambe verranno registrate nel manifest, quindi i run
restano confrontabili e riproducibili.
