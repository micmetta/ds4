# OVERVIEW — Predictive Expert Prefetch (branch `predictive-expert-prefetch`)

Questo documento è il **report di comprensione preliminare** richiesto prima di
qualunque modifica al codice. Non contiene ancora alcuna implementazione.
L'obiettivo è mappare esattamente cosa fa oggi DS4 lungo il percorso
`router → expert cache → SSD load → GPU transfer → CUDA expert kernel`
sul backend CUDA in modalità `--ssd-streaming`, per capire se — e dove — un
predictor MoE che gira in parallelo alla computazione del token `t` possa
mascherare la latenza di caricamento degli expert del token `t+1`.

Tutti i riferimenti `file:linea` sono presi dall'HEAD del branch attuale. Se
l'HEAD si sposta di qualche riga i punti restano semanticamente identici.

---

## 0. Ambiente e baseline

* Hardware: Intel i9-14900HX, 16 GB RAM (~11 GiB in WSL2), NVIDIA RTX 4080
  Laptop 12 GB VRAM (sm_89), NVMe SSD, Debian su WSL2, CUDA 12.6.
* Modello: DeepSeek V4 Flash Q2 (~80.76 GiB GGUF, 284B parametri logici,
  43 layer, 256 routed expert totali, 6 attivi per token — `DS4_SHAPE_FLASH` in
  `ds4.c:557`; costante `DS4_N_EXPERT_USED` in `ds4.c:716`).
* Comando di riferimento (baseline decode):

  ```
  DS4_CUDA_MMQ=0 ./ds4 --cuda --ssd-streaming \
      --ssd-streaming-cache-experts 8GB -c 1024 -n 64 --nothink --temp 0 \
      -p "Explain what a binary search is in one short paragraph."
  ```

  Osservato: prefill ≈ 1.4 t/s, generation ≈ 0.75 t/s. Baseline decode: **~0.75 t/s**.

---

## 1. Architettura DS4 rilevante per MoE + SSD streaming + CUDA

DS4 è un motore di inferenza C monolitico (senza C++, come da `AGENT.md`),
specializzato per DeepSeek V4 Flash / PRO e GLM 5.2. Il grafo di computazione
per token è costruito da funzioni C `metal_graph_*` che, malgrado il nome,
sono **il builder GPU unificato** — Metal, CUDA e ROCm passano tutte da lì.
La scelta del backend avviene nelle primitive `ds4_gpu_*`; per CUDA sono
implementate in `ds4_cuda.cu`.

Layout dei file rilevanti:

| File | Ruolo |
|---|---|
| `ds4.c` | Motore, session, layer loop, MoE encode, SSD-streaming orchestration, expert profiler, hotlist preload |
| `ds4.h` | `ds4_engine_options` (config runtime) |
| `ds4_ssd.c` / `.h` | Solo bookkeeping byte-budget cache + parse `NGB`; NON gestisce l'SSD I/O reale |
| `ds4_cuda.cu` | Backend CUDA: kernel, stream, cache streaming compact, dispatcher `routed_moe_launch` |
| `ds4_gpu.h`, `ds4_gpu_mgpu.h` | API pubblica GPU (astratta backend) |
| `ds4_metal.m` | Backend Apple: possiede la vera **expert cache LRU multi-slot** |
| `cuda/mmq/*.cu*` | Tier prefill mutuato da llama.cpp (`mmid`, `mmvq`, `ds4_mmq_*`) |
| `ds4_cli.c`, `ds4_help.c`, `ds4_gpu_args.c` | Parsing flag e messa a terra delle opzioni |

Punto architetturale critico che condiziona tutto ciò che segue:

> **La expert-cache "vera" (LRU, multi-slot, con hotness e pool di pread
> worker) esiste solo nel backend Metal.** Il backend CUDA ha una cache di
> forma diversa: un singolo scratch device *per-layer / per-selected-set*
> (`cuda_stream_selected_cache`, `ds4_cuda.cu:130-152`). Il round-trip
> per-layer H2D→D2H → SSD load → H2D è la sequenza attuale.

Questa asimmetria è la chiave per capire tanto il potenziale quanto i limiti
dell'idea.

---

## 2. File e funzioni responsabili di ogni fase (CUDA)

### 2.1 Routing MoE (produzione degli expert ID)

* Per-layer decode encoder: `metal_graph_encode_decode_layer_phase()` in
  `ds4.c:21835`. La sezione router è a `ds4.c:23833-23849`:
  1. Logits: `metal_graph_matmul_plain_tensor(router_logits, model,
     layer->ffn_gate_inp, N_EMBD, N_EXPERT, ffn_norm, 1)` — matvec F16
     `4096×256`.
  2. Top-k + gate weights: `ds4_gpu_router_select_tensor(...)` — host wrapper
     in `ds4_cuda.cu:18714`; kernel `router_select_warp_topk_kernel` in
     `ds4_cuda.cu:11947`.
  3. Variante fusa: `ds4_gpu_router_project_select_fused_tensor`
     (`ds4.c:23783`).
* Prefill (n_tokens>1): router in `metal_graph_encode_layer_ffn_batch`
  (`ds4.c:30007-30039`), select in `ds4_gpu_router_select_batch_tensor`
  (`ds4_cuda.cu:18753`).

### 2.2 Strutture dati che rappresentano "expert IDs per questo token"

Allocate una volta per tier in `ds4.c:17222-17225`:

* `router_logits_by_tier[t]` — device `float[256]`
* `router_probs_by_tier[t]`  — device `float[256]`
* `router_selected_by_tier[t]` — **device `int32_t[6]`** ← è la lista degli ID
* `router_weights_by_tier[t]` — device `float[6]`

Prefill batch: `batch_router_selected` — device `int32_t[n_tokens*6]` (token-major).
Accessor: `metal_graph_router_selected(g)`.

### 2.3 Cache lookup (backend CUDA)

Il percorso "trova/carica gli expert selezionati" per il layer corrente è:

* `metal_graph_decode_cuda_selected_load()` in `ds4.c:21123-21193`. Dopo il
  `router_select`, questa funzione:
  1. Chiama `ds4_gpu_end_commands()` (linea 21145) → `cudaDeviceSynchronize` —
     **hard sync**.
  2. Legge dal device i 6 ID selezionati con `ds4_gpu_tensor_read(router_selected, …)`
     (linea 21149) — `cudaMemcpy(cudaMemcpyDeviceToHost)` bloccante.
  3. Costruisce la `ds4_gpu_stream_expert_table` per il layer `il`
     (linea 21157).
  4. Chiama `ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids, 6)`
     (linea 21163). Su CUDA il thunk è in `ds4_cuda.cu:30387`.
* La cache CUDA è `g_stream_selected_cache` (`ds4_cuda.cu:130-152`, singleton
  `g_stream_selected_cache` a `ds4_cuda.cu:154`). I suoi `gate_ptr / up_ptr /
  down_ptr` sono tre buffer VRAM dimensionati per l'insieme compatto
  degli expert selezionati per QUESTO layer.

### 2.4 Cache hit / miss (CUDA)

Sul percorso CUDA il concetto di "hit" nel senso Metal non esiste
propriamente. La cache Metal (`ds4_metal.m:663`, `g_stream_expert_cache[80][384]`)
ha un vero slot per (layer, expert) con hotness/LRU e worker pool pread
(`ds4_metal.m:12076`). Il percorso CUDA invece **ricostruisce** ogni layer un
"compact selected cache" e ricopia gli expert scelti da mmap → pinned staging
→ VRAM. Il refill di questi 6 expert (3 sub-tensori ciascuno: gate/up/down)
è la sorgente di latenza principale.

Nota: la costruzione del compact cache CUDA (`cuda_stream_selected_cache_begin_load`)
non implementa una politica di eviction in stile LRU: ogni layer sostanzialmente
riparte con l'insieme scelto per quel layer. Un possibile riuso incidentale c'è
solo se il set di 6 expert non cambia da un layer all'altro, ma di norma cambia.

### 2.5 SSD load (host)

* Il "vero" I/O SSD nel motore vive nella cache Metal e nel worker pool pread
  (`ds4_metal.m:12004-12183`, 18 thread `pthread` + condvars).
* Sul percorso CUDA il file GGUF è mmap-backed; l'accesso è pread/mmap
  descritto dal record `model_map + offset`. Non c'è (oggi) un pool di worker
  CUDA dedicato equivalente a quello Metal: il caricamento è driven dal thread
  di encode.

### 2.6 Trasferimento CPU → GPU (H2D)

* Funzione centrale: `cuda_model_copy_to_device_streamed`
  (`ds4_cuda.cu:2193-2282`). Fast path: `cudaMemcpyAsync` su
  `g_stream_selected_upload_stream` (linea 2245), staging pinned a 4 vie
  (`g_stream_selected_stage_event[4]`, `ds4_cuda.cu:446-448`), evento
  `cudaEventRecord` a linea 2256. Chiusura: `cudaStreamSynchronize` linea
  2273.
* Copy stream dedicato: `g_stream_selected_upload_stream` (dichiarato
  `ds4_cuda.cu:450`, creato con `cudaStreamCreateWithFlags(...,
  cudaStreamNonBlocking)` a linea 2157/2158).
* Chiamato tre volte per expert (gate/up/down) dentro il loop
  `cuda_stream_selected_cache_begin_load` (`ds4_cuda.cu:26165-26193`).
* **Fatto importante**: dopo `cudaStreamSynchronize` la funzione ritorna
  bloccante — il compute stream non è sincronizzato via `cudaStreamWaitEvent`
  con la copy stream; il handshake è fatto lato host. Questo limita
  l'overlap effettivo oggi.

### 2.7 Cache insertion / eviction (CUDA)

* Insertion: `cuda_stream_selected_cache_begin_load` (`ds4_cuda.cu:26088-26220`)
  ricostruisce e riempie i tre slab compatti; l'ordine e la posizione degli
  expert è determinata dagli ID selezionati.
* Eviction: nessuna politica LRU; il compact cache si azzera/rimappa per
  layer. Esiste un `slot_selected_ptr` (mappa expert-id → slot compatto)
  copiato in device (`ds4_cuda.cu:26194`).
* (Confronto Metal, per riferimento futuro: `prune_global` in
  `ds4_metal.m:14559-14601` — **hotness prima, LRU come tiebreak**; hotness
  in `g_stream_expert_cache_route_hotness[80][384]` con decadimento ogni 16
  token.)

### 2.8 Esecuzione CUDA dell'expert

Chain per decode di un token:

```
metal_graph_eval_token_raw_swa_streaming  ds4.c:30547
 └─ metal_graph_encode_decode_layer        ds4.c:25100
     └─ metal_graph_encode_decode_layer_phase  ds4.c:21835
         ├─ ds4_gpu_router_select_tensor       ds4_cuda.cu:18714
         ├─ metal_graph_decode_cuda_selected_load  ds4.c:21123  (SSD cache)
         └─ ds4_gpu_routed_moe_one_tensor      ds4_cuda.cu:25386
             └─ routed_moe_launch              ds4_cuda.cu:23543
                 ├─ IQ2_XXS + Q2_K (Flash Q2):
                 │   ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec  ds4_cuda.cu:23623
                 │   ds4_mmq_q2_K_aligned_moe_vec                 ds4_cuda.cu:23634
                 │   (impl in cuda/mmq/ds4_mmq.cu:856 / 1811)
                 ├─ Q4_K decode: routed_moe_decode_q4_graph_launch  ds4_cuda.cu:22004
                 └─ MXFP4: *_mxfp4_moe_gate_up_mid_vec / _down_sum6_vec  ds4_cuda.cu:23753/23761
```

Terminologia mmq: **mmvq** = mat-vec quantized (decode, n_tokens=1); **mmid**
= mat-mat with expert IDs (prefill, n_tokens>1).

---

## 3. Call graph semplificato di un token di decode

```
ds4_session_step  (ds4.c)
 └─ ds4_session_eval → ds4_session_eval_internal  (ds4.c:61462)
     └─ metal_graph_eval_token_raw_swa            (ds4.c:30727)
         └─ if (g->ssd_streaming) metal_graph_eval_token_raw_swa_streaming  (ds4.c:30547)
             for il = 0 .. 42:
                 ├─ metal_graph_stream_map_layer_decode(model, weights, il)
                 └─ metal_graph_encode_decode_layer                (ds4.c:25100)
                     └─ metal_graph_encode_decode_layer_phase       (ds4.c:21835)
                         ├─ attention: QKV, KV cache, softmax, out proj    [device]
                         ├─ router matmul + top-k                           [device]
                         ├─ **metal_graph_decode_cuda_selected_load**       [HOST STALL]
                         │     ├─ ds4_gpu_end_commands  → cudaDeviceSynchronize
                         │     ├─ cudaMemcpy D2H  (6 int32 selected IDs)
                         │     └─ ds4_gpu_stream_expert_cache_begin_selected_load
                         │           └─ for e in selected_ids:
                         │                 for sub in {gate,up,down}:
                         │                     cuda_model_copy_to_device_streamed
                         │                        (pinned stage → cudaMemcpyAsync → cudaStreamSync)
                         ├─ shared expert (gate/up/down su tensore residente)
                         ├─ routed MoE:  ds4_gpu_routed_moe_one_tensor → routed_moe_launch
                         └─ swap cur_hc / after_ffn_hc                      [device]
         ds4_gpu_end_commands  → cudaDeviceSynchronize   (fine token)
         ds4_gpu_tensor_read(logits, …)  → cudaMemcpy D2H bloccante
 └─ ds4_session_sample  (ds4.c:60745)  → CPU top-p/min-p over vocab
 └─ append token
 └─ prossimo token
```

Osservazioni chiave:

* Il ciclo layer è un banale `for il = 0..42` sequenziale (no scheduler).
* La barriera più visibile per il decode è dentro il layer stesso, in
  `metal_graph_decode_cuda_selected_load` (uno stall host per layer).
* La seconda barriera è a fine token (`ds4_gpu_end_commands` +
  `cudaMemcpy` logits) — bloccanti entrambi.

---

## 4. Come funziona attualmente la expert cache

Due implementazioni molto diverse:

**Metal (`ds4_metal.m`)** — expert cache "canonica":
* Tabella statica `g_stream_expert_cache[80][384]` (linea 663), entry con
  `MTLBuffer` gate/up/down, offset assoluti, `last_used`, `use_count`,
  `inflight_seq`, `slab_slot`, `valid` (linea 631-652).
* Pool di slab (`g_stream_expert_cache_slabs[256]`, linea 683).
* Lookup: `ds4_gpu_stream_expert_cache_get_protected` (linea 14769). Miss →
  `prepare_load_buffers` → `pread` (18-thread pool) → `didModifyRange:`.
* Eviction: `prune_global` (linea 14559) — hotness prima, LRU come tiebreak.
* Hotness: `g_stream_expert_cache_route_hotness[80][384]` (linea 679),
  decadimento shift-right ogni 16 token
  (`_maybe_decay_route_hotness`, linea 12880).
* Preload popolarità all'avvio: hotlist da `ds4_streaming_hotlist*.inc`
  caricata via `metal_graph_streaming_expert_hotlist_load_default`
  (`ds4.c:20701-20744`); numero controllato da
  `metal_graph_streaming_expert_preload_count` (`ds4.c:20746-20782`).
* Concorrenza: single-writer cooperativa tra encode thread e worker async
  (`metal_graph_selected_async_load_worker_main` in `ds4.c:21371`) —
  nessun lock fine-grain sulla tabella.

**CUDA (`ds4_cuda.cu`)** — cache "compact per-layer":
* `cuda_stream_selected_cache` (linea 130-152, singleton linea 154).
* Non è una tabella per (layer, expert). È un buffer VRAM riscrivibile che a
  ogni layer contiene solo gli expert selezionati (6 su 256).
* Nessuna hotness, nessun LRU, nessun refcount, nessun preload popolarità
  attivo (l'hotlist esiste ma è cablata sulla cache Metal).
* Overlap begin_load/compute non implementato: `cudaStreamSynchronize` chiude
  la copia prima di procedere.

Conseguenza: **sul percorso CUDA praticamente ogni layer ripaga il costo di
ricostruire in VRAM gli slot dei 6 expert scelti, con il thread di encode
bloccato al sync**. Ogni layer paga inoltre almeno un `cudaDeviceSynchronize`
(all'ingresso di `metal_graph_decode_cuda_selected_load`) per poter leggere gli
ID.

---

## 5. I caricamenti SSD sono sincroni o asincroni?

**In sostanza, sul percorso CUDA in decode: sincroni dal punto di vista
del thread di encode.** Nel dettaglio:

* Il thread di encode chiama `metal_graph_decode_cuda_selected_load` che
  esegue: `cudaDeviceSynchronize` → `cudaMemcpy` D2H bloccante degli ID →
  `begin_selected_load` (che a sua volta lancia `cudaMemcpyAsync` sulla copy
  stream ma poi la richiude con `cudaStreamSynchronize` — vedi
  `cuda_model_copy_to_device_streamed` linea 2273).
* Esiste un handshake host-side dedicato al selected-load
  (`ds4_gpu_signal_selected_readback_ready` in `ds4_cuda.cu:30382`,
  `_wait_...` in 30467, `_commit_and_wait_...` in 30476) ma tutti riducono
  a `cudaDeviceSynchronize`. Non c'è un vero `cudaStreamWaitEvent` che
  ordini la copia rispetto al compute stream in modalità decode.
* Il worker async esiste in `ds4.c:21371` (`metal_graph_selected_async_load_worker_main`)
  ma è pensato per l'overlap Metal e sul percorso CUDA non porta un
  vantaggio strutturale finché la sequenza dentro `_begin_load` è
  auto-sincronizzata.

Conclusione: **il thread di encode è oggi bloccato per la durata del
caricamento SSD del layer corrente prima di poter eseguire la MoE del layer
stesso**. La compute del token `t` NON è pipelinata con l'I/O del token
`t+1`.

---

## 6. Esiste già un meccanismo di prefetch?

Tre forme, ma tutte su Metal / non su CUDA per il caso decode:

1. **Startup popularity preload (hotlist)** — carica l'insieme "hot" più
   frequente all'avvio nella cache. Metal-side: sì
   (`metal_graph_streaming_expert_hotlist_load_default`, `ds4.c:20701`).
   Sul percorso CUDA in decode il compact cache non riusa questa hotlist:
   non c'è un residente hot in VRAM equivalente.

2. **Overlap selected-load ↔ shared expert compute** — Metal:
   `metal_graph_selected_async_load_start` (`ds4.c:24505`) parte la
   `begin_selected_load` in worker, poi il main encode `shared_gate_up` /
   `shared_down` (linee 24518-24557), infine `metal_graph_selected_async_load_finish`
   (24559-24582). Su CUDA la stessa struttura c'è nel codice ma di fatto la
   `begin_selected_load` termina con `cudaStreamSynchronize`, quindi
   l'overlap effettivo con `shared_gate_up` è nullo/limitato.

3. **Overlap layer N+1 dietro layer N (prefill)** — per prefill il begin_load
   del layer successivo viene emesso mentre il layer corrente è ancora in
   flight, sfruttando il pipelining di command buffer Metal
   (`ds4.c:41787-41928`). In decode a n_tokens=1 questa forma non si applica
   nello stesso modo.

4. **`g_model_prefetch_stream`** (`ds4_cuda.cu:102`, `cudaMemPrefetchAsync` a
   linea 1956) — è UM/HMM prefetch dell'intero model map, **non** un
   prefetch per expert.

**Non esiste alcun prefetch cross-token in decode** — cioè: nessun meccanismo
che, mentre il token `t` computa i suoi 43 layer, provi ad anticipare il
caricamento di qualche expert che potrebbe servire al token `t+1`. Questo è
precisamente il buco che l'esperimento vuole esplorare.

---

## 7. Come funzionano `--ssd-streaming-cache-experts` e `--ssd-streaming-preload-experts`

Definiti in `ds4_cli.c:1878-1903`; anche in `ds4_server.c:13187`,
`ds4_bench.c:287`, `ds4_agent.c:703`, `ds4_eval.c:1610`. Help in
`ds4_help.c:174, 193, 276`.

| Flag | Campo `ds4_engine_options` | Semantica |
|---|---|---|
| `--ssd-streaming` | `ssd_streaming` (`ds4.h:155`) | Abilita la modalità streaming |
| `--ssd-streaming-cache-experts <N \| NGB>` | `ssd_streaming_cache_experts` + `ssd_streaming_cache_bytes` (`ds4.h:142-143`) | Se `NGB`: budget di memoria per la cache di expert dinamici (dopo aver riservato la headroom per il pipeline di prefill); se `N` puro: numero **esatto** di slot expert dinamici (senza extra accounting). Parsing in `ds4_ssd.c` (`ds4_parse_streaming_cache_experts_arg`) |
| `--ssd-streaming-preload-experts N` | `ssd_streaming_preload_experts` (`ds4.h:145`) | Numero di expert caricati all'avvio dall'hotlist popolarità |
| `--ssd-streaming-cold` | `ssd_streaming_cold` (`ds4.h:156`) | Disabilita il preload hot (per misure/cold-start) |
| `--ssd-streaming-full-layers N` | `ssd_streaming_full_layers` (`ds4.h:144`) | Numero di layer interi tenuti residenti (headroom prefill) |

Il valore `NGB` non è un byte-cache generico: DS4 prima si riserva la
headroom per due layer routed pieni usati dal prefill sovrapposto, poi
converte i byte rimanenti in numero di slot per il GGUF corrente
(README.md sezione "Running models larger than RAM").

Il budget automatico prende l'80% del working set raccomandato dal backend,
sottrae i pesi non-routed, riapplica la headroom prefill e dimensiona la
dynamic cache.

Nella baseline (`--ssd-streaming-cache-experts 8GB`) si sta chiedendo un
budget molto stretto per 12 GB VRAM: la cache dinamica sarà proporzionalmente
piccola.

---

## 8. `--expert-profile` — stato attuale e riusabilità su CUDA

* CLI: `ds4_cli.c:1926-1927` e `ds4_bench.c:34, 261, 595` mappano
  `--expert-profile FILE` in `ds4_engine_options.expert_profile_path`.
* Help: "Metal-only" (`ds4_help.c:193, 276`) — dichiarato esplicitamente.
* Stato + implementazione: struct `ds4_expert_profile` e singleton
  `g_expert_profile` in `ds4.c:1231-1262`; init `ds4.c:1318-1348`;
  registrazione `ds4_expert_profile_record` (`ds4.c:1386-1436`); scrittura
  JSON per (layer, expert), adjacency inter-token, simulazione hit-rate per
  cap ∈ {1,2,4,…,384}.
* Call site di collection: `metal_graph_profile_router_selection`
  (`ds4.c:21686-21728`) chiamato da `ds4.c:23860` (decode Metal),
  `glm_graph_profile_router_selection` (`ds4.c:41115`) chiamato da
  `ds4.c:41773`, batch variant a `41159`. **Zero occorrenze di
  `expert_profile` in `ds4_cuda.cu`** (verificato con grep).

**Riusabilità su CUDA — molto alta**: la funzione
`metal_graph_profile_router_selection` fa `ds4_gpu_end_commands()` +
`ds4_gpu_tensor_read(metal_graph_router_selected(g), …)` +
`ds4_gpu_tensor_read(metal_graph_router_weights(g), …)` + record. Entrambe
le primitive `ds4_gpu_*` funzionano identiche su CUDA (implementate in
`ds4_cuda.cu:3142`). L'unica ragione per cui oggi non gira su CUDA è che il
call site nel decode phase `ds4.c:23860` non è raggiunto sul percorso
`--cuda --ssd-streaming` per un vincolo del flow. La strumentazione CUDA
può quindi essere ottenuta chiamando la stessa `_record` da un nuovo sito
CUDA-side, oppure — meglio — riusando `metal_graph_decode_cuda_selected_load`
che GIÀ legge i 6 selected_ids in host memory (linea 21148) e ha già pagato
il sync + D2H. Aggiungere `ds4_expert_profile_record(il, selected_ids,
weights, ...)` qui è quasi a costo zero.

---

## 9. Dove inserire la strumentazione CUDA con il minor impatto

Ordine dal più economico al più invasivo:

1. **Zero-cost — attivare i profiler nascosti esistenti**:
   * `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE` (`ds4.c:21142`) — stampa
     `sync=…ms read=…ms load=…ms resume=…ms total=…ms` per layer per
     `metal_graph_decode_cuda_selected_load`. Attivabile via env var,
     nessuna ricompilazione.
   * `DS4_METAL_GRAPH_TOKEN_PROFILE` / `DS4_ROCM_GRAPH_TOKEN_PROFILE`
     (`ds4.c:30627, 30759-30768`) — encode/execute/read per-token. Il
     percorso è comune al backend CUDA e usa solo `now_sec()`
     (`clock_gettime(CLOCK_MONOTONIC)`).
   * `DS4_CUDA_MOE_PROF` — gate della catena `cudaEventRecord` in
     `routed_moe_launch` (`ds4_cuda.cu:24127-24928`).

2. **Costo minimo (poche righe di codice) — record del routing decision**:
   in `metal_graph_decode_cuda_selected_load` a `ds4.c:21148-21167`, subito
   dopo la lettura di `selected_ids`, aggiungere una call a un nuovo
   `ds4_expert_profile_record_cuda(il, selected_ids, DS4_N_EXPERT_USED,
   token_index)` che scrive su file (append) una riga
   `token,layer,e0,e1,…,e5,w0,…,w5,sync_ms,load_ms`. Questo cattura ID +
   pesi + costo I/O di quel layer senza aggiungere sync ulteriori (il sync
   c'era già).

3. **Costo minimo lato host-only — hit/miss log per il compact cache CUDA**:
   in `cuda_stream_selected_cache_begin_load` (`ds4_cuda.cu:26088`)
   registrare, per ogni expert, se il suo byte-range era già coincidente con
   lo slot precedente (proxy di "hit locale"). Utile per capire quanto
   frequentemente lo stesso expert torna in layer consecutivi.

4. **Costo moderato — timing device-side**: `cudaEventRecord` prima/dopo
   `cudaMemcpyAsync` dentro `cuda_model_copy_to_device_streamed`
   (`ds4_cuda.cu:2245-2273`). Occhio a non lasciare event `EnableTiming`
   sul percorso caldo se non gated da env — l'overhead di `EnableTiming`
   può disturbare la baseline.

L'ordine consigliato per raccogliere i primi trace è **1 + 2**: fornisce
già la sequenza `(token, layer, expert_ids, gate_weights, ssd_ms)` senza
toccare kernel o stream.

---

## 10. Dove eseguire un predictor IN PARALLELO alla MoE del token t

Requisiti per un predictor asincrono:

* Deve leggere un "contesto" che è disponibile DURANTE la computazione di
  layer del token `t` (idealmente prima possibile, o addirittura appena
  finito il token `t-1`).
* Deve produrre `(layer_target, expert_id_list)` per uno o più layer del
  token `t+1`.
* Deve poter chiamare in modo asincrono un caricamento SSD/H2D senza
  bloccare il compute stream.

Punti di innesto naturali che il codice offre già:

1. **Su thread separato**: il repository ha già un pattern di worker
   thread — `metal_graph_selected_async_load_worker_main`
   (`ds4.c:21371`) — con mutex/condvar; è il modello a cui affiancare un
   `predictor_worker`. Il thread di encode non paga nulla, il predictor gira
   fuori dalla catena critica.

2. **Trigger di partenza — subito dopo il sampling del token corrente**:
   a valle di `ds4_session_sample` (`ds4.c:60745`), prima del prossimo
   `ds4_session_eval`. Qui il token appena generato è noto: il predictor
   può usarlo come feature per stimare la distribuzione di routing del
   token successivo. Vantaggio: massimo lookahead.

3. **Trigger alternativo — dentro il layer stesso**: appena i router IDs
   del layer `L` del token `t` sono disponibili (in
   `metal_graph_decode_cuda_selected_load` linea 21149), passarli al
   predictor per predire i router IDs del layer `L+k` del token `t`
   (short-range intra-token) o del layer `L` del token `t+1` (cross-token).

4. **Prefetch API**: il predictor spinge le sue predizioni in una funzione
   che sotto chiama
   `ds4_gpu_stream_expert_cache_begin_selected_load(&table, predicted_ids,
   K)` (`ds4_gpu.h:218`) su un layer che non è ancora quello corrente. Su
   Metal questa API già oggi produce il comportamento overlap desiderato.
   Su CUDA occorre una piccola variante che NON chiude con
   `cudaStreamSynchronize` sulla `g_stream_selected_upload_stream`, ma
   registra un `cudaEvent` in modo che il vero begin_selected_load (quello
   emesso dal thread di encode al momento reale) possa `cudaStreamWaitEvent`
   invece di ripartire da mmap. In pratica: **il compact cache CUDA deve
   diventare almeno "reusable" con un handshake evento invece di
   auto-sincronizzarsi**.

Suggerimento operativo per il primo prototipo funzionante: **cross-token
per il layer 0**. Appena finito il decode del token `t`, mentre il main
thread sta ancora facendo sampling + append + housekeeping, il predictor
lancia il begin_selected_load del layer 0 del prossimo token. È il caso più
semplice a implementare, e mette a nudo se serve davvero un `cudaEvent` di
handshake o basta rimuovere il `cudaStreamSynchronize` interno.

---

## 11. L'architettura consente davvero l'overlap compute(t) ↔ predict(t+1) + prefetch(t+1)?

Risposta breve: **sì, ma con due modifiche architetturali obbligatorie**.

Analisi:

**Positivi**
* CUDA ha già uno stream separato per la copia H2D degli expert
  (`g_stream_selected_upload_stream`, `ds4_cuda.cu:450`) — è la primitiva
  giusta per overlap.
* Esiste già un handshake con `cudaEvent` (`g_stream_selected_stage_event[4]`,
  `ds4_cuda.cu:448`) e un ring di 4 pinned staging buffer — è la struttura
  giusta per non serializzare.
* Il worker async pattern esiste già in `ds4.c:21371`; adattarlo per un
  predictor è naturale.
* Il router select in decode è un piccolo matvec F16 4096×256 + top-k su
  256 — pochissime FLOPs; un predictor "shallow" può girare in parallelo
  sul device senza rubare risorse significative all'expert compute.

**Blocchi da rimuovere**
* **B1 — la coppia `cudaDeviceSynchronize` + `cudaMemcpy(D2H)` in
  `metal_graph_decode_cuda_selected_load` (`ds4.c:21145,21149`)** ferma il
  compute stream a ogni layer. Il predictor può ridurre il costo del layer
  successivo solo se il layer corrente non è già serializzato in questo
  modo. Serve una variante che passi gli ID via `cudaMemcpyAsync` +
  `cudaStreamWaitEvent`, oppure che tenga gli ID device-side e li usi
  direttamente come indici (indexing-in-device del compact cache).
* **B2 — il `cudaStreamSynchronize(g_stream_selected_upload_stream)` in
  `cuda_model_copy_to_device_streamed` (`ds4_cuda.cu:2273`)** annulla
  l'overlap tra copia degli expert e compute. Serve un
  `cudaStreamWaitEvent(compute_stream, upload_event)` posto SOLO sul
  kernel che consuma quegli slot.

Se B1 e B2 restano, il predictor riduce a valle solo il costo dell'I/O
SSD "cache-miss" ma NON riduce il tempo di roundtrip perché il thread di
encode continuerà a bloccarsi al successivo layer. Un predictor perfetto
in questo scenario darebbe zero speedup.

Rimossi B1 e B2 (che è una modifica localizzata, non un refactor di
architettura), la struttura per pipeline `compute(t) || prefetch(t+1)`
esiste già.

Un'osservazione importante sulle strutture dati: gli hidden state tra
layer sono solo device-side (`cur_hc_by_tier` / `after_ffn_hc_by_tier`,
swap in `ds4.c:27544-27546` e `30677-30679`, shape `4 * 4096` = 16 KiB).
Un predictor host-side che voglia usarli come feature deve leggerli via
`cudaMemcpy(D2H)` per 16 KiB — non caro, ma va gated. Un predictor
device-side (kernel piccolo) può leggerli senza copia.

**DSpark/MTP non fornisce hint sugli expert futuri.** DSpark cattura hidden
state per proporre TOKEN futuri, non expert (verificato: nessun match
`expert` in output DSpark; capture in `metal_graph_dspark_capture_decode_layer`
`ds4.c:27229`). Non è quindi un'alternativa gratis al predictor.

---

## 12. Rischi principali

| # | Rischio | Descrizione | Mitigazione preliminare |
|---|---|---|---|
| R1 | **Overhead del predictor** | Se il predictor gira sul compute stream o su device saturato, ruba tempo alla MoE reale | Girare su thread host o su stream CUDA separato; se device-side, kernel minuscolo gated da flag |
| R2 | **Cache pollution** | Prefetch sbagliato può evictare expert che ci sarebbero serviti | Cache CUDA compact è per-layer, oggi non c'è vero LRU, quindi il rischio è basso ora ma cresce se introduciamo persistence multi-layer/multi-token |
| R3 | **I/O sprecato** | Predizione errata = SSD bandwidth sprecata | Misurare hit-rate offline PRIMA di avviare il prefetch; introdurre gate per confidenza |
| R4 | **Competizione bandwidth PCIe** | Se prefetch usa la stessa banda H2D del load reale, il load reale rallenta | Un solo `g_stream_selected_upload_stream` oggi; potrebbe servire un secondo copy stream per il prefetch, o rate-limit |
| R5 | **Sincronizzazioni residue (B1/B2)** | Vedi §11: senza rimuoverle il predictor non produce speedup | Sono modifiche localizzate, ma richiedono correttezza sotto CUDA graph capture |
| R6 | **Lookahead insufficiente** | Se prediction(t+1) parte dopo che l'expert è già servito, tardi | Prototipo cross-token layer-0: il gap sampling+bookkeeping è la finestra utile |
| R7 | **Correttezza del routing** | Il predictor NON deve mai sostituire il router reale | Vincolo esplicito: il router reale rimane authoritative; il predictor tocca solo la cache/prefetch |
| R8 | **Interazione con DSpark/MTP** | DSpark modifica il pattern di computazione multi-token — un predictor sbagliato può interagire con l'accept/reject | Nel primo prototipo tenere DSpark **spento** |
| R9 | **Piccola cache VRAM** | 8 GB su 12 significa slot compatti piccoli; false positive del predictor sono più costose relativamente | Preferire predictor conservativi (alto precision, basso recall) all'inizio |
| R10 | **Confronto misure** | La baseline 0.75 t/s ha varianza; misure senza sample size adeguato fuorviano | Runs multipli, seed fisso, prompt fisso, temperatura 0 |

---

## 13. Piano minimale per raccogliere i primi trace (SOLO profiling)

Sequenza a **impatto codice minimo**, in ordine:

### Fase 0 — misure a zero righe di codice
1. Attivare `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE=1` sulla baseline;
   girare il decode di 64 token con il prompt di riferimento; raccogliere
   per ogni layer di ogni token: `sync_ms, read_ms, load_ms, resume_ms,
   total_ms`. Aggregare per layer: distribuzione dei tempi.
2. Attivare `DS4_ROCM_GRAPH_TOKEN_PROFILE=1` (funziona anche su CUDA in
   quanto la funzione host è comune) per ottenere encode/execute/read
   per-token.
3. Salvare stderr in `traces/baseline_ssd_profile.txt`. Estrarre in CSV.

Questa fase risponde a: "quanto della latenza per token è dovuta al
selected-load?" e permette di stimare il tetto teorico di miglioramento
prima di scrivere codice.

### Fase 1 — trace routing (impatto minimo: ~30 righe)
1. Aggiungere in `ds4.c:21167` (subito dopo la lettura di `selected_ids`
   in `metal_graph_decode_cuda_selected_load`) una chiamata a un nuovo
   `expert_trace_record_cuda(token_index, il, selected_ids,
   DS4_N_EXPERT_USED)` che scrive su file gated dall'env
   `DS4_EXPERT_TRACE_PATH=/tmp/ds4_trace.tsv`.
2. Se possibile, leggere anche i router weights (già device-resident in
   `router_weights_by_tier`): un secondo `ds4_gpu_tensor_read` di 6 float
   allo stesso layer.
3. Formato consigliato per riga TSV:
   `token_idx\tlayer\te0\te1\te2\te3\te4\te5\tw0\tw1\tw2\tw3\tw4\tw5\tload_ms`

### Fase 2 — analisi offline (nessun codice DS4)
1. Costruire la sequenza `(token, layer) → set-of-6-experts`.
2. Statistiche di base:
   * frequenza expert per layer (skew hotness);
   * `P(expert ∈ selected(t+1, L) | expert ∈ selected(t, L))` — semplice
     Markov ordine 1 cross-token same-layer;
   * `P(expert ∈ selected(t, L+1) | expert ∈ selected(t, L))` — intra-token;
   * cache hit-rate simulata per cap ∈ {6, 12, 24, 48, 96} slot con
     politica LRU pura vs LRU+hotness.
3. Confrontare con il baseline "hotlist statica di popolarità" fornito da
   `ds4_expert_profile_write_hotlist_file` (`ds4.c:1540`) — la hotlist
   Metal-side dà un naive "predictor" di riferimento.

### Fase 3 — decisione go/no-go
* Se la hit-rate simulata di un predictor Markov ordine-1 è >> del baseline
  hotlist e la latenza di `selected_load` è >> 20% del tempo per-token
  (verifica con Fase 0), il predictor ha spazio.
* Altrimenti fermarsi qui: il collo di bottiglia non è il routing miss ma
  il compute o il baseline hotlist copre già bene.

Solo dopo Fase 3 (positiva) si prosegue con:
* rimozione mirata di B1 e B2 (§11),
* implementazione del predictor runtime (thread separato o kernel),
* integrazione con `ds4_gpu_stream_expert_cache_begin_selected_load` come
  API di prefetch.

---

## Riepilogo dei pointer file per riferimento rapido

* Layer loop decode SSD: `ds4.c:30547` (`metal_graph_eval_token_raw_swa_streaming`)
* Encode di un layer: `ds4.c:21835` (`metal_graph_encode_decode_layer_phase`)
* Router select CUDA: `ds4_cuda.cu:18714`, kernel `ds4_cuda.cu:11947`
* Selected IDs on device: `ds4.c:17222-17225`; host snapshot in
  `ds4.c:21148` (`selected_ids[6]`)
* Cache CUDA compact: `ds4_cuda.cu:130-152` (`cuda_stream_selected_cache`),
  begin_load `ds4_cuda.cu:26088-26220`
* Copy stream: `ds4_cuda.cu:450` (`g_stream_selected_upload_stream`)
* H2D expert: `ds4_cuda.cu:2193-2282` (`cuda_model_copy_to_device_streamed`)
* MoE dispatcher: `ds4_cuda.cu:23543` (`routed_moe_launch`)
* End-of-encode sync: `ds4_cuda.cu:3685-3690` (`ds4_gpu_end_commands` →
  `cudaDeviceSynchronize`)
* Logits readback: `ds4_cuda.cu:3142-3152` (`ds4_gpu_tensor_read`), chiamato
  da `ds4.c:30756`
* Selected-load driver: `ds4.c:21123-21193` (con env profiler già presente:
  `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE`)
* Prefetch API pubblica: `ds4_gpu.h:218`
  (`ds4_gpu_stream_expert_cache_begin_selected_load`)
* Expert profiler (Metal): `ds4.c:1231-1665` (`ds4_expert_profile*`),
  call site Metal `ds4.c:21686-21728` (`metal_graph_profile_router_selection`)
* Cache Metal LRU (per riferimento): `ds4_metal.m:663`, prune
  `ds4_metal.m:14559`, worker pool `ds4_metal.m:12076`, hotness
  `ds4_metal.m:679`
* Async worker pattern in ds4.c: `ds4.c:21371`
  (`metal_graph_selected_async_load_worker_main`)
* DSpark capture (nessun expert hint): `ds4.c:27229`
  (`metal_graph_dspark_capture_decode_layer`)

---

## Conclusione preliminare (non ancora un impegno implementativo)

1. Il problema che il progetto vuole studiare **esiste e è ben localizzato**
   nel percorso `--cuda --ssd-streaming` decode: ogni layer paga uno stall
   host per `sync + read + load` prima di poter eseguire la MoE.
2. Il codice espone già le primitive giuste (copy stream separato, eventi,
   staging pinned, worker pattern, API `begin_selected_load`).
3. Ci sono due sync bloccanti (B1 e B2 in §11) che oggi impediscono
   l'overlap. Se non rimossi, nessun predictor darà speedup misurabile.
4. Un profiler nascosto (`DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE`) permette
   di misurare **subito**, senza modifiche, quanto pesano quei sync — è il
   primo passo raccomandato.
5. L'expert profiler Metal-only `--expert-profile` è **quasi tutto
   riusabile** su CUDA: solo il call site va aggiunto. È il modo più
   economico per ottenere trace di routing per l'analisi offline della
   prevedibilità.
6. Il vincolo di correttezza è rispettabile by design: la nostra pipeline
   può solo popolare la cache, mai riscrivere la selezione — la
   `routed_moe_launch` continua a consumare `metal_graph_router_selected(g)`
   invariato.
