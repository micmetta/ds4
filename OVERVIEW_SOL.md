# DS4 — analisi per predictive expert prefetch

Data dell'analisi: 2026-08-17
Branch verificato: `predictive-expert-prefetch`

## 1. Risposta breve

Un predictive expert prefetch corretto è realisticamente realizzabile, ma non può essere aggiunto chiamando in parallelo l'attuale API CUDA `ds4_gpu_stream_expert_cache_begin_selected_load()`.

Nel backend CUDA corrente la cosiddetta expert cache è un solo scratch compatto contenente gli expert selezionati per un layer. Ogni nuovo caricamento invalida e sovrascrive il contenuto precedente. Un prefetch per `(token t+1, layer L)` eseguito mentre il token `t` usa quello scratch potrebbe quindi corrompere la disponibilità degli expert correnti o contendere le strutture globali di staging.

È invece plausibile aggiungere un livello separato di prefetch, con chiavi `(generation, token, layer, expert)`, buffer propri e una politica che non possa espellere i dati demand del token corrente. La prima versione dovrebbe usare un predictor CPU molto leggero e una cache di staging pinned-host; una seconda versione può aggiungere slot GPU persistenti o due banchi demand/prefetch.

La correttezza può essere mantenuta integralmente: gli ID reali prodotti dal router continuano a determinare gli expert eseguiti; il risultato del predictor viene usato soltanto per preparare dati. Al lookup reale si promuovono esclusivamente gli expert previsti che coincidono con gli ID autoritativi.

## 2. Fonti verificate e correzioni principali a `OVERVIEW.md`

Sono stati letti `README.md`, `AGENT.md`, `CONTRIBUTING.md`, `MODEL_CARD.md` e `OVERVIEW.md`, poi sono stati verificati direttamente `ds4.c`, `ds4_cuda.cu`, `ds4.h` e, come confronto, `ds4_metal.m`.

Le correzioni più importanti rispetto a `OVERVIEW.md` sono:

- Il backend CUDA non possiede oggi una cache expert persistente dimensionata dal budget da 8 GB. Le API CUDA per budget, configurazione e seed/preload sono stub/no-op. Il valore configurato viene elaborato dal motore, ma non crea una cache CUDA LRU.
- Il percorso CUDA predefinito esegue già un overlap utile per gli expert correnti dei layer non-hash: il worker carica gli expert routed mentre il thread principale calcola lo shared expert.
- La `cudaStreamSynchronize()` interna al caricamento blocca il worker, ma non annulla questo overlap con lo shared expert. Il join/barrier avviene prima dell'esecuzione routed.
- Il profiler `--expert-profile` non registra una sequenza raw token-layer e viene inizializzato soltanto sul backend Metal. Il call site comune esiste anche nel percorso CUDA, ma il profiler resta inattivo.
- `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE` misura il percorso di fallback sincrono. Con l'overlap CUDA predefinito attivo, non misura il percorso normale; disabilitare l'overlap per raggiungerlo cambia il comportamento che si vuole profilare.
- `DS4_CUDA_MMQ=0` disabilita il tier MMQ di prefill/batched verify. Il commento del codice precisa che il decode a token singolo non è interessato da quel tier.

I dati hardware, la dimensione effettiva del file Q2 e i circa 0,75 token/s sono assunti come misure sperimentali fornite, non sono verificabili dal solo sorgente.

## 3. Forma del modello rilevante

Per `DS4_SHAPE_FLASH`, `ds4.c:541-577` definisce:

- 43 layer;
- hidden size 4096;
- 256 routed expert;
- top-k di 6 expert;
- 3 layer hash (`n_hash_layer = 3`);
- expert FFN dimension 2048.

Nei layer 0-2 la selezione può essere determinata tramite la tabella token-to-expert. Negli altri layer dipende dal router calcolato sullo stato del layer corrente.

## 4. Call flow reale del decode CUDA + SSD streaming

Il percorso principale osservato è:

```text
ds4_session_eval_internal()                         ds4.c:61462
  └─ metal_graph_eval_token_raw_swa()              ds4.c:30727
      └─ metal_graph_eval_token_raw_swa_streaming() ds4.c:30547
          ├─ begin token commands
          ├─ per layer 0..42
          │   └─ metal_graph_encode_decode_layer()
          │       └─ metal_graph_encode_decode_layer_phase()
          │           ├─ attention
          │           ├─ FFN norm
          │           ├─ router projection
          │           ├─ router select top-k
          │           ├─ selected-expert load
          │           │   || shared-expert compute (non-hash, default)
          │           ├─ routed MoE
          │           └─ residual/output del layer
          ├─ output norm + LM head
          ├─ end commands / device synchronization
          └─ logits D2H
```

La modalità streaming usa normalmente la static decode map e il loop batched sui layer (`ds4.c:30597-30624`). Il fallback non-batched chiude i comandi per ogni layer (`ds4.c:30650-30689`) ma non è il percorso predefinito nelle condizioni della baseline.

### 4.1 Router e momento in cui gli expert diventano noti

In `metal_graph_encode_decode_layer_phase()`:

1. `ffn_norm` viene prodotto sul device (`ds4.c:23741-23749`).
2. La proiezione del router scrive i logits tramite `metal_graph_matmul_plain_tensor()` (`ds4.c:23833-23834`).
3. `ds4_gpu_router_select_tensor()` (`ds4.c:23836-23849`, implementazione CUDA `ds4_cuda.cu:18714`) calcola top-k e pesi sul device. Il kernel principale è `router_select_warp_topk_kernel()` (`ds4_cuda.cu:11947`).

I risultati risiedono nei tensori GPU allocati in `ds4.c:17222-17225`:

- `router_logits_by_tier`: 256 float;
- `router_probs_by_tier`: 256 float;
- `router_selected_by_tier`: 6 `int32_t`;
- `router_weights_by_tier`: 6 float.

Per i layer non-hash, questo è il punto esatto in cui gli ID reali sono noti sul device. Diventano disponibili alla CPU dopo la sincronizzazione/readback usata dal loader.

Per i layer hash 0-2, `layer_hash_selected_experts()` (`ds4.c:10692-10710`) legge direttamente sei ID dalla tabella indicizzata dal token. `metal_graph_decode_set_hash_selected_override()` (`ds4.c:20877-20927`) effettua il caricamento selezionato. Dopo il sampling CPU del prossimo token, gli expert hash dei primi tre layer del token successivo sono quindi conoscibili esattamente, senza predictor; questo però avviene alla fine del compute di `t`, non abbastanza presto per sovrapporli al grosso di `compute(t)`.

### 4.2 Lookup, miss e caricamento nel backend CUDA

La struttura `cuda_stream_selected_cache` è definita in `ds4_cuda.cu:130-152`; ne esiste una sola istanza globale, `g_stream_selected_cache`.

`cuda_stream_selected_cache_begin_load()` (`ds4_cuda.cu:26088-26221`):

1. invalida subito lo stato precedente;
2. deduplica i sei ID selezionati nel layer corrente;
3. alloca o amplia tre slab GPU per gate/up/down;
4. per ogni expert unico carica gate, up e down con `cuda_model_copy_to_device_streamed()`;
5. copia sul device la rimappatura dagli ID originali agli slot compatti;
6. marca lo scratch valido per il layer corrente.

Non c'è un lookup persistente `(layer, expert)`, né un hit tra token o tra layer. La sola forma di riuso è la deduplicazione di ID ripetuti nello stesso top-k. La verifica in `routed_moe_launch()` (`ds4_cuda.cu:23543`, controlli intorno a `23703-23722` e `23897-23938`) controlla che lo scratch appena riempito appartenga al layer/tabella attesi e usa la rimappatura compatta; non è un lookup di cache tradizionale.

Di conseguenza, nella baseline CUDA:

- ogni expert unico richiesto è un demand load;
- l'insertion equivale al riempimento dello scratch corrente;
- l'eviction logica avviene a ogni nuova `begin_load()` e sostituisce in blocco il contenuto precedente;
- non esiste LRU, LFU o hotness policy CUDA;
- i buffer fisici vengono riusati e crescono quando serve, ma i pesi contenuti non sono conservati semanticamente come cache.

### 4.3 Punto esatto della lettura SSD e del trasferimento GPU

Il file del modello viene registrato nel backend CUDA da `ds4_gpu_set_model_fd()` (`ds4_cuda.cu:4313-4347`). Su Linux il codice tenta normalmente di riaprire `/proc/self/fd/N` con `O_DIRECT`, salvo `DS4_CUDA_NO_DIRECT_IO`, e impone l'allineamento richiesto.

La catena di caricamento è:

```text
cuda_stream_selected_cache_begin_load()             ds4_cuda.cu:26088
  └─ cuda_model_copy_to_device_streamed()           ds4_cuda.cu:2193
      ├─ cuda_model_stage_read()                     ds4_cuda.cu:2098
      │   └─ pread() O_DIRECT o fallback pread()
      ├─ cudaMemcpyAsync(..., H2D, upload_stream)
      ├─ cudaEventRecord(...)
      └─ cudaStreamSynchronize(upload_stream)        ds4_cuda.cu:2272-2280
```

Esistono quattro staging buffer host pinned e uno stream CUDA non-blocking dedicato all'upload. Tuttavia `pread()` è sincrona per il thread che esegue il loader e ogni invocazione di `cuda_model_copy_to_device_streamed()` sincronizza lo stream prima di restituire. Poiché gate/up/down vengono caricati con chiamate separate per ogni expert, il decode può effettuare fino a 18 sequenze sostanzialmente serializzate per un top-k senza duplicati.

Se il file descriptor non è disponibile, il fallback copia dalla mmap con una `cudaMemcpy()` bloccante (`ds4_cuda.cu:2205-2211`).

## 5. Cosa è sincrono, cosa è asincrono e overlap già presente

Per un layer non-hash nel percorso CUDA predefinito:

```text
default/decode stream          CPU worker + upload stream
---------------------          --------------------------
router
cudaDeviceSynchronize()
start worker          -------> readback ID
shared expert                  pread gate/up/down
                              cudaMemcpyAsync H2D
flush/device sync              stream sync per copia
join worker           <------- load completo
routed experts
```

I punti chiave sono:

- `metal_graph_use_cuda_selected_shared_overlap()` (`ds4.c:20284-20293`) abilita di default l'overlap CUDA.
- `ds4_gpu_signal_selected_readback_ready()` (`ds4.c:24498`, CUDA `ds4_cuda.cu:30382-30385`) usa attualmente `cudaDeviceSynchronize()`: è una barriera dopo il router.
- `metal_graph_selected_async_load_start()` (`ds4.c:24505`, worker in `ds4.c:21280-21492`) avvia il caricamento su un thread CPU.
- Il main thread calcola lo shared expert (`ds4.c:24518-24557`) mentre il worker esegue I/O e upload.
- `ds4_gpu_flush_commands()` (`ds4.c:24560`) sul backend CUDA esegue una sincronizzazione device, poi il main fa join del worker (`ds4.c:24562`).
- I routed expert vengono eseguiti soltanto dopo che i pesi selezionati sono pronti (`ds4.c:24608-24629`).

Quindi I/O e H2D degli expert correnti sono asincroni rispetto al thread principale e possono sovrapporsi allo shared-expert compute. Non sono però una pipeline completamente asincrona: `pread` blocca il worker, ogni copia sincronizza il proprio stream e il layer non procede ai routed expert finché il load non termina.

Per i layer hash l'overlap sopra non si applica: il caricamento viene avviato sincronicamente in `metal_graph_decode_set_hash_selected_override()` prima dello shared expert.

Alla fine del token, `ds4_gpu_end_commands()` sincronizza l'esecuzione e i logits vengono copiati verso host (`ds4.c:30620-30624`). `ds4_session_sample()` (`ds4.c:60745`) esegue poi il sampling CPU.

## 6. Caching, preload e prefetch esistenti

### CUDA

- Le API `ds4_gpu_stream_expert_cache_budget_for_expert_size()`, setter del budget/configurazione e seed/preload sono stub o no-op (`ds4_cuda.cu:30395-30566`).
- `--ssd-streaming-cache-experts 8GB` viene parsato e contabilizzato dal motore (`ds4.c:55414-55693`, configurazione intorno a `ds4.c:57706-57731`), ma il backend CUDA non usa quel budget per una cache persistente.
- La hotlist/preload machinery in `ds4.c:20585-20782` e `ds4.c:31042-31175` ritorna senza caricare expert sul CUDA corrente, perché il budget riportato dal backend è zero e le seed API non fanno lavoro.
- `cudaMemPrefetchAsync` e il model prefetch stream presenti nel backend riguardano mapping/HMM del modello, non la previsione di routed expert.
- Il readahead Apple in `metal_graph_stream_readahead_range_impl()` non fornisce un prefetch Linux equivalente per il percorso selezionato; il normale `O_DIRECT` bypassa comunque la page cache.

### Metal, come riferimento progettuale

Il backend Metal ha invece una vera cache per `(layer, expert)`, contatori hit/miss/eviction/pread, worker di caricamento, preload da hotlist e una politica di pruning basata prima sulla hotness e poi su `last_used` (`ds4_metal.m`, struttura intorno a `631-704`, lookup/load intorno a `14559-14769`, seed intorno a `16553-16710`). È un riferimento utile per semantica e statistiche, ma non è un'implementazione direttamente riutilizzabile nel backend CUDA.

## 7. Informazioni disponibili al tempo t

Per ciascun layer, durante il token `t`, sono disponibili:

- token ID corrente e posizione assoluta `pos`;
- sei expert ID realmente selezionati;
- sei route weights;
- i 256 router logits e le 256 probabilità sul device;
- `ffn_norm`/input del router, 4096 float sul device;
- hidden state prima/dopo attention e dopo FFN sul device;
- la sequenza storica di selezioni, se registrata lato host.

Per un primo predictor Markov sono sufficienti `(pos, layer, expert IDs[, weights])`. Portare ogni hidden state da 4096 float su CPU aggiungerebbe circa 16 KiB per layer e nuove dipendenze di sincronizzazione; non è consigliabile nel primo esperimento. Un predictor device-side eviterebbe il D2H ma competerebbe per risorse GPU con un decode già lento. Il primo prototipo dovrebbe quindi essere CPU, aggiornato dagli ID già letti dal loader.

Limite autoregressivo: il token `t+1` e il suo hidden state non esistono finché i logits di `t` non sono stati prodotti e campionati. Un predictor lanciato durante `compute(t)` può usare soltanto stato e storia fino a `t`; non può eseguire in anticipo il router reale di `t+1`.

## 8. Inserimento della pipeline predittiva

### A. Raccolta delle feature

Punto comune: subito dopo `ds4_gpu_router_select_tensor()` e il call site di profiling in `ds4.c:23836-23860`.

Punto a minore duplicazione del readback nel percorso normale:

- layer non-hash: nel worker async, dopo il readback degli ID già necessario al demand load (`ds4.c:21280-21492`), aggiungendo `pos` al job;
- layer hash: in `metal_graph_decode_set_hash_selected_override()` dopo la lookup della tabella (`ds4.c:20877-20927`).

Per la sola strumentazione iniziale è più semplice un hook comune post-router, purché sia indipendente da `g_expert_profile.active` e riusi una sincronizzazione già inevitabile.

### B. Lancio del predictor

Un worker CPU persistente può ricevere il record `(t, layer L, E(t,L), weights)` e produrre immediatamente una lista corta per `(t+1,L)`. Questo lascia come finestra utile il resto dei layer di `t`, la testa di output e il sampling. Il predictor deve essere non bloccante: coda bounded, possibilità di scartare lavoro in ritardo e nessun join sul percorso di `t`.

### C. Avvio del prefetch

Non usare l'attuale `begin_selected_load`, che invalida lo scratch demand. Serve una nuova API CUDA con:

- staging pool pinned separato oppure un I/O scheduler unico thread-safe;
- slot indicizzati per layer/expert e generation/token target;
- stream/eventi CUDA separati se si carica già in VRAM;
- limite di byte/in-flight request e cancellazione logica dei risultati tardivi;
- priorità assoluta ai miss demand rispetto al lavoro speculativo.

Prima scelta sperimentale: leggere SSD in buffer host pinned durante `compute(t)`. Riduce il rischio e consente di misurare il valore della previsione, anche se l'H2D resta sul percorso di `t+1`. Seconda scelta: caricare anche in una cache GPU bounded per eliminare sia SSD sia H2D sui hit.

### D. Integrazione con la cache/demand path

Il lookup deve avvenire in `cuda_stream_selected_cache_begin_load()` dopo che il router reale ha fornito i sei ID e prima di emettere i normali `pread`.

Per ogni ID reale:

1. cercare un entry pronto con la chiave corretta;
2. promuovere/copiare soltanto gli entry coincidenti;
3. caricare normalmente i miss;
4. costruire la rimappatura compatta per i soli expert reali;
5. lasciare gli expert previsti erroneamente fuori dall'esecuzione e farli scadere/evictare.

La separazione già esistente fra `router_selected` e `slot_selected_tensor` è utile: gli ID autoritativi possono rimanere immutati mentre cambia soltanto l'indirizzamento dei pesi nella cache.

## 9. Ostacoli al vero overlap e rischio di regressione

1. **Scratch singleton.** È il blocco architetturale principale: non può contenere contemporaneamente il demand corrente e il prefetch futuro.
2. **Globali non predisposte alla concorrenza.** Staging ring, upload stream, eventi e stato cache sono globali; un secondo producer concorrente sarebbe soggetto a race senza separazione o scheduler.
3. **Sincronizzazioni device-wide.** Il segnale post-router e il flush usano `cudaDeviceSynchronize()`, serializzando più lavoro del necessario.
4. **Granularità del loader.** Gate/up/down e i diversi expert vengono caricati con chiamate separate, ciascuna conclusa da `cudaStreamSynchronize()`.
5. **Contesa SSD/PCIe/copy engine.** Un falso positivo può ritardare un demand load o lo shared compute. Il lavoro speculativo deve essere throttled e preemptibile almeno a livello di scheduling I/O.
6. **Finestra temporale variabile.** La previsione per i primi layer di `t+1` ha meno tempo per completarsi di quella per gli ultimi layer, a meno di iniziare non appena si conoscono gli expert corrispondenti di `t`.
7. **Beneficio massimo inferiore alla latenza SSD totale.** DS4 nasconde già parte del load corrente dietro lo shared expert. Il prefetch può rimuovere soprattutto la coda non coperta, oltre al percorso hash oggi sincrono.
8. **Budget VRAM reale.** Il valore 8 GB non è oggi una cache CUDA allocata; trasformarlo in slot residenti richiede contabilizzazione rispetto a KV cache, static spans, scratch e headroom.

Prima di ottimizzare il predictor conviene inoltre misurare separatamente il costo delle 18 possibili sequenze read/copy/sync: una migliore pipeline del demand loader è un'ottimizzazione ortogonale e può cambiare il margine disponibile per il predictive prefetch.

## 10. Strumentazione minima proposta

La strumentazione deve essere opt-in e non deve attivare `g_expert_profile.active`, perché quel flag disabilita il layer batching/static path e altererebbe il percorso misurato.

### Trace delle route

Nuovo flag o variabile, per esempio `--expert-trace FILE` / `DS4_EXPERT_TRACE_PATH`, con una riga JSONL o TSV per `(token, layer)`:

```text
pos  token_id  layer  is_hash  e0 e1 e2 e3 e4 e5  w0 w1 w2 w3 w4 w5
```

Per l'analisi iniziale di `P(E(t+1,L) | E(t,L))` gli ID sono obbligatori; i weights sono utili ma opzionali. Logits completi e hidden state vanno esclusi dalla prima iterazione.

Per minimizzare la perturbazione:

- riusare il readback degli ID già eseguito dal demand loader;
- accumulare record in un buffer memory e scrivere in blocco;
- includere `pos` come indice token stabile;
- non fare `fprintf` sincroni nel loop di ogni layer.

### Trace cache/I/O

In `cuda_stream_selected_cache_begin_load()` registrare:

```text
pos, layer, requested_ids, unique_ids,
hit_ids, miss_ids, replaced_layer, replaced_ids,
ssd_bytes, ssd_ms, h2d_bytes, h2d_wait_ms, load_total_ms
```

Nello stato attuale `hit_ids` sarà vuoto salvo la deduplicazione intra-top-k; questo dato è importante perché rende esplicita la baseline. La sostituzione corrisponde all'intero scratch precedente.

In `cuda_model_copy_to_device_streamed()` separare almeno:

- tempo host di `cuda_model_stage_read()`/`pread`;
- tempo fino al completamento H2D;
- byte letti/copiati;
- eventuale fallback da `O_DIRECT`.

Per la prima versione si possono usare clock monotonic host attorno a operazioni che già bloccano. CUDA events servono successivamente per isolare il tempo device senza aggiungere nuove sincronizzazioni. A livello token si possono affiancare i timing esistenti `DS4_ROCM_GRAPH_TOKEN_PROFILE`/`DS4_METAL_GRAPH_TOKEN_PROFILE` del percorso comune e `DS4_CUDA_MOE_PROFILE`; le etichette “Metal/ROCm” nel primo caso sono storiche, ma il controllo è nel codice condiviso.

### Eventi minimi necessari

- `ROUTE_SELECTED(pos, layer, IDs, weights)`;
- `CACHE_LOOKUP(pos, layer, hit/miss IDs)`;
- `SSD_READ_BEGIN/END(layer, expert, tensor, bytes)`;
- `H2D_BEGIN/END(layer, expert, tensor, bytes)`;
- `CACHE_REPLACE(old layer/IDs, new layer/IDs)`;
- opzionale `SHARED_EXPERT_BEGIN/END` e `ROUTED_EXPERT_BEGIN/END` per misurare la coda di load non nascosta.

Tutti gli eventi devono avere timestamp monotonic e un generation/session identifier se si profilano richieste concorrenti.

## 11. Piano concreto in cinque step

1. **Trace baseline.** Implementare soltanto gli eventi sopra, verificare che a tracing disattivato il percorso sia invariato e raccogliere route, I/O e intervalli di overlap sulla RTX 4080 Laptop.
2. **Analisi offline.** Calcolare reuse diretto, matrici Markov per layer, top-k precision/recall, copertura pesata per byte, distanza temporale disponibile e simulazioni di cache con diversi budget. Confrontare anche hotlist e oracle/replay.
3. **Substrato di prefetch senza predictor.** Implementare staging pinned separato e lookup sicuro; pilotarlo prima con una trace replay/oracle o con gli expert hash esatti post-sampling. Dimostrare equivalenza dei token/logits e misurare hit pronti vs tardivi.
4. **Primo predictor runtime.** Aggiungere reuse/Markov CPU con coda bounded, confidence threshold e limite I/O. Il router reale resta l'unica sorgente degli ID eseguiti; i miss continuano sul demand path.
5. **Cache GPU e rimozione delle barriere selettive.** Solo se i dati mostrano beneficio, aggiungere slot GPU bounded/double-buffered ed eventi stream-specific al posto delle sincronizzazioni device-wide dove sicuro. Misurare token/s, latenza p50/p95, byte sprecati, pollution e consumo VRAM.

## 12. Criteri di correttezza per ogni esperimento

- Il tensore degli ID del router reale non viene sostituito dal predictor.
- L'ordine e i pesi degli expert realmente eseguiti restano quelli del router.
- Un entry previsto viene consumato soltanto dopo match esatto di layer, expert, formato/layout, device e generation.
- Un prefetch incompleto o tardivo degrada a demand load, senza attesa aggiuntiva obbligatoria.
- A configurazione deterministica, token e logits devono rispettare la stessa tolleranza della baseline vanilla.
- Il tracing disattivato deve avere costo nullo o trascurabile; quello attivo va quantificato confrontando token/s e timing per layer.

## Conclusione

L'architettura dispone già dei mattoni utili — selezione top-k sul device, readback degli ID, worker CPU, pinned staging, upload stream e remapping compatto — ma non della struttura dati necessaria al prefetch cross-token. Il punto migliore per la previsione è immediatamente dopo che gli ID di un layer del token `t` sono stati letti; il punto corretto per consumare il risultato è il demand loader dopo la selezione reale di `t+1`.

Il primo obiettivo non deve essere un predictor sofisticato: deve essere una trace affidabile che misuri la prevedibilità e, soprattutto, quanta parte del load resta oggi fuori dall'overlap con lo shared expert. Solo quel residuo rappresenta il beneficio immediatamente ottenibile dal predictive expert prefetch.
