# TEST_3 — Oracle prefetch e misura della capacità VRAM

TEST_3 studierà il beneficio massimo teorico di un predictive expert prefetch
nel percorso CUDA + SSD streaming. Prima di introdurre un predictor ML,
usiamo l'oracolo di routing prodotto da Run A: conosce gli expert realmente
selezionati nel futuro e permette di misurare separatamente il limite della
memoria e il beneficio potenziale del prefetch.

È implementato il primo sotto-test di capacità VRAM. Il banco prefetch e Run C
non sono ancora implementati: saranno il passo successivo, dopo avere letto le
misure del sotto-test.

## Obiettivo

Il test finale confronterà tre processi DS4 separati:

~~~text
Run A  -> produce il routing oracle del percorso di riferimento
Run B  -> replay oracle, nessun prefetch: controllo riproducibile
Run C  -> replay oracle, prefetch oracle: misura del beneficio ideale
~~~

Il router reale non verrà mai rimosso dal flusso. Come in TEST_1, il replay
serve soltanto a rendere B e C confrontabili: DS4 può continuare a registrare
la scelta originale del router, ma nel test usa gli ID dell'oracolo per
garantire lo stesso percorso. L'implementazione finale con predictor ML non
dovrà invece sostituire gli ID del router reale: potrà solo prefetchare pesi.

## Glossario essenziale

| Termine | Significato in TEST_3 |
|---|---|
| **Banco domanda** | Buffer CUDA che contiene `E_l(t)`, gli expert necessari subito per eseguire il layer corrente. È l'attuale scratch cache di DS4. |
| **Banco prefetch** | Nuovo buffer/cache CUDA separato che conserva gli expert previsti per un uso futuro. Non deve invalidare il banco domanda. |
| **`E_l(t)`** | Sei expert realmente usati dal layer `l` sul token corrente `t`. |
| **`E_l(t+1)`** | Expert che saranno richiesti dallo stesso layer sul token successivo. In TEST_3 arrivano dall'oracolo; in futuro arriveranno dal predictor. |
| **Prefetch hit** | Gli expert necessari al token corrente sono già pronti nel banco prefetch e non richiedono la lettura SSD sul percorso critico. |
| **Prefetch miss** | La previsione non è disponibile, è incompleta o non corrisponde agli expert richiesti: DS4 deve usare il normale demand load. |
| **VRAM libera** | Memoria GPU effettivamente disponibile nel momento della misura. È diversa dalla memoria totale della GPU e può cambiare durante l'inferenza. |

## Perché il sotto-test VRAM viene prima

Durante l'esecuzione di `E_l(t)`, i pesi `gate`, `up` e `down` dei sei expert
correnti sono già nella VRAM, nel buffer temporaneo corrente. TEST_2 ha
misurato circa **40.5 MiB** per tale gruppo di sei expert.

Un prefetch per `E_l(t+1)` richiede ulteriore VRAM. Inoltre non basta un solo
secondo buffer: dopo aver prefetchato `E_0(t+1)`, gli expert devono restare
disponibili fino a quando il token `t+1` raggiungerà il layer 0. Durante il
token `t` dovrebbero essere conservati anche `E_1(t+1)`, `E_2(t+1)` e così
via. Nel caso ideale dei 43 layer osservati in TEST_2, l'ordine di grandezza è:

~~~text
token t, layer 0  -> conserva E_0(t+1)
token t, layer 1  -> conserva E_1(t+1)
...
token t, layer 42 -> conserva E_42(t+1)

token t+1, layer 0 -> riusa E_0(t+1)
~~~

~~~text
43 layer x 40.5 MiB ~= 1.7 GiB
~~~

La stima non include metadata, workspace CUDA, buffer temporanei o eventuali
duplicati. Per questo non possiamo assumere che la RTX 4080 Laptop da 12 GB
abbia sempre spazio sufficiente: dobbiamo misurare la VRAM libera nel punto
esatto in cui il banco corrente è già residente.

## Flusso 1 — sotto-test: misura VRAM con il percorso domanda corrente

**Obiettivo:** misurare quanta VRAM è realmente libera mentre DS4 ha già
caricato gli expert correnti, senza prefetchare nulla e senza produrre uno
speedup artificiale.

~~~text
test_predictor/test_3/test_3_runner.c
    |
    |  crea output incrementale e avvia Run A / Run B come processi distinti
    v
Run A: ./ds4 con DS4_ROUTING_ORACLE_WRITE_PATH
    |
    |  ds4.c: ds4_routing_oracle_*()
    |  salva (token_index, layer) -> expert IDs + router weights
    v
output_N/routing_oracle.bin
    |
    v
Run B: ./ds4 con DS4_ROUTING_ORACLE_REPLAY_PATH
    |
    |  ds4.c: metal_graph_decode_cuda_selected_load(...)
    |  legge gli ID del router reale, poi applica il replay solo per il test
    v
ds4_cuda.cu: cuda_stream_selected_cache_begin_load(...)
    |
    |  carica gate/up/down di E_l(t): SSD -> RAM pinned -> VRAM
    |  rende valido g_stream_selected_cache, il banco domanda attuale
    v
ds4_cuda.cu: ds4_gpu_get_memory_info(...)
    |
    |  cudaMemGetInfo(&free_bytes, &total_bytes)
    |  misura: prima del demand load, dopo il demand load di E_l(t),
    |  dopo l'invio del MoE allo stream CUDA
    v
ds4.c: trace JSONL TEST_3 (DS4_TEST_3_VRAM_TRACE_PATH)
    |
    |  salva token, layer, bytes di E_l(t), VRAM libera/totale,
    |  stima bytes richiesti dal prefetch e margine disponibile
    v
output_N/vram_ram_trace.jsonl + summary.json
~~~

Le funzioni già esistenti nel percorso sono:

| File | Funzione | Ruolo nel sotto-test |
|---|---|---|
| `ds4.c` | `metal_graph_decode_cuda_selected_load(...)` | Punto in cui gli expert del layer corrente diventano disponibili al lato CPU e viene richiesto il loro load. |
| `ds4_cuda.cu` | `cuda_stream_selected_cache_begin_load(...)` | Costruisce il banco domanda temporaneo con gli expert selezionati. |
| `ds4_cuda.cu` | `cuda_model_copy_to_device_streamed(...)` | Per ogni gruppo `gate`/`up`/`down`, legge dall'SSD nei buffer RAM pinned e trasferisce in VRAM. |
| `ds4_cuda.cu` | `ds4_gpu_routed_moe_one_tensor(...)` | Esegue il MoE usando il banco domanda già pronto. |
| CUDA runtime | `cudaMemGetInfo(...)` | API che verrà invocata dalla nuova sonda per ottenere VRAM libera e totale. |

`after_moe_queued` non fa una sincronizzazione aggiuntiva: indica che il MoE
è stato inviato allo stream CUDA. In questo modo la diagnostica non forza il
completamento del lavoro GPU e non altera intenzionalmente la velocità.

Questo sotto-test non crea ancora il banco prefetch e non modifica routing,
logits o expert eseguiti. Serve a scegliere un budget realistico per il test
completo: tutti i layer, un sottoinsieme di layer, oppure un numero limitato
di expert per layer.

## Esecuzione del sotto-test

In CLion crea una configurazione Native Application chiamata, ad esempio,
`TEST_3 VRAM Capacity Probe`:

| Campo | Valore |
|---|---|
| Target | `test_3_runner` |
| Executable | `/home/michele/ds4/test_predictor/test_3/test_3_runner` |
| Working directory | `/home/michele/ds4/` |
| Environment variables | `DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |

Il target ricompila `ds4` in modo seriale e poi avvia il runner. Ogni avvio
crea `output`, poi `output_2`, `output_3` e così via.

## Risultati osservati — sotto-test VRAM

Sono stati eseguiti tre esperimenti completi: `output/`, `output_2/` e
`output_3/`. Tutti hanno eseguito Run A per creare l'oracolo e Run B per
riprodurre quel percorso con le sonde VRAM attive. `output_3/` aggiunge la
mappa dettagliata descritta più sotto.

| Misura | `output` | `output_2` | Significato |
|---|---:|---:|---|
| Esito Run A | 0 | 0 | L'oracolo è stato creato correttamente. |
| Esito Run B | 0 | 0 | Il replay e la raccolta delle sonde sono terminati correttamente. |
| Token di decode osservati | 63 | 63 | Token dall'indice 20 all'indice 82. |
| Layer per token | 43 | 43 | Tutti i layer del modello sono stati attraversati. |
| Eventi di demand load | 2.709 | 2.709 | `63 × 43`: un caricamento domanda per ogni coppia token/layer. |
| Sonde VRAM totali | 8.127 | 8.127 | Tre sonde per ogni demand load. |
| Eventi `is_hash=true` | 567 | 567 | `63 × 3`: i tre layer hash del modello. |
| Eventi MoE ordinari | 2.520 | 2.520 | `63 × 40`: i layer MoE non-hash. |
| Byte demand load dopo il caricamento | 42.467.328 | 42.467.328 | Circa 40,5 MiB per il gruppo di sei expert. |
| VRAM libera minima | 0 byte | 0 byte | Nessun margine libero misurabile. |
| VRAM libera massima | 0 byte | 0 byte | Il valore resta zero lungo tutto il decode. |
| VRAM totale rilevata | 12.878.086.144 byte | 12.878.086.144 byte | Circa 12 GiB, coerente con la RTX 4080 Laptop. |

I due output sono quindi coerenti fra loro: la struttura del decode e i byte
degli expert sono stabili, ma DS4 arriva al decode senza VRAM libera riportata
da CUDA. Questo non dimostra che ogni byte della GPU sia necessariamente
occupato da un singolo tensore: mostra però che, per il runtime CUDA,
`cudaMemGetInfo()` non vede spazio allocabile aggiuntivo. Un nuovo banco di
prefetch non può quindi essere aggiunto alla configurazione corrente senza
ridurre o ripartizionare qualche budget già assegnato.

La stima iniziale di un banco che trattenga un gruppo da circa 40,5 MiB per
ciascuno dei 43 layer è circa 1,7 GiB. Con zero VRAM libera, questa forma
completa di prefetch non è praticabile così com'è; dovremo valutare un budget
esplicito e limitato oppure una ripartizione dell'attuale memoria expert.

### Mappa dettagliata — `output_3`

`output_3/` ripete lo stesso sotto-test con i nuovi campi contabili. I 8.127
eventi hanno la medesima fotografia di memoria: nessuna delle categorie cambia
tra `before_demand_load`, `after_demand_load` e `after_moe_queued`. Questo è
coerente con il loader DS4: il demand load sovrascrive un banco già allocato,
non esegue una nuova `cudaMalloc` per ogni layer.

| Categoria nel trace | Valore in `output_3` | Interpretazione |
|---|---:|---|
| `total_bytes` / `used_bytes` | 12.281,5 MiB / 12.281,5 MiB | CUDA considera tutta la VRAM visibile già usata o riservata. |
| `free_bytes` | 0 MiB | Non esiste spazio immediatamente allocabile per aggiungere un secondo banco. |
| `model_arena_reserved_bytes` | 8.960,0 MiB | Chunk CUDA riservati per pesi residenti del modello. |
| `model_arena_used_bytes` | 8.393,7 MiB | Parte delle arena effettivamente assegnata ai pesi. |
| Riserva interna delle model arena | 566,4 MiB | Differenza `reserved - used`. È spazio già allocato dentro chunk che contengono anche pesi necessari; non è automaticamente sicuro da liberare o riusare. |
| `q8_f16_cache_bytes` | 2.652,0 MiB | Cache GPU di pesi Q8 convertiti in FP16. È una delle maggiori categorie modificabili solo dopo un benchmark del costo prestazionale. |
| `demand_cache_bytes` | 621,0 MiB | Capacità già allocata del banco domanda per `gate`, `up`, `down` e ID slot. È molto maggiore dei 40,5 MiB trasferiti per il gruppo corrente perché il buffer è stato dimensionato in precedenza, anche durante il prefill. |
| `tensor_bytes` | 67,8 MiB | Tensori DS4 tracciati: attivazioni, KV cache, grafo e scratch allocati con le API tensor. |
| `scratch_bytes` | 2,7 MiB | Scratch CUDA persistente noto al backend. |
| `device_cache_bytes`, `q8_f32_cache_bytes`, `derived_weights_bytes` | 0 MiB | Queste categorie non hanno allocazioni attive in questa configurazione. |

Il campo `known_allocated_bytes` vale circa 12.303,5 MiB, cioè circa 22 MiB
in più di `used_bytes`. Questo piccolo scarto non indica VRAM disponibile: è
un limite atteso di una mappa contabile che somma capacità logiche/richieste
DS4, mentre CUDA riporta la propria visione dell'allocatore, con allineamenti,
overhead e possibili relazioni interne fra cache. Per questo
`unattributed_bytes` è riportato come zero, ma non va letto come “tutta la
VRAM è spiegata perfettamente”.

La deduzione utile è più circoscritta e solida:

~~~text
VRAM nuova allocabile per un banco prefetch: 0 MiB

Possibili riserve da studiare, ma non ancora riutilizzabili automaticamente:
    - 566,4 MiB di capacità non assegnata dentro le model arena;
    - 621,0 MiB di capacità già allocata nel banco domanda;
    - 2.652,0 MiB della cache Q8 → FP16.
~~~

Il prossimo design non dovrà tentare una seconda allocazione “in più”. Dovrà
prima ripartire una di queste categorie o scegliere il sotto-test di prefetch
SSD → RAM pinned, che non richiede ulteriore VRAM. Questo esperimento operativo
prosegue in [TEST_4](../test_4/README.md), dove il banco pinned separato e il
suo dizionario `(target_token, layer) -> slot` sono descritti prima
dell'implementazione.

## Risultati osservati — sotto-test RAM

`output_4/` ha introdotto il monitoraggio RAM; `output_5/` aggiunge
`RssShmem` e completa la scomposizione della RAM residente di DS4. Entrambi
gli esperimenti sono terminati correttamente, con 8.127 sonde ciascuno:
2.709 eventi per ognuno dei tre stage del decode.

| Misura | `output_4` | `output_5` | Interpretazione |
|---|---:|---:|---|
| RAM totale Linux/WSL | 11.680 MiB | 11.680 MiB | Memoria visibile al sistema che ospita DS4. |
| RAM disponibile | 9.950--9.958 MiB | 9.945--9.958 MiB | `MemAvailable` resta molto alto e varia solo di circa 13 MiB nel decode. |
| RAM completamente libera | 9.518--9.534 MiB | 9.506--9.529 MiB | Pagine non occupate; e minore della RAM disponibile perche Linux usa RAM come cache. |
| Page cache Linux | 669--679 MiB | 674--685 MiB | Cache file stabile; il Direct I/O del demand load non dipende da essa per la lettura degli expert. |
| Swap libera | 8.191 MiB | 8.191 MiB | Swap praticamente inutilizzato. |
| RSS del processo DS4 | 1.045--1.046 MiB | 1.046--1.047 MiB | Memoria fisica residente di DS4, stabile fra i tre stage. |
| Pool DS4 gia pinned | 256,02 MiB | 256,02 MiB | Quattro buffer CUDA di staging gia allocati e riusati dal loader SSD corrente. |
| VRAM CUDA libera | 0 MiB | 0 MiB | Coerente con la mappa VRAM: la strategia RAM pinned non richiede un secondo banco GPU. |

Le piccole differenze tra i due output sono normali oscillazioni del sistema
operativo e della page cache. Non c'e alcun aumento sistematico di RSS, consumo
swap o perdita di `MemAvailable` tra `before_demand_load`,
`after_demand_load` e `after_moe_queued`: il demand load esistente riusa il
pool pinned anziche allocare nuova RAM a ogni layer.

### Mappa dettagliata — `output_5`

`output_5/` e la misura di riferimento per la RAM. I valori seguenti sono
intervalli minimi/massimi sull'intero decode; il loro intervallo ristretto
indica che la memoria host rimane stabile durante i 63 token osservati.

| Campo nel trace | Valore in `output_5` | Cosa rappresenta e cosa indica |
|---|---:|---|
| `ram_total_bytes` | 11.680 MiB | RAM totale del Linux/WSL. E il limite host complessivo, non il budget assegnabile automaticamente a CUDA. |
| `ram_available_bytes` | 9.945--9.958 MiB | Stima Linux della memoria allocabile senza iniziare swap. E il campo piu utile per scegliere il budget del futuro banco pinned. |
| `ram_free_bytes` | 9.506--9.529 MiB | RAM priva di dati. La differenza rispetto a `ram_available_bytes` comprende soprattutto memoria che Linux puo recuperare dalla cache. |
| `ram_page_cache_bytes` | 674--685 MiB | Page cache di file. E informativa, ma non viene considerata un budget sicuro per pagine pinned. |
| `swap_total_bytes` | 8.192 MiB | Swap configurata nel Linux/WSL. |
| `swap_free_bytes` | 8.191 MiB | Swap quasi interamente libera: circa 1 MiB e in uso, quindi il decode non sta lavorando sotto pressione RAM. |
| `process_rss_bytes` | 1.045,9--1.046,9 MiB | Tutta la RAM fisica residente del processo DS4. |
| `process_vmsize_bytes` | 117.412,2--117.412,7 MiB | Spazio virtuale del processo. Il valore grande e atteso: DS4 mappa il grande GGUF; non significa che 114 GiB siano occupati nella RAM fisica. |
| `process_locked_bytes` | 0 MiB | `VmLck` non espone il pool CUDA host-pinned in questo ambiente WSL. Non va quindi usato da solo per stabilire quanta RAM pinned usa DS4. |
| `process_rss_anon_bytes` | 352,1--352,6 MiB | Heap, stack e altre pagine anonime residenti del processo. |
| `process_rss_file_bytes` | 91,8--92,3 MiB | Pagine residenti supportate da file, incluse porzioni del GGUF mappato. |
| `process_rss_shmem_bytes` | 602,0 MiB | Pagine residenti mappate come memoria condivisa. Non sono automaticamente liberabili o identiche alla RAM pinned, ma spiegano la parte prima non classificata dell'RSS. |
| `selected_stage_pinned_bytes` | 256,02 MiB | Capacita richiesta dagli attuali quattro buffer CUDA pinned per lo staging SSD -> VRAM. E la prova diretta che DS4 usa gia un pool host-pinned di questa dimensione. |

La scomposizione ora e completa:

~~~text
process_rss_bytes ~= process_rss_anon_bytes
                   + process_rss_file_bytes
                   + process_rss_shmem_bytes

1.046 MiB ~= 352 MiB + 92 MiB + 602 MiB
~~~

Il residuo e zero in tutte le righe di `output_5`: il monitoraggio non lascia
piu una grande parte della RAM residente senza categoria. `RssShmem` non deve
pero essere trattata come memoria libera: e una classificazione della RAM
residente, non un invito a riutilizzarla.

### Conclusione: passaggio al vero sotto-test SSD -> RAM pinned

La configurazione ha zero VRAM nuova allocabile, ma circa 9,7 GiB di
`ram_available_bytes` stabile e un pool pinned esistente da 256 MiB. Il
prossimo sotto-test puo quindi usare l'oracolo per creare un **banco pinned
separato da 256 MiB**, senza occupare VRAM aggiuntiva mentre la lettura SSD
avviene in background. Per non sovraccaricare questo README di misure e codice
successivi, il sotto-test e stato promosso a
[TEST_4 — Oracle SSD -> RAM pinned](../test_4/README.md).

Un gruppo di sei expert pesa circa 40,5 MiB. Per lasciare margine a
allineamenti, metadata e gestione del worker, il primo esperimento conservera
una finestra prudente di **cinque gruppi futuri** (circa 202,5 MiB), non tutti
i 43 layer. Il flusso sara:

~~~text
token t, layer l
    -> oracle legge E_l(t+1)
    -> worker: SSD -> banco RAM pinned separato
    || GPU esegue E_l(t)
token t+1, layer l
    -> hit: banco pinned -> VRAM
    -> miss/non pronto: demand load SSD -> RAM pinned -> VRAM
~~~

Il router reale restera autorevole. L'oracolo servira solo a creare un upper
bound controllato: il sotto-test misurera quanta della lettura SSD puo essere
nascosta prima di introdurre un predictor reale.

## File prodotti da un'esecuzione

| File | Contenuto |
|---|---|
| `routing_oracle.bin` | File binario compatto prodotto da Run A. Associa `(token_index, layer)` agli ID e ai pesi degli expert di riferimento; Run B lo legge per il replay. |
| `run_A.stdout.txt`, `run_A.stderr.txt` | Output normale e diagnostico della prima inferenza. |
| `run_B.stdout.txt`, `run_B.stderr.txt` | Output normale e diagnostico dell'inferenza con replay e sonde VRAM. |
| `vram_ram_trace.jsonl` | Una riga JSON per ogni sonda VRAM/RAM; è il dataset principale del sotto-test. I vecchi output mantengono il nome storico `vram_trace.jsonl`. |
| `summary.json` | Riepilogo con timestamp, exit code e conteggio degli stage del trace. |

## Eventi intercettati in `vram_ram_trace.jsonl`

Ogni riga rappresenta un evento `vram_probe`: una fotografia della memoria
CUDA in un punto preciso del percorso di un token in un layer. Non è un nuovo
caricamento né una modifica del routing; è solo osservazione.

Per ogni `(token_index, layer)` il trace contiene tre eventi, nell'ordine:

| `stage` | Punto del flusso | Cosa indica |
|---|---|---|
| `before_demand_load` | Il router/replay ha già reso noti i sei expert, ma il loader non ha ancora richiesto i loro pesi. | VRAM disponibile immediatamente prima del caricamento necessario a `E_l(t)`. `demand_bytes` è `0` perché il loader non ha ancora prodotto la statistica del trasferimento. |
| `after_demand_load` | Il loader DS4 ha completato il percorso SSD → RAM pinned → VRAM del gruppo corrente. | VRAM dopo che `gate`, `up` e `down` dei sei expert sono nel banco domanda. `demand_bytes` riporta i byte effettivamente trasferiti. |
| `after_moe_queued` | DS4 ha chiamato `ds4_gpu_routed_moe_one_tensor(...)`: i kernel CUDA che applicano gli expert `gate`, `up` e `down` al hidden state del token sono stati aggiunti allo stream CUDA. | Significa **lavoro consegnato alla GPU**, non necessariamente lavoro già concluso: CPU e GPU possono procedere in parallelo e CUDA esegue le operazioni nello stream nel loro ordine. La sonda viene scritta subito dopo l'invio, senza `cudaStreamSynchronize()`, per non fermare artificialmente la CPU in attesa della GPU. Il valore serve quindi a vedere il margine VRAM nel punto in cui il MoE corrente è già programmato e il futuro prefetch potrebbe essere avviato. |

In breve, `after_moe_queued` è il punto concettualmente più vicino a:

~~~text
expert E_l(t) pronto nel banco domanda
    ↓
DS4 invia alla GPU il calcolo MoE di E_l(t)
    ↓
after_moe_queued: la CPU potrebbe ora avviare il prefetch di E_l(t+1)
    ↓
la GPU esegue il MoE corrente nel proprio stream
~~~

`before_demand_load`, `after_demand_load` e `after_moe_queued` compaiono
2.709 volte ciascuno nei due output. Il totale è perciò 8.127 righe per
esecuzione.

### Campi di ogni riga JSONL

Esempio abbreviato:

~~~json
{"phase":"decode","event":"vram_probe","stage":"after_demand_load",
 "token_index":20,"token_id":35,"layer":0,"is_hash":true,
 "experts":[219,165,218,5,20,136],"free_bytes":0,
 "total_bytes":12878086144,"demand_bytes":42467328}
~~~

| Campo | Descrizione |
|---|---|
| `phase` | Vale `decode`: il trace riguarda solo la generazione autoregressiva, non il prefill del prompt. |
| `event` | Vale `vram_probe`: identifica una fotografia diagnostica della VRAM. |
| `stage` | Uno dei tre punti descritti nella tabella precedente. |
| `token_index` | Posizione assoluta del token nella sequenza elaborata da DS4. Include il prompt: per questo il decode qui inizia dal token 20. |
| `token_id` | ID numerico del token elaborato in quella posizione. È utile per collegare il trace al tokenizer, ma non è una misura della VRAM. |
| `layer` | Indice del layer Transformer, da 0 a 42. |
| `is_hash` | `true` per un layer con routing hash deterministico; `false` per un normale layer MoE con router a score. Entrambi usano il normale demand load degli expert per questo sotto-test. |
| `experts` | I sei ID expert effettivamente caricati nel banco domanda. In Run B provengono dall'oracolo di Run A, così il percorso è riproducibile. |
| `free_bytes` | VRAM che CUDA dichiara libera e ulteriormente allocabile in quel preciso istante. Nei primi due esperimenti vale sempre zero. |
| `total_bytes` | VRAM totale visibile a CUDA. Non è la VRAM libera e non varia durante l'esecuzione. |
| `used_bytes` | `total_bytes - free_bytes`: VRAM che CUDA considera complessivamente già usata o riservata. |
| `model_arena_reserved_bytes` | Spazio totale dei chunk CUDA riservati da DS4 per pesi del modello residenti. Può essere maggiore dei pesi effettivamente inseriti perché le arena crescono a chunk. |
| `model_arena_used_bytes` | Parte delle model arena che DS4 ha effettivamente assegnato a pesi residenti. La differenza fra `reserved` e `used` è spazio interno alle arena, non automaticamente memoria sicura da liberare. |
| `device_cache_bytes` | Slab CUDA usati dalla cache diretta di intervalli del modello. |
| `tensor_bytes` | Allocazioni possedute dai normali oggetti `ds4_gpu_tensor`: attivazioni, KV cache, tensori del grafo e scratch creati con la relativa API DS4. |
| `q8_f16_cache_bytes` | Pesi Q8 convertiti e conservati in FP16 sulla GPU per kernel più veloci. |
| `q8_f32_cache_bytes` | Pesi Q8 convertiti e conservati in FP32, se presenti. |
| `derived_weights_bytes` | Pesi derivati/preparati da DS4 per formati o kernel specializzati. |
| `demand_cache_bytes` | Capacità allocata dal banco domanda temporaneo: pesi `gate`, `up`, `down` e ID slot degli expert correnti. |
| `scratch_bytes` | Scratch CUDA persistente noto a DS4, inclusi buffer temporanei del backend. Non include i buffer RAM pinned, perché quelli risiedono nella RAM host. |
| `known_allocated_bytes` | Somma contabile delle categorie DS4 precedenti. Alcune esprimono capacità richiesta/logica, mentre CUDA riporta l'uso del proprio allocatore: il valore può quindi differire leggermente da `used_bytes`. Non è una seconda misura della memoria totale. |
| `unattributed_bytes` | Parte di `used_bytes` che non rientra ancora nelle categorie DS4 tracciate. Può includere workspace cuBLAS/CUDA, overhead/allineamento dell'allocatore, grafi CUDA e altre allocazioni backend non ancora classificate. Non va interpretata automaticamente come memoria libera. |
| `demand_bytes` | Byte del gruppo di expert appena caricato. Vale zero prima del load e 42.467.328 byte dopo il load e dopo l'accodamento del MoE. |

### Nuovi campi RAM e RAM pinned

Le esecuzioni precedenti hanno il formato storico `vram_trace.jsonl`. Dalla
prossima esecuzione il runner scrive `vram_ram_trace.jsonl`, che conserva tutti
i campi VRAM precedenti e aggiunge i seguenti campi RAM.

| Campo | Descrizione |
|---|---|
| `selected_stage_pinned_bytes` | Dimensione richiesta dai quattro buffer RAM pinned gia usati dal loader SSD corrente per lo staging degli expert selezionati. E memoria host, non VRAM. Mostra il punto di partenza reale per un futuro banco SSD -> RAM pinned. |
| `ram_total_bytes` | RAM totale visibile al Linux/WSL che ospita DS4, letta da `/proc/meminfo: MemTotal`. |
| `ram_available_bytes` | Stima Linux della RAM utilizzabile senza iniziare swap, letta da `MemAvailable`. E il campo principale per scegliere un budget prudente di RAM pinned; non garantisce da solo che `cudaMallocHost` riesca. |
| `ram_free_bytes` | Pagine RAM completamente non usate (`MemFree`). E normalmente minore di `ram_available_bytes`, perche Linux usa la RAM libera come cache. |
| `ram_page_cache_bytes` | Stima della page cache Linux (`Cached + SReclaimable - Shmem`). E utile per capire quanta RAM e gia occupata da cache file, ma non va sommata automaticamente al budget pinned. |
| `swap_total_bytes`, `swap_free_bytes` | Dimensione totale e ancora libera dello swap Linux/WSL. Un prefetch pinned non deve dipendere dallo swap, perche le sue pagine non possono essere spostate li. |
| `process_rss_bytes` | RAM fisica residente nel processo DS4 (`VmRSS`). |
| `process_vmsize_bytes` | Spazio di indirizzamento virtuale di DS4 (`VmSize`), inclusi mmap e riserve; non equivale alla RAM fisica consumata. |
| `process_locked_bytes` | Pagine bloccate che Linux espone per il processo (`VmLck`). E complementare a `selected_stage_pinned_bytes`: CUDA/WSL puo non rendere visibile ogni pagina host-pinned tramite `VmLck`. |
| `process_rss_anon_bytes` | Parte anonima della RAM residente (`RssAnon`): heap, stack e allocazioni senza file di backing. |
| `process_rss_file_bytes` | Parte file-backed della RAM residente (`RssFile`), ad esempio porzioni del GGUF mappato e librerie. |
| `process_rss_shmem_bytes` | Parte della RAM residente mappata come memoria condivisa (`RssShmem`). Insieme a `process_rss_anon_bytes` e `process_rss_file_bytes` consente di spiegare la quasi totalita di `process_rss_bytes`. |

## Campi di `summary.json`

| Campo | Descrizione |
|---|---|
| `test` | Nome del sotto-test eseguito. |
| `timestamp_utc` | Istante UTC in cui il runner ha scritto il riepilogo. |
| `run_a_exit_code` | Codice di uscita di Run A; `0` indica successo. |
| `run_b_exit_code` | Codice di uscita di Run B; `0` indica successo. |
| `before_demand_load_events` | Numero di sonde prima del demand load. |
| `after_demand_load_events` | Numero di sonde dopo il demand load. |
| `after_moe_queued_events` | Numero di sonde dopo l'accodamento del MoE. |
| `trace` | Nome del file JSONL a cui si riferiscono i conteggi. |

## Prossimi sotto-test, in ordine

Il valore `free_bytes = 0` è un vincolo importante, ma non è ancora
sufficiente per decidere come dividere la memoria. `cudaMemGetInfo()` mostra
la memoria che CUDA considera ulteriormente allocabile; non distingue da solo
tra memoria che contiene pesi/dati attivi e memoria che DS4 o CUDA hanno già
riservato per riutilizzarla più avanti. I sotto-test seguenti rispondono a
questa domanda prima di costruire il prefetch completo.

### 1. Mappa dettagliata della VRAM per stage — completato in `output_3`

**Obiettivo:** misurare per ogni stage del decode quanta VRAM è attribuibile
alle principali categorie DS4 e quanta risulta riservata ma non attivamente
occupata da dati utili in quel momento.

~~~text
inizializzazione CUDA / modello
    ↓
prefill
    ↓
decode: before_demand_load
    ↓
decode: after_demand_load
    ↓
decode: after_moe_queued
    ↓
report per categoria + VRAM totale / libera / riservata
~~~

Le categorie da rendere visibili saranno almeno: modello residente, KV cache,
buffer del grafo CUDA, banco domanda degli expert, cache di pesi convertiti,
arena CUDA e memoria riservata dal runtime ma non attribuita a una categoria
DS4. Lo scopo non è presumere che la memoria riservata sia liberabile: è
capire se esiste una parte che possa essere ripartita in sicurezza per un
banco prefetch, senza aumentare il consumo totale.

Il risultato è documentato nella sezione **Mappa dettagliata — `output_3`**:
esistono riserve già allocate che meritano studio, ma nessuna VRAM nuova
allocabile. Prima di usare una riserva per il prefetch occorre dimostrare che
può essere ripartita senza danneggiare il percorso domanda.

### 2. Prefetch SSD → RAM pinned

**Obiettivo:** nascondere prima la lettura SSD, senza richiedere un secondo
banco grande in VRAM. Un worker separato usa l'oracolo nel test; in futuro
riceverà candidati dal predictor.

> Stato: la misura preliminare di RAM e completata in TEST_3; l'implementazione
> del worker, del banco pinned separato e del dizionario di lookup prosegue in
> [TEST_4](../test_4/README.md). Questa sezione conserva la motivazione e il
> flusso che hanno portato a quel test.

~~~text
token t, layer l
    ↓
oracle/predictor produce E_l(t+1)
    ↓
worker legge SSD → RAM pinned in background
    ||
GPU esegue E_l(t)
    ↓
token t+1, layer l
    ↓
se pronto: RAM pinned → VRAM
altrimenti: normale SSD → RAM pinned → VRAM
~~~

Questo sotto-test conserva i pesi prefetched in RAM pinned, indicizzati da
`(target_token, layer)`. Non cambia mai il router reale: se la previsione non
è pronta o non coincide con gli expert richiesti, DS4 ricade nel demand load
normale. Il possibile vantaggio è nascondere la parte SSD del caricamento;
rimane comunque il trasferimento RAM → VRAM.

#### Monitoraggio RAM implementato prima del prefetch

Prima di creare il worker o un nuovo banco pinned, TEST_3 ora raccoglie una
sonda RAM negli stessi tre stage del trace VRAM. Il nuovo file e
`vram_ram_trace.jsonl`: il nome rende esplicito che contiene entrambe le
memorie.

Questo sotto-test di monitoraggio e **gia implementato ed eseguito**. I
risultati di `output_4/` e, in forma completa, di `output_5/` sono riportati
nelle sezioni **Risultati osservati — sotto-test RAM** e **Mappa dettagliata —
`output_5`** piu sopra. Essi mostrano RAM disponibile stabile intorno a
9,7 GiB e un pool pinned DS4 esistente da circa 256 MiB.

~~~text
Run B con replay oracle
    |
    +--> before_demand_load
    |      legge VRAM CUDA + /proc/meminfo + /proc/self/status
    |
    +--> SSD -> pool RAM pinned esistente -> VRAM
    |
    +--> after_demand_load
    |      misura RAM disponibile, RSS DS4 e pool pinned corrente
    |
    +--> after_moe_queued
           misura lo stesso insieme mentre il MoE e stato inviato alla GPU
    v
vram_ram_trace.jsonl
    |
    +--> confronto tra MemAvailable, page cache, RSS, VmLck e
         selected_stage_pinned_bytes
    v
decisione: scegliere un budget pinned prudente oppure non avviare il prefetch
~~~

Questo monitoring non alloca RAM pinned aggiuntiva e non modifica SSD
streaming, routing, cache expert o scheduling CUDA. Serve a verificare se
esiste abbastanza `ram_available_bytes` stabile per tenere uno o piu gruppi
futuri di expert in RAM pinned senza mettere sotto pressione Linux/WSL.

### 3. Ridurre/ripartire il budget VRAM DS4 e creare un banco CUDA

**Obiettivo:** riservare volutamente una quantità fissa di VRAM a un banco
prefetch, confrontando il costo sulla cache corrente con il beneficio degli
expert già pronti sulla GPU.

~~~text
riduciamo esplicitamente il budget destinato alla cache expert corrente
    ↓
otteniamo, per esempio, 256 MiB / 512 MiB / 1 GiB liberi
    ↓
allochiamo un banco prefetch CUDA di dimensione fissa
    ↓
prefetchiamo tutti oppure solo una finestra di layer futuri
~~~

Con gruppi da circa 40,5 MiB, i budget iniziali corrispondono
approssimativamente a 6, 12 e 25 gruppi expert/layer. Il banco dovrà essere
separato dal banco domanda attuale: il demand load continua a servire
`E_l(t)`, mentre il banco prefetch conserva `E_l(t+1)` fino al suo consumo.
Il test confronterà sempre Run B e Run C con lo **stesso budget totale**,
altrimenti un eventuale speedup non sarebbe confrontabile.

## Flusso 2 — TEST_3 completo: prefetch oracle controllato

**Obiettivo:** confrontare il demand load normale con un prefetch perfetto,
ma senza attribuire al predictor un vantaggio di correttezza sul router.

~~~text
Run A
    |
    |  ds4.c: ds4_routing_oracle_*()
    |  registra il percorso di routing di riferimento
    v
routing_oracle.bin
    |
    +---------------------------------------------------+
    |                                                   |
    v                                                   v
Run B                                               Run C
replay oracle, nessun prefetch                      replay oracle + prefetch oracle
    |                                                   |
    v                                                   v
ds4.c: metal_graph_decode_cuda_selected_load(...)   router/replay ottiene E_l(t)
    |                                                   |
    v                                                   +--> banco domanda: carica/usa E_l(t)
banco domanda: carica E_l(t)                         |       tramite il loader DS4 esistente
    |                                                   |
    v                                                   +--> oracolo legge E_l(t+1)
ds4_gpu_routed_moe_one_tensor(...)                   |       per lo stesso layer l
    |                                                   |
    v                                                   +--> NUOVA prefetch cache separata       [da aggiungere]
misura baseline                                         |       conserva (layer l, token t+1, expert IDs)
                                                        |       e carica SSD -> RAM pinned -> VRAM
                                                        |       su stream/worker separato
                                                        v
                                                    esecuzione normale di E_l(t)
                                                        |
                                                        v
                                                    token t+1, layer l
                                                        |
                                                        +--> hit: promuove/usa pesi prefetched
                                                        `--> miss: demand load normale
~~~

Il banco prefetch dovrà essere indicizzato almeno da `(target_token, layer)`.
Un solo buffer globale non è corretto: sarebbe sovrascritto dal prefetch del
layer seguente prima di raggiungere il token successivo.

Le estensioni previste, ma non ancora presenti, sono:

| File previsto | Estensione | Scopo |
|---|---|---|
| `ds4_cuda.cu` | Seconda struttura cache, separata da `g_stream_selected_cache` | Conservare gli expert prefetched senza invalidare quelli necessari ora. |
| `ds4_cuda.cu` | API di start/query/promozione del prefetch | Caricare, verificare e trasformare un hit prefetched in banco utilizzabile dal MoE. |
| `ds4.c` | Scheduling vicino al punto che chiama `ds4_gpu_routed_moe_one_tensor(...)` | Avviare il prefetch mentre il MoE corrente è in esecuzione, senza aspettarlo sul percorso critico. |
| `ds4.c` | Trace TEST_3 | Registrare `prefetch_start`, `prefetch_ready`, `hit`, `miss`, byte, attesa residua e tempi per token. |
| `test_predictor/test_3/test_3_runner.c` | Orchestrazione A/B/C e confronto finale | Avviare i tre processi, mantenere configurazione identica e produrre il report di speedup. |

Il confronto finale dovrà riportare almeno:

~~~text
Run B vs Run C
    - token sequence e routing replay equivalenti
    - tok/s e tempo totale di decode
    - prefetch hit rate e miss rate
    - byte SSD letti on-demand e in prefetch
    - tempo di attesa residua al consumo del prefetch
    - VRAM massima usata e prefetch rifiutati per budget
~~~

Il risultato di Run C sarà un **upper bound sperimentale**: usa un oracolo
perfetto e quindi misura quanto può valere l'idea se le predizioni fossero
corrette. Solo dopo introdurremo un predictor reale, che dovrà degradare in
modo sicuro a demand load quando sbaglia o quando il prefetch non è pronto.
