# TEST_2 — validazione SSD streaming prima del prefetch

TEST_2 è una fase di analisi, non ancora un benchmark né un'implementazione
del predictive prefetch. Serve a verificare con precisione come DS4 legge gli
expert dal modello su SSD, attraverso la RAM temporanea di DS4, e poi in VRAM,
così da evitare che
un esperimento futuro misuri per errore un vantaggio della cache RAM del
sistema operativo invece del prefetch reale.

## Obiettivo

Prima di costruire TEST_3, vogliamo rispondere a queste domande:

1. gli expert sotto SSD streaming vengono davvero letti tramite direct I/O;
2. dopo ogni lettura DS4 prova a rilasciare le pagine del file dalla page cache
   del sistema operativo;
3. il caricamento degli expert usa un buffer distinto da quello necessario
   agli expert del layer corrente;
4. esiste già un punto sicuro in cui si possa caricare `E_l(t+1)` senza
   sovrascrivere o rallentare `E_l(t)`;
5. quali tempi e quali hit/miss dovranno essere registrati per confrontare una
   baseline con una simulazione di prefetch.

## Glossario

| Termine | Significato nel contesto di TEST_2 |
|---|---|
| **CUDA** | Piattaforma NVIDIA usata da DS4 per eseguire calcoli e gestire la memoria della GPU. |
| **GPU / VRAM** | La GPU esegue i kernel del modello. La VRAM è la memoria veloce fisicamente presente sulla GPU, dove i pesi devono essere disponibili prima di eseguire un expert. |
| **RAM host** | La normale memoria principale del computer, usata dal processo DS4 sul lato CPU. |
| **Buffer** | Area di memoria temporanea che contiene dati durante un passaggio del flusso, per esempio pesi letti dal disco prima dell'upload alla GPU. |
| **RAM pinned** | Buffer le cui pagine **fisiche** di RAM restano in una posizione stabile finché il buffer esiste: Linux non può spostarle, rimapparle o inviarle in swap durante un trasferimento. Questo è necessario perché la GPU usa DMA e deve leggere indirizzi fisici stabili; CUDA può così trasferire i dati host→GPU in modo efficiente e prevedibile. |
| **Staging** | Fase e buffer intermedi: in DS4 i pesi passano dall'SSD ai buffer RAM pinned e solo dopo alla VRAM. |
| **SSD streaming** | Modalità in cui DS4 non mantiene tutti i pesi MoE in VRAM: legge da SSD gli expert necessari quando servono. |
| **Direct I/O (`O_DIRECT`)** | Modalità di lettura file che cerca di evitare la normale page cache Linux. Non elimina la RAM: DS4 usa comunque buffer pinned controllati per il passaggio SSD→RAM→VRAM. |
| **Page cache Linux** | Cache automatica in RAM del sistema operativo per i file letti. Può rendere letture successive artificialmente veloci; TEST_2 cerca di ridurne l'influenza. |
| **Expert MoE** | Una FFN indipendente all'interno di un layer Mixture-of-Experts. Ogni expert ha i propri pesi `gate`, `up` e `down`. |
| **Router MoE** | Parte del layer che calcola quali expert usare per il token corrente e i relativi pesi di combinazione. Rimane sempre autorevole. |
| **`E_l(t)`** | Insieme ordinato degli expert selezionati dal router nel layer `l` durante l'elaborazione del token al tempo `t`. Nel modello osservato contiene normalmente sei expert. |
| **`E_l(t+1)`** | Insieme degli expert che il router selezionerà nello stesso layer `l` al token successivo. Il futuro predictor proverà solo a prepararlo in anticipo, senza sostituire il router reale. |
| **Decode** | Fase autoregressiva in cui il modello genera un token alla volta dopo il prompt/prefill. È la fase su cui vogliamo ridurre la latenza. |
| **Prefill** | Elaborazione iniziale dell'intero prompt prima della generazione autoregressiva. Nel parallelismo con la tesi, è il forward pass del Transformer applicato alla sequenza di token del prompt: embedding → layer Transformer ripetuti (attenzione causale + FFN/MoE) → logits dell'ultimo token. L'attenzione resta causale, quindi ogni posizione vede solo i token precedenti, ma molte posizioni del prompt possono essere calcolate insieme; nel MoE il router seleziona gli expert per le posizioni elaborate. Durante il prefill DS4 costruisce anche la **KV cache**, cioè chiavi e valori dell'attenzione dei token del prompt. Nel decode successivo, il nuovo token consulta questa cache invece di ricalcolare tutto il prompt. Il predictive prefetch studiato qui mira soprattutto al decode, dove si genera un token alla volta e la latenza SSD degli expert pesa maggiormente. |
| **Scratch cache CUDA** | Buffer GPU temporaneo `g_stream_selected_cache` che contiene i pesi degli expert del layer corrente. Non è una cache persistente: il caricamento successivo la invalida. |
| **Cache hit / miss** | Un hit indica che pesi già preparati possono essere riusati; un miss richiede una nuova lettura e upload. Nel backend CUDA attuale di TEST_2 gli expert selezionati hanno sempre miss fra layer/token, perché la scratch cache non persiste. |
| **Upload H2D** | Trasferimento Host-to-Device: dalla RAM host alla VRAM della GPU. |
| **Stream CUDA** | Coda di operazioni CUDA che permette di inviare copie o kernel in modo asincrono rispetto alla CPU. Un successivo `cudaStreamSynchronize()` attende però il completamento delle operazioni accodate. |
| **Oracolo di routing** | File prodotto dai test precedenti con gli expert realmente scelti in Run A. In TEST_3 servirà per misurare il massimo beneficio del prefetch prima di usare un predictor ML reale. |
| **Predictive prefetch** | Caricamento anticipato in un banco VRAM separato di expert previsti per il futuro. Può influenzare solo caricamento, caching e scheduling: non modifica router, logits o token generati. |

## Mappa dei flussi importanti

### Flusso 1 — avvio diagnostico di TEST_2

**Obiettivo:** eseguire una singola inferenza CUDA + SSD streaming e salvare
una misura per ogni caricamento di expert del decode.

~~~text
Configurazione IDE: TEST_2 SSD Streaming Trace
    |
    v
test_predictor/test_2/test_2_runner.c
    |  controlla DS4_MODEL, crea output incrementale e avvia un processo ./ds4
    |  imposta DS4_SSD_TRACE_PATH=<output>/ssd_trace.jsonl
    v
ds4 --cuda --ssd-streaming ...
    |
    v
ds4.c: ds4_ssd_trace_*()
    |  apre il JSONL solo quando la variabile diagnostica è presente
    v
output_N/ssd_trace.jsonl + summary.json + run.stdout.txt + run.stderr.txt
~~~

Il runner non confronta due processi e non forza il routing: TEST_2 osserva
un solo percorso di decode reale. Con `DS4_SSD_TRACE_PATH` DS4 disabilita
soltanto l'overlap interno del layer corrente, per attribuire in modo limpido
il tempo al loader esplicito. Per questo TEST_2 non è un benchmark di speedup.

### Flusso 2 — dal router agli expert pronti in VRAM

**Obiettivo:** identificare il percorso che TEST_3 dovrà accelerare senza
alterare il router autorevole.

~~~text
router CUDA del layer l
    |
    |  produce ID e pesi degli expert E_l(t)
    v
ds4.c: metal_graph_decode_cuda_selected_load(...)
    |  sincronizza quanto basta e legge ID/pesi dalla GPU
    |  in TEST_2 registra router_sync_ms e router_read_ms
    v
ds4_cuda.cu: cuda_stream_selected_cache_begin_load(...)
    |  invalida il buffer compatto precedente e deduplica i sei ID
    v
cuda_model_copy_to_device_streamed(...) x gate, up, down
    |  per ogni tensore: SSD -> RAM pinned -> VRAM
    |  legge i pesi selezionati e riempie la cache scratch CUDA
    v
ds4_gpu_routed_moe_one_tensor(...)
    |  esegue il MoE usando i pesi ora presenti in VRAM
    v
uscita del layer l -> layer l+1
~~~

Gli ID sono quindi noti dopo il router del layer corrente. Il router resta
l'unica sorgente che decide gli expert effettivamente eseguiti; la futura
predizione potrà solo preparare pesi per un possibile uso successivo.

### Flusso 3 — lettura SSD e trasferimento GPU

**Obiettivo:** distinguere il costo della lettura dal costo del trasferimento,
evitando di confondere una cache calda del sistema con lo streaming reale.

~~~text
ds4_cuda.cu: ds4_gpu_set_model_fd(...)
    |  apre /proc/self/fd/<fd> con O_DIRECT, se supportato
    v
g_model_direct_fd
    |
    v
cuda_model_stage_read(...)
    |  pread() su O_DIRECT: SSD -> buffer RAM pinned di DS4
    |  -> TEST_2 misura ssd_read_ms
    v
g_stream_selected_stage[4]
    |  quattro buffer nella RAM host, pinned, usati come staging
    |  non sono la page cache Linux
    v
cudaMemcpyAsync(..., g_stream_selected_upload_stream)
    |  trasferimento host -> VRAM
    v
cudaStreamSynchronize(g_stream_selected_upload_stream)
    |  il chiamante aspetta: il layer non può usare pesi incompleti
    v
expert pronti nel buffer CUDA compatto
~~~

Dopo ciascun chunk, `cuda_model_drop_file_pages()` e
`cuda_model_discard_source_pages()` chiedono rispettivamente
`POSIX_FADV_DONTNEED` e `POSIX_MADV_DONTNEED`. Sono richieste advisory:
riducono l'influenza della page cache Linux, ma non possono svuotare le cache
interne dell'SSD o del controller.

In sintesi, il Direct I/O **non** crea un percorso fisico `SSD -> VRAM` senza
RAM. Evita principalmente la normale page cache Linux; DS4 usa comunque i
propri buffer temporanei in RAM pinned, necessari per trasferire i dati alla
GPU in modo efficiente e controllato:

~~~text
lettura normale: SSD -> page cache Linux -> RAM applicazione -> VRAM
DS4 Direct I/O:  SSD -> RAM pinned gestita da DS4 -> VRAM
~~~

### Flusso 4 — limite della cache attuale e direzione di TEST_3

~~~text
g_stream_selected_cache  (un solo banco scratch CUDA)
    |
    |  contiene gate/up/down + remapping per E_l(t)
    v
nuovo cuda_stream_selected_cache_begin_load(...)
    |
    |  invalida immediatamente il contenuto precedente
    v
non può contenere contemporaneamente E_l(t) e E_l(t+1)

TEST_3 necessario:
    banco domanda     = E_l(t), usato dal MoE corrente
    banco prefetch    = previsione/oracolo E_l(t+1), caricato in anticipo
    al passo seguente  = hit nel banco prefetch oppure miss normale
~~~

Questo è il motivo per cui il futuro prefetch richiederà un secondo banco,
oppure una struttura equivalente, e non il semplice riuso della scratch cache
esistente.

## Osservazioni già verificate

Con la configurazione CUDA + SSD streaming di riferimento è stato osservato
nel log:

~~~text
ds4: CUDA model direct I/O enabled (align=4096)
~~~

Questo conferma che DS4 ha aperto il file del modello con direct I/O per il
percorso CUDA di streaming. La lettura degli expert selezionati evita quindi,
per quanto possibile, la normale page cache Linux, ma passa comunque nei
buffer RAM pinned gestiti esplicitamente da DS4 prima dell'upload in VRAM.
DS4 inoltre richiede il rilascio delle pagine del modello dopo le letture; tale
richiesta è però advisory, cioè dipende dal sistema operativo.

I messaggi verbosi `CUDA fd-cached ...` descrivono soprattutto tensori comuni
del modello caricati durante l'avvio. Non dimostrano che gli expert MoE di un
processo precedente restino disponibili per un processo successivo.

## Flusso CUDA + SSD trovato nel codice

Il percorso di decode rilevante passa principalmente attraverso `ds4.c` e
`ds4_cuda.cu`.

~~~text
router CUDA
  ↓
selected expert IDs e pesi
  ↓
ds4.c: metal_graph_decode_cuda_selected_load(...)
  ↓
ds4_cuda.cu: cuda_stream_selected_cache_begin_load(...)
  ↓
lettura SSD → staging nella RAM host pinned → copia in VRAM
  ↓
ds4_cuda.cu: ds4_gpu_routed_moe_one_tensor(...)
  ↓
kernel MoE che usa gli expert presenti nel buffer compatto
~~~

### 1. Apertura del modello e direct I/O

`ds4.c` passa il file del modello al backend CUDA. In
`ds4_cuda.cu`, la funzione `ds4_gpu_set_model_fd()` prova ad aprire una seconda
handle del medesimo file tramite `/proc/self/fd/<fd>` con il flag Linux
`O_DIRECT`.

Se l'operazione riesce, salva la handle in `g_model_direct_fd` e stampa:

~~~text
CUDA model direct I/O enabled (align=4096)
~~~

La variabile `DS4_CUDA_NO_DIRECT_IO` disabilita esplicitamente questo tentativo.
Nella configurazione di riferimento non è impostata, quindi gli expert
selezionati usano direct I/O quando il file system e il driver lo permettono.

### 2. Il router rende noti gli expert del layer corrente

Nel decode, `ds4.c: metal_graph_decode_cuda_selected_load(...)` prima chiude
la sequenza CUDA in corso con `ds4_gpu_end_commands()`. Poi legge dalla GPU:

- gli ID in `metal_graph_router_selected(g)`;
- i pesi in `metal_graph_router_weights(g)`;
- opzionalmente le probabilità del router, se è attivo il trace.

In TEST_1, nello stesso punto avviene l'eventuale sostituzione diagnostica con
l'oracolo. Fuori da TEST_1, gli ID letti qui sono precisamente quelli scelti
dal router reale per il token e layer correnti.

### 3. Buffer temporaneo degli expert selezionati

`ds4_cuda.cu` mantiene una sola struttura globale:

~~~c
static cuda_stream_selected_cache g_stream_selected_cache;
~~~

Contiene tre buffer GPU (`gate_ptr`, `up_ptr`, `down_ptr`) e una tabella GPU di
remapping (`slot_selected_ptr`). La funzione
`cuda_stream_selected_cache_begin_load(...)`:

1. chiama subito `cuda_stream_selected_cache_invalidate()`;
2. elimina i duplicati dai sei expert selezionati;
3. alloca o riusa spazio GPU sufficiente per gli expert unici;
4. carica gate, up e down di ogni expert nel buffer compatto;
5. carica in GPU la tabella che traduce ogni slot del router nell'indice del
   buffer compatto;
6. marca il buffer `valid=1` per quello specifico layer.

Questo è il punto chiave: una nuova chiamata di caricamento invalida subito la
struttura precedente. Non esistono, nell'implementazione CUDA attuale, due
banche indipendenti — una per la domanda corrente e una per il prefetch.

Inoltre, le API CUDA `ds4_gpu_set_streaming_expert_cache_budget()` e
`ds4_gpu_stream_expert_cache_configured_count()` sono attualmente no-op /
ritornano zero. Il budget `--ssd-streaming-cache-experts 8GB` viene usato per
la pianificazione e la riserva di memoria riportata nei log, ma non equivale a
una cache CUDA persistente con hit/miss, LRU o eviction degli expert tra token.
Il percorso effettivo di decode carica la tabella compatta degli expert richiesti
dal layer corrente.

### 3. Pesi di un expert: `gate`, `up` e `down`

Ogni expert MoE è una FFN indipendente e completa. Non è diviso in tre expert:
`gate`, `up` e `down` sono i tre gruppi di pesi — in pratica tre matrici o
tensori di pesi — che formano la stessa FFN dell'expert.

~~~text
expert e
    |- W_gate,e
    |- W_up,e
    `- W_down,e
~~~

Rispetto alla FFN classica descritta nella tesi, DeepSeek usa la variante
**SwiGLU**, che per un hidden state `x` e per l'expert `e` può essere
rappresentata così:

`x` è il **hidden state** in ingresso all'expert: il vettore di attivazioni
prodotto dalle sottostrutture precedenti del layer Transformer per il token
corrente. Contiene una rappresentazione numerica contestuale del token, non
l'ID del token stesso. L'expert lo trasforma e restituisce un nuovo vettore
con la stessa dimensione, che prosegue nel resto del layer.

~~~text
output_e = ( SiLU(x * W_gate,e) * (x * W_up,e) ) * W_down,e
~~~

~~~text
hidden state x
    |
    +--> gate(x) --> SiLU --> filtro/porta delle componenti interne
    |
    +--> up(x) ----------------> caratteristiche interne candidate
                                      |
                                      v
                              prodotto elemento per elemento
                                      |
                                      v
                                  down(...)
                                      |
                                      v
                              output dell'expert
~~~

| Gruppo di pesi | Ruolo nella FFN SwiGLU |
|---|---|
| `gate` | Proiezione che, dopo `SiLU`, controlla quali componenti interne risultano attive. |
| `up` | Proiezione dell'hidden state verso la dimensione interna più ampia della FFN. |
| `down` | Proiezione finale che riporta il risultato alla dimensione dell'hidden state del Transformer. |

Sono quindi necessari proprio tre gruppi di pesi: due proiezioni parallele in
ingresso (`gate` e `up`) e una proiezione di ritorno (`down`). Quando il router
seleziona l'expert `e`, DS4 deve caricare `W_gate,e`, `W_up,e` e `W_down,e`.
Se mancasse anche uno solo dei tre, l'intera FFN dell'expert non potrebbe essere
eseguita correttamente.

Il parallelismo con l'architettura della tesi è:

~~~text
FFN classica del layer l
    -> un singolo insieme di pesi condiviso dal layer

FFN MoE di DeepSeek del layer l
    -> il router sceglie sei expert
    -> ogni expert e è una FFN completa con W_gate,e, W_up,e e W_down,e
~~~

### 4. Lettura SSD, staging e trasferimento verso GPU

Per ogni expert unico, `cuda_stream_selected_cache_begin_load(...)` invoca tre
volte `cuda_model_copy_to_device_streamed(...)`: una per gate, una per up e
una per down.

Questa funzione usa:

- `cuda_model_stage_read(...)` per leggere un chunk dal file;
- `cuda_pread_full(...)` come lettura completa con `pread`;
- quattro buffer host pinned `g_stream_selected_stage[...]`;
- uno stream CUDA non bloccante `g_stream_selected_upload_stream`;
- `cudaMemcpyAsync(...)` per trasferire ogni chunk nella destinazione VRAM.

Il passaggio nella RAM host è intenzionale e necessario: `pread` deposita i
dati nei buffer pinned `g_stream_selected_stage[...]`, poi CUDA li copia in
VRAM. Il Direct I/O evita la page cache Linux, non la RAM usata come staging.

Quando `g_model_direct_fd` è disponibile, `cuda_model_stage_read(...)` usa
`pread` su quella handle `O_DIRECT`; altrimenti ripiega sulla normale handle
del file. Il fallback è esplicito e, con log verbosi, DS4 lo segnala come
`CUDA direct model read disabled`.

Al termine di ogni chunk DS4 chiama:

- `cuda_model_drop_file_pages(...)`, che usa `posix_fadvise(...,
  POSIX_FADV_DONTNEED)`;
- `cuda_model_discard_source_pages(...)`, che usa
  `posix_madvise(..., POSIX_MADV_DONTNEED)` sulla mappatura del modello.

Entrambe le richieste vengono saltate solo se è impostata
`DS4_CUDA_KEEP_MODEL_PAGES`. Sono richieste advisory: riducono il rischio di
misurare la page cache Linux, ma non possono controllare completamente cache
interne di SSD, controller o sistema operativo.

### 5. Quando il trasferimento è davvero pronto

Il singolo `cudaMemcpyAsync` è asincrono rispetto al thread host, ma
`cuda_model_copy_to_device_streamed(...)` termina con:

~~~c
cudaStreamSynchronize(g_stream_selected_upload_stream)
~~~

Quindi il caricamento completo degli expert selezionati è sincrono per il
chiamante. Dopo il caricamento, `metal_graph_decode_cuda_selected_load(...)`
richiama `ds4_gpu_begin_commands()` e il decode prosegue con il kernel MoE.

`ds4_gpu_routed_moe_one_tensor(...)` verifica che la cache sia valida per quel
layer e usa `gate_ptr`, `up_ptr`, `down_ptr` e la tabella di remapping al posto
dei pesi completi del modello. Se la cache non corrisponde al layer corrente,
lo streaming CUDA fallisce esplicitamente con un messaggio del tipo:

~~~text
CUDA streaming selected experts are unavailable for layer ...
~~~

### 6. Overlap già presente e suo limite

`ds4.c` possiede un worker CPU,
`metal_graph_selected_async_load_worker_main(...)`, avviato tramite
`metal_graph_selected_async_load_start(...)`. Può leggere gli ID selezionati e
caricare gli expert del *layer corrente* mentre il main thread esegue lo shared
expert dello stesso layer. Il main thread poi chiama
`metal_graph_selected_async_load_finish(...)` e aspetta il completamento prima
del routed MoE.

Nel nostro percorso CUDA questo overlap è disponibile solo in determinate
condizioni. In particolare viene disabilitato quando è attivo il replay oracle
di TEST_1 (`DS4_ROUTING_ORACLE_REPLAY_PATH` o
`DS4_ROUTING_ORACLE_WRITE_PATH`). Il controllo senza overlap resta invece
`metal_graph_decode_cuda_selected_load(...)`.

Questo meccanismo non è ancora predictive prefetch: conosce solo gli expert
di `E_l(t)` dopo il router del layer attuale e carica gli stessi expert. Non
può conservare contemporaneamente `E_l(t)` e `E_l(t+1)` nella singola cache
globale esistente.

## Configurazione manuale usata per la verifica

La configurazione IDE **DS4 Baseline CUDA** può essere usata per osservare il
comportamento del backend. I parametri sono:

| Campo | Valore |
|---|---|
| Target | `cuda-generic` |
| Executable | `/home/michele/ds4/ds4` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | `--cuda --ssd-streaming --ssd-streaming-cache-experts 8GB -c 1024 -n 64 --nothink --temp 0 -p "Explain what a binary search is in one short paragraph."` |
| Environment variables | `DS4_CUDA_MMQ=0;DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` |

`DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` va usata solo per osservare i log e può
essere rimossa dopo la verifica, perché rende l'output molto lungo. Per i test
di prefetch non imposteremo `DS4_CUDA_NO_DIRECT_IO` né
`DS4_CUDA_KEEP_MODEL_PAGES`.

## Risultato importante per TEST_3

Run A e Run B sono processi distinti: quando Run A termina, la sua VRAM e la
sua cache CUDA vengono liberate. Perciò non è possibile lasciare gli expert di
Run A in VRAM e usarli direttamente in Run B.

La simulazione corretta dovrà avvenire dentro un singolo processo di misura:

~~~text
Run C, token t, layer l
  router/replay determina E_l(t)
  oracle fornisce E_l(t+1)
  carica in anticipo E_l(t+1) in un buffer di prefetch separato
  esegue normalmente E_l(t)
  al token successivo riusa il buffer se la previsione è corretta
~~~

Il buffer di prefetch deve essere separato da quello degli expert correnti. Il
loader selezionato esistente usa una cache scratch compatta del layer corrente:
riutilizzarla direttamente per `E_l(t+1)` potrebbe invalidare gli expert ancora
necessari a `E_l(t)` e rendere il benchmark scorretto.

TEST_2 non modifica il router, il contenuto degli expert, lo SSD streaming o
i logits. Attiva esclusivamente tracing e, per chiarezza della misura,
disabilita l'overlap già esistente del layer corrente.

## Implementazione della traccia

TEST_2 ora dispone di un runner C, `test_2_runner`, e di una sola variabile
diagnostica interna:

~~~text
DS4_SSD_TRACE_PATH=<output>/ssd_trace.jsonl
~~~

Il runner avvia **un solo processo DS4** con la configurazione CUDA + SSD
streaming di riferimento. Non è un confronto fra due run: serve a misurare il
caricamento reale degli expert richiesti durante il decode.

La traccia è attivabile solo tramite questa variabile; senza di essa DS4 non
scrive JSONL e segue il normale flusso esistente.

Per ottenere tempi chiari del caricamento corrente, quando la variabile è
attiva DS4 non usa il worker di overlap già presente per gli expert del layer
corrente. TEST_2 misura quindi il loader esplicito, dal router fino agli
expert pronti in VRAM. Non va usato per dichiarare uno speedup end-to-end:
quello sarà il compito di TEST_3.

### Campi di `ssd_trace.jsonl`

Ogni riga rappresenta il caricamento degli expert di un layer MoE durante il
decode.

| Campo | Significato |
|---|---|
| `phase` / `event` | Sempre `decode` / `selected_load`. |
| `token_index`, `token_id`, `layer` | Token e layer a cui appartiene il caricamento. |
| `is_hash` | `true` per un layer con selezione hash deterministica; `false` per un layer top-k del router. |
| `experts` | I sei expert richiesti in ordine. |
| `slot_count` | Numero di slot richiesti dal router, normalmente sei. |
| `unique_experts` | Expert effettivamente letti dopo aver eliminato eventuali duplicati. |
| `bytes` | Byte totali gate + up + down trasferiti per gli expert unici. |
| `direct_io` | Se la lettura ha usato la handle `O_DIRECT`. |
| `cache_reused` | Attualmente sempre `false`: il backend CUDA usa il buffer temporaneo del layer corrente e non una cache LRU persistente. |
| `router_sync_ms` | Attesa per rendere leggibile dal CPU l'output CUDA del router. |
| `router_read_ms` | Copia CPU degli ID/pesi del router e lavoro diagnostico locale. |
| `ssd_read_ms` | Tempo host impiegato dalle letture `pread` degli expert. |
| `upload_and_sync_ms` | Staging pinned, enqueue H2D e sincronizzazione dello stream di upload. |
| `load_ms` | Tempo totale di `cuda_stream_selected_cache_begin_load(...)`. |
| `resume_ms` | Tempo necessario per riprendere il flusso CUDA dopo il loader. |

`upload_and_sync_ms` include anche una piccola parte di gestione host; è più
corretto leggerlo come costo complessivo di staging + trasferimento che come
tempo puro del bus PCIe.

## Risultati osservati

Tutte le esecuzioni hanno usato la stessa configurazione e hanno completato
correttamente l'inferenza. Ogni run ha generato 63 token di decode e quindi:

~~~text
63 token x 43 layer per token = 2709 selected-load events
                             = 189 hash + 2520 MoE
~~~

| Output | Timestamp UTC | Eventi | Direct I/O | Riusi cache | Lettura SSD media | Upload + sync medio | Load totale medio | Intervallo load |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `output/` | 2026-08-18 18:10:54 | 2709 | 2709 | 0 | 19.412 ms | 7.566 ms | 26.993 ms | 22.458-48.444 ms |
| `output_2/` | 2026-08-18 18:20:07 | 2709 | 2709 | 0 | 19.623 ms | 9.149 ms | 28.796 ms | — |
| `output_3/` | 2026-08-18 18:23:28 | 2709 | 2709 | 0 | 19.341 ms | 7.375 ms | 26.733 ms | — |
| `output_4/` | 2026-08-18 18:25:15 | 2709 | 2709 | 0 | 19.646 ms | 8.947 ms | 28.614 ms | — |

In tutti i 10836 eventi complessivi sono stati richiesti sei expert unici e
sono stati trasferiti `42,467,328` byte (circa 40.5 MiB) per layer. Non vi è
alcun `cache_reused=true`: è il comportamento atteso dalla singola scratch
cache CUDA corrente.

I risultati sono coerenti fra loro. La lettura SSD rimane stabile, circa
19.3-19.6 ms per layer; il totale oscilla fra 26.7 e 28.8 ms soprattutto per
la parte di staging/upload e per la normale variabilità del sistema GPU/SSD.
La lettura SSD rappresenta approssimativamente il 68-72% del loader totale:
è quindi il principale candidato da nascondere tramite prefetch.

`output/` è stato creato prima della correzione del solo campo diagnostico
`resume_ms`: i 2520 layer MoE presentano lì valori negativi impossibili. I
tempi `ssd_read_ms`, `upload_and_sync_ms` e `load_ms` sono validi. Da
`output_2/` in poi `resume_ms` non è più negativo. La correzione non cambia
l'inferenza né le misure di caricamento.

## Prossimo step — progetto controllato di TEST_3

Prima di implementare il predictor ML, TEST_3 dovrà isolare il beneficio
massimo teorico del prefetch usando l'oracolo di routing di Run A:

~~~text
Run A: produce routing_oracle per il percorso di riferimento
    |
    +--> Run B: replay oracle, nessun prefetch
    |          -> controllo corretto e riproducibile
    |
    +--> Run C: replay oracle + banco prefetch separato
               -> prepara E_l(t+1) durante l'esecuzione di E_l(t)
               -> misura hit, miss, byte, attesa residua e tok/s
~~~

Il prossimo lavoro tecnico è progettare il secondo banco CUDA e definire i
suoi eventi (`prefetch_start`, `prefetch_ready`, `prefetch_hit`,
`prefetch_miss`, eventuale eviction). Solo dopo confronteremo i tempi
end-to-end di B e C: TEST_2 non è, da solo, una misura di speedup.

### File originali modificati

- `ds4_gpu.h`: definisce una piccola struttura di statistiche diagnostiche.
- `ds4_cuda.cu`: misura letture SSD, upload e byte nel loader degli expert
  selezionati; non cambia gli expert scelti né il contenuto trasferito.
- `ds4.c`: scrive `ssd_trace.jsonl` nel punto in cui gli expert sono già noti
  e il caricamento è concluso; disabilita soltanto l'overlap corrente mentre
  la traccia è richiesta, per avere misure non ambigue.
- `Makefile`: aggiunge i target `test_2_runner` e `test-predictor-2`.

## Esecuzione

Dalla root della repository:

~~~sh
make test-predictor-2
~~~

Il runner richiede `DS4_MODEL`, come i test precedenti. Nell'IDE JetBrains crea
una configurazione **Native Application** chiamata:

~~~text
TEST_2 SSD Streaming Trace
~~~

| Campo | Valore |
|---|---|
| Target | `test_2_runner` |
| Executable | `/home/michele/ds4/test_predictor/test_2/test_2_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | vuoto |
| Environment variables | `DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Before launch | `Build` |

Ogni esecuzione crea `output/`, poi `output_2/`, `output_3/` e così via, senza
sovrascrivere i risultati precedenti. La cartella contiene:

- `ssd_trace.jsonl`: un record per ciascun caricamento di expert nel decode;
- `summary.json`: timestamp UTC, numero di eventi e quanti hanno usato direct
  I/O;
- `run.stdout.txt` e `run.stderr.txt`: output completo del processo DS4.
