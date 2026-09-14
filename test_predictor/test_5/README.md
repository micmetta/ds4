# TEST_5 — Oracle prefetch da HDD esterno

## Obiettivo

TEST_5 misurera se separare fisicamente le letture speculative del prefetch
dalle letture urgenti del decode riduce la competizione I/O osservata in
TEST_4.

Il modello principale continuera a risiedere sull'SSD NVMe interno. Il disco
esterno USB sara usato **solo** dal worker di prefetch come seconda sorgente
di lettura. Per chiarezza, il dispositivo esterno e un HDD: non e una
soluzione finale desiderata, ma uno strumento sperimentale disponibile ora.

```text
NVMe interno: modello DS4 principale
    demand load di E_l(t): NVMe -> RAM pinned -> VRAM

HDD esterno USB: clone identico del modello
    worker prefetch di E_l(t+1): HDD -> banco RAM pinned
```

L'idea non e dimostrare che un HDD sia veloce. L'HDD e piu lento dell'NVMe,
specialmente per letture sparse, ma e un dispositivo fisicamente distinto. Se
il worker non rallenta piu le richieste urgenti del NVMe, potremo isolare il
costo della competizione sul singolo SSD osservata in TEST_4.

## Glossario

| Termine | Significato nel test |
|---|---|
| **NVMe interno** | L'SSD interno WD sul quale DS4 legge normalmente il modello e gli expert richiesti dal router. |
| **HDD esterno** | Disco meccanico collegato via USB, visibile in Windows come `D:`. Ha 302 GB liberi e sara la sorgente del solo prefetch. |
| **Clone del GGUF** | Copia byte-per-byte del file modello. Poiche conserva gli stessi offset, il worker puo leggere gli stessi pesi senza una nuova mappa `(layer, expert) -> offset`. |
| **Demand load** | Caricamento urgente degli expert necessari al token/layer corrente. Deve restare prioritario e usa l'NVMe interno. |
| **Prefetch** | Lettura anticipata di expert candidati per il token successivo. In TEST_5 usa il clone sul disco esterno. |
| **Oracolo** | Trace prodotto da Run A che dice quali expert sono stati realmente usati nel percorso di riferimento. Serve solo a simulare una previsione perfetta. |
| **RAM pinned** | RAM host bloccata temporaneamente dal runtime CUDA; il sistema operativo non puo spostarne le pagine finche sono registrate. Serve per trasferimenti piu prevedibili verso la GPU. |
| **Hit / miss** | Un hit trova il gruppo futuro gia pronto nel banco pinned; un miss usa il demand load normale dall'NVMe. |
| **Direct I/O** | Lettura che evita la page cache Linux. Il test deve verificarne il comportamento anche sul percorso del disco esterno, senza attribuire alla cache RAM risultati non reali. |

## Verifica del mount WSL dell'HDD esterno

La configurazione di TEST_5 usa un percorso Linux, ad esempio:

```text
DS4_TEST_5_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-...gguf
```

La lettera Windows `D:` deve quindi essere montata e visibile in WSL come
`/mnt/d`. Prima di eseguire il test e utile verificare che il clone sia
realmente accessibile:

```bash
ls -lh /mnt/d/ds4/gguf/
```

Se il file e presente in Esplora File ma `/mnt/d` risulta vuota in WSL, TEST_5
non puo confrontare le dimensioni dei due GGUF e termina con un errore. In
questo caso chiudi CLion e i terminali WSL, poi da PowerShell esegui:

```powershell
wsl --shutdown
```

Al successivo avvio WSL ricrea i mount automatici delle unita Windows. Questo
non modifica ne cancella file: riavvia soltanto WSL e ripristina il
collegamento tra `D:` e `/mnt/d`.

## Perche TEST_5 viene dopo TEST_4

In TEST_4 worker e demand load leggevano lo stesso file sullo stesso NVMe:

```text
worker:      SSD interno -> RAM pinned, gruppo futuro
                          ||
demand load: SSD interno -> RAM pinned -> VRAM, gruppo urgente
```

Questo puo allungare i miss demand: le due richieste condividono il controller,
la banda e la coda del dispositivo.

TEST_5 separa le due sorgenti:

```text
worker:      HDD esterno -> RAM pinned, gruppo futuro
                          ||
demand load: NVMe interno -> RAM pinned -> VRAM, gruppo urgente
```

Rimangono condivisi CPU, RAM pinned, PCIe e GPU; viene isolata soltanto la
competizione per il dispositivo di archiviazione. Pertanto un risultato
positivo indicherebbe che la contesa sul singolo NVMe era una componente
importante, non che l'HDD sia intrinsecamente una sorgente preferibile.

## Perche due dispositivi fisici possono aiutare

### Banda di lettura e richieste contemporanee

La **banda di lettura** e la quantita massima di dati che un dispositivo puo
fornire in un secondo. Se worker e demand load leggono dallo stesso SSD,
possono essere avviati da thread diversi nello stesso momento, ma usano la
stessa capacita fisica del dispositivo:

```text
stesso NVMe
    |
    +--> demand load urgente:     legge E_l(t)
    `--> worker prefetch:         legge E_l(t+1)

risorse condivise: controller, canali NAND, cache interna,
banda di lettura e collegamento PCIe
```

Un NVMe puo mantenere molte richieste in volo e sovrapporle parzialmente. Non
e pero possibile ottenere il doppio della banda soltanto perche esistono due
thread: se il worker usa parte della banda o mantiene richieste sparse in coda,
il demand load puo attendere di piu. Questo e la competizione I/O osservata in
TEST_4.

Con due dispositivi fisici la situazione cambia:

```text
NVMe interno                          HDD esterno USB
-------------                          ----------------
demand load E_l(t)                    worker per E_l(t+1)
controller NVMe proprio               controller del disco USB proprio
canali NAND propri                    piatti/testina propri
banda NVMe propria                    banda USB/HDD propria
```

Le letture non devono piu dividere lo stesso controller, la stessa coda del
dispositivo o la stessa banda di lettura. Il PC puo avere ancora qualche
risorsa condivisa a monte, per esempio CPU, RAM pinned, chipset/PCIe e GPU,
ma viene eliminata la componente principale della contesa sullo storage.

### HDD meccanico e SSD/NVMe: differenza pratica

Un HDD meccanico contiene piatti rotanti e una testina fisica. Se deve leggere
due zone lontane del file, la testina deve spostarsi: per questo le letture
sparse tendono a diventare quasi seriali.

```text
HDD
richiesta A -> spostamento testina -> lettura A
richiesta B -> nuovo spostamento   -> lettura B
```

Un SSD/NVMe non ha testine: puo gestire molte richieste contemporanee molto
meglio e sovrapporre parte del lavoro. Tuttavia le richieste continuano a
condividere controller, memoria NAND, cache interna e banda PCIe, quindi la
concorrenza non e gratuita.

```text
HDD meccanico: quasi seriale per letture sparse
SSD/NVMe:      parallelismo parziale, molto piu efficiente
due dischi:    letture molto piu indipendenti
```

Nel loader MoE un gruppo nominale di sei expert contiene circa **40,5 MiB**,
ma non corrisponde necessariamente a un unico blocco fisico contiguo: per ogni
expert DS4 legge i pesi `gate`, `up` e `down`. Un HDD puo quindi soffrire dei
seek tra piu zone del GGUF. Per questo TEST_5 iniziera da una finestra piccola,
non perche si desideri usare un HDD in produzione, ma per misurare in modo
controllato la separazione della contesa I/O.

## Prerequisiti verificati

1. Il GGUF completo e stato copiato sul disco esterno in
   `D:\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
   Il clone occupa circa 81 GiB; lo spazio libero disponibile e sufficiente.
2. WSL lo vede come
   `/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
3. Il runner verifica prima di partire che modello principale e clone abbiano
   la stessa dimensione. Prima di usare i risultati come misura definitiva e
   comunque consigliato verificare una volta anche lo SHA-256 del clone.
4. Una lettura manuale di prova con `iflag=direct` sul clone in `/mnt/d` e
   riuscita. Run C ripete comunque la stessa verifica in modo autonomo: apre
   il clone con Direct I/O ed esegue subito una lettura allineata. Se non
   riesce, termina con errore; non esiste un fallback silenzioso alla page
   cache Linux.

Non modificheremo il GGUF originale e non useremo il clone per il demand load.

## Run previste

| Run | Scopo | Router/esecuzione | Origine delle letture expert |
|---|---|---|---|
| **A** | Costruisce il riferimento | Routing normale; registra tutti gli expert e pesi, inclusi i layer `is_hash=true`. | NVMe interno, flusso DS4 normale. |
| **B** | Baseline controllata | Replay completo dell'oracolo di A; nessun worker prefetch. | NVMe interno per ogni demand load. |
| **C** | Esperimento a due dischi | Stesso replay completo di B; il worker legge dal clone su HDD. | Demand: NVMe interno. Prefetch: HDD esterno. |

Come in TEST_4, B e C calcoleranno comunque il router locale a fini
diagnostici, ma eseguiranno gli expert e i pesi dell'oracolo di A. Questo rende
identici il percorso MoE, i token e l'output: la differenza misurata e il
percorso di caricamento.

## Flusso previsto

```text
test_predictor/test_5/test_5_runner.c                 [da creare]
    |
    +--> Run A, nuovo processo DS4
    |      |
    |      `--> ds4.c: ds4_routing_oracle_*()
    |             registra (token, layer) -> expert IDs + router weights
    |             nel routing_oracle.bin
    |
    +--> Run B, nuovo processo DS4, replay senza prefetch
    |      |
    |      `--> ds4_cuda.cu: cuda_stream_selected_cache_begin_load(...)
    |             demand: NVMe interno -> RAM pinned -> VRAM
    |
    `--> Run C, nuovo processo DS4, replay + prefetch da HDD
           |
           +--> demand corrente E_l(t)
           |      NVMe interno -> RAM pinned -> VRAM
           |
           +--> dopo l'accodamento CUDA del MoE corrente
           |      l'oracolo fornisce soltanto il candidato E_l(t+1)
           |      il worker accoda la lettura futura
           |
           +================ ATTIVITA' IN PARALLELO ================+
           | GPU: esegue E_l(t)                                     |
           | worker: HDD esterno -> banco pinned per E_l(t+1)       |
           +=========================================================+
           |
           `--> token t+1, layer l
                  |
                  +--> hit: banco pinned -> VRAM, nessuna nuova lettura SSD
                  `--> miss: NVMe interno -> RAM pinned -> VRAM
```

## Parametri previsti

Il runner introdurra una variabile separata, per esempio:

```text
DS4_TEST_5_PREFETCH_MODEL=/mnt/d/ds4-model/DeepSeek-...gguf
```

La variabile indichera il clone usato esclusivamente dal worker di Run C. Il
percorso del modello principale continuera a essere `DS4_MODEL` e non verra
sostituito.

Anche la finestra di layer dovra essere configurabile, con valore iniziale
prudente pari a un layer:

```text
DS4_TEST_5_PREFETCH_LAYERS=1
```

La distanza temporale e configurabile separatamente. Il valore predefinito e
`1`: durante `token t` il worker prepara `E_l(t+1)`. Per dare piu tempo a una
sorgente lenta, come l'HDD USB, TEST_5 puo preparare un token piu lontano:

```text
DS4_TEST_5_PREFETCH_DISTANCE=2
```

Con questo valore, dopo il MoE di `token t, layer l`, Run C legge
`E_l(t+2)`. Il banco pinned mantiene fino a otto gruppi futuri per ciascun
layer attivo, cosi una richiesta per `t+2` non sovrascrive una richiesta gia
in corso per `t+1`. I valori validi sono `1..8`; valori assenti, non validi o
zero tornano a `1`.

Con distanza `2`, l'anticipo resta costante a ogni token: non viene richiesta
la coppia `(t+1, layer l)`; viene sempre richiesta quella distante due token.

```text
dopo E_l(t)     -> il worker riceve la richiesta di E_l(t+2)
dopo E_l(t+1)   -> il worker riceve la richiesta di E_l(t+3)
dopo E_l(t+2)   -> il worker riceve la richiesta di E_l(t+4)
...
```

Un HDD USB legge sequenzialmente circa 83 MB/s nel benchmark effettuato, ma
ha prestazioni molto peggiori nelle letture sparse. DS4 legge i pesi `gate`,
`up` e `down` per sei expert, quindi il primo test deve limitare la pressione
sul worker. Solo dopo i risultati con `1` layer valuteremo `3` e `5`.

## Misure e criteri di interpretazione

Il test produrra routing trace, prefetch trace, stdout/stderr e un summary per
le tre run. Almeno questi valori saranno confrontati tra B e C:

- token/s di generazione;
- numero di hit, miss e richieste accodate;
- `ssd_read_ms` dei miss demand sul NVMe;
- tempo di lettura del worker dall'HDD;
- `transfer_from_RAM_to_VRAM_ms`;
- correttezza: output, eventi routing, expert e pesi A/B/C identici.

Interpretazione desiderata:

```text
hit C elevati + miss C simili/piu rapidi di B
    -> la separazione fisica attenua la contesa I/O

hit C bassi
    -> l'HDD non completa il lavoro entro la finestra disponibile

miss C comunque piu lenti di B
    -> il limite e probabilmente fuori dal singolo SSD:
       CPU, RAM pinned, PCIe, trasferimento CUDA o sincronizzazioni
```

## Implementazione

Il test e ora implementato con il minimo riuso necessario di TEST_4.

| File | Modifica | Motivo |
|---|---|---|
| `test_predictor/test_5/test_5_runner.c` | Nuovo runner A/B/C con output incrementali, trace e `summary.json`. | Esegue l'esperimento su tre processi DS4 realmente distinti. |
| `ds4_cuda.cu` | Il worker del banco pinned apre facoltativamente `DS4_TEST_5_PREFETCH_MODEL` e legge da quel descrittore. | In Run C separa il percorso worker (HDD) dal demand loader (NVMe). |
| `Makefile` | Aggiunti `test_5_runner` e `test-predictor-5`. | Permette a CLion di individuare e compilare il runner. |

`ds4.c` non e stato modificato per TEST_5: continua a usare le API di banco
pinned e replay gia validate in TEST_4. Senza `DS4_TEST_5_PREFETCH_MODEL` il
codice CUDA conserva esattamente il comportamento di TEST_4.

## Configurazione CLion

Crea una **Native Application** chiamata, per esempio,
`TEST_5 External-Device Oracle Prefetch`.

| Campo | Valore |
|---|---|
| Target | `test_5_runner` |
| Executable | `/home/michele/ds4/test_predictor/test_5/test_5_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | vuoto |
| Environment variables | vedi sotto |

Variabili di ambiente, su una sola riga separata da `;`:

```text
DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_5_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_5_PREFETCH_LAYERS=1
```

Per il prossimo esperimento con anticipo di due token, sostituisci il valore
con questa riga:

```text
DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_5_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_5_PREFETCH_LAYERS=10;DS4_TEST_5_PREFETCH_DISTANCE=2
```

Il valore `1` e il punto di partenza prudente per l'HDD USB. Il runner forza
internamente `DS4_CUDA_MMQ=0`, crea `output`, poi `output_2`, e cosi via.

## Risultati osservati

### `output_2` — HDD esterno, finestra di 10 layer

Configurazione rilevante:

```text
DS4_TEST_5_PREFETCH_LAYERS=10
```

Il clone sul disco esterno e stato effettivamente aperto dal worker con Direct
I/O. Lo conferma `run_C.stderr.txt`:

```text
ds4: TEST_5 prefetch worker source opened with Direct I/O: /mnt/d/...gguf (align=512)
```

Quindi il worker non ha usato per errore la page cache Linux e il demand load
ha continuato a leggere il modello originale dall'NVMe interno.

| Misura | Run B: replay senza prefetch | Run C: prefetch da HDD su layer 0..9 |
|---|---:|---:|
| Generazione | 0,80 token/s | 0,77 token/s |
| `prefetch_hit` | — | 0 |
| `prefetch_miss_demand` | 2709 | 2709 |
| `prefetch_enqueue` | — | 620 |
| Lettura SSD media dei demand miss | 19,34 ms | 19,25 ms |
| Trasferimento RAM → VRAM medio dei demand miss | 7,47 ms | 8,28 ms |
| Tempo totale medio del demand miss | 26,83 ms | 27,54 ms |

I 620 eventi `prefetch_enqueue` sono coerenti con 62 token per i quali esiste
un token futuro da predire, moltiplicati per i 10 layer attivi (`0..9`).
Indicano che Run C ha inoltrato le richieste speculative al worker; non
dimostrano che i dati fossero gia pronti nella RAM pinned.

Il dato decisivo e infatti `prefetch_hit = 0`: nessuno dei gruppi di expert
letti dal disco esterno e diventato `ready` prima della richiesta reale del
token successivo. Per tutte le 2709 esecuzioni MoE, Run C ha quindi seguito
comunque il demand load standard:

```text
NVMe interno -> RAM pinned -> VRAM
```

Il lieve peggioramento da 0,80 a 0,77 token/s non e uno speedup negativo
attribuibile a hit falliti: non ci sono stati hit. E il costo del worker e
normale variabilita tra run. La lettura demand dall'NVMe resta quasi uguale;
la differenza osservata e soprattutto nel trasferimento RAM -> VRAM e nelle
oscillazioni normali di sistema/GPU.

La conclusione e che aumentare la finestra da 1 a 10 layer ha aumentato le
richieste speculative, ma non ha reso il prefetch abbastanza tempestivo. Il
disco esterno e un HDD USB: per ogni gruppo di sei expert il worker deve
leggere blocchi `gate`, `up` e `down` distribuiti nel GGUF. Le letture sparse
e i movimenti della testina rendono il completamento troppo lento rispetto al
tempo disponibile di un token. Il test resta utile: dimostra che i due
percorsi sono su dispositivi distinti e che, in questa configurazione, il
limite e la latenza di completamento del prefetch su HDD, non il routing o
l'oracolo.

### Confronto misurato: NVMe interno contro HDD USB esterno

I benchmark `winsat disk` eseguiti su Windows quantificano la differenza tra
il drive `C:` interno, un NVMe WD PC SN560, e il drive `D:` esterno, un HDD
USB. Sono misure generali del dispositivo: il loader MoE reale usa letture
sparse, quindi il vantaggio del NVMe nella pratica puo essere ancora piu
importante di quanto suggerisca la sola lettura sequenziale.

| Misura `winsat` | NVMe interno `C:` | HDD USB esterno `D:` | Rapporto NVMe/HDD circa |
|---|---:|---:|---:|
| Lettura sequenziale 64 KiB | 3349,54 MB/s | 83,38 MB/s | 40× |
| Lettura casuale 16 KiB | 376,90 MB/s | 1,01 MB/s | 373× |
| Latenza media lettura casuale | 0,077 ms | 16,741 ms | 217× piu bassa |
| Latenza al 95° percentile | 0,411 ms | 36,553 ms | 89× piu bassa |

Questo confronto spiega il risultato di `output_2`. Il worker deve portare
dal disco esterno alla RAM pinned un gruppo di pesi previsto mentre il modello
sta gia generando il token corrente. Un HDD fornisce circa 83 MB/s in
sequenziale, ma per le letture sparse raggiunge solo circa 1 MB/s e introduce
latenze dell'ordine delle decine di millisecondi. Non riesce quindi a rendere
pronto il gruppo `E_l(t+1)` entro la breve finestra disponibile prima di
`token t+1, layer l`.

La separazione fisica tra NVMe e HDD elimina la contesa diretta sullo stesso
dispositivo, ma non rende l'HDD abbastanza rapido da produrre hit. Per un
esperimento futuro con possibilita realistica di hit servirebbe un secondo
SSD/NVMe veloce, idealmente su un controller e un collegamento PCIe/USB ad
alta banda realmente indipendenti; un HDD USB resta utile soprattutto come
controllo sperimentale di separazione dei percorsi I/O.

## Traccia diagnostica del worker

Dopo `output_6` il test registra anche il ciclo di vita effettivo delle
richieste speculative nel file `run_C.prefetch.jsonl`. Lo scopo e distinguere
un semplice `prefetch_enqueue` (intenzione di prefetch) da una richiesta che
il worker ha davvero letto e reso pronta nella RAM pinned.

| Evento | Significato |
|---|---|
| `prefetch_worker_queued` | uno slot del banco pinned e stato riservato per `(target_token, layer)`; il worker puo prenderlo. |
| `prefetch_worker_started` | il worker ha preso lo slot e ha iniziato le letture Direct I/O dal modello esterno. |
| `prefetch_worker_ready` | tutte le letture `gate`, `up` e `down` sono terminate: il gruppo e pronto in RAM pinned. Se poi il router/replay richiede lo stesso gruppo, ci aspettiamo un `prefetch_hit`. |
| `prefetch_worker_failed` | la lettura o la preparazione del banco pinned e fallita; il decode continuera con il demand load normale. |
| `prefetch_bank_full` | non era disponibile uno slot per quella richiesta. Non e un errore di inferenza: il prefetch viene saltato e resta il fallback demand. |
| `prefetch_lookup_loading` | il token ha richiesto il gruppo mentre il worker lo stava ancora leggendo. Questo e il segnale diretto che il worker non ha finito in tempo. |
| `prefetch_lookup_failed` | il token ha richiesto un gruppo la cui lettura speculative era fallita. |
| `prefetch_lookup_absent` | al momento della richiesta non esisteva uno slot corrispondente: per esempio la richiesta non era stata accodata, e stata scartata per capacita, oppure apparteneva a un layer fuori finestra. |
| `prefetch_lookup_expert_mismatch` | esiste uno slot per lo stesso token/layer, ma contiene expert diversi. Con il replay completo A dovrebbe restare zero: sarebbe un segnale di errore di correlazione. |
| `prefetch_stale_discarded` | uno slot rimasto indietro rispetto al token corrente viene liberato in sicurezza. |

I nuovi campi temporali sono espressi in millisecondi e permettono di
separare attesa del worker e tempo di I/O:

| Campo | Significato |
|---|---|
| `queue_to_worker_ms` | tempo trascorso tra l'accodamento e l'inizio reale del worker. Misura la coda interna del banco. |
| `worker_read_ms` | durata della lettura completa del gruppo dal disco alla RAM pinned. |
| `queue_to_ready_ms` | tempo totale tra accodamento e disponibilita del gruppo in RAM pinned. |
| `slot_age_ms` | eta dello slot quando l'evento e stato registrato; nei lookup mostra quanto tempo il worker aveva gia avuto a disposizione. |

Ogni record contiene ora anche `failure_reason`. Vale `none` per gli eventi
che non rappresentano un errore; per `prefetch_worker_failed` rende esplicita
la causa osservata dal worker:

| Valore | Significato |
|---|---|
| `direct_io_offset_alignment` | l'offset nel GGUF non rispetta l'allineamento richiesto dal Direct I/O. |
| `direct_io_size_alignment` | la dimensione richiesta non e multipla dell'allineamento Direct I/O. |
| `direct_io_destination_alignment` | la porzione di RAM pinned scelta come destinazione non e allineata. |
| `read_gate`, `read_up`, `read_down` | il controllo di allineamento e passato, ma la lettura fisica del rispettivo blocco e fallita. |
| `pinned_allocation` | CUDA non e riuscita ad allocare il buffer host pinned dello slot. |
| `invalid_slot` / `invalid_expert` | metadati dello slot o ID expert non validi. |

### Correzione Direct I/O dopo `output_8`

`output_8` ha mostrato 49 errori `direct_io_offset_alignment`: una singola
matrice di un expert non e necessariamente collocata nel GGUF a un offset
multiplo di 512 byte. Il worker non legge piu direttamente quella porzione
non allineata nel buffer finale.

Ora, per ciascun blocco `gate`, `up` o `down`, applica questo flusso:

```text
offset e dimensione logici della matrice
    ↓
intervallo fisico piu piccolo che li contiene ed e allineato al Direct I/O
    ↓
SSD/HDD -> scratch buffer RAM pinned e allineato
    ↓
copia della sola porzione logica richiesta -> slot del banco pinned
```

Lo scratch e riutilizzabile per slot e viene allocato con `cudaMallocHost`,
quindi resta RAM pinned. La correzione non modifica router, expert eseguiti,
cache demand o percorso SSD normale: riguarda soltanto il worker speculativo
di TEST_5. Dopo questa modifica, un eventuale `read_gate`, `read_up` o
`read_down` indichera un errore fisico di lettura e non piu il semplice fatto
che il tensore partiva da un offset non allineato.

## Risultati osservati — `output_9`

`output_9` e la prima esecuzione dopo la correzione Direct I/O. Usa il modello
demand sull'NVMe interno, il modello identico sull'HDD USB esterno per il
worker, `DS4_TEST_5_PREFETCH_LAYERS=1` e
`DS4_TEST_5_PREFETCH_DISTANCE=8`.

| Misura | Run B: replay senza prefetch | Run C: replay con prefetch HDD -> RAM pinned |
|---|---:|---:|
| Generazione | 0,77 token/s | 0,83 token/s |
| Demand miss | 2709 | 2654 |
| Prefetch hit | 0 | 55 |
| Worker queued | — | 55 |
| Worker ready | — | 55 |
| Worker failed | — | 0 |
| Lookup mentre `loading` | — | 0 |
| Banco pinned pieno | — | 0 |

I risultati sono coerenti anche per il layer 0, l'unico layer attivo in
questo esperimento. Run B ha 63 demand miss nel layer 0. Run C ha invece:

```text
8 demand miss iniziali: token 20..27, fase di warm-up
55 prefetch hit:        token 28 e successivi
-----------------------------------------------
63 utilizzi del layer 0
```

Gli otto miss iniziali sono attesi: con distanza 8, durante il token 20 il
worker prepara il gruppo per il token 28. Non puo quindi esistere un gruppo
gia pronto per i token 20..27.

### Perche prima l'HDD esterno non produceva hit

Nei precedenti output il contatore degli hit restava a zero, ma la causa non
era ancora dimostrata. `output_8` l'ha resa esplicita: tutte le 49 richieste
del worker fallivano con:

```text
failure_reason = direct_io_offset_alignment
```

Il file GGUF contiene le matrici `gate`, `up` e `down` degli expert a offset
che non sono necessariamente multipli dell'allineamento richiesto dal Direct
I/O dell'HDD. Il worker tentava di leggere direttamente la singola matrice;
Linux rifiutava quindi la richiesta prima che il blocco potesse diventare
`ready`. La sequenza era:

```text
prefetch_worker_queued
    -> prefetch_worker_started
    -> prefetch_worker_failed (offset non allineato)
    -> prefetch_lookup_failed
    -> demand load normale da NVMe
```

Di conseguenza non era possibile misurare la vera capacita dell'HDD di
completare in anticipo: il prefetch non arrivava mai neppure in RAM pinned.

Dopo la lettura allineata tramite scratch pinned, `output_9` mostra invece:

```text
prefetch_worker_queued
    -> prefetch_worker_started
    -> prefetch_worker_ready
    -> prefetch_hit
    -> RAM pinned -> VRAM
```

La lettura HDD rimane lenta: `worker_read_ms` ha mediana circa 434 ms. La
distanza 8 gli lascia pero un anticipo sufficiente, e con un solo layer il
worker riesce a completare una richiesta prima del suo consumo. Al momento
dell'hit viene evitata la lettura demand dall'NVMe: il caricamento passa dalla
mediana di circa 26,26 ms per `NVMe -> RAM pinned -> VRAM` a circa 6,58 ms
per il solo trasferimento `RAM pinned -> VRAM`.

Il passaggio da 0,77 a 0,83 token/s e circa +7,8% in questa singola run. E un
segnale sperimentale importante, ma non una stima definitiva: va ripetuto su
piu run e con piu layer, separando lo speedup reale dalla normale variabilita
di GPU e I/O.

## Risultati osservati - `output_10` e `output_11`

Gli esperimenti successivi mantengono `DS4_TEST_5_PREFETCH_DISTANCE=8` e
aumentano il numero di layer per i quali Run C accoda il prefetch. Servono a
capire fino a che punto il worker dell'HDD e il banco pinned riescono a seguire
la generazione.

| Output | Layer prefetch | Hit | Demand miss | Banco pieno | Run B | Run C |
|---|---:|---:|---:|---:|---:|---:|
| `output_10` | 2 (layer 0..1) | 110 | 2599 | 0 | 0,81 token/s | 0,80 token/s |
| `output_11` | 10 (layer 0..9) | 145 | 2564 | 341 | 0,80 token/s | 0,79 token/s |

### `output_10`: due layer completati

Con due layer il risultato e completo: i 55 utilizzi in decode del layer 0 e
i 55 del layer 1, dopo gli otto token iniziali di warm-up, diventano tutti
hit. Non sono presenti fallimenti del worker, lookup mentre il dato era ancora
in lettura o rifiuti per banco pieno. Il worker dell'HDD ha quindi abbastanza
tempo, grazie alla distanza 8, per preparare entrambi i gruppi futuri.

Il valore di generazione di una singola run resta molto vicino alla baseline
(0,80 contro 0,81 token/s). Gli hit evitano comunque il demand load per quei
gruppi: il loro trasferimento medio RAM pinned -> VRAM e circa 7 ms, contro un
demand load medio di circa 26,6 ms. La differenza finale piccola richiede piu
ripetizioni per separare il beneficio reale dalla variabilita normale di I/O e
GPU.

### `output_11`: dieci layer saturano il banco e la coda

Con dieci layer, Run C prova ad accodare 10 gruppi futuri per ogni token di
decode. Gli eventi mostrano:

| Evento Run C | Conteggio | Cosa mostra |
|---|---:|---|
| `prefetch_enqueue` | 550 | richieste speculative tentate per i layer 0..9. |
| `prefetch_worker_queued` | 209 | richieste che hanno realmente ottenuto uno slot nel banco pinned. |
| `prefetch_worker_ready` | 177 | gruppi letti completamente dall'HDD e pronti in RAM pinned. |
| `prefetch_hit` | 145 | gruppi pronti consumati con trasferimento RAM pinned -> VRAM. |
| `prefetch_bank_full` | 341 | richieste rifiutate perche non esisteva uno slot libero. |
| `prefetch_lookup_loading` | 64 | gruppi richiesti mentre il worker li stava ancora leggendo. |
| `prefetch_stale_discarded` | 30 | gruppi diventati troppo vecchi, liberati senza usarli. |

Gli hit non sono distribuiti uniformemente: layer 0 ha 55 hit, layer 1 ha 55,
layer 2 ne ha 32 e layer 3 solo 3; i layer 4..9 non ne hanno. Il worker legge
un gruppo alla volta. La lettura stessa dura in media circa 441 ms, ma con
dieci richieste per token la coda cresce: una richiesta aspetta in media circa
11,1 s prima che il worker cominci a leggerla e diventa pronta dopo circa
11,55 s. Per molti gruppi questo e troppo tardi rispetto al token bersaglio.

#### Quando e perche il banco prefetch diventa pieno

Il banco pinned non contiene tutti gli expert del modello: contiene un numero
fisso e piccolo di **gruppi futuri**. Ogni gruppo occupa uno slot e rappresenta
la coppia `(target_token, layer)`, con i pesi gate, up e down dei sei expert
previsti. Uno slot resta occupato dall'istante in cui la richiesta viene
accettata fino a uno di questi momenti:

```text
slot loading -> worker finisce -> slot ready -> token bersaglio lo consuma
                                                     -> slot libero

oppure

slot loading/ready -> il token bersaglio e gia passato -> stale discarded
                                                        -> slot libero
```

Con `PREFETCH_LAYERS=2`, per ogni token vengono creati solo due gruppi futuri:
il consumer riesce a liberarli regolarmente quando raggiunge il token
bersaglio. Con `PREFETCH_LAYERS=10`, invece, il thread di decode prova ad
aggiungerne dieci nello stesso intervallo. Il worker dell'HDD e seriale e ne
puo leggere solo uno per volta; mentre i primi slot sono ancora `loading` o
`ready` in attesa del loro token, gli altri slot non sono disponibili. Quando
Run C tenta un nuovo `prefetch_enqueue` e tutti gli slot sono in questo stato,
registra `prefetch_bank_full` e non avvia alcuna lettura speculativa per quella
coppia `(target_token, layer)`.

`prefetch_bank_full` non compromette la correttezza: il gruppo non viene
prefetchato e, quando serve, DS4 esegue il normale demand load dall'NVMe. E
pero un limite di capacita e di scheduling: solo 145 hit su 550 richieste
tentate non compensano il costo del worker, perciò Run C scende leggermente a
0,79 token/s rispetto agli 0,80 token/s di Run B.

Il prossimo intervallo utile da testare e quindi intermedio, per esempio 3 o
5 layer con distanza 8: abbastanza gruppi da aumentare la copertura degli hit,
ma non tanti da mantenere il banco pinned occupato e creare una lunga coda nel
worker.

## Risultati osservati - `output_12`: finestra di 3 layer

`output_12` prova proprio la prima finestra intermedia suggerita sopra:

```text
DS4_TEST_5_PREFETCH_LAYERS=3
DS4_TEST_5_PREFETCH_DISTANCE=8
```

| Misura | Valore |
|---|---:|
| Prefetch tentati | 165 |
| Richieste accettate nel banco pinned | 147 |
| Worker ready | 147 |
| Hit effettivi | 138 |
| Hit potenziali dopo warm-up | 165 |
| Copertura hit | circa 83,6% |
| Banco pieno | 18 |
| Lookup mentre `loading` | 9 |
| Run B | 0,73 token/s |
| Run C | 0,82 token/s |

La differenza osservata tra B e C e circa:

```text
(0,82 - 0,73) / 0,73 = circa +12,3%
```

In questa singola esecuzione la finestra di tre layer e quindi il miglior
compromesso osservato. Il risultato e incoraggiante, ma richiede ripetizioni:
la generazione di Run B puo variare tra run, per I/O e GPU, quindi il +12,3%
non e ancora una stima statistica definitiva.

### Copertura per layer

| Layer | Hit | Hit possibili dopo warm-up |
|---|---:|---:|
| 0 | 55 | 55 |
| 1 | 53 | 55 |
| 2 | 30 | 55 |

Il worker riesce a preparare sempre il layer 0, quasi sempre il layer 1 e piu
di meta del layer 2. Questo e coerente con un worker seriale: le richieste per
i layer iniziali entrano prima nella coda e hanno piu tempo per completarsi.

### Coda e banco pinned

| Tempo medio degli slot arrivati a `ready` | `output_11`: 10 layer | `output_12`: 3 layer |
|---|---:|---:|
| `worker_read_ms` | 441 ms | 448 ms |
| `queue_to_worker_ms` | 11,1 s | 2,56 s |
| `queue_to_ready_ms` | 11,55 s | 3,01 s |

La velocita fisica dell'HDD non cambia: la lettura resta circa 0,45 s per un
gruppo. Cambia invece la quantita di lavoro accodata. Con tre layer la coda e
molto piu corta rispetto ai dieci layer, percio molti piu gruppi arrivano
pronti prima del token bersaglio.

I conteggi sono coerenti tra loro:

```text
165 prefetch tentati
- 18 rifiutati per prefetch_bank_full
= 147 slot accettati, poi arrivati ready

147 ready
- 9 richiesti mentre erano ancora loading
= 138 prefetch_hit
```

Anche `prefetch_lookup_absent = 42` ha una spiegazione precisa:

```text
24 lookup iniziali di warm-up = 8 token x 3 layer
18 richieste rifiutate per banco pieno
-------------------------------------
42 lookup absent
```

Un hit continua a ridurre significativamente il singolo caricamento:

| Percorso | Tempo medio evento |
|---|---:|
| Miss demand: NVMe -> RAM pinned -> VRAM | circa 26,95 ms |
| Hit: RAM pinned -> VRAM | circa 6,81 ms |

Il prossimo passo utile e ripetere piu volte questa stessa configurazione
`3/8` per valutarne la stabilita. Solo dopo conviene confrontarla con una
finestra da 5 layer alla stessa distanza.

## Risultati osservati - `output_13`: finestra di 5 layer

`output_13` esegue il confronto successivo con:

```text
DS4_TEST_5_PREFETCH_LAYERS=5
DS4_TEST_5_PREFETCH_DISTANCE=8
```

| Misura | `output_12`: 3 layer | `output_13`: 5 layer |
|---|---:|---:|
| Prefetch tentati | 165 | 275 |
| Hit | 138 | 149 |
| Copertura hit sui gruppi post warm-up | 83,6% | 54,2% |
| `prefetch_bank_full` | 18 | 102 |
| `prefetch_lookup_loading` | 9 | 24 |
| Attesa media prima del worker | 2,56 s | 7,64 s |
| Run B | 0,73 token/s | 0,75 token/s |
| Run C | 0,82 token/s | 0,82 token/s |

Run C resta a 0,82 token/s anche con cinque layer, ma non migliora rispetto
alla finestra da tre. Il rapporto osservato rispetto a Run B e:

```text
(0,82 - 0,75) / 0,75 = circa +9,3%
```

Anche questo e un singolo campione: non va confrontato come una misura
statistica definitiva con il +12,3% di `output_12`, perche la baseline Run B
e variata da 0,73 a 0,75 token/s. Il dato robusto e che aggiungere due layer
ha portato solo 11 hit in piu (149 contro 138), ma molta piu pressione sul
banco e sul worker.

### Distribuzione degli hit

| Layer | Hit | Hit possibili dopo warm-up |
|---|---:|---:|
| 0 | 55 | 55 |
| 1 | 55 | 55 |
| 2 | 32 | 55 |
| 3 | 7 | 55 |
| 4 | 0 | 55 |

Il pattern conferma la priorita naturale della coda seriale: il worker
completa i layer iniziali, arriva solo in parte al layer 2 e quasi mai al layer
3; non raggiunge il layer 4 prima del consumo. Non e un errore del modello o
del Direct I/O: tutti i 173 gruppi che entrano nel banco diventano `ready`.

### Perche la finestra da 5 layer peggiora la copertura

Con cinque layer ci sono 275 gruppi potenzialmente utili dopo il warm-up.
Tuttavia 102 richieste vengono rifiutate subito per `prefetch_bank_full`, per
cui soltanto 173 gruppi entrano nel banco e vengono letti dal worker. Di quei
173 gruppi, 24 vengono richiesti quando la lettura e ancora `loading`; restano
quindi 149 hit:

```text
275 prefetch tentati
- 102 banco pinned pieno
= 173 richieste accettate e diventate ready

173 ready
- 24 richiesti ancora loading
= 149 prefetch hit
```

`prefetch_lookup_absent = 142` e coerente con 40 lookup di warm-up
(`8 token x 5 layer`) piu le 102 richieste che non hanno trovato posto nel
banco. Gli slot restano occupati piu a lungo e l'attesa media in coda sale a
circa 7,64 s, contro 2,56 s con tre layer. La lettura fisica dell'HDD rimane
invece simile, circa 421 ms: e la coda, non la singola lettura, a crescere.

Il singolo hit resta conveniente (`6,21 ms` contro `26,44 ms` per un demand
load medio), ma la copertura piu bassa impedisce di ottenere un aumento
ulteriore dei token/s. Per il banco e il worker attuali, tre layer con distanza
otto rimangono il candidato piu equilibrato; servono ripetizioni di `3/8`
prima di valutare modifiche alla capacita del banco o al numero di worker.
