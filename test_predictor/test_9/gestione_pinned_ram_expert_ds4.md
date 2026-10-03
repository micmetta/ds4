# Gestione della RAM pinned per gli expert in DS4

> Nota di ambito: questo documento descrive principalmente il **caricamento
> standard su domanda** di DS4, cioè il percorso usato quando il router reale
> richiede un expert immediatamente. Nel branch sperimentale del predictor è
> presente inoltre una banca pinned distinta, usata da TEST_4/TEST_5 e
> analizzata da TEST_9. I suoi slot non condividono i quattro buffer temporanei
> descritti qui. Per la politica di quella banca vedere `README.md` nella stessa
> cartella.

## Introduzione

Nel path CUDA di DS4, la **pinned RAM** usata per gli expert non significa, di per sé, che gli expert vengano mantenuti permanentemente in RAM.

Nel `main` attuale di DS4, la pinned RAM serve principalmente come **area di staging** tra SSD e VRAM.

Il flusso mentale corretto è:

```text
SSD
 │
 │ lettura dei pesi dell'expert
 ▼
Pinned RAM
 │
 │ cudaMemcpyAsync
 ▼
VRAM
 │
 ▼
MoE kernel
```

---

## Passo 1 — Che cos'è la pinned RAM

La RAM normale è *pageable*: il sistema operativo può spostarne le pagine, eventualmente anche nello swap.

La **pinned RAM**, invece, è memoria host le cui pagine vengono bloccate in memoria fisica. CUDA può quindi usarla in modo molto più efficiente per trasferimenti CPU → GPU, soprattutto con `cudaMemcpyAsync`.

In DS4 viene allocata tramite:

```cpp
cudaMallocHost(...)
```

Per lo streaming degli expert, DS4 crea **4 buffer pinned**:

```cpp
g_stream_selected_stage_raw[0]
g_stream_selected_stage_raw[1]
g_stream_selected_stage_raw[2]
g_stream_selected_stage_raw[3]
```

Quindi possiamo immaginare:

```text
RAM pinned

stage[0]  ─────────
stage[1]  ─────────
stage[2]  ─────────
stage[3]  ─────────
```

Questi non sono quattro cache persistenti di expert. Sono quattro **buffer temporanei di trasporto**.

---

## Passo 2 — Il router sceglie gli expert

Supponiamo che, per un certo token `t` e per il layer `l`, il router scelga:

```text
[12, 47, 83, 47, 120, 12]
```

DS4 elimina implicitamente i duplicati costruendo una rappresentazione compatta:

```text
compact_ids = [12, 47, 83, 120]
```

e costruisce anche una mappa che gli permette di sapere che le varie selezioni originali corrispondono agli slot compatti.

Questo avviene in `cuda_stream_selected_cache_begin_load()`: DS4 crea strutture come `expert_to_slot`, `compact_ids` e `slot_ids`, inserendo ciascun expert una sola volta.

Questa parte è importante: **se lo stesso expert è selezionato più volte nello stesso caricamento, DS4 non deve trasferirne più copie.**

---

## Passo 3 — DS4 prepara lo spazio in VRAM

A questo punto DS4 sa quanti expert distinti deve usare.

Per ogni expert MoE ci sono sostanzialmente i pesi delle tre proiezioni:

```text
gate
up
down
```

DS4 alloca quindi in VRAM tre regioni:

```text
gate_ptr
up_ptr
down_ptr
```

dimensionate per contenere gli expert selezionati.

Queste allocazioni sono fatte con `cudaMalloc`, quindi **questa è VRAM**, non RAM pinned.

Possiamo quindi già distinguere due cose:

```text
Pinned RAM
stage[0..3]
       │
       │ trasferimento
       ▼
VRAM
gate_ptr
up_ptr
down_ptr
```

---

## Passo 4 — DS4 trova dove si trova l'expert nel GGUF

Supponiamo che dobbiamo caricare l'expert `47`.

DS4 conosce gli offset di partenza delle tre grandi matrici degli expert:

```text
gate_offset
up_offset
down_offset
```

e conosce la dimensione di un singolo expert.

Calcola quindi, concettualmente:

```text
gate_src = gate_offset + 47 * gate_expert_bytes
up_src   = up_offset   + 47 * gate_expert_bytes
down_src = down_offset + 47 * down_expert_bytes
```

Dopodiché richiama le routine di copia per trasferire i dati corrispondenti alle tre componenti dell'expert.

Quindi DS4 sa precisamente:

> I pesi `gate` dell'expert 47 si trovano da questo byte a questo byte del file GGUF.

---

## Passo 5 — SSD → pinned RAM

Ora entra davvero in gioco la pinned RAM.

DS4 non trasferisce necessariamente tutto l'expert in un colpo solo. Usa dei **chunk**.

Il default attuale è:

```text
64 MB
```

configurabile tramite:

```text
DS4_CUDA_MODEL_COPY_CHUNK_MB
```

con limiti tra 16 MB e 4096 MB.

Quindi, se una certa porzione di expert fosse per esempio 150 MB, concettualmente DS4 potrebbe fare:

```text
64 MB
64 MB
22 MB
```

Ogni chunk viene letto dall'SSD in uno dei quattro buffer pinned.

---

## Passo 6 — Perché ci sono quattro buffer pinned

DS4 usa un meccanismo del tipo:

```cpp
bi = chunk_idx % 4;
```

Quindi:

```text
chunk 0 → stage[0]
chunk 1 → stage[1]
chunk 2 → stage[2]
chunk 3 → stage[3]
chunk 4 → stage[0]
chunk 5 → stage[1]
...
```

Il motivo è permettere di sovrapporre almeno in parte I/O e trasferimento PCIe.

Per esempio:

```text
tempo ───────────────────────────────────────────────►

stage[0]: SSD READ ──► H2D async ──────────────►
stage[1]:              SSD READ ──► H2D async ──────►
stage[2]:                         SSD READ ──► H2D ──►
stage[3]:                                    SSD READ...
```

Mentre la GPU sta trasferendo il contenuto di `stage[0]` verso la VRAM, la CPU può già riempire un altro staging buffer con i dati provenienti dall'SSD.

Questo è uno dei motivi fondamentali per cui la memoria è **pinned**: il trasferimento H2D può essere accodato attraverso `cudaMemcpyAsync()` su uno stream CUDA non bloccante.

---

## Passo 7 — Come DS4 evita di sovrascrivere un buffer ancora in uso

Supponiamo di essere arrivati al chunk 4.

Vorremmo riutilizzare:

```text
stage[0]
```

ma magari la GPU non ha ancora terminato il trasferimento precedente da `stage[0]`.

DS4 associa quindi un **CUDA event a ogni staging buffer**.

Prima di riutilizzare il buffer fa, concettualmente:

```cpp
cudaEventSynchronize(g_stream_selected_stage_event[bi]);
```

e dopo avere lanciato il nuovo `cudaMemcpyAsync()` registra:

```cpp
cudaEventRecord(...)
```

Quindi:

```text
stage[0]
   │
   ├── SSD mette chunk A
   │
   ├── GPU inizia a leggere chunk A
   │
   ├── event[0]
   │
   ├── CPU vuole metterci chunk E
   │
   ├── aspetta event[0]
   │
   └── ora può sovrascrivere stage[0]
```

Questo evita una race condition del tipo:

> Sto riscrivendo il buffer mentre la GPU sta ancora leggendo i dati precedenti.

---

## Passo 8 — Pinned RAM → VRAM

Una volta che il chunk è nella pinned RAM, DS4 fa qualcosa concettualmente equivalente a:

```cpp
cudaMemcpyAsync(
    dst + copied,
    payload,
    n,
    cudaMemcpyHostToDevice,
    g_stream_selected_upload_stream
);
```

Quindi il percorso completo di un pezzo dell'expert 47 è:

```text
GGUF sull'SSD
      │
      │ pread / O_DIRECT quando disponibile
      ▼
stage[2] in pinned RAM
      │
      │ cudaMemcpyAsync
      ▼
gate_ptr / up_ptr / down_ptr in VRAM
      │
      ▼
kernel MoE
```

Solo quando i pesi sono nella parte corretta della VRAM l'expert può essere utilizzato dal kernel CUDA.

---

## Passo 9 — La cosa fondamentale per la tesi: questa non è ancora una cache RAM persistente degli expert

Questo è il concetto più importante da fissare.

Nel CUDA upstream considerato, la pinned RAM è essenzialmente:

> **un'autostrada SSD → RAM → VRAM**

e non:

> **un magazzino persistente SSD → RAM in cui tengo gli expert che probabilmente serviranno dopo.**

Gli expert selezionati vengono caricati nei buffer VRAM compatti, mentre i buffer pinned vengono continuamente riutilizzati come staging area.

Inoltre, nel codice CUDA attuale, le API relative a una cache persistente degli expert lato RAM/streaming risultano sostanzialmente non implementate o stub.

---

## Collegamento con il prefetch predictor

Il caricamento standard appena descritto non offre di per sé una cache
persistente. Il progetto di prefetch predictor ha quindi introdotto, in modo
sperimentale, una banca pinned separata che mira a ottenere il comportamento
seguente:

```text
                  ┌───────────────┐
                  │  VRAM cache   │  ← expert necessari ORA
                  └───────▲───────┘
                          │
                     RAM → VRAM
                          │
                ┌─────────┴─────────┐
                │  Pinned RAM cache │  ← expert previsti per DOPO
                └─────────▲─────────┘
                          │
                     SSD → RAM
                          │
                       ┌──┴──┐
                       │ SSD │
                       └─────┘
```

mentre il path CUDA base è più vicino a:

```text
SSD
 │
 ▼
pinned staging buffer
 │
 ▼
VRAM selected-expert buffer
 │
 ▼
compute

pinned staging buffer viene poi riutilizzato
```

Questa distinzione è molto importante per il predictor.

Se il predictor stima che al prossimo token serviranno, ad esempio:

```text
{17, 42, 91, ...}
```

limitarsi a caricarli nei normali `stage[0..3]` di DS4 **non equivale a prefetcharli in una vera cache RAM**, perché quei buffer sono pensati per essere continuamente riciclati.

Nel branch del progetto questa banca separata è già una banca **a slot
persistenti e riusabili**: l'allocazione pinned rimane associata allo slot e
viene riutilizzata da richieste successive, mentre la proprietà logica dello
slot viene liberata quando il gruppo è consumato o diventa stale. Non è però
ancora una vera cache semantica di expert: non conserva né indicizza gli expert
globalmente tramite una chiave `(layer_id, expert_id)` oltre la vita del gruppo
prefetchato. Il passo architetturale successivo è quindi evolverla verso una
**vera cache persistente di expert in pinned RAM**, con una struttura
concettuale del tipo:

```text
(layer_id, expert_id) → indirizzo in pinned RAM
```

gestendo:

- cache hit;
- cache miss;
- eviction;
- politica di rimpiazzo;
- caricamento SSD → pinned RAM;
- caricamento pinned RAM → VRAM;
- eventuale prefetch asincrono guidato dal predictor.

Questa è precisamente la parte in cui il prefetch predictor può diventare utile: anticipare quali expert serviranno e cercare di averli già disponibili in RAM pinned prima che il layer/token corrispondente ne richieda il caricamento in VRAM.

---

## Configurazione record attuale — TEST_14 `output_2`

Il miglior speedup verificato finora è **+29,65%**: il tempo medio della fase
C learned-prefetch è **72,588 s**, contro **103,170 s** della baseline B sullo
stesso protocollo. Il risultato è documentato in
[`TEST_14/output_2`](../test_14/output_2/benchmark_summary.json).

La configurazione conserva il meccanismo di inferenza e il bridge Windows che
hanno reso efficace `TEST_13/output_9`; sostituisce però il predictor Build 5
con il miglior checkpoint Build 7:

```text
checkpoint: Build 7 Top-4 gated residual
state corrente fixed-NNZ: 2048D
raw hidden l:             4096D -> Dense(64) -> ReLU
raw hidden l-1:           4096D -> Dense(64) -> ReLU
q di transizione:         256D, residuale gated
fusione:                  2048D -> ReLU -> Dropout(0.3)
```

Il predictor restituisce il **Top-6** degli expert per richiesta. Questa è la
scelta record: il router richiede sei expert per layer, quindi un gruppo
predetto Top-6 può essere un hit completo oppure offrire un riuso parziale;
gli expert mancanti restano coperti dal caricamento on-demand standard.
L'esperimento Top-10 ha aumentato il volume I/O e la pressione sulla banca
pinned senza superare il risultato Top-6.

### Finestra, slot e bootstrap

La banca contiene una finestra fisica di **43 layer, da 0 a 42**, con **due
slot futuri per layer**. Ci sono quindi `43 x 2 = 86` slot logici, allocati
lazy: un buffer pinned viene materializzato soltanto quando uno slot deve
leggere un gruppo e resta poi disponibile per essere riutilizzato da un gruppo
successivo nello stesso slot.

### Struttura concreta degli slot del record

La banca CUDA non è una lista dinamica né un dizionario di expert. È un array
bidimensionale a capacità fissa:

```text
slots[layer fisico][indice di distanza]
```

Il binario può contenere al massimo `slots[43][8]`, cioè 43 layer fisici e
otto posizioni future per layer. `output_2` usa solo le prime due posizioni;
le altre sei sono capacità compilata ma inattiva in questa configurazione.

L'indice di distanza non identifica un token specifico: `[0]` non significa
necessariamente “t+1” e `[1]` non significa necessariamente “t+2”. Sono due
posti fisici in cui trattenere richieste dello stesso layer. L'identità logica
è conservata *dentro* lo slot dai campi `target_token` e `layer`.

Nel record Top-6, uno slot contiene concettualmente:

```text
slot = {
    stato,
    target_token,
    layer fisico,
    6 ID expert predetti,
    tabella degli offset gate/up/down nel GGUF,
    buffer di RAM pinned host gate/up/down,
    tempi di enqueue, inizio lettura e disponibilità,
    worker assegnato e codice di errore
}
```

Nel percorso record, il bridge Windows legge dal GGUF e invia i blocchi al
worker DS4, che riempie i buffer di RAM pinned `gate`, `up` e `down` dello
slot. Lo scratch previsto dal percorso direct-I/O non fa parte di questo
percorso Windows. Le allocazioni sono lazy alla prima lettura e rimangono
associate allo slot fino allo shutdown di DS4: riportare lo slot a `EMPTY` lo
libera logicamente, senza restituire subito la memoria host pinned al sistema.

Quando Build 7 restituisce il Top-6 per il token successivo `t+1` nel layer
`l`, DS4 cerca il primo slot `EMPTY` fra `slots[l][0]` e `slots[l][1]`. Se una
richiesta per lo stesso `target_token` è già presente, non accoda un duplicato.
Se entrambi gli slot sono occupati, registra `prefetch_bank_full` e prosegue
con il forward: la saturazione della banca è sempre non bloccante.

Per esempio, dopo il token 21 al layer `l`:

```text
slots[l][0] = {
    stato        = QUEUED, READING oppure READY,
    target_token = 22,
    expert Top-6 = [15, 90, 96, 112, 119, 202]
}
slots[l][1] = EMPTY
```

Se una nuova predizione per il token 23 arriva prima del consumo del gruppo
per il token 22, può usare il secondo posto fisico:

```text
slots[l][0] = { stato = READY,  target_token = 22, ... }
slots[l][1] = { stato = QUEUED, target_token = 23, ... }
```

Alla domanda reale `(token, layer)`, DS4 scorre entrambi gli slot del layer e
cerca `slot.target_token == token` e `slot.layer == layer`. Un gruppo `READY`
con gli stessi sei expert è un hit completo; un gruppo `READY` con soltanto
alcuni expert in comune è un hit parziale, dal quale DS4 riusa i pesi
coincidenti dalla RAM pinned e carica on-demand solo i mancanti. Uno slot non
pronto, stale, fallito o senza intersezione utile lascia invariato il fallback
standard. Il prefetch non modifica mai gli expert scelti dal router reale né
ritarda il forward quando la previsione non è utilizzabile.

Le richieste learned sono emesse per i layer fisici **5--42**. I layer 0--2
sono deterministici; i layer 3 e 4 costituiscono il bootstrap necessario a
Build 7, che per predire il layer 5 usa congiuntamente `hidden(l)` e
`hidden(l-1)`. Per 3 e 4 non viene quindi emesso un prefetch learned: i loro
expert vengono risolti dal percorso on-demand, ma i relativi hidden vengono
comunque prodotti e conservati per l'inferenza del predictor.

```text
token futuro / layer 5--42
        |
        v
Build 7 Top-4 -> sei expert predetti
        |
        v
slot (layer, distanza futura) nella banca pinned
        |
        +-- worker --> bridge Windows: richiesta layer + expert Top-6
        |   bridge legge gate/up/down dal GGUF e li invia via socket
        |   worker riceve nei buffer di RAM pinned dello slot
        v
READY -> upload pinned RAM -> VRAM al momento del forward
        |
        +-- hit completo: sei expert riusati
        +-- hit parziale: riuso degli expert coincidenti + fallback on-demand
        +-- non pronto/miss: fallback on-demand
```

Ogni slot ha uno stato (`EMPTY`, `QUEUED`, `READING`, `READY`, `CONSUMING`,
`TOO_LATE` o `FAILED`). Lo stato descrive chi possiede logicamente lo slot e
se i suoi buffer pinned possono essere usati dal decode:

| Stato | Significato operativo | Cosa può fare il decode |
|---|---|---|
| `EMPTY` | Lo slot non contiene una richiesta valida. I buffer pinned, se sono già stati allocati, restano associati fisicamente allo slot ma la sua proprietà logica è libera per un nuovo `(token target, layer)`. | Non può riusarlo; continua con fallback oppure con gli altri slot. |
| `QUEUED` | Il predictor ha inserito token, layer, Top-K e offset GGUF; nessun worker ha ancora iniziato l'I/O su quei buffer. | Non attende. Se il token target è ormai passato, può scartarlo subito e renderlo `EMPTY`. |
| `READING` | Un worker ha preso in carico lo slot e sta riempiendo i suoi buffer pinned tramite il bridge Windows. | Non usa i dati ancora incompleti. Se quel forward è ormai arrivato, lo marca `TOO_LATE` e prosegue con fallback. |
| `READY` | La lettura di `gate`, `up` e `down` è terminata con successo; i pesi predetti sono disponibili nella RAM pinned dello slot. | Cerca gli expert coincidenti con il routing reale, li trasferisce in VRAM e, in caso di hit parziale, carica on-demand solo quelli mancanti. |
| `CONSUMING` | Il decode ha riservato temporaneamente uno slot `READY` per l'upload pinned RAM → VRAM, completo o parziale. | Lo sta già usando; nessun worker può sovrascriverlo. Al termine dell'upload lo slot torna `EMPTY`. |
| `FAILED` | Il worker ha terminato l'I/O con errore e i dati dello slot non sono affidabili. | Non riusa nulla; applica il fallback. Quando quello slot diventa vecchio viene liberato a `EMPTY`. |
| `TOO_LATE` | Il target `(token, layer)` è già stato superato mentre un worker era ancora in `READING`. Il worker conserva la proprietà dei buffer fino alla conclusione, per non scrivere in uno slot riutilizzato. | Non attende né riusa il risultato; usa il fallback. Alla fine della lettura il worker rende lo slot `EMPTY`. |

### Transizioni della banca: chi agisce e quando

La tabella seguente riprende la struttura descritta in `TEST_9/README.md`, ma
è aggiornata al percorso record Build 7: il **THREAD DI DECODE DS4** inoltra il
lavoro predittivo, mentre il **thread asincrono di prefetch DS4** riceve la
risposta del servizio modello prefetch Build 7 e accoda lo slot. Include anche
i rami di errore e di riciclo stale presenti nel codice attuale.

| Transizione implementata | Chi la esegue | Quando |
|---|---|---|
| `EMPTY → QUEUED` | thread asincrono di prefetch DS4 | Dopo la risposta del servizio modello prefetch Build 7, inserisce nello slot libero `(token target, layer, Top-6)` della previsione learned. |
| `QUEUED → READING` | worker di I/O | Trova una richiesta della propria partizione `layer % 6`, la prende in carico e inizia la lettura tramite il bridge Windows verso i buffer di RAM pinned dello slot. |
| `QUEUED → EMPTY` | THREAD DI DECODE DS4 oppure worker di I/O | La richiesta è diventata stale prima che il worker iniziasse l'I/O. Il decode la libera quando supera il token target; il worker può fare lo stesso se rileva che il token è già passato durante la propria scansione. I buffer non sono ancora in scrittura, quindi il riciclo è sicuro. |
| `READING → TOO_LATE` | THREAD DI DECODE DS4 | Raggiunge quel `(token target, layer)` prima che il worker abbia terminato la lettura. Non può liberare subito lo slot, perché il worker sta ancora scrivendo nei suoi buffer di RAM pinned. |
| `TOO_LATE → EMPTY` | worker di I/O | Conclude l'I/O ormai tardivo, scarta il risultato e libera logicamente lo slot senza pubblicarlo. |
| `READING → READY` | worker di I/O | Completa con successo una lettura ancora utile e pubblica `gate`, `up` e `down` nei buffer di RAM pinned dello slot. |
| `READING → FAILED` | worker di I/O | La lettura o il bridge Windows segnalano un errore: il worker registra l'esito e non pubblica dati riutilizzabili. |
| `READY → CONSUMING` | THREAD DI DECODE DS4 | Al forward reale trova uno slot del proprio `(token, layer)` con almeno un expert coincidente; riserva lo slot prima dell'upload RAM pinned → VRAM, completo o parziale. |
| `CONSUMING → EMPTY` | THREAD DI DECODE DS4 | Dopo aver sincronizzato la copia GPU e completato l'eventuale fallback degli expert mancanti, libera lo slot per una nuova richiesta. |
| `READY → EMPTY` oppure `FAILED → EMPTY` | THREAD DI DECODE DS4 | Lo slot è diventato stale senza essere stato consumato, oppure ha fallito e il token target è ormai passato; il decode lo scarta e lo rende riutilizzabile. |

Il decode non attende mai la lettura speculativa. Uno slot `READY` con hit
parziale è comunque utile e fornisce gli expert coincidenti; uno slot assente,
   `QUEUED`, `READING`, `FAILED` o `TOO_LATE` non blocca il forward e lascia
   operare il fallback standard.

### Ciclo di lavoro di un worker pinned

Nel record i sei worker sono **worker di I/O**: non eseguono il modello Build 7,
non decidono il routing reale e non effettuano l'upload in VRAM. Ricevono invece
richieste già prodotte dal thread asincrono di prefetch DS4 e trasferiscono nei
buffer pinned dello slot i tre tensori dell'expert predetto (`gate`, `up` e
`down`). Il loro ciclo effettivo è il seguente:

0. **Decode: il consumatore del prefetch e l'autorità sul routing reale.** Il
   thread di decode DS4 avanza in ordine il forward del token corrente attraverso i
   layer fisici del modello. Quando arriva a un layer MoE, calcola il routing
   reale e quindi gli expert che servono *davvero* per quel forward; né il
   predictor Build 7 né i worker possono sostituire questa decisione. Il decode
   cerca nella banca uno slot con la stessa coppia `(token target, layer)` e,
   se è `READY`, riusa tutti gli expert coincidenti dalla RAM pinned caricandoli
   in VRAM; se la corrispondenza è parziale, recupera on-demand soltanto gli
   expert reali mancanti. Il fallback completo si applica invece quando lo slot
   manca o non è pronto. Per questo il decode non attende mai l'I/O speculativo;
   se trova uno slot ancora `READING` ormai necessario, lo marca `TOO_LATE`
   sotto mutex affinché il worker non lo pubblichi inutilmente al termine della
   lettura.

   Il decode ha anche un secondo compito, distinto dal routing reale: quando
   per il layer corrente sono disponibili gli input richiesti dal predictor
   (state, hidden corrente, hidden precedente e probabilità di transizione),
   costruisce un job di prefetch e lo inserisce nella coda del **thread
   asincrono di prefetch DS4**. Non attende né l'inferenza né la lettura: dopo
   avere segnalato il job, continua immediatamente il forward.

   Il thread asincrono preleva il job dalla propria coda, lo invia via socket al
   **servizio modello prefetch Build 7** e attende la risposta Top-6 fuori dal
   percorso critico del decode. In TEST_14 questo servizio è un processo Python
   avviato automaticamente dal runner sul lato Linux/WSL: mantiene caricato il
   checkpoint Build 7 e risponde alle richieste di inferenza, ma non legge i
   pesi degli expert dal GGUF. Solo dopo avere ricevuto la risposta il thread
   asincrono acquisisce
   il mutex della banca, cerca uno slot `EMPTY` compatibile, registra token
   target, layer, Top-6 e metadati necessari alla lettura GGUF, quindi imposta
   lo slot a `QUEUED` e notifica i worker. Questo è il passaggio che apre il
   punto 1, dove la previsione diventa lavoro concreto per un worker di I/O.

   **Esempio, il punto di vista del decode.**

   ```text
   THREAD DI DECODE DS4 — token corrente 50, layer fisico 12
   |
   |  1. Consuma il prefetch preparato in precedenza per (token 50, layer 12)
   |      |
   |      v
   |  router reale --> expert realmente richiesti dal token 50
   |      |
   |      v
   |  cerca slot (token 50, layer 12) nella banca pinned
   |      |
   |      +--> READY    --> riuso expert coincidenti + fallback solo per i mancanti
   |      +--> READING  --> TOO_LATE + fallback immediato, senza attesa
   |      +--> assente / FAILED --> fallback immediato
   |
   |  2. Completa il forward reale del layer 12 per il token 50
   |      |
   |      v
   |  dispone di state, hidden(12), hidden(11) e q del token 50
   |
   |  3. Prepara il prefetch per il token futuro 51, stesso layer 12
   |      |
   |      v
   |  inserisce il job nella coda del thread asincrono di prefetch DS4
   |      |
   |      +--> il decode non attende e prosegue
   |
   |  THREAD ASINCRONO DI PREFETCH DS4
   |      |
   |      +--> preleva il job dalla coda
   |      +--> richiesta socket --> servizio modello prefetch Build 7
   |      +--> riceve risposta Top-6
   |      +--> mutex della banca --> accoda lo slot `QUEUED` (vedi punto 1)
   |
   +--> il decode continua subito con il layer successivo del token 50
   ```

1. **Attesa e ricerca del lavoro.** Il worker prende il mutex della banca prima
   di scandire gli slot attivi. Il lock è necessario perché, nello stesso
   istante, il thread asincrono di prefetch DS4 può accodare una richiesta, il
   decode può
   consumare o invalidare uno slot e un altro worker può completare una lettura:
   senza una vista coerente si potrebbe selezionare uno slot già cambiato.
   Ciascun *singolo* worker ha un proprio `worker_id` tra 0 e 5 e considera
   soltanto i layer per cui `layer % 6 == worker_id`; per ogni layer assegnato
   controlla i suoi due slot alle distanze future. Nel record non esiste una coda
   globale che riassegna liberamente il lavoro: questa partizione mantiene
   stabile l'affinità layer--worker. Se non trova uno slot `QUEUED`, rilascia il
   mutex mentre attende sulla condition variable, quindi non fa polling e non
   blocca le altre attività della banca.

   **Esempio, inizio del caso.**

   ```text
   THREAD DI DECODE DS4 — token corrente 50
          |
          v
   inserisce un job di prefetch nella coda del thread asincrono di prefetch DS4
          |
          +--> il decode prosegue senza attesa
   
   THREAD ASINCRONO DI PREFETCH DS4
          |
          +--> preleva il job dalla coda
          +--> invia una richiesta socket al servizio modello prefetch Build 7
          +--> riceve Top-6 per token futuro 51, layer fisico 12
          +--> acquisisce il mutex della banca
          +--> registra token, layer, Top-6 e metadati GGUF nello slot libero
          |
          v
   banca[12][distanza libera] = QUEUED
          |
          +--> 12 % 6 = 0  --> il lavoro appartiene al worker 0
                                      |
                                      v
                              mutex della banca acquisito
                                      |
                                      +--> vede lo slot 12 come QUEUED
                                      +--> vede in modo coerente anche gli
                                           aggiornamenti concorrenti della banca
   ```

   È quindi il **thread asincrono di prefetch DS4**, dopo aver ricevuto la
   risposta di Build 7, a impostare lo slot a `QUEUED` sotto mutex e a eseguire
   `pthread_cond_signal` sulla condition variable globale della banca. Lo fa
   perché la predizione da sola contiene soltanto gli identificativi degli
   expert: lo stato `QUEUED` trasforma quella predizione in un lavoro concreto
   di I/O ancora da svolgere. Nello stesso istante questo thread può accodare il
   layer 13 e il thread di decode DS4 può liberare uno slot del layer 6. Il
   mutex non sceglie il lavoro al posto del worker: permette a ciascuno di
   osservare questi cambiamenti senza leggere uno stato a metà aggiornamento.

   Il segnale non è indirizzato direttamente al worker 0: una condition
   variable POSIX può risvegliare **un worker in attesa qualunque**. Il worker
   risvegliato riacquisisce il mutex e applica poi la propria regola `layer % 6`.
   Se non possiede il layer 12, non prende quello slot e torna in attesa; se è
   il worker 0, lo porta a `READING`. Se invece il segnale risveglia un worker
   non assegnato al layer 12, il worker 0 può restare addormentato fino al
   successivo `pthread_cond_signal` prodotto da una nuova richiesta oppure a un
   `pthread_cond_broadcast` prodotto dal completamento/consumo di un altro
   slot. A basso carico, quando non esistono altri eventi, questo può ritardare
   davvero l'avvio dell'I/O corretto. Non compromette la correttezza — il decode
   userà il fallback se necessario — ma può ridurre la probabilità che lo slot
   diventi `READY` in tempo utile. Una condition variable per shard, segnalata
   direttamente al worker `layer % N`, sarebbe una possibile evoluzione
   coerente con i lock per shard discussi in fondo al documento.

2. **Presa in carico atomica.** Quando trova una richiesta, sempre sotto mutex,
   porta lo slot da `QUEUED` a `READING`, registra l'identificativo del worker e
   l'istante di inizio; l'audit conserva anche l'evento. Il mutex rende
   indivisibile il controllo dello stato e la sua transizione: due worker non
   possono quindi entrambi ritenere di possedere la stessa richiesta. Da quel
   momento lo slot non può essere assegnato a un altro worker. Il decode può
   tuttavia segnalarlo `TOO_LATE` se il forward lo raggiunge prima della
   conclusione della lettura.

   **Esempio, presa in carico dello stesso slot.**

   ```text
   worker 0 + mutex
          |
          v
   slot (token 51, layer 12): QUEUED
          |
          +--> registra worker_id = 0 e tempo di avvio
          |
          v
   slot (token 51, layer 12): READING
          |
          v
   mutex rilasciato
          |
          +--> un altro worker vede READING: non può prenderlo
          +--> il decode può solo segnalarlo TOO_LATE, mai aspettarlo
   ```

   Il passaggio di stato avviene sotto mutex come un'unica operazione: non può
   esistere un istante in cui due worker considerano contemporaneamente
   disponibile la medesima richiesta.

   Nella configurazione che ha ottenuto il record — **TEST_14 `output_2`**, con
   prefetch Top-6 e sei worker — la partizione fissa `layer % 6` assegna già
   ogni layer a un solo worker: per esempio, nessun worker diverso dal worker 0
   può prendere gli slot del layer 12. Per la proprietà di quei singoli slot il
   worker 0 deve quindi coordinarsi soprattutto con il thread di decode DS4 e
   con il thread asincrono di prefetch DS4. L'unico mutex globale attuale
   serializza comunque anche i worker che stanno operando su layer diversi; il
   **miglioramento architetturale 1** discute i lock per shard `layer % N`
   proprio per eliminare questa sincronizzazione superflua senza rinunciare
   alla sicurezza sugli slot.

3. **Lettura senza trattenere il lock.** Dopo aver riservato in modo sicuro lo
   slot, il worker rilascia il mutex prima dell'I/O: il lock serve a proteggere
   metadati e transizioni di stato condivisi, non a proteggere per tutta la
   durata i buffer già esclusivamente associati a quello slot. Gli altri cinque
   worker possono quindi leggere in parallelo i propri slot; mantenere il mutex
   durante l'I/O renderebbe invece seriali le letture e annullerebbe il
   vantaggio della banca concorrente.

   Il **bridge Windows** è il processo Python nativo sul lato Windows, distinto
   sia dal **servizio modello prefetch Build 7** sia dai worker di I/O DS4. Il
   servizio Build 7, avviato automaticamente dal runner in Linux/WSL, fornisce
   soltanto gli identificativi degli expert predetti. Il bridge Windows, invece,
   è il processo da avviare separatamente in PowerShell: legge i loro pesi dal
   GGUF sull'SSD.
   Il worker, dopo avere riservato lo slot, si assicura che i suoi buffer di
   **RAM pinned** siano disponibili (sono allocati lazy alla prima lettura e poi
   riutilizzati dallo stesso slot), apre/usa la connessione socket e invia al
   bridge il layer e i sei identificativi Top-6.

   Il bridge usa layer e identificativi per ricavare nel GGUF gli offset di
   `gate`, `up` e `down`, legge i relativi blocchi dall'SSD e invia i byte di
   risposta sul socket. Non riceve un puntatore né un "canale della RAM pinned":
   quella memoria appartiene al processo DS4 in WSL e non è indirizzabile dal
   processo Windows. È quindi il worker che chiama la ricezione socket indicando
   come destinazione i buffer di RAM pinned del proprio slot. Il bridge non
   accede alla VRAM; l'upload RAM pinned → VRAM avviene soltanto più tardi, nel
   forward reale, se lo slot viene pubblicato `READY`.

   **Esempio, lettura dello stesso slot.**

   ```text
   worker 0, slot (token 51, layer 12) già READING
          |
          +--> mutex già rilasciato
          |
          v
   worker 0 --> verifica/riusa i buffer di RAM pinned dello slot [12][...]
          |
          v
   worker 0 --> richiesta socket al bridge: layer 12 + 6 ID expert Top-6
          |
          v
   bridge Windows --> ricava offset nel GGUF e legge gate / up / down dall'SSD
          |
          v
   bridge Windows --> invia i byte gate / up / down sul socket
          |
          v
   worker 0 --> recv() nei buffer di RAM pinned dello slot [12][...]
          |
          +--> in parallelo: worker 1 legge il layer 13
          +--> in parallelo: worker 2 legge il layer 14
          +--> il decode può continuare a consultare la banca
   ```

   Il risultato non è ancora utilizzabile dal forward: fino alla pubblicazione
   finale lo slot resta `READING`. La concorrenza è possibile perché ciascun
   worker riempie buffer appartenenti a uno slot distinto.

4. **Pubblicazione dell'esito.** Questo punto riprende esattamente dal termine
   del `recv()` mostrato al punto 3: il bridge Windows ha inviato tutti i byte e
   il worker ha riempito i buffer di RAM pinned dello **stesso slot**, che è
   ancora `READING`. Il worker riacquisisce il mutex perché il decode potrebbe
   avere nel frattempo consumato, invalidato o marcato tardivo quello slot. Solo
   sotto lock può verificare in modo affidabile se è ancora `READING`, renderlo
   `READY` in caso di successo o `FAILED` in caso di errore, registrare il tempo
   di disponibilità e svegliare chi sta attendendo. Se nel frattempo il decode
   lo ha marcato `TOO_LATE`, non lo pubblica come pronto: annota il completamento
   tardivo e lo riporta a `EMPTY`, evitando che un risultato ormai fuori tempo
   venga riutilizzato.

   **Esempio, esito dello stesso slot.**

   ```text
   punto 3 concluso: worker 0 ha ricevuto tutti i byte del layer 12
          |
          v
   buffer di RAM pinned dello slot (token 51, layer 12) completi
          |
          v
   worker 0 riacquisisce il mutex
          |
          +--> slot ancora READING
          |       |
          |       v
          |   READY --> slot pubblicato: il decode potrà consultarlo al forward
          |
          +--> slot già TOO_LATE
                  |
                  v
              EMPTY --> pesi arrivati dopo il forward richiesto
   ```

   Il mutex finale è ciò che distingue in modo affidabile i due rami: evita che
   il worker pubblichi come `READY` un risultato che il decode ha già giudicato
   tardivo.

   La ricezione terminata non avvia da sola alcun upload in VRAM: il worker ha
   soltanto pubblicato lo slot come `READY`. Il ciclo completo del **THREAD DI
   DECODE DS4** — forward reale, router reale e richiesta del prefetch futuro —
   è descritto al **punto 1**. Qui si chiude quel flusso: quando il decode
   esegue davvero il forward del token 51 e raggiunge il layer 12, il router
   reale conosce i sei expert effettivamente necessari e il decode cerca lo slot
   `(token 51, layer 12)`:

   ```text
   THREAD DI DECODE DS4: forward reale token 51 / layer 12
          |
          v
   router reale --> 6 expert effettivamente richiesti
          |
          v
   cerca slot (token 51, layer 12) nella banca di RAM pinned
          |
          +--> READY
          |      |
          |      +--> confronta Top-6 predetto e 6 expert reali
          |      +--> expert coincidenti: RAM pinned --> VRAM
          |      +--> expert mancanti: fallback on-demand
          |
          +--> assente / QUEUED / READING / FAILED / TOO_LATE
                 |
                 +--> fallback on-demand per tutti gli expert necessari
   ```

   Se trova uno slot `READY` utile, è il **THREAD DI DECODE DS4** a riportarlo
   disponibile per il riuso. Prima lo marca `CONSUMING`, così nessun worker può
   sovrascriverne i buffer mentre gli expert coincidenti vengono copiati dalla
   RAM pinned alla VRAM; completa poi gli expert mancanti con il fallback
   on-demand. Soltanto dopo il completamento della copia GPU lo riporta a
   `EMPTY` e chiama `pthread_cond_broadcast(...)`, rendendolo disponibile per
   una nuova richiesta e risvegliando i worker eventualmente in attesa:

   ```text
   READY
      |
      v
   THREAD DI DECODE DS4 --> CONSUMING
      |                       |
      |                       +--> expert coincidenti: RAM pinned --> VRAM
      |                       +--> expert mancanti: fallback on-demand
      |                       +--> attende il completamento della copia GPU
      v
   EMPTY --> slot riutilizzabile per una nuova richiesta
   ```

   Non viene quindi liberato immediatamente dopo l'avvio di `cudaMemcpyAsync`:
   il codice sincronizza lo stream prima di permettere il riuso dei buffer di
   RAM pinned, evitando che una nuova lettura li sovrascriva mentre la GPU li
   sta ancora leggendo. L'eccezione è uno slot `READING` che il decode ha già
   marcato `TOO_LATE`: non viene consumato dal decode e sarà il worker, al
   termine della sua lettura, a riportarlo in sicurezza a `EMPTY`.

   Il broadcast non sceglie un worker specifico: risveglia tutti quelli in
   attesa sulla condition variable della banca. Ciascuno riacquisisce poi il
   mutex uno alla volta e cerca lavoro solo nella propria partizione
   `layer % 6`; se non trova uno slot `QUEUED`, torna ad attendere. In questo
   caso `EMPTY` non crea di per sé una nuova lettura: il broadcast attuale è
   una notifica generica del cambiamento di stato, non l'assegnazione di un
   lavoro preciso.

   Con il **miglioramento architetturale 2** — una condition variable per shard
   `layer % N` — la notifica mirata sarebbe soprattutto quella prodotta quando
   il thread asincrono di prefetch DS4 accoda uno slot `QUEUED`: segnalerebbe
   soltanto la condition dello shard del layer e risveglierebbe quindi il worker
   che ne è responsabile. Quando il decode libera uno slot a `EMPTY`, invece,
   non sarebbe normalmente necessario risvegliare alcun worker, salvo che una
   futura progettazione introduca produttori in attesa di capacità in quello
   shard. Questo evita risvegli inutili senza cambiare la sicurezza degli slot.

In forma compatta, il percorso normale è:

```text
QUEUED -- worker assegnato dal layer % 6 --> READING
       -- bridge Windows, fuori dal mutex --> READY
       -- forward reale --> upload pinned RAM -> VRAM / riuso parziale

READING -- il forward arriva prima --> TOO_LATE -- worker conclude --> EMPTY
```

Questo spiega perché aumentare i worker non modifica il Top-K né il routing: può
solo ridurre il tempo necessario a trasformare più slot `QUEUED` in slot
`READY` prima che il forward reale ne abbia bisogno.

### Sei worker e perché sono parte del record

Le letture speculative vengono svolte da **sei worker** indipendenti. La
partizione è `layer % 6`: ciascun worker è responsabile di sei o sette layer
della finestra learned 5--42. La selezione dello slot avviene sotto mutex,
mentre le letture Windows/SSD e il riempimento dei buffer pinned avvengono in
parallelo su slot distinti.

Questa scelta riduce il tempo di attesa in coda senza aumentare inutilmente la
concorrenza I/O. Con Top-6, due slot per layer e sei worker il benchmark ha
osservato in media circa **7.170 expert** riusati dalla RAM pinned per fase C,
ottenendo il record attuale. Il risultato non implica che tutti gli expert
richiesti siano sempre già pronti: misura proprio il compromesso riuscito tra
qualità predittiva, capacità della banca, banda SSD/bridge e parallelismo dei
worker.

### Approfondimento operativo: una richiesta Top-6, dall'inferenza al forward

La configurazione record separa volutamente tre attività, così che la
generazione non rimanga in attesa di un I/O speculativo:

1. il thread di decode DS4 produce gli input già disponibili per il predictor;
2. il thread asincrono di prefetch DS4 comunica con il servizio modello prefetch Build 7;
3. i sei worker della banca pinned leggono e preparano i gruppi che il
   predictor ha restituito.

I sei worker non eseguono TensorFlow e non scelgono gli expert. Il loro unico
compito è trasformare una richiesta già predetta in un gruppo `READY` nella
RAM pinned. Questo evita di mettere nel percorso critico sia l'inferenza del
modello sia la lettura del GGUF.

Di seguito un esempio semplificato. Supponiamo che DS4 stia eseguendo il token
`t` al layer fisico 12. Al termine del suo calcolo dispone dello state,
`hidden(12)`, `hidden(11)` e della probabilità di transizione `q` necessari a
Build 7. L'obiettivo non è accelerare il layer 12 già in corso: è preparare lo
stesso layer per il token successivo `t + 1`.

```text
Decode token t, layer 12
        |
        | copia degli input già materializzati nella coda asincrona bounded
        v
Thread predictor Build 7
        |
        | richiesta Unix al servizio Python; il decode continua nel frattempo
        v
Top-6 predetto per (t + 1, layer 12)
        |
        | esempio: [17, 42, 91, 104, 8, 203]
        v
slot libero (layer 12, una delle due distanze future)
        |
        v
QUEUED -> worker assegnato da 12 % 6 = 0 -> READING
        |
        v
reader Windows legge gate/up/down dal GGUF nel buffer pinned dello slot
        |
        v
READY per (t + 1, layer 12)
```

Nel caso dell'esempio, il layer 12 è assegnato al worker `0`, perché `12 % 6`
vale zero. Un layer 13 sarebbe gestito dal worker `1`, e così via. Ogni worker
scansiona soltanto i layer della propria classe modulo sei; può quindi leggere
in parallelo agli altri cinque worker, ma non entra in conflitto con loro sullo
stesso slot. La scelta e il cambio di stato dello slot sono sincronizzati;
lettura e riempimento del buffer avvengono invece fuori da quella breve sezione
critica, in modo indipendente per ogni slot.

Quando il decode raggiunge davvero `(t + 1, layer 12)`, il router reale produce
i sei expert effettivamente necessari. DS4 controlla allora i due slot del
layer 12 associati a quel token futuro.

```text
Expert reali del router:       [17, 42, 91, 77, 8, 203]
Gruppo READY predetto:         [17, 42, 91, 104, 8, 203]
                                └──── cinque expert in comune ────┘
```

In questo esempio il gruppo non è un hit completo, ma è un **hit parziale**:
DS4 trasferisce in VRAM dal buffer pinned i cinque expert coincidenti e usa il
loader on-demand solo per l'expert `77` mancante. Se tutti e sei coincidono,
il gruppo `READY` copre interamente il forward; se non coincide alcun expert,
oppure il gruppo è ancora `QUEUED`/`READING`, il forward non attende e usa
subito il percorso standard SSD → staging pinned → VRAM.

Dopo l'upload, lo slot passa temporaneamente a `CONSUMING`, quindi torna
`EMPTY` dal punto di vista logico. Le sue allocazioni pinned non vengono
rilasciate: resteranno riutilizzabili per la prossima richiesta assegnata a
quella stessa posizione `(layer, distanza futura)`. Se invece il decode supera
il token previsto prima che una lettura sia completata, una richiesta ancora
in coda può essere scartata immediatamente; una richiesta già in lettura viene
marcata `TOO_LATE` e liberata dal worker non appena termina in sicurezza.

Le due distanze future per layer sono importanti proprio in questa sequenza:
permettono di non sovrascrivere un gruppo già `READY` per `t + 1` mentre una
nuova predizione viene preparata per un token più avanti. Se entrambi gli slot
di un layer sono occupati, DS4 registra la banca piena e lascia che quel caso
segua il fallback on-demand: la correttezza del forward prevale sempre sul
prefetch.

## Miglioramenti architetturali

### 1. Lock per shard `layer % N`

#### Problema: contesa sul mutex globale della banca

La configurazione record usa un **unico mutex globale** per proteggere gli
stati e i metadati di tutti gli slot della banca pinned. Il lock viene tenuto
solo per operazioni brevi — scelta di uno slot, transizioni come
`QUEUED → READING` o `READING → READY`, accodamento della predizione e
liberazione/invalidazione da parte del decode — e viene rilasciato durante
l'I/O Windows/SSD. Per questo i sei worker possono già leggere in parallelo i
propri buffer pinned.

Resta però una possibile limitazione al crescere del numero di worker: ogni
worker aggiuntivo aumenta il numero di accessi allo stesso mutex per prendere
o pubblicare lavoro. La contesa non è soltanto tra worker: sul medesimo lock
intervengono anche il **thread di decode DS4**, che consulta, consuma, libera o
marca `TOO_LATE` gli slot, e il **thread asincrono di prefetch DS4**, che
accoda le nuove predizioni come `QUEUED`. Anche se i worker operano stabilmente
su layer diversi, tutte queste brevi operazioni di coordinamento restano
serializzate dal lock globale. Di conseguenza, oltre un certo punto, più worker
potrebbero non tradursi in più slot `READY` in tempo utile; il limite potrebbe
dipendere dalla contesa sul mutex, oltre che dalla banda del bridge Windows,
dell'SSD, della RAM pinned, del PCIe e della GPU.

#### Soluzione proposta

Una possibile evoluzione consiste nel dividere la banca in `N` shard, con un
mutex distinto per ogni classe `layer % N`. Con sei worker, per esempio, il
worker 0 coordinerebbe solo i layer `0, 6, 12, …`, il worker 1 i layer
`1, 7, 13, …` e così via. Il thread di decode DS4 e il thread asincrono di
prefetch DS4 acquisirebbero il lock dello shard relativo al layer che stanno
consultando o accodando. In questo modo worker assegnati a shard diversi non
dovrebbero competere tra loro durante le transizioni di stato.

Questa soluzione richiederebbe però di riprogettare con attenzione le
condition variable, l'audit e le statistiche aggregate della banca, mantenendo
la stessa garanzia di non bloccare mai il decode. Non va quindi introdotta solo
per intuizione: **Per attribuirlo con certezza andrebbe misurato il tempo di
attesa sul mutex; solo allora avrebbe senso passare a lock per shard `layer %
N`.**

### 2. Condition variable per shard `layer % N`

#### Problema: risveglio non indirizzato della condition variable globale

La banca usa oggi una sola condition variable globale. Quando il thread
asincrono di prefetch DS4 riceve dal servizio modello prefetch Build 7 una
nuova predizione, scrive lo slot come `QUEUED` e chiama
`pthread_cond_signal`. Il segnale risveglia un worker in attesa, ma non può
selezionare il worker proprietario del layer secondo `layer % N`.

Per esempio, una richiesta per il layer 12 con sei worker appartiene al worker
0 (`12 % 6 = 0`). Il `signal` potrebbe però risvegliare il worker 3: questo
riacquisisce il mutex, scandisce soltanto il proprio shard, non trova il layer
12 e torna in attesa. Il worker 0 può quindi restare addormentato fino a una
nuova richiesta che produce un altro `signal`, oppure fino a un
`pthread_cond_broadcast` emesso quando un worker completa una lettura o il
decode consuma uno slot. Se non si verificano altri eventi, l'avvio della
lettura giusta viene ritardato.

Il comportamento non compromette la correttezza: il thread di decode DS4 non
attende mai e applica il fallback on-demand. Può però diminuire la probabilità
che lo slot diventi `READY` prima del forward reale, e quindi ridurre il
beneficio del prefetch.

Una soluzione immediata sarebbe sostituire il `pthread_cond_signal` successivo
a un nuovo `QUEUED` con `pthread_cond_broadcast`: tutti i worker si
risveglierebbero e quello assegnato al layer troverebbe certamente il lavoro.
Il costo è il *thundering herd*: anche i worker non interessati si contendono
il mutex globale solo per scoprire di non avere lavoro.

La soluzione architetturalmente più pulita è affiancare agli shard `layer % N`
una condition variable per shard. Il thread asincrono di prefetch DS4
segnalerebbe allora direttamente la condition dello shard del layer accodato;
il worker competente si risveglierebbe senza coinvolgere gli altri. Questa
evoluzione andrebbe progettata insieme ai lock per shard e verificata con
metriche di latenza tra `QUEUED`, `READING` e `READY`, per confermare che il
ritardo di risveglio limita realmente lo speedup.

### 3. Riuso degli expert già presenti nella RAM pinned tra richieste future

#### Problema: gli slot sono deduplicati per richiesta, non per expert

La banca attuale tratta ogni slot come un gruppo indivisibile associato a
`(token target, layer, Top-K)`. Quando il **thread asincrono di prefetch DS4**
riceve il Top-6 dal servizio modello prefetch Build 7, verifica già se nei due
slot del medesimo layer esiste una richiesta per **lo stesso token target**:
in quel caso non accoda un duplicato. Non confronta però gli ID degli expert
con i gruppi `READY` riferiti a token futuri diversi. Di conseguenza può
riportare dall'SSD nella RAM pinned un expert che è ancora disponibile nei
buffer di un altro slot del medesimo layer.

Esempio: il layer 12 ha due slot futuri, entrambi ancora validi.

```text
slot A: token target 51, layer 12, READY
        Top-6 = [4, 9, 23, 41, 70, 88]

arriva la previsione per:
slot B: token target 52, layer 12
        Top-6 = [9, 23, 41, 56, 91, 101]

oggi
----
thread asincrono di prefetch DS4 --> accoda B come QUEUED
worker di I/O --> bridge Windows legge tutti e 6 gli expert di B dall'SSD
                 [9, 23, 41, 56, 91, 101]

duplicazione evitabile
----------------------
[9, 23, 41] sono già nella RAM pinned dello slot A
solo [56, 91, 101] richiederebbero una nuova lettura dal GGUF
```

Il controllo non può essere svolto dal **THREAD DI DECODE DS4** prima di
accodare il job: in quel momento conosce le feature del token/layer futuro, ma
non conosce ancora il Top-6. Il punto corretto per confrontare gli expert è
quindi dopo la risposta del servizio Build 7 e prima di affidare la richiesta
al worker di I/O.

Non basta tuttavia considerare lo slot A una cache libera. Lo slot A può essere
consumato dal decode per il token 51 e tornare `EMPTY` mentre si sta preparando
lo slot B. La soluzione deve perciò proteggere la vita dei buffer e non deve
mai ritardare il forward reale.

#### Soluzione A: composizione del nuovo slot da expert riusati e mancanti

Dopo avere ottenuto il Top-6 di B, il thread asincrono cercherebbe, sotto il
mutex della banca, uno slot `READY` dello stesso layer con expert in comune.
Creerebbe comunque lo slot B separato, ma annoterebbe quali posizioni possono
essere copiate da A e quali devono essere lette. Il worker riserverebbe B,
copierebbe nella RAM pinned di B i blocchi `gate/up/down` dei tre expert
coincidenti da A e chiederebbe al bridge Windows soltanto i tre expert mancanti.
Quando B è completo, diventerebbe `READY` come oggi.

```text
slot A READY: [4, 9, 23, 41, 70, 88]
                 |   |   |
                 |   |   +--> expert 41: copia RAM pinned A --> RAM pinned B
                 |   +------> expert 23: copia RAM pinned A --> RAM pinned B
                 +----------> expert 9:  copia RAM pinned A --> RAM pinned B

slot B in preparazione: [9, 23, 41, 56, 91, 101]
                                      |   |   |
                                      +---+---+--> bridge Windows legge dal GGUF
                                                     e il worker riceve in B
```

**Perché può essere utile.** Riduce direttamente i byte richiesti al bridge,
all'SSD e alla rete locale senza cambiare il contratto finale del decode: al
momento del forward esso vede ancora un normale slot B `READY` e può riusarlo
con la logica attuale.

**Costo e limiti.** Occorre estendere il protocollo bridge per leggere un
sottoinsieme di expert e mantenere le corrette posizioni nei buffer di B. Lo
slot A deve inoltre restare protetto fino al termine della copia host-to-host:
non può essere liberato o sovrascritto dal decode nel frattempo. La soluzione
duplica comunque i byte riusati nei buffer di B e consuma due slot durante la
composizione. È dunque l'approccio meno invasivo, ma il beneficio dipende da
quanto spesso l'overlap è alto e A è ancora `READY` al momento giusto.

#### Soluzione B: cache pinned condivisa per `(layer, expert)`

In alternativa, la banca può evolvere da cache di *gruppi per token* a cache
di singoli expert. Ogni ingresso sarebbe indicizzato da `(layer, expert_id)` e
conterrebbe i buffer `gate/up/down` in RAM pinned. Gli slot futuri
conserverebbero soltanto i riferimenti ai sei ingressi necessari; B potrebbe
quindi riferirsi direttamente agli ingressi 9, 23 e 41 già residenti e
richiedere solo 56, 91 e 101.

```text
cache pinned del layer 12
    expert 9   --> buffer gate/up/down condivisi
    expert 23  --> buffer gate/up/down condivisi
    expert 41  --> buffer gate/up/down condivisi

slot A (token 51) --> riferimenti a [4, 9, 23, 41, 70, 88]
slot B (token 52) --> riferimenti a [9, 23, 41, 56, 91, 101]
                                  ^   ^   ^
                                  stessi buffer, nessuna seconda lettura
```

**Perché può essere utile.** Evita sia la seconda lettura SSD sia la copia
RAM pinned → RAM pinned, e può riusare un expert tra più token futuri dello
stesso layer. È la soluzione più efficiente se l'overlap temporale delle
predizioni è frequente.

**Costo e limiti.** Cambia in profondità la proprietà dei buffer: servono
stati per-expert, riferimenti o contatori di utilizzo, un criterio di
eviction, protezione contro la liberazione durante un upload verso VRAM e una
gestione rigorosa degli errori e dei dati `TOO_LATE`. Aumenta quindi la
complessità di concorrenza e la superficie per bug; non è una modifica da
introdurre senza misure preliminari.

#### Misura necessaria prima di scegliere

Prima di implementare una delle due opzioni conviene registrare, per ogni
layer, l'intersezione tra i Top-6 di richieste future consecutive, lo stato
degli altri slot quando arriva la nuova previsione e i byte effettivamente
risparmiabili. Solo se una quota rilevante dell'overlap è ancora `READY` in
tempo utile il costo di questa ottimizzazione può trasformarsi in uno speedup
end-to-end misurabile.
