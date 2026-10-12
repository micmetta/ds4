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

La misura è ora isolata in **TEST_15**. Riusa senza modifiche la
configurazione record di `TEST_14/output_2` (Build 7 Top-6, due slot, sei
worker), esegue A una volta solo per l'oracolo e poi tre sole fasi C. Per ogni
`prefetch_worker_queued` la trace registra anche gli ID degli expert degli
altri slot dello stesso layer; il report calcola l'intersezione, lo stato dello
slot fratello e i byte immediatamente riusabili soltanto se esso è `READY`.

Solo se una quota rilevante dell'overlap è ancora `READY` in tempo utile il
costo di questa ottimizzazione può trasformarsi in uno speedup end-to-end
misurabile. Il report di TEST_15 è dunque la base quantitativa per scegliere
fra composizione di slot e cache per-expert, non l'implementazione di una delle
due politiche.

## Approfondimento sulla soluzione B

> Questa sezione descrive un possibile sviluppo architetturale; **non è il
> comportamento attualmente implementato** dalla banca pinned record. La banca
> attuale conserva gruppi Top-6 per `(token target, layer)`. La soluzione B
> sostituirebbe la proprietà esclusiva degli expert da parte di uno slot con una
> cache di buffer condivisi, indicizzata per `(layer fisico, expert_id)`.

### Sincronizzazione per shard `layer % N`: un worker proprietario per gruppo

La cache per-expert non dovrebbe conservare un solo mutex e una sola condition
variable globali. Fissato `N` al numero dei worker di I/O (nel record, `N = 6`),
ogni layer fisico appartiene a uno shard deterministico:

```text
shard(layer) = layer % N

layer 12 -> shard 0 -> worker di I/O 0
layer 13 -> shard 1 -> worker di I/O 1
...
layer 17 -> shard 5 -> worker di I/O 5
layer 18 -> shard 0 -> worker di I/O 0
```

Ogni shard possiede la propria coda di entry `QUEUED`, il proprio mutex e la
propria condition variable. Il **THREAD DI DECODE DS4** e il **thread
asincrono di prefetch DS4** calcolano sempre `shard(layer)` prima di consultare
o aggiornare una entry `(layer, expert)`; prendono quindi soltanto il lock di
quello shard. Il worker associato è l'unico worker di I/O che estrae lavori da
quella coda e attende sulla sua condition variable.

```text
THREAD DI DECODE DS4: richiesta futura per layer 12
        |
        v
thread asincrono di prefetch DS4
        |
        +--> shard(12) = 12 % 6 = 0
        +--> lock shard 0: crea/aggiorna le entry (12, expert)
        +--> accoda solo i miss nello shard 0
        +--> signal condition variable dello shard 0
        v
worker di I/O 0
        |
        +--> prende il mutex dello shard 0: verifica la coda e riserva
        |    atomicamente una entry QUEUED
        |    (serve a impedire che decode o thread asincrono modifichino,
        |     riusino o espellano proprio quell'entry mentre il worker la prende)
        +--> QUEUED -> READING, poi rilascia il mutex
        +--> bridge Windows
```

Così il lavoro del layer 12 non può risvegliare per errore il worker 1--5, né
deve contendere il mutex della propria coda con essi. La concorrenza resta
intenzionalmente possibile **dentro il singolo shard**: decode, thread
asincrono e il suo worker devono coordinare stato, lease, riferimenti futuri e
possibile riuso dei buffer della medesima entry. Operazioni davvero globali, come il
pool di capacità pinned o le statistiche aggregate, richiederebbero una
sincronizzazione separata e breve; non devono riportare la scansione ordinaria
degli shard sotto un lock unico.

Il worker non tiene il mutex mentre attende il bridge Windows né mentre i byte
sono trasferiti nei buffer RAM pinned: in quel periodo non deve proteggere
alcuna modifica ai metadati, e tenere il lock bloccherebbe inutilmente decode e
thread asincrono del medesimo shard. Lo riprende solo per operazioni indivisibili
sullo stato condiviso: scegliere/rimuovere una `QUEUED`, pubblicare
`READING → READY` o `READING → FAILED` e aggiornare i riferimenti. Il riuso di
un buffer per un nuovo expert è una sostituzione atomica della chiave sotto
questo stesso lock: non esiste uno stato intermedio osservabile dagli altri
attori.

#### Risveglio del worker: il segnale non è il lavoro

La condition variable non trasporta una richiesta e non deve essere trattata
come una coda di eventi. Il lavoro reale è l'entry `QUEUED`, che resta
materialmente nella coda dello shard finché il suo worker non la riserva. Il
**thread asincrono di prefetch DS4** deve quindi, sotto il mutex dello shard,
prima inserire l'entry nella coda e poi eseguire `pthread_cond_signal()` sulla
condition variable di quello stesso shard.

Il worker attende sempre ricontrollando il predicato “la mia coda contiene
almeno una `QUEUED`” in un ciclo `while`:

```text
worker di I/O 0
        |
        +--> lock mutex shard 0
        |
        +--> finché la coda dello shard 0 è vuota:
        |      pthread_cond_wait(cond shard 0, mutex shard 0)
        |
        +--> trova una QUEUED, la riserva: QUEUED -> READING
        +--> unlock mutex shard 0
        v
bridge Windows
```

`pthread_cond_wait()` rilascia il mutex e mette il worker in attesa in modo
atomico. Perciò non esiste una finestra in cui il thread asincrono possa
inserire il lavoro e il worker possa poi addormentarsi senza averlo visto.
Anche un risveglio spurio è innocuo: il `while` ricontrolla la coda prima di
prendere lavoro.

Può accadere che il worker sia stato segnalato ma non riesca subito a
riacquisire il mutex perché il **THREAD DI DECODE DS4** sta aggiornando lease o
riferimenti futuri dello stesso shard. In tal caso il worker resta bloccato
sul mutex; **non torna a dormire e non perde il segnale**. Quando il decode
rilascia il lock, il worker lo acquisisce, trova ancora la `QUEUED` persistente
e inizia la lettura:

```text
thread asincrono di prefetch DS4
        |
        +--> lock shard 0
        +--> inserisce (12, X) = QUEUED nella coda shard 0
        +--> pthread_cond_signal(cond shard 0)
        +--> unlock shard 0
        |
        v
worker di I/O 0 si risveglia
        |
        +--> mutex shard 0 temporaneamente detenuto dal THREAD DI DECODE DS4
        |      (aggiornamento breve di lease/future_refcount)
        |
        +--> attende il mutex, non la condition variable
        |
        v
THREAD DI DECODE DS4 rilascia il mutex shard 0
        |
        v
worker di I/O 0: vede ancora (12, X) = QUEUED
        |
        +--> QUEUED -> READING
```

Questo può introdurre soltanto un ritardo breve, non la perdita del lavoro. È
proprio per limitarlo che il decode non deve trattenere il mutex durante
l'upload RAM pinned → VRAM, né il worker durante I/O bridge/SSD: il lock serve
solo alle transizioni e agli aggiornamenti di metadati.

### Il token pianifica il prefetch, ma non identifica il buffer dell'expert

Un expert di un dato layer ha gli stessi pesi per tutti i token. Perciò la
chiave corretta della cache non deve contenere il token:

```text
richiesta futura
  (token target 52, layer 12, Top-6 predetto)
                         |
                         v
cache pinned condivisa
  chiave (layer 12, expert 9)  --> buffer RAM pinned gate/up/down
  chiave (layer 12, expert 23) --> buffer RAM pinned gate/up/down
  chiave (layer 12, expert 41) --> buffer RAM pinned gate/up/down
```

Il token target resta utile nel record della **richiesta futura**: permette al
thread asincrono di prefetch DS4 di sapere per quale forward la previsione è
stata prodotta, di assegnare una priorità e di scrivere l'audit. Non è invece
necessario per cercare un expert già caricato. Quando il **THREAD DI DECODE
DS4** raggiunge davvero il layer 12 del token 52, il router reale produce i sei
expert necessari e il decode cerca semplicemente le sei chiavi `(12, expert)`
nella cache.

```text
THREAD DI DECODE DS4: forward reale token 52, layer 12
        |
        v
router reale --> [9, 23, 41, 56, 91, 101]
        |
        v
cache --> (12,9), (12,23), (12,41) = READY
        |       (12,56), (12,91), (12,101) = assenti/non pronte
        |
        +--> expert READY: RAM pinned --> VRAM
        +--> altri expert: fallback on-demand, senza attendere
```

Quindi una previsione precedente per il token 51 può rendere riusabile
`(12, 9)` anche per il token 52: non occorre né una lettura SSD duplicata né
una copia RAM pinned dallo slot A allo slot B. I due slot futuri conserverebbero
riferimenti allo stesso buffer cache, non due copie dei suoi byte.

### Capacità minima per layer: due richieste Top-K, fino a `2 × Top-K` entry

La cache non dovrebbe affidarsi a una sola capacità globale senza garantire
spazio ai layer che hanno già ricevuto una previsione. Mantiene infatti le due
richieste future per layer della configurazione record. Se ciascuna contiene al
massimo `Top-K` expert, la quota massima riservata a un layer è:

```text
capacità_massima_layer = 2 × Top-K

Top-K = 6
capacità_massima_layer = 2 × 6 = 12 entry cache distinte
```

Per il layer 12 ciò significa al massimo dodici porzioni di RAM pinned, una
per ciascun `(12, expert)` distinto; ogni porzione contiene i buffer
`gate/up/down` di quell'expert. Non sono dodici copie per forza: se le due
Top-6 condividono expert, ad esempio `9`, `23` e `41`, quei tre buffer sono
una sola entry condivisa e la quota effettivamente occupata è inferiore a 12.

```text
richiesta futura A, layer 12 -> [4, 9, 23, 41, 70, 88]
richiesta futura B, layer 12 -> [9, 23, 41, 56, 91, 101]

entry distinte necessarie per layer 12 =
  [4, 9, 23, 41, 70, 88, 56, 91, 101] = 9 <= 12
```

Questa quota è un limite di sicurezza, non un invito a riempire sempre dodici
entry. Evita che una nuova previsione del medesimo layer debba espellere un
expert `READY` ancora pianificato solo perché il decode non ha ancora raggiunto
il forward corrispondente. Se una terza richiesta o un insieme di expert oltre
la quota non può essere ammesso senza espellere dati protetti, il sistema non
deve forzare l'eviction: lascia quegli expert al fallback on-demand. La
capacità totale pinned andrà dimensionata in modo compatibile con la somma
delle quote dei layer learned effettivamente gestiti, non soltanto con il
numero di worker.

### Due strutture con responsabilità diverse

La separazione evita di confondere *per quale token è stata fatta una
previsione* con *quale peso è fisicamente residente*.

| Struttura | Chiave / contenuto | Responsabilità |
|---|---|---|
| Richiesta futura | `(token target, layer, Top-6 predetto)` | Pianifica il prefetch, conserva priorità e audit; può riferirsi a sei entry cache. |
| Entry della cache pinned | `(layer fisico, expert_id)` → buffer `gate/up/down` | Rappresenta una sola copia in RAM pinned dei pesi dell'expert, riusabile da più richieste e da più forward. |

Una entry dovrebbe quindi contenere almeno il proprio stato, i buffer pinned,
la dimensione, **due contatori distinti** e metadati per l'eviction, come
`last_used` o priorità. Il `token_predetto` può essere mantenuto come
diagnostica o indizio per la priorità, ma non è parte della chiave né deve
impedire il riuso tra token diversi.

| Contatore | Chi lo modifica | Perché esiste |
|---|---|---|
| `future_refcount` (`0, 1, 2, ...`) | thread asincrono di prefetch DS4, quando registra o rimuove una richiesta futura | Protegge l'entry prima del forward reale: una o più richieste future pianificate la potrebbero ancora usare. |
| `lease_count` (`0, 1, 2, ...`) | THREAD DI DECODE DS4, durante l'uso reale | Protegge l'entry soltanto mentre il decode sta trasferendo i suoi byte RAM pinned → VRAM. |

Una entry diventa candidata all'eviction esclusivamente se entrambi i
contatori sono zero:

```text
evictable = (future_refcount == 0) AND (lease_count == 0)
```

Il lease non protegge quindi una previsione appena registrata ma non ancora
raggiunta; quel compito è di `future_refcount`. Viceversa, il
`future_refcount` non sostituisce il lease: una richiesta può essere stata già
consumata mentre l'upload verso VRAM è ancora in corso.

### Stati proposti per una entry `(layer, expert)`

Gli stati non coinciderebbero esattamente con quelli degli slot attuali.
In particolare non conviene trasformare una entry `READY` in `CONSUMING`:
un buffer condiviso può essere letto dal decode per un upload verso VRAM e,
nello stesso tempo, restare disponibile per una richiesta futura. Il suo uso
temporaneo è espresso dai lease, non da uno stato esclusivo.

| Stato | Significato | Il decode può usarla? |
|---|---|---|
| `QUEUED` | Il thread asincrono di prefetch DS4 ha creato l'entry, incrementato `future_refcount` per la richiesta che la pianifica e accodato il lavoro, ma nessun worker di I/O ha ancora iniziato a leggere dal GGUF. Mantiene la stessa separazione fra accodamento e I/O della banca record attuale. | No: fallback on-demand. Una seconda richiesta può incrementare il riferimento alla stessa entry, senza accodare un duplicato. |
| `READING` | Un worker di I/O ha riservato l'entry e il bridge Windows sta leggendo `gate/up/down` dal GGUF nei suoi buffer RAM pinned. È lo stesso significato di `READING` nella banca record attuale, applicato a un singolo expert invece che a un gruppo Top-6. | No: il decode non attende la lettura e usa il fallback. Una richiesta futura può però agganciarsi alla stessa entry in caricamento, senza avviare una seconda lettura. |
| `READY` | I buffer pinned sono completi, coerenti e riusabili. Quando occorre capacità per un nuovo expert dello stesso layer, se entrambi i contatori sono zero i buffer possono essere riassegnati atomicamente alla nuova chiave. | Sì: il decode acquisisce un lease, trasferisce i byte necessari in VRAM e poi rilascia il lease. |
| `FAILED` | La lettura o il protocollo bridge è fallito; i buffer non sono affidabili. | No: fallback on-demand. Una richiesta futura può decidere un nuovo tentativo controllato. |

`ABSENT` non è uno stato memorizzato: indica soltanto che la chiave non è nella
mappa. Anche `EVICTING` non è uno stato persistente: la rimozione della vecchia
chiave e l'inserimento della nuova avvengono nella stessa sezione critica dello
shard. Può restare un evento di audit, ad esempio “`(12,9)` sostituito da
`(12,X)`”, ma non una condizione osservabile dalla cache.

### Transizioni proposte della cache condivisa

La tabella segue lo stile delle transizioni della banca attuale, ma descrive la
semantica proposta per una singola entry `(layer, expert)`. Tutte le modifiche
a stato, lease e mappa devono avvenire sotto il mutex dello **shard del
layer**, non sotto un lock globale: `shard(layer) = layer % N`. Il
trasferimento SSD → RAM pinned e RAM pinned → VRAM resta invece fuori dal
lock dello shard, come nel disegno record, per non serializzare l'I/O.

| Transizione proposta | Chi la esegue | Quando |
|---|---|---|
| Chiave assente nella mappa → `QUEUED` | thread asincrono di prefetch DS4 | Ricevuto il Top-6 dal servizio modello prefetch Build 7, calcola `shard(layer)`, prende il solo lock di quello shard e trova una chiave `(layer, expert)` non residente. Verifica prima la quota `2 × Top-K` del layer; se può ammetterla, crea una sola entry, porta `future_refcount` da `0` a `1`, la accoda nella coda dello shard e segnala la condition variable di quello shard. Se la quota è satura di entry protette, non espelle dati pianificati: lascia quell'expert al fallback. |
| `QUEUED → READING` | worker di I/O proprietario dello shard | Prende il mutex del proprio shard per rimuovere e riservare atomicamente la `QUEUED`: così decode e thread asincrono vedono subito che l'entry è in carico e non possono né accodarla di nuovo né riassegnarne i buffer a un'altra chiave. Rilascia poi il mutex e solo in quel momento avvia la richiesta al bridge Windows per leggere l'expert dal GGUF. |
| `QUEUED` o `READING` invariato, con nuovo riferimento | thread asincrono di prefetch DS4 | Una seconda richiesta futura predice lo stesso `(layer, expert)` mentre il primo lavoro è già accodato o in lettura. Sotto il lock dello stesso shard non reimposta lo stato e non avvia una seconda lettura: incrementa soltanto `future_refcount` per la stessa entry. |
| `READING → READY` | worker di I/O proprietario dello shard | Terminato il bridge Windows, riprende il mutex del proprio shard per pubblicare insieme completezza dei buffer e cambio di stato: senza questo passaggio atomico il decode potrebbe osservare `READY` prima che i metadati siano coerenti, oppure il thread asincrono potrebbe riassegnare i buffer durante la pubblicazione. Non deve risvegliare il decode: il THREAD DI DECODE DS4 non aspetta mai questa lettura e, quando consulterà la cache, userà l'entry solo se la troverà già `READY`. |
| `READING → FAILED` | worker di I/O / bridge Windows | Dopo un errore, il worker riprende il mutex del proprio shard e marca l'entry `FAILED`, così nessun altro attore può scambiarne buffer parziali per dati riusabili. |
| `READY` invariato, con lease acquisito | THREAD DI DECODE DS4 | Calcola `shard(layer)`, prende quel lock, verifica l'entry richiesta dal router reale e incrementa il lease; poi rilascia il lock e trasferisce RAM pinned → VRAM. L'entry resta `READY` e altri utenti possono ancora riferirsi agli stessi byte immutabili. |
| `READY` invariato, con lease rilasciato | THREAD DI DECODE DS4 | Terminato l'upload/uso sicuro nel forward, riprende il lock dello stesso shard e decrementa il lease. Non deve "rimettere" l'entry in `READY`, perché non ne è mai uscita; potrà essere espulsa solo dopo la rimozione di tutte le richieste future che la referenziano. |
| Rimozione del riferimento futuro | thread asincrono di prefetch DS4 / THREAD DI DECODE DS4 | Quando una richiesta futura è stata consumata o è diventata obsoleta, rimuove i suoi riferimenti alle entry del proprio layer e decrementa i rispettivi `future_refcount`, senza cambiare per forza lo stato `READY`. |
| Sostituzione atomica `READY(layer, vecchio) → QUEUED(layer, nuovo)` | thread asincrono di prefetch DS4 / gestore della cache | Serve capacità per un nuovo expert dello stesso layer. Sotto il mutex del suo shard seleziona solo un'entry con `lease_count = 0` e `future_refcount = 0`, rimuove la chiave vecchia dalla mappa, conserva i suoi buffer RAM pinned e li associa subito alla chiave nuova con `future_refcount = 1` e stato `QUEUED`. Non esistono passaggi osservabili `EVICTING` o `ABSENT`; rilasciato il lock, segnala il worker di quello shard, che farà `QUEUED → READING`. |
| `FAILED → QUEUED` | thread asincrono di prefetch DS4 | Dopo audit e pulizia dei metadati dell'errore, una nuova richiesta futura per la stessa chiave può riusare l'entry e i suoi buffer, aggiornare `future_refcount`, riportarla a `QUEUED` e segnalare il worker dello shard. Se invece occorre davvero ridurre la cache, la chiave viene semplicemente rimossa dalla mappa: non serve registrare uno stato `ABSENT`. |

### Esempio completo: riuso di un expert anche mentre un'altra richiesta lo usa

```text
token 51, layer 12: richiesta A -> [4, 9, 23, 41, 70, 88]
token 52, layer 12: richiesta B -> [9, 23, 41, 56, 91, 101]

cache del layer 12
  (12, 9)  = READY, future_refcount = 1, lease_count = 0
  (12,23)  = READY, future_refcount = 1, lease_count = 0
  (12,41)  = READY, future_refcount = 1, lease_count = 0

THREAD DI DECODE DS4 esegue A
  -> acquisisce lease per 9, 23, 41
  -> copia gli expert necessari RAM pinned -> VRAM
  -> consuma la richiesta futura A: future_refcount 1 -> 0
  -> al termine degli upload: lease_count 1 -> 0
  -> gli expert restano READY: non vengono svuotati dopo il forward A

thread asincrono di prefetch DS4 riceve B
  -> calcola shard(12) e prende soltanto il suo lock
  -> B referenzia 9, 23, 41 gia' READY: future_refcount 0 -> 1
  -> crea QUEUED soltanto per 56, 91, 101
     con future_refcount = 1, entro la quota massima 2 x Top-K = 12
  -> segnala soltanto la condition variable di shard(12)

worker di I/O proprietario di shard(12)
  -> prende il mutex del proprio shard per riservare le entry 56, 91, 101
     e portarle atomicamente QUEUED -> READING
  -> rilascia il mutex: l'I/O non deve bloccare decode e thread asincrono
  -> il bridge Windows legge soltanto 56, 91, 101 dal GGUF
  -> al termine riprende il mutex per pubblicare ogni READING -> READY

THREAD DI DECODE DS4 esegue B
  -> usa 9, 23, 41 dalla cache READY
  -> usa 56, 91, 101 se diventati READY; altrimenti fallback on-demand
  -> consuma B: rimuove i suoi future_refcount e rilascia i lease
```

#### Esempio separato: vita di una entry e significato del lease

Un **lease** è un permesso temporaneo di usare un buffer cache. Non duplica i
byte e non cambia `READY` in un altro stato: incrementa soltanto un contatore
protetto dalla cache. La sua funzione è impedire che la politica di eviction
liberi o riutilizzi i buffer RAM pinned mentre il **THREAD DI DECODE DS4** sta
ancora trasferendo quei pesi verso la VRAM. Al termine dell'upload il decode
rilascia il lease; l'entry può comunque restare `READY` per un riuso futuro.

L'esempio seguente segue la singola entry `(layer 12, expert 9)` dal primo
prefetch fino all'eventuale riassegnazione dei suoi buffer a un nuovo expert.

```text
1. La previsione richiede per la prima volta (12, 9)

THREAD DI DECODE DS4: durante il forward del token corrente
        |
        +--> delega una richiesta per un token/layer futuro
             al thread asincrono di prefetch DS4
        v
thread asincrono di prefetch DS4
        |
        +--> cache: (12, 9) assente
        |
        +--> shard(12) = 0: lock shard 0
        +--> crea l'entry e la marca QUEUED
        |    (nessun altro thread puo' creare una seconda lettura per 12,9)
        +--> registra la richiesta futura: future_refcount 0 -> 1
        +--> accoda il lavoro nello shard 0 e segnala solo la sua condition
        v
worker di I/O 0, proprietario dello shard 0
        |
        +--> prende mutex shard 0: QUEUED -> READING e riserva il lavoro
        |    (evita che decode/thread asincrono cambino o espellano l'entry
        |     mentre il worker ne assume la proprietà)
        +--> rilascia mutex shard 0
        +--> bridge Windows legge gate/up/down dal GGUF
        |    nei buffer RAM pinned della entry (12, 9)
        +--> riprende mutex shard 0: pubblica READING -> READY solo se
        |    metadati e buffer sono completi
        +--> rilascia mutex shard 0
        v
cache: (12, 9) = READY, future_refcount = 1, lease_count = 0

2. Il forward reale ne ha bisogno

THREAD DI DECODE DS4: router reale richiede expert 9 al layer 12
        |
        +--> trova (12, 9) = READY
        +--> acquisisce lease: lease_count 0 -> 1
        +--> consuma la richiesta futura: future_refcount 1 -> 0
        +--> avvia RAM pinned -> VRAM
        |
        +--> stato resta READY: una richiesta futura puo' gia' referenziare
             gli stessi byte; non viene avviata una seconda lettura SSD

3. L'uso del forward termina

THREAD DI DECODE DS4
        |
        +--> upload terminato in modo sicuro
        +--> rilascia lease: lease_count 1 -> 0
        v
cache: (12, 9) resta READY
        |
        +--> e' ancora riusabile da un altro token/layer request
        +--> non puo' essere espulsa se future_refcount > 0
        +--> diventa candidata solo se future_refcount = 0 e lease_count = 0

4. Serve spazio per un nuovo expert del layer richiesto, ma non esistono piu'
   utilizzatori dell'entry candidata: riuso diretto dei buffer

THREAD DI DECODE DS4: forward token corrente
        |
        +--> delega una richiesta futura (token target, layer 12)
             al thread asincrono di prefetch DS4
        v
thread asincrono di prefetch DS4
        |
        +--> riceve il Top-6 per il layer 12
        +--> shard(12) = 0: lavora sotto il lock dello shard 0
        +--> cerca/riusa le entry (12, expert) gia' residenti
        |
        +--> se serve capacita' nel pool pinned:
             1. pulisce subito le entry FAILED
             2. considera solo entry con lease_count = 0
                e future_refcount = 0
             3. sceglie la meno utile: frequenza d'uso con aging piu' bassa,
                poi last_used piu' vecchio
        |
        +--> sceglie (12, 9), che è READY, lease_count = 0
        |    e future_refcount = 0
        +--> nello stesso lock rimuove la chiave (12, 9) dalla mappa
        |    ma conserva i suoi buffer RAM pinned
        +--> associa gli stessi buffer alla nuova chiave (12, X)
        |    con future_refcount 0 -> 1 e stato QUEUED
        +--> accoda (12, X) nella coda dello shard 0
        +--> rilascia mutex shard 0
        +--> segnala soltanto la condition variable dello shard 0
        v
worker di I/O 0
        |
        +--> QUEUED -> READING
        +--> bridge Windows sovrascrive i buffer riusati con i byte di X
        +--> READING -> READY
        v
cache: (12, X) = READY
```

Il caso di errore segue un ramo alternativo, senza mai esporre dati parziali al
decode:

```text
worker di I/O / bridge Windows
        |
        +--> errore durante la lettura di (12, 9)
        v
worker di I/O 0 prende mutex shard 0: READING -> FAILED
        |
        +--> il mutex impedisce al decode di usare buffer incompleti come READY
        +--> rilascia mutex shard 0
        +--> il THREAD DI DECODE DS4 applica il fallback on-demand
        +--> il gestore registra l'audit e conserva l'entry FAILED finché non
        |    arriva un nuovo tentativo oppure serve riutilizzarne i buffer
        +--> una previsione futura potra' riportare la stessa chiave a QUEUED
             oppure sostituirla atomicamente con una nuova chiave dello stesso layer
```

#### Frequenza d'uso con aging: scegliere cosa espellere senza premiare il passato remoto

Il contatore di utilizzo non dovrebbe essere il semplice totale cumulato
dall'avvio. Un expert molto usato all'inizio della generazione, ma mai più
richiesto, non deve restare in RAM pinned per sempre solo grazie alla sua
storia passata. La cache può usare una frequenza esponenzialmente decadente
(*decayed LFU*), calcolata sui token generati e non sul tempo di parete: rende
la politica riproducibile e legata al vero avanzamento del decode.

Per ogni entry la cache conserva `score_salvato` e
`token_ultimo_aggiornamento`. Al token corrente `t`, il valore utile per
confrontare due candidate è:

```text
score_effettivo(t) = score_salvato × 2 ^ (-(t - token_ultimo_aggiornamento) / H)
```

`H` è l'*half-life* in token: dopo `H` token senza un nuovo utilizzo, lo score
si dimezza. L'aggiornamento può essere lazy, cioè eseguito soltanto quando il
**THREAD DI DECODE DS4** usa davvero l'entry o quando il gestore deve
confrontarla per l'eviction:

```text
quando il decode usa (layer, expert) al token t:

score_salvato = score_salvato × 2 ^ (-(t - token_ultimo_aggiornamento) / H) + 1
token_ultimo_aggiornamento = t
```

L'incremento deve avvenire soltanto per un vero hit del router reale: entry
`READY`, lease acquisito e upload RAM pinned → VRAM avviato. Una semplice
predizione del modello non deve aumentare il punteggio, altrimenti expert
predetti male ma mai usati dal forward sembrerebbero artificialmente popolari.

Esempio con `H = 64` token:

```text
expert A
  score_salvato = 32
  ultimo uso = 128 token fa
  score_effettivo = 32 × 2 ^ (-128 / 64) = 8

expert B
  score_salvato = 4
  ultimo uso = token corrente
  score_effettivo = 4 × 2 ^ (0) = 4

oggi: A resta preferibile a B (8 > 4)

dopo altri 64 token senza nuovi utilizzi di A:
  A = 8 × 2 ^ (-64 / 64) = 4
  B = 4 se nel frattempo non viene usato

a parita' di score: si espelle quello con last_used piu' vecchio
```

La politica proposta per liberare memoria diventa quindi:

```text
1. pulire subito FAILED;
2. considerare solo entry con lease_count = 0 e senza riferimenti futuri;
3. scegliere score_effettivo minimo;
4. usare last_used piu' vecchio come spareggio.
```

Un valore iniziale ragionevole da misurare sarebbe `H = 64` o `H = 128`
token: abbastanza lungo da trattenere un expert riusato in una finestra vicina,
ma non tanto da rendere permanente il vantaggio di utilizzi molto vecchi.

Questo è il vantaggio principale rispetto alla composizione A → B: gli expert
in comune non vengono copiati tra buffer slot-specifici e possono rimanere
validi anche quando A è nel suo consumo. In cambio serve una politica di
capacità/eviction accurata: trattenere ogni expert troppo a lungo può saturare
la RAM pinned e ridurre, anziché aumentare, il prefetch utile.

I dati di TEST_15 mostrano che il riuso già `READY` nel preciso istante di
accodamento è piccolo; hanno però evidenziato molti overlap con slot
`CONSUMING`. La cache per-expert potrebbe rendere sfruttabile parte di questi
ultimi proprio perché separa la vita del buffer dalla vita di uno slot. Prima
di implementarla resta necessario misurare per quanti di tali match l'entry
sarebbe `READY` entro il forward reale della richiesta successiva e quale
capacità pinned aggiuntiva richiederebbe.

## Approfondimento sulla soluzione B.1

### Correzione minimale della soluzione B: riallocare i buffer già sicuri da riusare

La soluzione B introduce già la cache condivisa per `(layer, expert)`, gli
shard, `future_refcount` e `lease_count`. TEST_15 ha però mostrato un limite
residuo: un buffer RAM pinned può restare associato a un'entry `READY` ormai
inutile per il suo layer, mentre un altro layer fallisce la creazione di una
nuova entry con `persistent_pinned_allocation_failed`.

La soluzione B.1 non richiede più RAM pinned e non modifica il comportamento
del decode o del fallback. Aggiunge soltanto la possibilità di **riassegnare
in sicurezza un buffer fisico già allocato** quando la nuova richiesta non
trova capacità libera.

```text
prima della correzione

layer 12: (12, 9) = READY, future_refcount = 0, lease_count = 0
          buffer pinned occupato ma inattivo

layer 27: vuole accodare (27, X)
          -> nessuna capacità locale disponibile
          -> persistent_pinned_allocation_failed

dopo B.1

layer 12: (12, 9) sicuro da rimuovere
          -> conserva il suo buffer fisico

layer 27: (27, X) = QUEUED sullo stesso buffer fisico
          -> il worker di shard(27) legge X e lo pubblica READY
```

Un'entry `READY` è una candidata sicura alla riallocazione **solo** se:

```text
future_refcount == 0    # nessuna previsione futura la richiede
lease_count == 0        # il THREAD DI DECODE DS4 non la sta caricando in VRAM
```

Questo non significa necessariamente che l'expert non sia mai stato utile:
significa che il decode lo ha già consumato, oppure che non esiste più alcuna
richiesta futura che possa usarlo. In entrambi i casi i suoi byte possono
essere sovrascritti senza compromettere il forward.

### Modifiche minime necessarie

1. **Separare chiave logica e buffer fisico.** Ogni entry `(layer, expert)`
   deve conservare un `buffer_id`; i buffer `gate/up/down` già allocati
   confluiscono in un piccolo registro/pool fisico. Non occorre riallocarli né
   aumentare il budget pinned della macchina.

2. **Cercare una candidata solo quando manca spazio.** Il thread asincrono di
   prefetch DS4 prova prima un buffer libero. Solo se non ce n'è uno, cerca:
   prima una entry `FAILED` che trattiene ancora buffer, poi una `READY` con
   entrambi i contatori a zero. La normale richiesta non paga quindi il costo
   della scansione globale quando il pool ha ancora capacità libera.

3. **Sostituire direttamente la mappa sotto lock.** La vecchia chiave viene
   rimossa e la nuova viene associata allo stesso `buffer_id` nella stessa
   sezione critica. La nuova entry nasce già `QUEUED`, con
   `future_refcount = 1`; dopo il rilascio del lock viene segnalata soltanto la
   condition variable del suo shard. Non serve esporre stati intermedi
   `ABSENT` o `EVICTING`.

4. **Mantenere i lock per shard.** Se la vecchia e la nuova entry appartengono
   a shard diversi, si prendono i due mutex in ordine crescente di indice;
   il registro dei buffer richiede un lock brevissimo solo per trasferire la
   proprietà del `buffer_id`. Il worker continua a fare l'I/O senza lock e
   resta proprietario del lavoro per `layer % N`.

5. **Non toccare il percorso critico del decode.** Il THREAD DI DECODE DS4
   continua a usare soltanto entry trovate già `READY`. Se una richiesta è
   ancora `QUEUED`, `READING`, assente o non sicura da riallocare, il fallback
   on-demand resta immediato: B.1 non può introdurre attese nel forward.

### Flusso di riallocazione

```text
THREAD DI DECODE DS4
        |
        +--> delega il prefetch futuro di (layer 27, expert X)
        v
thread asincrono di prefetch DS4
        |
        +--> nessun buffer libero nel pool fisico
        +--> trova (12, 9) = READY,
        |    future_refcount = 0, lease_count = 0
        |
        +--> lock shard(12) e shard(27), in ordine canonico
        +--> rimuove (12, 9) dalla mappa
        +--> associa il suo buffer_id a (27, X) = QUEUED
        +--> future_refcount di (27, X): 0 -> 1
        +--> rilascia i lock
        +--> segnala solo il worker di shard(27)
        v
worker di shard(27)
        |
        +--> QUEUED -> READING
        +--> bridge Windows sovrascrive il buffer con i byte di X
        +--> READING -> READY
```

### Cosa non cambia: I/O raggruppato e upload del forward reale

B.1 cambia esclusivamente **quale buffer pinned già esistente** viene
assegnato a una nuova entry. Non cambia la modalità con cui il worker riempie
il buffer né allarga il gruppo realmente eseguito dal modello.

```text
SSD -> RAM pinned

worker dello shard
        |
        +--> raccoglie il normale batch di entry QUEUED del proprio layer
        +--> invia al bridge Windows una richiesta raggruppata
        +--> il bridge legge gate/up/down dal GGUF direttamente nei buffer
             RAM pinned già associati alle entry del batch

RAM pinned -> VRAM

THREAD DI DECODE DS4
        |
        +--> router reale sceglie sempre e soltanto i 6 expert del forward
        +--> riusa i cache-hit READY tra quei 6
        +--> usa il fallback on-demand solo per i mancanti
```

In particolare, un `Top-K = 10` è soltanto la previsione da mettere in cache:
non fa eseguire dieci expert e non trasferisce automaticamente dieci expert
in VRAM. Al forward il gruppo reale resta Top-6. Nell'implementazione attuale
l'upload dei cache-hit RAM pinned -> VRAM è ancora effettuato per singolo
expert pronto; l'eventuale batching di queste copie è una possibile
ottimizzazione successiva, distinta da B.1 e non necessaria per risolvere la
saturazione.

### Perché questa è la priorità per la saturazione osservata in TEST_15

I dati raccolti da TEST_15 mostrano che, mentre avvengono i fallimenti,
esistono in media circa 2 GiB di buffer già occupati da entry sicure da
riassegnare. Prima di chiedere altra RAM pinned al sistema, B.1 permette di
trasformare questa capacità logicamente bloccata in capacità utile per le
nuove predizioni.

L'effetto atteso non è soltanto ridurre il contatore dei `FAILED`: più expert
potranno arrivare a `READY` entro il loro forward reale, quindi il decode
dovrà ricorrere meno spesso al fallback on-demand. Il guadagno di speedup va
poi dimostrato sperimentalmente, perché dipende anche da accuratezza del
predictor, ritardo del bridge Windows e tempi di upload RAM pinned -> VRAM.
La metrica di successo primaria è comunque chiara: i fallimenti di allocazione
persistente devono diminuire senza mai riallocare entry con riferimento futuro
o lease attivo. Questa verifica è stata eseguita successivamente in TEST_15
`output_12`.

## Risultati TEST_15 `output_11`: saturazione logica, non assenza fisica di RAM pinned

L'esecuzione `output_11` usa la cache condivisa per `(layer, expert)`,
`Top-K = 10` e otto worker. Ha confermato che i `FAILED` osservati in TEST_15
non sono causati dal bridge Windows: in tutte le ripetizioni il contatore
`bridge_failed` è zero. Il problema è invece
`persistent_pinned_allocation_failed`: la cache non riesce ad ammettere nuovi
expert pur avendo ancora molti buffer RAM pinned residenti ma già inutili.

Il sintomo è visibile anche end-to-end: in questa configurazione la fase C
richiede in media **121,282 s** contro **98,235 s** della baseline B
(-23,45% di speedup). La tabella seguente spiega perché il peggioramento non
va attribuito al modello di prefetch o al bridge: la cache cade spesso nel
fallback proprio mentre conserva capacità pinned riutilizzabile.

| Misura della fase C | Rip. 1 | Rip. 2 | Rip. 3 |
|---|---:|---:|---:|
| `persistent_pinned_allocation_failed` | 16.404 | 16.004 | 16.171 |
| RAM pinned occupata al primo fallimento | 2,947 GiB | 2,960 GiB | 2,960 GiB |
| Coppie subito riallocabili al primo fallimento | 238 | 81 | 208 |
| Byte subito riallocabili al primo fallimento | 1,607 GiB | 0,547 GiB | 1,404 GiB |
| Byte riallocabili medi durante i fallimenti | 1,983 GiB | 2,014 GiB | 1,969 GiB |
| Byte riallocabili massimi | 2,248 GiB | 2,282 GiB | 2,248 GiB |
| Entry protette da `future_refcount > 0` (media) | 113,2 | 121,2 | 117,8 |
| Entry con `lease_count > 0` (media) | 1,2 | 1,3 | 1,3 |
| `bridge_failed` | 0 | 0 | 0 |

Il primo campione varia perché il primo fallimento può arrivare in momenti
diversi della generazione. La misura stabile è quindi quella media: circa
**2 GiB di RAM pinned** risultano contemporaneamente occupati da entry `READY`
senza riferimenti futuri e senza lease. Sono byte già allocati, ma non sono
riutilizzabili da un layer che sta tentando di accodare un nuovo expert.

Le coppie candidate sono inoltre riportate nei file
`shared_expert_cache_metrics.json` di `output_11`, con il motivo
`ready_unreferenced_and_unleased` oppure
`failed_entry_retains_pinned_buffer`. Questo rende il problema verificabile:
non è necessario aumentare subito la memoria pinned del PC; prima bisogna
riutilizzare correttamente quella che è già stata allocata.

## Risultati TEST_15 `output_12`: B.1 elimina la saturazione pinned

TEST_15 `output_12` applica B.1 al caso che aveva evidenziato il problema:
cache condivisa `(layer, expert)`, `Top-K = 10`, capacità logica `2 × Top-10`
per layer e **otto worker I/O / otto shard**. Il risultato conferma il
meccanismo previsto: un buffer fisico viene spostato da una entry `READY`
sicura a una nuova entry `QUEUED`, senza una nuova allocazione pinned.

| Ripetizione | Baseline B | Cache condivisa B.1 C | Speedup C |
|---|---:|---:|---:|
| 1 | 101,158 s | 77,685 s | +23,21% |
| 2 | 106,446 s | 79,501 s | +25,31% |
| 3 | 100,514 s | 80,733 s | +19,68% |
| Media | **102,706 s** | **79,306 s** | **+22,73%** |

| Contatore B.1, media per ripetizione | Risultato |
|---|---:|
| `persistent_pinned_allocation_failed` | **0** |
| `bridge_failed` | **0** |
| `scatter_invariant_failed` | **0** |
| Riallocazioni fisiche cross-layer | ~13.020 |
| Byte trasferiti fra proprietà logiche | ~85,8 GiB |
| Donor `READY` riutilizzati | ~13.020 |
| Donor `FAILED` riutilizzati | 0 |
| Nessun donor sicuro disponibile | 380 |
| Hit `READY` del decode | ~53,65% |

La diagnosi cambia in modo netto. In `output_11` i buffer già allocati
rimanevano legati a entry ormai inutili e provocavano oltre sedicimila
fallimenti di allocazione persistente per ripetizione; in `output_12` non si
verifica più alcun `FAILED` né del bridge né dell'allocazione. Le
riallocazioni provengono tutte da entry `READY` con `future_refcount = 0` e
`lease_count = 0`: byte che il THREAD DI DECODE DS4 aveva già usato, oppure
che nessuna previsione futura poteva più richiedere.

Lo speedup torna positivo in tutte le ripetizioni, passando da `-23,45%` di
`output_11` a **+22,73%**. Non è ancora il record di TEST_14 `output_2`
(+29,65%): il percorso per-expert conserva circa 2.387 chiamate bridge per
run e un'attesa media in coda di circa 155 ms per job. B.1 risolve quindi la
saturazione della RAM pinned; il limite rimanente è il costo di coda,
pianificazione e I/O della cache per-expert, non la capacità fisica pinned.

I dati verificabili sono in
`test_predictor/test_15/output_12/benchmark_summary.json` e in ciascun
`output_12/repetition_*/shared_expert_cache_metrics.json`.

## Come B.1 realizza il pool fisico di buffer RAM pinned

Prima di B.1 la cache era logica per `(layer, expert)`, ma nella pratica i
buffer RAM pinned restavano vincolati alla capacità locale del layer. Quando
il layer 27 era saturo, non poteva usare i buffer che il layer 12 manteneva
`READY` ma non avrebbe più utilizzato. B.1 elimina questo vincolo, come
confermato da `output_12`.

La modifica applicata separa concettualmente **entry logica** e
**buffer fisico**:

```text
prima di B.1
  entry (layer, expert)  ---> possiede direttamente i suoi buffer RAM pinned

B.1
  entry (layer, expert)  ---> buffer_id opzionale
                                  |
                                  v
                       pool globale di buffer RAM pinned riutilizzabili
```

Il numero totale di buffer resta fissato a un limite prudente per la macchina;
non viene quindi richiesta più RAM pinned del budget attuale. Cambia solo chi
può usare un buffer diventato inutile: qualunque layer, non soltanto il layer
che lo aveva ottenuto per primo.

### Cosa cambia nell'implementazione B.1

1. **Pool fisico condiviso.** I blocchi pinned già allocati possono essere
   trasferiti fra entry logiche. Nel codice l'identità fisica è il gruppo di
   buffer `gate/up/down` associato allo slot; non serve una nuova allocazione
   quando quel gruppo viene passato a un altro layer.

2. **Riassegnazione globale sicura.** Quando una nuova predizione per
   `(layer_nuovo, expert_nuovo)` non trova un buffer libero, il thread
   asincrono di prefetch cerca in tutto il pool prima le entry `FAILED`, poi le
   entry `READY` con `future_refcount = 0` e `lease_count = 0`. Se trova una
   candidata, scollega in modo atomico la vecchia chiave, mantiene il gruppo
   di buffer fisici e lo collega subito alla nuova chiave, che passa a
   `QUEUED`.
   Il worker proprietario di `layer_nuovo % N` viene quindi segnalato e fa
   `QUEUED -> READING` nel buffer appena riusato.

3. **Lock con ordine deterministico.** La mappa di ogni layer continua a
   essere protetta dal mutex del suo shard. Se una riassegnazione coinvolge un
   layer vecchio e uno nuovo, i due shard vanno presi sempre in ordine di
   indice crescente; una piccola protezione del pool serve solo per trasferire
   la proprietà del gruppo di buffer. L'I/O del bridge Windows e la copia RAM
   pinned -> VRAM restano fuori dai lock.

4. **Le garanzie esistenti non cambiano.** Un buffer non può essere
   riassegnato se è ancora richiesto da una previsione futura
   (`future_refcount > 0`) o se il THREAD DI DECODE DS4 lo sta caricando in
   VRAM (`lease_count > 0`). Il fallback resta identico quando non c'è una
   candidata sicura: il decode non deve mai attendere la cache.

```text
nuova richiesta: (layer 27, expert X)
        |
        +--> nessun buffer libero nel pool
        |
        +--> trova (layer 12, expert 9): READY,
        |    future_refcount = 0, lease_count = 0
        |
        +--> scollega (12, 9) dal suo buffer_id
        +--> collega lo stesso buffer_id a (27, X) = QUEUED
        +--> segnala soltanto il worker di shard(27)
        v
worker di shard(27): QUEUED -> READING -> READY
```

### Miglioramento atteso e come verificarlo

Questa correzione dovrebbe trasformare una quota importante dei fallimenti di
allocazione persistente in normali richieste `QUEUED -> READING -> READY`.
I circa 1,97--2,01 GiB medi rilevati come riutilizzabili equivalgono a
centinaia di buffer-expert che oggi restano inutilmente bloccati. Ci si aspetta
quindi più entry `READY` al momento del decode, meno fallback on-demand e una
maggiore possibilità di sfruttare davvero Top-K elevati.

Non è però corretto promettere automaticamente il record di speedup: restano
da misurare accuratezza del predictor, ritardo del bridge, numero di chiamate
di I/O e costo di upload verso VRAM. La correzione è soprattutto necessaria
per rimuovere un limite artificiale di capacità; lo speedup end-to-end dovrà
essere verificato contro TEST_14 `output_2`.

## Batching per layer/shard: stato effettivamente implementato

La soluzione B.1 non ha trasformato il percorso SSD in una lettura per singolo
expert. L'implementazione corrente conserva il batching che precedeva B.1:
il thread asincrono di prefetch DS4 valuta un Top-K per un certo layer, crea o
riusa le corrispondenti entry logiche e accoda **solo gli expert mancanti**
nello shard `layer % N`. L'accodamento avviene sotto il mutex di quello shard e
la condition variable risveglia esclusivamente il worker proprietario.

```text
THREAD ASINCRONO DI PREFETCH DS4: layer 12, Top-10
        |
        +--> entry gia' READY / QUEUED / READING: riuso, nessuna lettura
        |
        +--> expert mancanti: entry (12, e) = QUEUED
        |    rank 1--6: accodate subito come lavoro primario
        |    rank 7--10: tail differito finche' la coda primaria non e' vuota
        v
worker dello shard 12
        |
        +--> sotto mutex: prende una sequenza FIFO contigua
        |    di entry QUEUED dello stesso layer e della stessa priorita'
        |    fino a Top-K
        +--> QUEUED -> READING per il batch
        +--> rilascia il mutex
        +--> una chiamata bridge Windows con tutti gli expert del batch
        +--> disperde gate/up/down nei buffer RAM pinned delle entry
        +--> pubblica ogni entry come READY
```

Il batch viene formato soltanto dagli expert che necessitano davvero della
lettura; gli expert gia' residenti non vengono ri-letti. I buffer RAM pinned
restano per-entry per consentire riuso, lease e riallocazione B.1 sicuri, ma
la chiamata SSD/bridge e' una sola per il gruppo dello stesso layer.

### Ammissione Top-10 con Top-6 primario

Quando `Top-K = 10`, non tutti i rank hanno il medesimo valore nel percorso
critico: il router reale richiede al massimo sei expert. L'implementazione
assegna quindi priorita' **primaria** ai rank 1--6 e priorita' **tail** ai rank
7--10. Il thread asincrono accoda subito soltanto i primari; conserva i tail
nella cache e li rilascia verso il worker proprietario solo quando la coda
primaria dello shard e' vuota. Se una nuova previsione porta un tail fra i
primi sei, l'entry viene promossa senza essere ri-letta.

```text
Top-10 predetto per layer 12
    [e1 e2 e3 e4 e5 e6 | e7 e8 e9 e10]
     \________ primari ________/ \_____ tail _____/
                |                         |
                v                         v
        coda immediata shard 12%N   QUEUED nella cache
                |                         |
                v                         +--> rilasciato soltanto quando
          batch bridge prioritario              la coda primaria e' vuota
```

La telemetria distingue `primary_queue_pushes`, `tail_queue_pushes` e
`deferred_tail_releases`, cosi' da separare il costo della speculazione
aggiuntiva dalla disponibilita' dei sei expert che possono essere davvero
consumati dal forward.

### Verifica sperimentale: TEST_15 `output_13`

Con Top-10, B.1 e dieci worker/shard, `output_13` ha letto 13.866 expert
mediante 2.391 chiamate bridge nella prima ripetizione: **5,80 expert per
chiamata**, non 13.866 chiamate indipendenti. Le tre ripetizioni non hanno
mostrato errori bridge o saturazione pinned; l'attesa media di coda e' scesa a
circa 94 ms per expert, contro circa 155 ms con otto worker. Il tempo C medio
e' migliorato da 79,306 s a 77,690 s, mentre lo speedup medio e' +20,55% a
causa di una baseline di quella run piu' veloce e quindi non direttamente
confrontabile.

Di conseguenza, un'ulteriore implementazione di batching sarebbe ridondante:
il margine residuo va ricercato nella quota di expert ancora assenti/non READY
al decode, nei relativi upload on-demand e nella varianza della baseline,
non in una supposta chiamata bridge per ogni expert.

### Verifica sperimentale dell'ammissione prioritaria: TEST_15 `output_16`

Con cache B.1, sei worker, Top-10 e `primary_top_k = 6`, `output_16` ha
ottenuto **77,826 s** medi per C contro **105,525 s** della baseline,
equivalenti a **+26,25%**. I tail non hanno piu' il diritto di precedere un
batch primario gia' accodato, non sono comparsi `FAILED`, ma la copertura
`READY` e' rimasta circa **49,1%**. Il risultato non supera quindi il record
di TEST_14/output_2 (+29,65%): il limite osservato e' soprattutto la quota di
expert reali non ancora residenti al decode, non la contesa fra i rank 7--10
e i primi sei.

Dopo l'implementazione, il test deve mostrare contemporaneamente:

1. una forte riduzione di `persistent_pinned_allocation_failed`;
2. contatori di buffer effettivamente riutilizzati tra layer diversi;
3. crescita di hit e copertura `READY` al decode;
4. riduzione del fallback on-demand e del tempo della fase C;
5. assenza di riassegnazioni con `future_refcount > 0` o `lease_count > 0`.
