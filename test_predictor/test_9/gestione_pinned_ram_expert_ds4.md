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

Nel branch del progetto questa banca separata esiste già, ma la sua politica è
ancora volutamente semplice: per ogni combinazione attiva di layer e distanza
futura usa un solo slot, senza sostituzione quando è occupato. Il passo
architetturale successivo è quindi migliorare tale banca verso una **vera cache
persistente di expert in pinned RAM**, con una struttura concettuale del tipo:

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
