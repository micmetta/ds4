# TEST_9 — Benchmark ripetuto con audit automatico della RAM pinned

## Scopo

TEST_9 esegue autonomamente lo stesso confronto controllato A/B/C di TEST_8:

- A registra una sola volta l'oracolo di routing;
- B esegue il replay baseline senza predictor né prefetch;
- C esegue predictor CPU asincrono e learned prefetch;
- B e C vengono alternati per il numero di ripetizioni richiesto, di default
  tre.

La differenza rispetto a TEST_8 è che TEST_9, subito dopo ogni fase C,
analizza automaticamente `run_C.prefetch.jsonl` e salva il relativo
`pinned_ram_policy_audit.json`. Non occorre quindi un secondo comando né una
lettura manuale della trace per contare i casi `prefetch_bank_full`, stale e
riuso parziale.

In particolare, distingue `prefetch_bank_full` dalla lentezza del predictor o
dell'SSD e cerca le sequenze in cui una nuova richiesta viene rifiutata mentre
uno slot precedente verrà liberato soltanto più tardi come `stale`.

## Due usi distinti della RAM pinned in DS4

Il file [`gestione_pinned_ram_expert_ds4.md`](gestione_pinned_ram_expert_ds4.md)
descrive correttamente il caricamento **normale** degli expert nel path CUDA di
DS4. Per capire TEST_9 è però indispensabile non confonderlo con il meccanismo
aggiunto per il learned prefetch: sono due insiemi di buffer separati.

```text
Caricamento normale, quando un expert serve adesso

GGUF del modello principale
        |
        v
4 buffer pinned temporanei di staging
        |
        v
VRAM: buffer compatto degli expert richiesti dal router reale
        |
        v
kernel MoE


Learned prefetch, quando il predictor stima un expert per un token futuro

GGUF usato dal worker di prefetch
        |
        v
banca pinned TEST_4/TEST_5: slot[target_token][layer][distanza]
        |
        | soltanto quando arriva la domanda reale compatibile
        v
VRAM: buffer compatto degli expert richiesti dal router reale
        |
        v
kernel MoE
```

Nel caricamento normale DS4 usa esattamente quattro buffer
`g_stream_selected_stage_raw[0..3]`, allocati con `cudaMallocHost`. Sono un
anello di trasporto: per ogni chunk il codice usa `chunk_idx % 4`, aspetta il
CUDA event del buffer prima di riutilizzarlo e poi avvia il trasferimento
`cudaMemcpyAsync` verso la GPU. Questi quattro buffer non sono una cache
persistente di expert: dopo il trasferimento vengono riutilizzati per il chunk
successivo.

La banca analizzata da TEST_9 è intenzionalmente separata: il commento nel
codice CUDA dice esplicitamente che non può condividere i quattro buffer di
staging, perché il worker di prefetch può leggere mentre il thread di decode
sta usando il caricatore normale. Ogni suo slot possiede invece i propri buffer
pinned `gate`, `up`, `down` e, quando si usa direct I/O, `scratch`. Quindi:

- la disponibilità dei quattro buffer di staging normali **non** determina un
  evento `prefetch_bank_full`;
- `prefetch_bank_full` riguarda solo gli slot logici della banca learned per
  quel layer e quelle distanze future;
- aumentare la distanza futura può creare più posizioni nella banca learned e
  quindi più memoria pinned potenzialmente allocabile, ma non modifica il
  numero fisso di quattro buffer temporanei del caricatore normale.

Prima di ogni caricamento, il router reale compatta anche gli ID degli expert
selezionati e non trasferisce due volte lo stesso expert nella stessa richiesta.
Per esempio, la sestina `[12, 47, 83, 47, 120, 12]` diventa gli expert distinti
`[12, 47, 83, 120]` nel caricamento normale. Questa deduplicazione riduce il
traffico di una singola domanda, ma non risolve la contesa dello slot learned
osservata da TEST_9: quella dipende dal fatto che lo slot del layer sia ancora
`QUEUED`, `READING`, `READY`, `CONSUMING`, `FAILED` o `TOO_LATE` quando arriva
la nuova previsione.

## Struttura reale della banca pinned

L'implementazione CUDA è in `ds4_cuda.cu` e usa un **array bidimensionale
fisso**, non una lista o un dizionario. La forma concettuale è:

```text
slots[layer fisico][indice di distanza]
```

In C è un array con capacità massima:

```text
slots[43][8]
```

Il primo indice identifica il layer fisico di DS4; il secondo identifica una
delle posizioni che quel layer può usare per conservare richieste future. Il
`target_token` non è quindi un terzo indice della matrice: è un valore salvato
**dentro** lo slot scelto.

In forma concettuale, uno slot contiene:

```text
slot = {
    stato,
    target_token,
    layer,
    sei expert predetti,
    buffer pinned e metadati della lettura
}
```

Quando, mentre DS4 elabora il token `t`, il predictor prevede sei expert per
il token `t+1` nel layer fisico `l`, DS4 cerca una posizione `EMPTY` tra:

```text
slots[l][0], slots[l][1], ..., slots[l][distanza_attiva - 1]
```

e memorizza in quella posizione `target_token = t+1`. L'indice di distanza non
significa rigidamente “questo è il token t+1” oppure “questo è il token t+2”:
è soprattutto un numero di posizioni fisiche disponibili per mantenere più
previsioni future dello stesso layer. L'implementazione corrente sceglie la
prima posizione vuota che trova.

I limiti massimi compilati sono 43 layer e 8 posizioni future per layer. Non
significa però che tutte queste posizioni siano attive: i limiti runtime sono
impostati da:

- `DS4_TEST_4_PREFETCH_LAYERS`;
- `DS4_TEST_4_PREFETCH_INIT_DISTANCE`.

Con la configurazione learned attuale di TEST_8:

```text
DS4_TEST_8_PREFETCH_LAYERS = 8
DS4_TEST_8_PREFETCH_INIT_DISTANCE = 1
```

il runner traduce il primo valore in `DS4_TEST_4_PREFETCH_LAYERS=11`, perché
riserva gli indici fisici 0–2 e usa i layer MoE fisici 3–10. La distanza uno
attiva una sola posizione per ciascun layer fisico:

```text
layer 3  -> slots[3][0]
layer 4  -> slots[4][0]
...
layer 10 -> slots[10][0]
```

Per esempio, durante il token `t`, la previsione per `(t+1, layer 3)` può
essere inserita soltanto in `slots[3][0]`; quello slot salva poi
`target_token = t+1`. Se è `QUEUED`, `READING`, `READY`, `CONSUMING`, `FAILED`
oppure `TOO_LATE`, non esiste una seconda posizione dello stesso layer in cui
inserire la nuova previsione: DS4 registra `prefetch_bank_full`. Con distanza
due sarebbero invece disponibili sia `slots[3][0]` sia `slots[3][1]`.

### Esempio concreto con distanza due

Impostando:

```text
DS4_TEST_8_PREFETCH_INIT_DISTANCE = 2
```

ogni layer learned avrebbe due posizioni fisiche. Per esempio, mentre DS4
elabora il token 21, il predictor può accodare la previsione per il token 22
nel layer `l`:

```text
slots[l][0] = {
    stato          = QUEUED, READING oppure READY,
    target_token   = 22,
    layer          = l,
    expert predetti = [15, 90, 96, 112, 119, 202],
    buffer pinned e metadati della lettura
}

slots[l][1] = EMPTY
```

Quando DS4 elabora il token 22, il predictor può poi produrre una nuova
previsione per il token 23 nello stesso layer. Se il primo slot è ancora
occupato, la nuova previsione può usare il secondo:

```text
slots[l][0] = {
    stato          = READY,
    target_token   = 22,
    layer          = l,
    expert predetti = [15, 90, 96, 112, 119, 202]
}

slots[l][1] = {
    stato          = QUEUED,
    target_token   = 23,
    layer          = l,
    expert predetti = [90, 96, 112, 155, 202, 240]
}
```

L'indice `[0]` oppure `[1]` non identifica il token. È soltanto il posto
fisico disponibile nella banca. Quando il router reale richiede gli expert del
token 22 nel layer `l`, DS4 scorre entrambi gli slot attivi e cerca quello che
soddisfa:

```text
slot.target_token == 22
e
slot.layer == l
```

In questo esempio seleziona `slots[l][0]`. Quando arriverà la domanda del token
23 per lo stesso layer, applica lo stesso confronto e seleziona `slots[l][1]`.
In altre parole, `target_token` è l'identità logica della previsione; l'indice
di distanza è solo capacità fisica aggiuntiva. La distanza due non obbliga il
predictor a prevedere direttamente “due token avanti”: permette di trattenere
due previsioni future in attesa, per esempio una per `t+1` e una per `t+2`
generate in momenti successivi.

Ogni slot contiene:

- `target_token`: token futuro per cui sono stati stimati gli expert;
- `layer`: layer fisico;
- sei ID expert predetti;
- riferimenti alla tabella degli offset nel GGUF;
- buffer host pinned per gate, up, down e, con direct I/O, scratch;
- tempi di enqueue, inizio lettura e disponibilità;
- stato e codice di errore.

I buffer sono allocati in modo pigro alla prima lettura e restano allocati fino
allo shutdown del processo DS4. Portare uno slot a `EMPTY` lo rende riusabile
logicamente, ma non restituisce immediatamente quei buffer al sistema.

## Stati possibili di uno slot

```text
EMPTY -> QUEUED -> READING -> READY -> CONSUMING -> EMPTY
                  |           |
                  |           +-> FAILED
                  +-> TOO_LATE -> EMPTY
```

- `EMPTY`: slot disponibile per un nuovo `(target token, layer)`;
- `QUEUED`: la previsione è nello slot e attende che il worker SSD la scelga;
- `READING`: il worker SSD sta leggendo i sei expert dal GGUF di prefetch nei
  buffer pinned dello slot;
- `READY`: i buffer pinned contengono gli expert e attendono la domanda GPU;
- `CONSUMING`: la domanda reale sta copiando tutti o parte degli expert verso
  la VRAM;
- `FAILED`: lettura o allocazione fallita; può essere scartato come stale.
- `TOO_LATE`: il thread di decoding ha già elaborato quel token e layer mentre
  il worker era in `READING`; il worker completerà l'I/O già iniziato, scarterà
  il risultato e libererà lo slot.

Uno slot non resta occupato fino alla fine dell'intera inferenza o della
generazione. Nel caso normale, dopo che la domanda reale ha riusato gli expert
e il trasferimento dalla RAM pinned alla VRAM è terminato, DS4 porta subito lo
slot da `CONSUMING` a `EMPTY`. A quel punto può ospitare una nuova previsione
per un altro token futuro.

Il riuso è però vincolato al layer fisico dello slot. Per esempio,
`slots[3][0]` può contenere prima la previsione per `(token 22, layer 3)` e,
dopo essere tornato `EMPTY`, quella per `(token 25, layer 3)`; non può invece
essere usato per una previsione del layer 4, che deve usare `slots[4][0]` o
un'altra sua posizione di distanza.

Uno slot può tornare `EMPTY` anche senza essere riusato dalla domanda reale:

- se è `READY` o `FAILED` per un `target_token` già superato, la successiva
  domanda reale dello stesso layer lo registra come `prefetch_stale_discarded`
  e lo libera;
- quando il processo DS4 termina, la chiusura libera anche i buffer pinned
  associati agli slot.

`EMPTY` indica soltanto che la posizione è libera **logicamente**. I buffer
pinned già allocati per quello slot restano invece disponibili per il riuso
fino allo shutdown del processo: non vengono liberati e riallocati dopo ogni
token.

## Perché `EMPTY` non libera subito la memoria pinned

Quando uno slot torna `EMPTY`, DS4 potrebbe teoricamente liberare i suoi
buffer con `cudaFreeHost(...)` e riallocarli alla previsione seguente con
`cudaMallocHost(...)`. Non lo fa perché queste operazioni sono relativamente
costose: CUDA e il sistema operativo devono allocare memoria host, bloccarne
le pagine fisiche e prepararla per trasferimenti efficienti verso la GPU.

La politica attuale è quindi:

```text
slot CONSUMING completato
        |
        v
slot.state = EMPTY             <- i dati precedenti non sono più validi
        |
        v
buffer pinned gate/up/down/scratch rimangono allocati
        |
        v
nuova previsione dello stesso slot: sovrascrive i buffer già pronti
```

Questo evita di pagare un ciclo `cudaFreeHost` → `cudaMallocHost` a ogni token
e rende più rapido e stabile il riuso dello slot. Il costo è che la memoria
pinned fisica resta occupata fino alla chiusura del processo DS4, anche quando
gli slot sono `EMPTY`.

### Quando può avvenire davvero una saturazione della RAM pinned

`prefetch_bank_full` non prova una saturazione fisica: nel nostro caso indica
soltanto che non c'è uno slot **logicamente vuoto** per quel layer. La
saturazione fisica è un problema diverso e può avvenire se `cudaMallocHost`
non riesce ad allocare nuovi buffer. Diventa più probabile quando si combinano:

- molti layer con learned prefetch attivo;
- una distanza futura elevata, quindi molti slot per ciascun layer;
- gruppi di expert grandi, perché ogni slot deve mantenere i buffer `gate`,
  `up` e `down` per i sei expert previsti;
- direct I/O, che può richiedere anche il buffer pinned `scratch` allineato;
- i quattro buffer pinned di staging usati dal caricamento normale degli
  expert, che sono separati dalla banca learned ma consumano ugualmente RAM
  pinned;
- altra memoria pinned CUDA già in uso nello stesso processo o da altri
  processi GPU.

Per esempio, passare da 8 layer con distanza 1 a 43 layer con distanza 8 non
aumenta soltanto il numero di metadati: nel caso in cui ogni posizione riceva
almeno una richiesta, può portare all'allocazione pigra di buffer pinned per
molti più slot. Per questo l'aumento della distanza va provato insieme a misure
di memoria e a una politica di priorità o sostituzione, non considerato come
una soluzione gratuita a `prefetch_bank_full`.

### Risposte lunghe: cosa cresce e cosa non cresce

Una risposta con molti token non fa crescere senza limite la memoria pinned
della banca learned. A configurazione invariata, il suo limite è dato da:

```text
numero di layer attivi
    × numero di posizioni di distanza attive per layer
    × dimensione dei buffer di un singolo slot
```

Durante i primi token, uno slot che riceve la sua prima richiesta può allocare
i buffer pinned in modo pigro. Dopo questa prima allocazione, gli stessi buffer
vengono sovrascritti e riusati per i token successivi. Una risposta da 1.000
token non crea quindi 1.000 copie dei buffer pinned: la crescita si arresta
quando tutti gli slot effettivamente usati hanno ricevuto la loro prima
richiesta.

Le risposte lunghe possono comunque evidenziare altri limiti, distinti dalla
saturazione pinned:

- gli eventi `prefetch_bank_full`, il riuso parziale e i risultati arrivati in
  ritardo possono ripetersi per molti token, riducendo ripetutamente le
  occasioni di speedup senza aumentare la memoria;
- la trace `run_C.prefetch.jsonl` cresce approssimativamente in proporzione ai
  token e agli eventi registrati, quindi può occupare molto spazio su disco;
- la cache Key-Value del modello cresce con il contesto generato. È un consumo
  separato della banca pinned degli expert e riguarda principalmente la memoria
  del modello/GPU;
- se la posizione del token supera molto l'intervallo osservato in training,
  la feature `token_position` standardizzata può essere fuori distribuzione.
  Questo può ridurre la qualità del predictor, ma non implica una saturazione
  della RAM pinned.

## Politica attuale di enqueue

Quando il predictor produce una sestina per `(token t+1, layer l)`, DS4 chiama
`ds4_gpu_test4_prefetch_enqueue`.

1. Controlla che il prefetch sia attivo, il layer sia entro il limite runtime e
   gli offset degli expert siano validi.
2. Esamina soltanto gli slot di quel singolo layer, nelle distanze attive.
3. Se trova lo stesso `target_token`, considera la richiesta già accodata e
   non la duplica.
4. Se trova uno slot `EMPTY`, lo riempie, lo porta in `QUEUED` e sveglia il
   worker. Il worker selezionerà poi lo slot e lo porterà in `READING` prima
   di iniziare l'I/O verso i buffer pinned.
5. Se non trova slot `EMPTY`, emette `prefetch_bank_full` e rifiuta la nuova
   richiesta. Il decode GPU non aspetta e prosegue; quando arriverà la domanda
   reale userà il caricamento normale come fallback.

Non esiste qui una sostituzione arbitraria: un `READY`, `READING`, `FAILED` o
`TOO_LATE` di un altro token occupa lo slot e non viene sovrascritto
dall'enqueue. Uno slot `QUEUED` che il decoding riconosce come già superato,
invece, viene liberato nella domanda reale successiva dello stesso layer.

## Politica attuale alla domanda reale

Quando il router reale richiede i sei expert per `(token t, layer l)`, DS4
chiama `ds4_gpu_test4_prefetch_try_selected_load`.

1. Scorre gli slot del layer `l`.
2. Se uno slot ha `target_token < t`, il token conservato nello slot è già
   stato elaborato per quel layer. DS4 agisce in base allo stato:
   `QUEUED → EMPTY` e registra `prefetch_stale_queued_discarded`; `READING →
   TOO_LATE` e lascia che sia il worker a liberarlo dopo l'I/O;
   `READY` o `FAILED → EMPTY` e registra `prefetch_stale_discarded`.
3. Se trova un gruppo `READY` identico, copia i sei expert dalla RAM pinned in
   VRAM: `prefetch_hit`, poi libera lo slot.
4. Se trova un gruppo `READY` dello stesso token/layer ma solo parzialmente
   sovrapposto, riusa gli expert comuni, carica gli altri da SSD e poi libera
   lo slot: `prefetch_partial_reuse`.
5. Se il gruppo è ancora `QUEUED` o `READING`, è fallito, assente o diverso,
   emette il relativo evento di lookup e DS4 esegue il fallback normale. Uno
   slot `READING` non viene sovrascritto: se è stale viene soltanto marcato
   `TOO_LATE`, perché il worker sta ancora usando i suoi buffer pinned.

## Problema osservato con distanza 1

Il codice libera uno slot consumato correttamente, quindi non è corretto dire
che tutti gli slot restino occupati dopo ogni domanda. Il problema è più
specifico: un risultato arrivato in ritardo può diventare `READY` per un token
che la GPU ha già superato; in quel caso viene pulito come stale soltanto
durante una domanda futura dello stesso layer. Uno slot può anche restare
temporaneamente `CONSUMING` mentre DS4 riusa una parte degli expert e completa
il trasferimento verso la VRAM.

Nel frattempo può accadere questa sequenza:

```text
GPU richiede (t+1, l), trova un riuso parziale e lo slot passa a CONSUMING
predictor prova a inserire (t+2, l) -> nessuno slot EMPTY -> prefetch_bank_full
il riuso parziale termina -> slot EMPTY, ma (t+2, l) era già stato rifiutato
```

Con distanza `1` non esiste un secondo slot nello stesso layer in cui mettere
la richiesta nuova. Questo spiega perché `prefetch_bank_full` può ridurre il
riuso anche con expert predetti correttamente. Non implica che la RAM fisica
totale sia esaurita: segnala che lo **slot logico del layer** è occupato. Il
caso stale resta possibile, ma TEST_9 lo misura separatamente.

## Cosa misura TEST_9

Il report `pinned_ram_policy_audit.json` contiene:

- conteggio di tutti gli eventi della trace;
- numero totale e per layer di `prefetch_bank_full`;
- numero totale e per layer di `prefetch_stale_discarded`;
- `bank_full_followed_by_stale_discard`: numero di richieste rifiutate per
  banca piena a cui segue una pulizia stale nello stesso layer. È un indicatore
  della sequenza descritta sopra, non una prova causale assoluta;
- `bank_full_during_previous_partial_reuse`: numero di richieste rifiutate per
  cui la trace mostra, subito dopo, il completamento di un riuso parziale del
  token precedente nello stesso layer. È l'indicatore del caso `CONSUMING`.
- interpretazione testuale del risultato.

## Esecuzione del benchmark automatico

TEST_9 usa variabili dedicate e non modifica TEST_8:

```text
DS4_MODEL                       percorso del GGUF usato da DS4
DS4_TEST_9_PREFETCH_MODEL       percorso del GGUF letto dal worker di prefetch
DS4_TEST_9_CHECKPOINT           checkpoint del predictor, facoltativo
DS4_TEST_9_PREFETCH_LAYERS      numero di layer MoE learned, default 1
DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER numero di slot futuri per layer, default 1
DS4_TEST_9_PREFETCH_NUM_WORKERS numero di worker SSD, da 1 a 4, default 3
DS4_TEST_9_PREFETCH_STEP        passo del prefetch, default 1
DS4_TEST_9_REPETITIONS          numero di coppie B/C, default 3
```

`DS4_TEST_9_PREFETCH_INIT_DISTANCE` è accettata solo come nome legacy per le
vecchie configurazioni: per ogni nuova run usare
`DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER`.

Dal terminale in `/home/michele/ds4`, oppure tramite la configurazione CLion
`TEST_9`, basta lanciare:

```bash
make test-predictor-9
```

La run crea una directory `test_predictor/test_9/output*/` contenente:

```text
benchmark_summary.json
routing_oracle.bin
repetition_01_B_then_C/
  run_C.prefetch.jsonl
  pinned_ram_policy_audit.json
repetition_02_C_then_B/
  run_C.prefetch.jsonl
  pinned_ram_policy_audit.json
repetition_03_B_then_C/
  run_C.prefetch.jsonl
  pinned_ram_policy_audit.json
```

Ogni audit è anche incorporato nella rispettiva sezione
`C_learned_prefetch.pinned_ram_policy_audit` del `benchmark_summary.json`.
Perciò il report principale contiene sia tempi B/C e speedup sia le cause
misurate dei rifiuti della banca pinned.

### Audit manuale facoltativo di una trace storica

Resta possibile analizzare una trace prodotta prima dell'introduzione del
benchmark automatico, senza eseguire DS4:

```bash
python test_predictor/test_9/test_9_runner.py \
  --audit-trace test_predictor/test_8/output_6/repetition_01_B_then_C/run_C.prefetch.jsonl
```

In questa modalità viene creato un singolo
`test_predictor/test_9/output*/pinned_ram_policy_audit.json`.

### Risultato verificato su `output_6`, prima ripetizione C

L'audit eseguito sulla trace
`output_6/repetition_01_B_then_C/run_C.prefetch.jsonl` ha rilevato:

| Misura | Valore |
|---|---:|
| Richieste learned | 504 |
| Richieste accodate al worker | 256 |
| Rifiuti `prefetch_bank_full` | 248 |
| Rifiuti durante un precedente riuso parziale | **240** |
| Pulizie `prefetch_stale_discarded` | 8 |

Quindi 240 dei 248 rifiuti (circa 96,8%) sono coerenti con uno slot ancora
`CONSUMING` per il riuso parziale del token precedente. Questo è il problema
dominante osservato: la nuova richiesta viene scartata invece di aspettare,
accodarsi o essere riprovata quando il trasferimento precedente libera lo slot.

## Modifiche future da valutare, non implementate qui

1. Durante l'enqueue, liberare subito gli slot `READY`/`FAILED` il cui token è
   già passato rispetto al token corrente, se questa informazione viene resa
   disponibile in modo sicuro al worker.
2. Quando il banco del layer è pieno, sostituire uno slot già scaduto invece di
   rifiutare automaticamente la nuova richiesta.
3. Separare una pulizia degli slot stale dalla domanda GPU, evitando che la
   pulizia avvenga dopo il rifiuto della nuova previsione.
4. Provare una distanza maggiore di uno solo insieme a una politica esplicita
   di priorità e memoria: aumentare la distanza senza controllo può aumentare
   i buffer pinned allocati e il traffico SSD.

Nessuna di queste modifiche viene fatta da TEST_9: il test misura e comprende
la politica esistente prima che venga cambiata.

## Risultati misurati — `output` (distanza 1, otto layer learned)

La directory [`output`](output) contiene il primo benchmark TEST_9 completo
con audit automatico. La configurazione usata era:

```text
checkpoint predictor     = best.weights.h5, Recall@6 offline sul test set 0,506056
layer learned            = 8 (layer fisici DS4 da 3 a 10)
distanza futura          = 1
passo                    = 1
ripetizioni B/C          = 3, con ordine alternato B-C, C-B, B-C
```

Il predictor calcola una previsione per il token successivo e per ciascuno
degli otto layer. Ogni previsione contiene sei expert. La fase B non usa né
predictor né prefetch; la fase C usa il predictor CPU asincrono e la banca
pinned descritta in questo documento.

### Tempi B e C

| Ripetizione | Ordine | B baseline | C learned prefetch | Differenza C rispetto a B |
|---|---|---:|---:|---:|
| 1 | B poi C | 99,930 s | 97,431 s | **-2,499 s (-2,50%)** |
| 2 | C poi B | 110,786 s | 104,727 s | **-6,059 s (-5,47%)** |
| 3 | B poi C | 100,439 s | 109,248 s | **+8,809 s (+8,77%)** |
| Media | — | 103,718 s | 103,802 s | **+0,084 s (+0,27%)** |

Il segno negativo nella colonna finale è positivo: significa che C ha
richiesto meno tempo di B. C è stata più veloce in due ripetizioni su tre,
ma la terza è stata sensibilmente più lenta. La media finale non dimostra
quindi ancora uno speedup stabile: è praticamente neutra e la variabilità tra
le ripetizioni resta elevata.

Questo è proprio il motivo per cui il benchmark alterna l'ordine B/C: una
singola coppia favorevole non basta per attribuire il miglioramento al
prefetch, perché cache, attività del sistema e tempi dell'I/O possono cambiare
tra una run e l'altra.

### Qualità delle previsioni e riuso effettivo

In tutte le tre ripetizioni la Recall@6 online è stata `0,435484`. Su 496
previsioni valutabili, il predictor ha indovinato in media 2,61 dei sei expert
proposti; soltanto quattro previsioni hanno contenuto tutti e sei gli expert
corretti. Questo valore online non coincide necessariamente con il risultato
offline sul test set: qui viene misurata una singola generazione DS4, con i
token e gli stati incontrati realmente durante quella risposta.

| Ripetizione | Gruppi con riuso parziale | Expert già riusati dalla RAM pinned | Expert ancora caricati da SSD | Hit completi |
|---|---:|---:|---:|---:|
| 1 | 286 | 793 | 2.231 | 0 |
| 2 | 276 | 753 | 2.271 | 0 |
| 3 | 259 | 697 | 2.327 | 0 |

Il learned prefetch non è inutile: centinaia di expert correttamente predetti
sono stati effettivamente riusati dalla RAM pinned. Tuttavia non c'è stato
nessun gruppo in cui tutti i sei expert richiesti dal router reale fossero già
pronti. Di conseguenza DS4 ha dovuto continuare a leggere molti expert dal
SSD e il guadagno resta limitato e instabile.

### Cosa rivela l'audit degli slot pinned

| Ripetizione | `prefetch_bank_full` | di cui durante riuso parziale precedente | `prefetch_stale_discarded` | Expert corretti non riusati per banca piena |
|---|---:|---:|---:|---:|
| 1 | 198 | 181 | 15 | 496 |
| 2 | 212 | 199 | 11 | 541 |
| 3 | 219 | 190 | 21 | 579 |

Il risultato più importante è `prefetch_bank_full`. Con distanza 1 esiste un
solo slot per ogni layer: in circa l'87–94% dei rifiuti lo slot era ancora
occupato dal riuso parziale del token precedente (`CONSUMING`). La nuova
previsione, anche quando avrebbe contenuto expert corretti, viene quindi
rifiutata prima di poter essere letta dal worker.

Questo non segnala che la RAM pinned fisica sia terminata. Indica invece una
contesa logica della politica corrente: per lo stesso layer non esiste un
secondo slot libero in cui conservare la previsione per il token successivo.
Il confronto successivo più utile è mantenere invariati modello, otto layer,
checkpoint e tre ripetizioni, cambiando solo:

```text
DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER=2
```

Con distanza 2 ciascun layer potrà avere due richieste future contemporanee.
TEST_9 permetterà di verificare se diminuiscono `prefetch_bank_full` e gli
expert corretti non riusati, e se questa riduzione si traduce finalmente in
uno speedup medio più alto e più stabile.

## Risultati misurati — `output_2` (distanza 2, otto layer learned)

La directory [`output_2`](output_2) ripete esattamente il benchmark di
[`output`](output), mantenendo invariati checkpoint, otto layer learned,
modello GGUF, passo e tre ripetizioni B/C. L'unica modifica è:

```text
DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER=2
```

Ogni layer fisico da 3 a 10 può quindi usare due slot pinned per richieste
future contemporanee, invece di uno solo. Il confronto è dunque adatto a
misurare l'effetto della capacità aggiuntiva della banca pinned.

### Tempo di esecuzione: speedup stabile

| Ripetizione | Ordine | B baseline | C learned prefetch | Speedup di C |
|---|---|---:|---:|---:|
| 1 | B poi C | 102,185 s | 96,547 s | **+5,52%** |
| 2 | C poi B | 96,854 s | 89,484 s | **+7,61%** |
| 3 | B poi C | 103,576 s | 92,358 s | **+10,83%** |
| Media | — | 100,872 s | 92,796 s | **+7,99%** |

In tutte e tre le coppie C è più veloce di B. In media il learned prefetch
risparmia **8,08 secondi** per la stessa generazione. La deviazione standard
dello speedup è circa 2,68 punti percentuali: c'è ancora normale variabilità
di I/O e sistema, ma non esiste più una ripetizione in cui C sia più lenta.

Non si devono confrontare direttamente i tempi assoluti di B fra `output` e
`output_2`, perché ogni benchmark ha cache e condizioni del sistema proprie.
Il confronto affidabile è sempre B contro C all'interno della stessa coppia.

### Effetto della distanza 2 sulla banca pinned

| Misura, media per run C | Distanza 1 (`output`) | Distanza 2 (`output_2`) | Effetto osservato |
|---|---:|---:|---|
| `prefetch_bank_full` | ~210 | ~11 | quasi eliminato |
| Rifiuti durante precedente `CONSUMING` | ~190 | ~1 | contesa del singolo slot rimossa |
| Gruppi con riuso parziale | ~274 | ~438 | molti più gruppi trovano almeno un expert pronto |
| Expert riusati dalla RAM pinned | ~748 | ~1.201 | circa 453 riusi aggiuntivi per run |
| Expert ancora letti da SSD | ~2.276 | ~1.823 | circa 453 letture SSD in meno per run |
| Richieste accodate al worker | ~294 | ~493 | quasi tutte le 504 previsioni vengono accettate |

Questo conferma l'ipotesi formulata dopo `output`: il collo di bottiglia
dominante con distanza 1 non era l'esaurimento fisico della RAM pinned, ma la
capacità logica di un solo slot per layer. Quando lo slot era ancora
`CONSUMING`, DS4 rifiutava la previsione seguente. Il secondo slot permette di
conservare la nuova previsione mentre il trasferimento del token precedente si
sta completando.

L'audit mostra anche che il problema `CONSUMING` quasi scompare: da circa 190
rifiuti per run a circa uno. I pochi `prefetch_bank_full` rimasti sono invece
associati principalmente a una successiva pulizia stale: la politica potrebbe
ancora migliorare liberando o sostituendo più presto gli slot scaduti, ma non
è più il limite dominante.

### Limiti che restano dopo la correzione della contesa

La qualità delle predizioni non cambia, perché checkpoint e sequenza sono gli
stessi: Recall@6 online `0,435484`, cioè in media 2,61 expert corretti sui sei
predetti. Anche con distanza 2 non esiste un hit completo dei sei expert;
tutte le 434–441 opportunità di riuso sono perciò riusi parziali.

Inoltre, fra i pochi expert corretti non ancora riusati, la causa più frequente
non è più `prefetch_bank_full`, ma `worker_not_ready`: in alcune richieste il
worker SSD/pinned non ha completato la lettura prima della domanda reale della
GPU. I prossimi miglioramenti possibili sono quindi aumentare prudentemente
la distanza, ottimizzare la tempestività del worker, oppure aumentare la
Recall@6 del predictor. Il risultato di `output_2` mostra però già che la
politica a due slot rende utile in modo affidabile il prefetch learned.

## Risultati misurati — `output_3` (distanza 2, sedici layer learned)

La directory [`output_3`](output_3) mantiene distanza futura 2, checkpoint,
modello e tre ripetizioni invariati rispetto a [`output_2`](output_2). La sola
modifica è:

```text
DS4_TEST_9_PREFETCH_LAYERS=16
```

Il predictor viene quindi invocato per i layer fisici DS4 da 3 a 18, invece
che da 3 a 10. Le previsioni valutabili passano da 496 a 992 e le richieste
learned da 504 a 1.008 per run C.

### Tempo di esecuzione: +15,15% medio

| Ripetizione | Ordine | B baseline | C learned prefetch | Speedup di C |
|---|---|---:|---:|---:|
| 1 | B poi C | 106,330 s | 92,757 s | **+12,76%** |
| 2 | C poi B | 104,427 s | 89,874 s | **+13,94%** |
| 3 | B poi C | 109,788 s | 89,220 s | **+18,73%** |
| Media | — | 106,848 s | 90,617 s | **+15,15%** |

C è più veloce della rispettiva baseline in tutte e tre le coppie. Il
risparmio medio è **16,23 secondi**; rispetto al benchmark con 8 layer e
distanza 2 (`output_2`, +7,99%), lo speedup medio è quasi raddoppiato.

Anche qui il confronto corretto è interno a ogni coppia B/C. I tempi assoluti
delle baseline appartengono a sessioni differenti e dipendono anche dalle
condizioni di cache e I/O presenti durante quella sessione.

### Cosa riesce ancora a riusare il prefetch

| Ripetizione | Gruppi con riuso parziale | Expert riusati dalla RAM pinned | Expert caricati da SSD | Hit completi |
|---|---:|---:|---:|---:|
| 1 | 401 | 1.092 | 4.956 | 0 |
| 2 | 441 | 1.220 | 4.828 | 0 |
| 3 | 404 | 1.109 | 4.939 | 0 |

La Recall@6 online è `0,438172`, molto simile a quella dell'esperimento a otto
layer. Il miglioramento di tempo non deriva quindi da una migliore accuratezza
del predictor, ma dal fatto che il prefetch viene applicato a più richieste
MoE. Restano assenti gli hit completi: i gruppi riusati sono tutti parziali.

### Nuovo collo di bottiglia osservato con sedici layer

Il raddoppio dei layer produce 1.008 richieste learned per run, ma la trace
registra soltanto 507, 527 e 507 richieste effettivamente accodate al worker.
Parallelamente, gli eventi `prefetch_bank_full` salgono a 501, 481 e 501 per
run, contro circa 11 per run con otto layer e distanza 2.

| Misura, media per run C | 8 layer, distanza 2 | 16 layer, distanza 2 |
|---|---:|---:|
| Richieste learned | 504 | 1.008 |
| Richieste accodate al worker | ~493 | ~514 |
| `prefetch_bank_full` | ~11 | ~494 |
| `prefetch_stale_discarded` | ~46 | ~72 |
| Rifiuti durante precedente `CONSUMING` | ~1 | ~3 |

Il problema non è tornato a essere lo slot `CONSUMING`: quel contatore resta
molto basso grazie alla distanza 2. La distribuzione per layer mostra invece
che i layer fisici 12–18 ricevono 61 rifiuti `prefetch_bank_full` ciascuno in
ogni ripetizione. Questo è un fatto misurato dalla trace; indica che, con la
politica e la capacità attuali, le richieste aggiuntive non riescono a entrare
o a progredire come quelle dei layer più bassi.

Non è ancora corretto attribuire automaticamente questa soglia alla RAM
pinned fisica: il report non mostra un errore di allocazione. Il candidato da
esaminare è piuttosto la capacità effettiva della coda/worker e la sua
interazione con gli slot stale, perché il numero di richieste accodate rimane
vicino a quello dell'esperimento a otto layer anche dopo aver raddoppiato le
richieste generate.

In sintesi, `output_3` dimostra che applicare il learned prefetch a più layer
può aumentare significativamente e stabilmente lo speedup. Prima di salire a
20 o più layer conviene però misurare e rimuovere il nuovo limite osservato:
permettere che anche i layer 12–18 vengano accodati e completati con la stessa
regolarità dei primi layer.

## Diagnostica automatica di TEST_9

TEST_9 esegue il benchmark con la politica di prefetch corrente e aggiunge
osservazioni automatiche per capire perché, con molti layer learned, una parte
delle richieste riceve `prefetch_bank_full`.

Il runner TEST_9 abilita internamente la variabile
`DS4_TEST_9_PINNED_DIAGNOSTICS=1` solo nella fase C. Non occorre aggiungerla
alla configurazione di CLion. Ogni evento della trace C che proviene dal worker
o da un rifiuto del banco contiene ora `pinned_bank_snapshot`; il file
`pinned_ram_policy_audit.json` aggrega questi dati e il
`benchmark_summary.json` incorpora l'audit della relativa ripetizione C.

Per ogni rifiuto `prefetch_bank_full` verranno registrati:

- quanti slot dell'intero banco erano `EMPTY`, `QUEUED`, `READING`, `READY`,
  `CONSUMING`, `FAILED` o `TOO_LATE`; per compatibilità conserva anche il
  contatore aggregato storico `loading = queued + reading`;
- lo stato, il token destinazione e l'età di entrambi gli slot del layer che
  ha subito il rifiuto, dato che in questo esperimento la distanza è 2;
- quanti slot `QUEUED` o `READING` appartenevano a layer fisici inferiori;
- quanti slot hanno già allocato buffer pinned e una stima dei byte pinned
  effettivamente trattenuti;
- i tempi da enqueue a inizio lettura, da inizio lettura a `READY` e da
  enqueue a `READY` del worker;
- per ciascun layer, quante richieste sono state generate, accettate nello
  stato `QUEUED`, avviate dal worker, rese `READY`, rifiutate e fallite;
- quanti slot stale sono stati liberati subito da `QUEUED`, quanti sono stati
  marcati `TOO_LATE` durante `READING` e quanti sono stati infine liberati dal
  worker SSD.

Questo permette di distinguere quattro cause pratiche:

| Causa osservata | Significato |
|---|---|
| Tutti gli slot del layer sono `QUEUED` | Il layer ha già due richieste accettate e il worker non ha ancora iniziato a leggerle. |
| Tutti gli slot del layer sono `READING` | Il worker sta leggendo gli expert nei buffer pinned; quegli slot non sono riutilizzabili fino al termine dell'I/O. |
| È presente uno slot `CONSUMING` | La GPU sta ancora trasferendo/consumando il gruppo precedente; non è ancora riutilizzabile. |
| Gli slot sono `READY` o `FAILED` | C'è una richiesta vecchia che non è stata ancora consumata o pulita; il problema riguarda la politica di ricambio. |
| Molti `QUEUED` nei layer inferiori | Il worker CUDA cerca gli slot in ordine di layer crescente. Senza una coda FIFO, i layer bassi possono essere serviti prima e ritardare quelli alti. |

`prefetch_worker_queued` va letto con attenzione: non rappresenta una coda
FIFO separata. Significa soltanto che DS4 ha potuto cambiare uno slot da
`EMPTY` a `QUEUED`. Il worker esamina poi gli slot `QUEUED` partendo dal layer
fisico più basso. I nuovi contatori servono proprio a verificare se questa
priorità fissa è la causa dei rifiuti nei layer 12–18.

## Risultati misurati — `output_4` (diagnostica automatica, 16 layer, distanza 2)

La directory [`output_4`](output_4) ripete la configurazione di `output_3`:
16 layer learned, due slot per layer e tre coppie B/C alternate. La differenza
è che TEST_9 abilita automaticamente gli snapshot del banco pinned-RAM nella
fase C. Il comportamento del prefetch non viene modificato: vengono soltanto
registrati più dati.

### Prestazioni

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 106,017 s | 94,095 s | **+11,25%** |
| 2 | 115,780 s | 90,262 s | **+22,04%** |
| 3 | 108,015 s | 91,649 s | **+15,15%** |
| Media | 109,937 s | 92,002 s | **+16,15%** |

C resta più veloce in tutte e tre le coppie. Le variazioni rispetto a
`output_3` dipendono dalle normali variazioni I/O e cache fra esecuzioni; la
configurazione e il checkpoint sono identici.

### Cosa significava `LOADING + LOADING` prima della separazione degli stati

I rifiuti `prefetch_bank_full` sono 475, 446 e 463 nelle tre fasi C. Questi
risultati appartengono alla run `output_4`, eseguita **prima** della
separazione attuale in `QUEUED`, `READING` e `TOO_LATE`. Di questi,
rispettivamente 467, 439 e 462 trovano entrambi gli slot dello stesso layer in
stato `LOADING`.

Lo snapshot dimostra che `LOADING` non equivale necessariamente a “il worker
sta leggendo ora dall'SSD”. Significa più precisamente: “la richiesta è stata
accettata, ma non è ancora `READY`”. Può quindi essere:

1. una lettura SSD effettivamente in corso;
2. una richiesta già accettata ma ancora in attesa che l'unico worker la
   selezioni.

All'inizio della generazione compare anche il caso intuitivo:

```text
layer 15, nuova richiesta target 23
slot [0]: LOADING, target 21
slot [1]: LOADING, target 22
```

La richiesta del token 23 non trova uno slot `EMPTY` e viene rifiutata. Questo
caso, però, non descrive la maggioranza della run. Calcolando la differenza fra
il token rifiutato e il più recente dei due token già presenti, i rifiuti con
uno scarto superiore a 20 token sono:

| Ripetizione | Rifiuti totali | Slot più recente oltre 20 token indietro |
|---|---:|---:|
| 1 | 475 | 252 |
| 2 | 446 | 259 |
| 3 | 463 | 215 |

Nei soli layer fisici 15–18, il valore è **164 rifiuti su 244** in ogni
ripetizione. Questi slot non contengono quindi semplicemente i due token
immediatamente precedenti: contengono richieste vecchie rimaste `LOADING` per
molto tempo.

### Causa confermata: starvation del worker, non saturazione fisica della RAM pinned

I layer bassi 3–8 ricevono normalmente tutte le 63 richieste e il worker le
avvia quasi tutte. Nei layer 15–18, invece, per ciascuna run sono accettate
soltanto due richieste per layer e il worker non ne avvia nessuna. I due slot
restano quindi `LOADING`; le successive 61 richieste di quel layer incontrano
`prefetch_bank_full`.

Durante un rifiuto il numero medio di slot `LOADING` appartenenti a layer più
bassi è circa 14–15, con percentili alti fra 21 e 24. Il worker, infatti,
scansiona gli slot in ordine crescente di layer e sceglie il primo `LOADING`:

```text
layer 0, poi 1, poi 2, ... poi 18
```

Non esiste una coda FIFO globale né una rotazione equa fra layer. I layer alti
possono perciò restare indefinitamente dietro al lavoro dei layer bassi.

La lettura SSD non è il limite primario: dura circa 136–145 ms in media. La
sola attesa prima che il worker inizi una lettura è invece 0,8–1,4 s in media,
con picchi fra 26 e 65 s. Inoltre la stima massima dei buffer pinned già
allocati è circa 0,85–1,04 GB e non sono presenti errori di allocazione. Il
problema osservato è quindi **logico e di scheduling**, non l'esaurimento della
RAM pinned del sistema.

La modifica tecnica successiva da valutare è sostituire la scansione fissa dal
layer più basso con una politica equa: una coda FIFO delle richieste `QUEUED`
oppure una selezione round-robin fra layer. Questo dovrebbe permettere anche ai
layer 12–18 di passare a `READY` senza aumentare il numero di slot o la RAM
pinned riservata.

## Gestione sicura delle richieste di prefetch arrivate troppo tardi

Questa gestione è ora implementata nella banca pinned di DS4. Nelle run
precedenti lo stato `LOADING` significava soltanto che la richiesta era stata
accettata ma gli expert non erano ancora `READY` nella RAM pinned. Il codice
attuale separa invece le due situazioni: `QUEUED` significa che la richiesta
attende il worker SSD, mentre `READING` significa che il worker sta già
scrivendo nei buffer pinned dello slot. Questa distinzione permette di
recuperare prima gli slot occupati da previsioni ormai troppo tardive, senza
introdurre corse tra il thread di decoding e il worker SSD.

Con *thread di decoding* si intende il flusso di inferenza del processo DS4
durante il decoding. È il codice host di DS4 che domanda gli expert reali per
la coppia `(token, layer)` e coordina il backend CUDA; non è il worker SSD e
non è il kernel eseguito dalla GPU.

### Esempio: perché una previsione vecchia dovrebbe lasciare spazio

Per il learned prefetch attuale, `target_token = 21` significa che i sei expert
predetti sono destinati al momento in cui DS4 eseguirà **il layer 15 del token
21**. Non indica tutta l'elaborazione del token 21, ma la coppia precisa:

```text
(token 21, layer 15)
```

Supponiamo che DS4 stia già elaborando `(token 22, layer 15)` e che arrivi una
nuova previsione per `(token 23, layer 15)`:

```text
slot[0]: QUEUED oppure READING, target 21
slot[1]: QUEUED oppure READING, target 22
nuova richiesta:        target 23
```

Il prefetch conservato nello `slot[0]` è ormai certamente arrivato troppo
tardi: contiene gli expert previsti per il layer 15 del token 21, ma DS4 ha
già completato quel layer per quel token. Anche se la lettura dell'SSD
terminasse ora, quegli expert non potrebbero più accelerare la run. Idealmente
lo slot dovrebbe lasciare spazio alla richiesta per il token 23.

Nel percorso learned, il target è sempre costruito come `token sorgente + 1`.
La variabile `DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER=2` **non** trasforma la
previsione in `t+2`: abilita soltanto due posti fisici, `slots[l][0]` e
`slots[l][1]`, per conservare più richieste future dello stesso layer.

La risposta del predictor CPU è asincrona. Perciò, quando una nuova risposta
arriva, DS4 può essere già al token sorgente, a un token successivo oppure aver
già superato il target. Il riferimento sicuro per decidere se una richiesta è
troppo tardiva è la domanda reale del thread di decoding: quando DS4 richiede
gli expert di `(token 22, layer 15)`, ogni slot dello stesso layer con
`target_token < 22` non può più essere utile.

### I due stati che distinguono l'attesa dalla lettura SSD

Nell'esempio precedente lo `slot[0]`, destinato a `(token 21, layer 15)`, è
ormai inutile. L'implementazione attuale distingue esplicitamente quale delle
due situazioni seguenti si stia verificando.

**A. Richiesta in attesa del worker SSD.** Il worker non ha ancora iniziato a
leggere gli expert del target 21. Nessun altro thread sta usando i buffer dello
`slot[0]`; il thread di decoding potrebbe quindi liberare subito lo slot,
portarlo a `EMPTY` e inserire senza rischi la nuova richiesta per il target 23.

**B. Lettura dell'SSD già in corso.** Il worker SSD ha già iniziato a leggere
gli expert del target 21 e sta scrivendo gate, up e down nei buffer pinned dello
`slot[0]`. In questo caso non è sicuro sostituire immediatamente target 21 con
target 23: il worker continuerebbe a scrivere i dati del token 21 in buffer che
il thread di decoding avrebbe appena assegnato al token 23. Sarebbe una race
condition e potrebbe rendere errati i dati di prefetch.

Nel caso A lo slot può essere liberato subito. Nel caso B il thread di decoding
deve solo segnalarlo come `TOO_LATE`: il risultato non sarà usato e il worker,
quando avrà concluso la lettura già avviata, svuoterà lo slot in sicurezza.

### Perché DS4 non può semplicemente sovrascrivere uno slot in `READING`

Nell'implementazione corrente il codice di enqueue usa soltanto slot `EMPTY`:
non rimpiazza uno slot occupato. Di conseguenza, nell'esempio, non può
recuperare `slot[0]` per inserire il target 23.

Quando la domanda reale incontra una richiesta con un token precedente, DS4
libera subito uno slot `QUEUED`; se lo slot è già `READING`, lo marca
`TOO_LATE`. Il worker lo svuota appena termina la lettura già iniziata. Questo
evita che richieste vecchie restino inutilmente occupate e, allo stesso tempo,
evita di riutilizzare buffer che il worker sta ancora scrivendo.

| Transizione implementata | Chi la esegue | Quando |
|---|---|---|
| `EMPTY → QUEUED` | thread di decoding | Riceve dal predictor CPU una nuova previsione learned e la inserisce nello slot libero. |
| `QUEUED → READING` | worker SSD | Seleziona la richiesta e inizia a leggere gli expert dall'SSD verso la RAM pinned. |
| `QUEUED → EMPTY` | thread di decoding | Scopre che il token presente in quello slot è già stato processato dal decoding prima che il worker abbia iniziato la sua lettura dall'SSD per portarlo in RAM pinned. |
| `READING → TOO_LATE` | thread di decoding | Gli expert del layer `l` per il token presente in quello slot sono ormai già stati elaborati dalla fase di decoding di DS4: DS4 ha già attraversato quel layer per quel token. |
| `TOO_LATE → EMPTY` | worker SSD | Termina la lettura degli expert del layer `l` per il token presente in quello slot, ormai già elaborato dalla fase di decoding di DS4, scarta il risultato e rende lo slot riusabile. |
| `READING → READY` | worker SSD | Completa una lettura che è ancora utile e pubblica gli expert nella RAM pinned. |
| `READY → CONSUMING` | thread di decoding | Trova nello slot gli expert già pronti per il token e il layer che DS4 sta elaborando e avvia il loro trasferimento dalla RAM pinned alla VRAM. |
| `CONSUMING → EMPTY` | thread di decoding | Il trasferimento in VRAM degli expert presenti nello slot, richiesti dal token e dal layer che DS4 sta elaborando, è completato; lo slot può quindi essere riusato. |

`TOO_LATE` è più chiaro di `CANCELLED`: non comunica soltanto che una richiesta
è stata annullata, ma il motivo. Il decoding ha già superato la coppia
`(token target, layer)` e il risultato della lettura non può più accelerare
l'inferenza.

Uno slot `READING` non può diventare subito `EMPTY`, perché il worker potrebbe
stare scrivendo nei suoi buffer pinned. Il thread di decoding lo marca quindi
`TOO_LATE`; il worker lo libera soltanto al termine della lettura. Uno slot
`QUEUED`, invece, non è ancora usato dal worker e può diventare subito `EMPTY`.

## Risultati misurati — `output_5` (riciclo sicuro degli slot stale)

La directory [`output_5`](output_5) usa 16 layer learned, distanza due e tre
coppie B/C alternate. È la prima run eseguita con gli stati `QUEUED`,
`READING` e `TOO_LATE` realmente attivi.

### Il riciclo degli slot funziona

| Ripetizione | `prefetch_bank_full` | `QUEUED → EMPTY` | `READING → TOO_LATE` | `TOO_LATE → EMPTY` |
|---|---:|---:|---:|---:|
| 1 | 3 | 458 | 5 | 5 |
| 2 | 1 | 469 | 3 | 3 |
| 3 | 2 | 435 | 2 | 2 |

Nella precedente `output_4`, con la vecchia gestione `LOADING`, i rifiuti
`prefetch_bank_full` erano rispettivamente 475, 446 e 463. Ora sono soltanto
3, 1 e 2: il nuovo percorso recupera quasi tutte le richieste che altrimenti
avrebbero trovato entrambi gli slot logicamente occupati.

I conteggi `READING → TOO_LATE` e `TOO_LATE → EMPTY` coincidono in ogni
ripetizione. Questo conferma il comportamento atteso: il thread di decoding
non sovrascrive un buffer mentre il worker SSD lo usa; il worker conclude la
lettura già iniziata, scarta il risultato tardivo e libera poi lo slot.

### Il limite rimasto è la scelta del worker SSD

Ridurre `prefetch_bank_full` non basta da solo a rendere pronti più gruppi di
expert. Il worker continua a scegliere lo slot `QUEUED` con il layer fisico più
basso. Nella prima ripetizione, per esempio, i layer 3–8 hanno completato 63
letture su 63; il layer 11 ne ha completate 25, il layer 12 soltanto 3 e i
layer 13–18 nessuna. Le loro richieste sono state liberate come stale prima che
il worker potesse iniziare l'I/O.

Questo spiega perché non ci sono ancora prefetch hit completi: nelle tre
ripetizioni `full_prefetch_hits` resta zero e il riuso continua a essere solo
parziale. I gruppi parzialmente riusati sono 463, 451 e 463, valori simili a
quelli della run precedente.

### Prestazioni osservate in `output_5`

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 95,116 s | 96,210 s | -1,15% |
| 2 | 102,404 s | 94,217 s | +7,99% |
| 3 | 100,297 s | 98,559 s | +1,73% |
| Media | 99,273 s | 96,329 s | **+2,86%** |

Due coppie su tre sono più veloci con il prefetch, ma la variabilità è ancora
alta e una coppia è più lenta. Questo esperimento dimostra con certezza che il
riciclo stale elimina il blocco logico del banco; non dimostra ancora uno
speedup stabile superiore a quello precedente.

## Variante sperimentale misurata: due worker SSD divisi fra layer pari e dispari

Per verificare se il collo di bottiglia è soprattutto il fatto che esiste un
solo lettore SSD, la variante che ha prodotto `output_6` e `output_7` impostava
internamente `DS4_TEST_9_PARITY_WORKERS=1`. Non occorreva aggiungere questa
variabile alla configurazione CLion.

DS4 avvia due thread che leggono dallo stesso GGUF di prefetch:

| Worker | Layer fisici che può selezionare |
|---|---|
| `worker_id = 0` | pari: 0, 2, 4, ..., 42 |
| `worker_id = 1` | dispari: 1, 3, 5, ..., 41 |

Con 16 layer learned, che nella configurazione attuale usano i layer fisici
3–18, ciascun worker gestisce otto layer: quello dispari 3, 5, ..., 17 e
quello pari 4, 6, ..., 18. Ogni worker continua a scegliere il layer più basso
del proprio gruppo: questa è quindi una prova del parallelismo I/O, non ancora
una politica FIFO o round-robin.

Le letture possono avvenire contemporaneamente, perché usano slot e buffer
pinned distinti. Il mutex comune continua però a proteggere tutti i passaggi
di stato `QUEUED`, `READING`, `TOO_LATE` e `EMPTY`: nessun buffer viene
riassegnato mentre un worker lo sta scrivendo.

Nei record `prefetch_worker_started`, `prefetch_worker_ready` e
`prefetch_worker_failed`, la trace C ora contiene `worker_id`. Il valore `0` o
`1` identifica il worker SSD che ha svolto la lettura; `4294967295` indica un
evento prodotto dal thread di decoding prima che una lettura fosse assegnata.
Nel confronto con le run precedenti bisogna verificare sia se aumentano le
letture completate nei layer alti, sia se la lettura media peggiora: due thread
sullo stesso SSD esterno possono aumentare il throughput oppure contendersi la
banda disponibile.

## Risultati preliminari — `output_6` (due worker pari/dispari)

La directory [`output_6`](output_6) è la prima esecuzione con due worker SSD,
16 layer learned, distanza due e tre coppie B/C alternate. I tempi della fase
di inferenza sono validi: la modifica relativa ai due worker agisce nel backend
CUDA e non dipende dai record diagnostici scritti successivamente nella trace.

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 101,847 s | 91,333 s | +10,32% |
| 2 | 111,326 s | 95,012 s | +14,65% |
| 3 | 101,392 s | 87,364 s | +13,84% |
| Media | 104,855 s | 91,236 s | **+12,94%** |

Il miglioramento è molto più alto e più stabile rispetto alla media `+2,86%`
di `output_5`. È quindi un'indicazione forte che due letture speculative in
parallelo possano aiutare il prefetch.

### Limite noto della telemetria di `output_6`

Questa run **non può essere usata** per analizzare il numero di letture di
ciascun worker, la latenza per worker o la copertura per layer. Durante la sua
esecuzione era stato aggiunto il campo `worker_id` alla struttura degli eventi
CUDA, ma il Makefile non ha ricompilato automaticamente l'oggetto C che
trasforma gli eventi nella trace JSON. I campi successivi nella struttura sono
quindi stati letti con offset errati: nell'audit compaiono layer e `worker_id`
impossibili.

Il problema riguarda soltanto l'interpretazione della trace, non il lavoro
svolto dai worker né i tempi B/C. Il passaggio del campo è stato corretto e
DS4 è stato ricompilato completamente. La successiva `output_7` dovrà
confermare il risultato temporale e fornire l'audit affidabile di worker e
layer.

## Risultati verificati — `output_7` (due worker pari/dispari)

La directory [`output_7`](output_7) ripete esattamente l'esperimento di
`output_6`, ma dopo la ricompilazione completa della struttura eventi. È quindi
la run di riferimento per valutare la variante a due worker: 16 layer learned,
distanza due e tre coppie B/C alternate.

### Speedup confermato

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 94,007 s | 89,642 s | +4,64% |
| 2 | 105,057 s | 87,801 s | +16,43% |
| 3 | 99,718 s | 88,084 s | +11,67% |
| Media | 99,594 s | 88,509 s | **+10,91%** |

Tutte e tre le coppie sono più veloci con il prefetch. Rispetto alla media
`+2,86%` dell'`output_5` a un worker, il guadagno è maggiore e non presenta
più una ripetizione negativa.

### I due worker lavorano davvero in parallelo

| Ripetizione | Worker pari: letture avviate / pronte | Worker dispari: letture avviate / pronte | Lettura media pari | Lettura media dispari |
|---|---:|---:|---:|---:|
| 1 | 419 / 415 | 405 / 401 | 166,85 ms | 172,25 ms |
| 2 | 412 / 410 | 401 / 399 | 169,01 ms | 173,71 ms |
| 3 | 419 / 417 | 409 / 406 | 163,25 ms | 167,57 ms |

La ripartizione è quasi bilanciata: non esiste più un unico worker che deve
servire tutti i 16 layer. Le letture individuali sono nell'ordine di 163–174
ms; sono leggermente più lente rispetto a un lettore isolato, ma le due letture
contemporanee aumentano abbastanza il throughput totale da migliorare il tempo
di decode.

### Copertura dei layer: migliorata, ma non uniforme

Con un worker, nella prima ripetizione di `output_5` i layer fisici 13–18 non
avevano completato neppure una lettura. In `output_7`, i layer 13 e 14 ne
completano rispettivamente 58 e 58 già nella prima ripetizione. I layer 15 e
16 ricevono 33 e 43 letture; i più alti 17 e 18 ne ricevono soltanto 3 e 7.

Il motivo è che ogni worker continua a scegliere il layer più basso **del
proprio gruppo di parità**. La divisione pari/dispari ha quindi rimosso la
starvation globale, ma conserva una starvation più debole all'interno di ogni
gruppo: il worker dispari favorisce 3, 5, 7, ... prima di 17 e quello pari
favorisce 4, 6, 8, ... prima di 18. Una futura coda FIFO o round-robin potrà
rendere equa anche questa scelta interna.

### Effetto sul riuso degli expert

| Indicatore per ripetizione | `output_5`, un worker | `output_7`, due worker |
|---|---:|---:|
| Gruppi con riuso parziale | 451–463 | **708–722** |
| Expert riusati dalla RAM pinned | 1.227–1.278 | **1.898–1.947** |
| Expert caricati dall'SSD nella domanda reale | 4.770–4.821 | **4.101–4.150** |
| `prefetch_bank_full` | 1–3 | 0–4 |

Non ci sono ancora `full_prefetch_hits`: con Recall@6 circa 0,506 è normale
che almeno un expert del gruppo richiesto manchi spesso. Tuttavia il maggiore
numero di riusi parziali evita molte letture SSD sincrone nella domanda reale,
ed è coerente con lo speedup misurato.

## Variante FIFO sperimentata: due worker con FIFO globale condivisa e controllo di freschezza

La variante a due worker ha migliorato molto il throughput, ma ogni worker
favorisce ancora i layer più bassi del proprio gruppo. TEST_9 combina ora i due
vantaggi: **due worker SSD** che scelgono dalla stessa coda FIFO globale le
richieste nell'ordine in cui sono create dal decoding. Il runner imposta
internamente `DS4_TEST_9_WORKER_POLICY=fifo` e
`DS4_TEST_9_FIFO_WORKERS=2`; non devi aggiungere queste variabili in CLion.

La selezione della prossima lettura avviene sotto il mutex della banca pinned:
il primo worker prende la richiesta QUEUED con `enqueue_sequence` più piccolo e
la porta a READING; il secondo worker prende quindi la più vecchia tra quelle
rimaste. I due accessi SSD successivi avvengono fuori dal mutex, in buffer
pinned appartenenti a slot distinti. Non c'è quindi una race condition né il
rischio che entrambi leggano la stessa richiesta.

L'obiettivo è separare due effetti diversi:

| Confronto | Cosa misura |
|---|---|
| Un worker a scansione per layer (`output_5`) | Effetto della priorità fissa ai layer bassi. |
| Un worker FIFO (`output_8`) | Effetto dell'ordine equo di arrivo, senza parallelismo I/O. |
| Due worker pari/dispari (`output_7`) | Effetto di due letture SSD in parallelo. |
| Due worker FIFO, variante attiva | Effetto combinato di ordine equo e parallelismo I/O. |

### Controllo di freschezza prima della lettura

Una FIFO ingenua potrebbe prendere una richiesta vecchia che non è stata
ancora letta, ma che è già inutile perché DS4 ha ormai attraversato quel layer
per il token a cui la richiesta si riferisce. Il worker FIFO dovrà quindi
memorizzare, per ogni layer fisico, l'ultimo token già elaborato dal thread di
decoding e applicare questo controllo **prima** di iniziare la lettura SSD:

```text
se slot.target_token < ultimo_token_elaborato_per_quel_layer:
    QUEUED → EMPTY
    non leggere nulla dall'SSD
altrimenti:
    QUEUED → READING
    leggere gli expert dall'SSD nella RAM pinned
```

`slot.target_token` è il token futuro per cui lo slot contiene la previsione;
`ultimo_token_elaborato_per_quel_layer` è il token più recente per cui il
thread di decoding di DS4 ha già attraversato quello stesso layer.

La disuguaglianza stretta è importante:

| Condizione | Significato |
|---|---|
| `target_token < ultimo_token_elaborato` | Lo slot è certamente troppo tardi: il worker lo libera senza I/O. |
| `target_token == ultimo_token_elaborato` | Il token è quello corrente per quel layer; il worker non lo scarta in base a questo solo controllo. |
| `target_token > ultimo_token_elaborato` | È una richiesta per un token futuro e può essere letta. |

Il normale percorso `READING → TOO_LATE → EMPTY` rimane necessario: anche una
richiesta valida al momento della selezione può diventare tardiva mentre il
worker sta già leggendo l'SSD. Il controllo FIFO evita invece le letture
inutili che sono già stale **prima** dell'inizio dell'I/O.

La trace e l'audit registrano questi casi come
`prefetch_stale_queued_discarded_by_worker`. Il relativo totale nel
`pinned_ram_policy_audit.json` misura quante letture SSD sono state risparmiate
dal controllo di freschezza del worker, separandole dagli slot stale liberati
dal thread di decoding (`prefetch_stale_queued_discarded`).

## Risultati verificati — `output_8` (un worker FIFO)

La directory [`output_8`](output_8) usa 16 layer learned, distanza due e la
nuova politica FIFO con un solo worker SSD. È il confronto diretto con
`output_5`: stessa quantità di parallelismo I/O, ma ordine di selezione FIFO
anziché scansione dal layer più basso al più alto.

### Risultato: FIFO da sola non tiene il passo

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 107,545 s | 111,572 s | -3,75% |
| 2 | 115,208 s | 112,671 s | +2,20% |
| 3 | 108,271 s | 108,498 s | -0,21% |
| Media | 110,341 s | 110,914 s | **-0,58%** |

Una sola ripetizione è più veloce con il prefetch; la media è leggermente
peggiore della baseline. Questo non dimostra che FIFO sia logicamente errata:
dimostra che **un solo worker non ha capacità I/O sufficiente** per mantenere
pronte le richieste FIFO con 16 layer e distanza due.

### Evidenze nell'audit

| Indicatore per ripetizione | `output_5`, un worker a scansione | `output_8`, un worker FIFO |
|---|---:|---:|
| Attesa media prima della lettura | 588–677 ms | **2.222–2.539 ms** |
| `prefetch_bank_full` | 1–3 | **106–118** |
| Gruppi con riuso parziale | 451–463 | **4–22** |
| Expert riusati dalla RAM pinned | 1.227–1.278 | **13–54** |
| Expert caricati dall'SSD nella domanda reale | 4.770–4.821 | **5.994–6.035** |

FIFO fa arrivare alcune letture anche ai layer alti — nella seconda
ripetizione i layer fisici 17 e 18 hanno rispettivamente 8 e 6 letture pronte,
mentre nella prima ripetizione di `output_5` ne avevano zero. Tuttavia queste
letture arrivano quasi sempre troppo tardi per essere riutilizzate dalla
domanda reale: per questo il riuso parziale crolla da circa 450 gruppi a meno
di 25.

Il nuovo contatore
`prefetch_stale_queued_discarded_by_worker_total` vale zero nelle tre
ripetizioni. Non è un malfunzionamento: il thread di decoding incontra gli
stessi layer prima che il worker FIFO scelga molte delle richieste vecchie e le
libera già tramite `prefetch_stale_queued_discarded`. Il controllo di
freschezza resta corretto come protezione aggiuntiva, ma in questo carico non
è lui a risparmiare I/O in modo misurabile.

### Conclusione operativa

`output_8` elimina l'ipotesi che basti cambiare l'ordine di un singolo
worker. Il risultato migliore misurato finora resta `output_7`: due worker SSD
pari/dispari, speedup medio `+10,91%`. Il confronto successivo è stato quindi
**due worker con una FIFO globale condivisa**, mantenendo sempre il controllo
di freschezza prima dell'I/O; i suoi risultati sono riportati subito sotto.

## Risultati verificati — `output_9` (due worker FIFO globali)

La directory [`output_9`](output_9) è il confronto diretto richiesto dopo
`output_8`: 16 layer learned, distanza due, tre coppie B/C alternate e **due
worker SSD che condividono la stessa FIFO globale**.

### Tempi B/C

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 104,240 s | 102,216 s | +1,94% |
| 2 | 108,256 s | 97,082 s | +10,32% |
| 3 | 104,797 s | 96,670 s | +7,75% |
| **Media** | **105,764 s** | **98,656 s** | **+6,67%** |

Tutte e tre le coppie migliorano con C: la combinazione FIFO + due worker è
quindi utile e molto migliore del FIFO a un solo worker (`output_8`, media
`-0,58%`). Non supera però la precedente divisione pari/dispari (`output_7`,
media `+10,91%`).

### Cosa mostrano i worker e la banca pinned

| Ripetizione | Letture pronte worker 0 / 1 | Attesa media prima della lettura | Lettura SSD media | Bank full |
|---|---:|---:|---:|---:|
| 1 | 367 / 373 | 1.873 ms | 185,9 ms | 82 |
| 2 | 486 / 487 | 1.401 ms | 162,3 ms | 6 |
| 3 | 474 / 476 | 1.431 ms | 157,1 ms | 10 |

I due worker si dividono il lavoro quasi esattamente a metà, dunque la FIFO
globale condivisa è implementata correttamente e le letture avvengono in
parallelo. La prima ripetizione ha però una coda più lunga e molti più eventi
`prefetch_bank_full`; è compatibile con una maggiore variabilità del tempo di
lettura dall'SSD, non con uno sbilanciamento tra worker.

FIFO distribuisce bene le letture anche in alto: nelle ripetizioni 2 e 3 i
layer fisici 3–18 ricevono quasi tutti 56–63 letture pronte. Il prezzo è che
le richieste urgenti dei layer bassi non hanno più precedenza. Il ritardo in
coda resta dell'ordine di 1,4 s e molte richieste diventano tardive prima di
essere utili.

| Ripetizione | Gruppi con riuso parziale | Expert riusati dalla RAM pinned | Expert ancora letti dall'SSD alla domanda |
|---|---:|---:|---:|
| 1 | 47 | 145 | 5.903 |
| 2 | 188 | 553 | 5.495 |
| 3 | 146 | 430 | 5.618 |

Il riuso è molto migliore di `output_8`, ma non raggiunge quello di
`output_7` (708–722 gruppi e 1.898–1.947 expert riusati). Per questo, con
questa macchina e distanza due, la politica pari/dispari rimane la migliore
tra quelle misurate. Il prossimo candidato da confrontare non è una FIFO pura,
ma una politica ibrida che preservi una parte della priorità ai layer vicini
alla domanda e garantisca comunque turni ai layer alti.

## Variante sperimentata: tre worker con scansione per layer modulo tre

`output_10` prova un'estensione diretta dell'`output_7`. La FIFO globale di
`output_9` resta disponibile come risultato sperimentale. Il runner usato per
`output_10` avviava internamente **tre worker SSD** e assegnava ogni layer
fisico a uno solo di loro:

```text
worker 0: layer fisico % 3 = 0  -> 3, 6, 9, 12, 15, 18
worker 1: layer fisico % 3 = 1  -> 4, 7, 10, 13, 16
worker 2: layer fisico % 3 = 2  -> 5, 8, 11, 14, 17
```

Ogni worker conserva la scansione dal layer più basso al più alto **solo nel
proprio gruppo**. Tre worker possono dunque eseguire fino a tre letture SSD in
parallelo, ciascuna verso buffer pinned di uno slot distinto. Il mutex della
banca protegge esclusivamente la scelta e i cambi di stato degli slot; non
serializza le letture dall'SSD.

Per riprodurre questa variante storica, il runner impostava internamente:

```text
DS4_TEST_9_PARTITION_WORKERS=3
DS4_TEST_9_WORKER_POLICY=layer_partition
```

Non devi aggiungere queste variabili in CLion. L'obiettivo è verificare se il
guadagno di throughput compensa l'eventuale rallentamento delle singole letture
dovuto a tre accessi concorrenti sullo stesso SSD esterno.

## Risultati verificati — `output_10` (tre worker modulo tre)

La directory [`output_10`](output_10) usa 16 layer learned, distanza due e la
variante a tre worker descritta sopra. È il confronto diretto con `output_7`:
stessa politica di scansione per layer, ma tre gruppi modulo tre invece di due
gruppi pari/dispari.

### Speedup

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 98,544 s | 82,308 s | +16,48% |
| 2 | 105,852 s | 88,221 s | +16,66% |
| 3 | 98,429 s | 81,933 s | +16,76% |
| **Media** | **100,942 s** | **84,154 s** | **+16,63%** |

Questo è il miglior risultato di speedup ottenuto finora in TEST_9. Tutte le
tre coppie B/C mostrano un miglioramento molto simile; rispetto alla media
`+10,91%` dei due worker pari/dispari (`output_7`), il guadagno aggiuntivo è
circa 5,7 punti percentuali.

### Perché tre worker aiutano

| Variante | Attesa media prima della lettura | Lettura SSD media | Speedup medio |
|---|---:|---:|---:|
| Due worker pari/dispari, `output_7` | 503–569 ms | 165–171 ms | +10,91% |
| Tre worker modulo tre, `output_10` | **361–398 ms** | 184–196 ms | **+16,63%** |

Una singola lettura con tre accessi concorrenti è un po' più lenta, come era
prevedibile perché tutti usano lo stesso SSD esterno. Tuttavia ogni worker ha
meno layer da servire: la coda si accorcia molto più di quanto peggiori la
lettura individuale. Il throughput complessivo è quindi superiore.

| Ripetizione | Worker 0: letture pronte | Worker 1: letture pronte | Worker 2: letture pronte | `prefetch_bank_full` |
|---|---:|---:|---:|---:|
| 1 | 322 | 306 | 306 | 4 |
| 2 | 347 | 313 | 314 | 0 |
| 3 | 342 | 306 | 306 | 3 |

Il worker 0 ha sei layer nel proprio gruppo, mentre worker 1 e 2 ne hanno
cinque; la lieve differenza nel numero di letture è quindi attesa. La banca
pinned non è più un ostacolo rilevante: gli eventi `prefetch_bank_full` sono
0–4 per ripetizione.

### Riuso ottenuto

| Ripetizione | Gruppi con riuso parziale | Expert riusati dalla RAM pinned | Expert ancora letti dall'SSD alla domanda |
|---|---:|---:|---:|
| 1 | 854 | 2.322 | 3.726 |
| 2 | 902 | 2.471 | 3.577 |
| 3 | 876 | 2.362 | 3.686 |

Il riuso supera anche `output_7` (1.898–1.947 expert riusati). Il layer 18
rimane il più difficile perché condivide il worker 0 con cinque layer più
bassi, ma gli altri layer ricevono quasi tutti 57–63 letture pronte per
ripetizione.

## Variante sperimentata: tre worker, FIFO separata per ciascun gruppo

`output_11` ha provato la politica ibrida richiesta. Restano gli stessi gruppi di
`output_10`, ma cambia la scelta all'interno di ciascun gruppo:

```text
worker 0: layer 3, 6, 9, 12, 15, 18  -> sceglie il più vecchio del proprio gruppo
worker 1: layer 4, 7, 10, 13, 16      -> sceglie il più vecchio del proprio gruppo
worker 2: layer 5, 8, 11, 14, 17      -> sceglie il più vecchio del proprio gruppo
```

Non è una FIFO globale: una richiesta del layer 3 non può sottrarre lavoro ai
worker 1 e 2. Non è neppure la scansione di `output_10`: una richiesta vecchia
del layer 18 non resta dietro a tutte le richieste più recenti dei layer 3, 6,
9, 12 e 15. Ogni worker esegue al massimo una lettura SSD alla volta, mentre
le tre letture dei tre worker continuano ad avvenire in parallelo.

Durante quella run, il runner impostava internamente:

```text
DS4_TEST_9_PARTITION_WORKERS=3
DS4_TEST_9_WORKER_POLICY=partition_fifo
```

Non devi aggiungere o cambiare queste variabili in CLion. Dopo `output_11` il
runner è stato ripristinato alla variante attiva di `output_10`, che resta la
migliore tra quelle misurate.

### Esempio concreto: cosa cambia quando arriva una nuova richiesta

Una predizione non viene assegnata direttamente a un worker. Quando DS4 la
riceve, inserisce nello slot pinned del relativo layer il token futuro, i sei
expert predetti e lo stato `QUEUED`. Il worker del gruppo che possiede quel
layer la leggerà dall'SSD quando diventa libero.

Supponiamo che DS4 stia decodificando il token `100`. Durante il layer fisico
3, il predictor produce la previsione per gli expert del layer 3 del token
futuro `101`:

```text
richiesta: token futuro 101, layer 3
→ slot del layer 3: QUEUED
→ gruppo del worker 0
```

Nel prosieguo del token 100 vengono create richieste analoghe per gli altri
layer. Con tre worker modulo tre, il lavoro è diviso così:

```text
worker 0 → layer 3, 6, 9, 12, 15, 18
worker 1 → layer 4, 7, 10, 13, 16
worker 2 → layer 5, 8, 11, 14, 17
```

Ogni worker può leggere un solo gruppo di sei expert alla volta, ma i tre
worker possono leggere in parallelo verso buffer pinned appartenenti a slot
distinti.

Immaginiamo ora che worker 0 abbia appena terminato una lettura. Nel suo gruppo
trova queste richieste già inserite da DS4:

```text
layer 3,  token futuro 102  → QUEUED, inserita più recentemente
layer 6,  token futuro 101  → QUEUED, inserita prima
layer 9,  token futuro 101  → QUEUED
layer 12, token futuro 101  → QUEUED
layer 15, token futuro 101  → QUEUED
layer 18, token futuro 101  → QUEUED
```

#### Scansione per layer di `output_10`

Senza FIFO, worker 0 cerca sempre il layer fisico numericamente più basso del
proprio gruppo che abbia uno slot `QUEUED`. In questo esempio sceglie quindi:

```text
layer 3, token futuro 102
```

anche se le richieste dei layer 6–18 per il token 101 sono più vecchie. Il
vantaggio è che i layer bassi ricevono priorità: durante il decoding DS4 li
incontra prima e hanno quindi una finestra temporale più corta per diventare
pronti. Lo svantaggio è che le richieste del layer 18 possono aspettare molto;
quando worker 0 torna libero può essere già comparsa un'altra richiesta del
layer 3 per un token ancora successivo.

#### FIFO per gruppo di `output_11`

Con FIFO locale, worker 0 osserva ancora soltanto i propri layer, ma sceglie lo
slot `QUEUED` inserito per primo. Nello stesso esempio sceglie:

```text
layer 6, token futuro 101
```

e prosegue concettualmente nell'ordine:

```text
layer 9,  token futuro 101
layer 12, token futuro 101
layer 15, token futuro 101
layer 18, token futuro 101
layer 3,  token futuro 102
```

FIFO impedisce quindi che le richieste dei layer alti restino continuamente
dietro alle nuove richieste del layer 3. Tuttavia la richiesta più vecchia non
è sempre quella più urgente: mentre worker 0 legge il layer 15 o 18 del token
101, DS4 può essere più vicina a richiedere il layer 3 del token 102. Questo
spiega perché la politica più equa di `output_11` resta positiva, ma produce
meno riuso e uno speedup medio inferiore rispetto alla priorità dei layer bassi
di `output_10`.

## Risultati verificati — `output_11` (tre worker, FIFO per gruppo)

La directory [`output_11`](output_11) usa le stesse impostazioni di
`output_10`: 16 layer learned, due slot per layer, tre ripetizioni B/C e tre
worker modulo tre. L'unica differenza è l'ordine locale: ciascun worker sceglie
la richiesta più vecchia del proprio gruppo invece del layer numericamente più
basso disponibile.

### Speedup

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 110,624 s | 95,708 s | +13,48% |
| 2 | 102,951 s | 88,532 s | +14,01% |
| 3 | 113,131 s | 93,169 s | +17,65% |
| **Media** | **108,902 s** | **92,470 s** | **+15,05%** |

Tutte e tre le coppie migliorano con il prefetch. La FIFO locale è quindi una
politica valida e supera sia i due worker pari/dispari di `output_7`
(`+10,91%`) sia il FIFO globale di `output_9` (`+6,67%`). Non supera però la
scansione per layer a tre worker di `output_10` (`+16,63%`), che resta la
migliore politica misurata.

### Confronto operativo con `output_10`

| Indicatore | `output_10`: scansione per layer | `output_11`: FIFO per gruppo |
|---|---:|---:|
| Speedup medio | **+16,63%** | +15,05% |
| Attesa media prima della lettura | **361–398 ms** | 725–929 ms |
| Lettura SSD media | **184–196 ms** | 204–232 ms |
| Eventi `prefetch_bank_full` | **0–4** | 12–31 |
| Expert riusati dalla RAM pinned | **2.322–2.471** | 1.550–1.776 |

La FIFO locale fa ottenere turni più regolari anche ai layer alti, ma porta i
worker a leggere richieste meno urgenti mentre alcuni layer bassi sono già più
vicini alla domanda reale. In particolare il worker 0, che gestisce sei layer
(3, 6, 9, 12, 15, 18), accumula più lavoro; nella prima ripetizione completa
268 letture contro 314 per ciascuno degli altri due worker.

La maggiore attesa in coda aumenta anche le richieste che diventano troppo
tardive durante l'I/O: 19–53 eventi `READING → TOO_LATE`, contro 2–5 in
`output_10`. Gli slot restano quindi occupati più a lungo e gli eventi
`prefetch_bank_full` aumentano.

### Riuso degli expert

| Ripetizione | Gruppi con riuso parziale | Expert riusati dalla RAM pinned | Expert ancora letti dall'SSD alla domanda |
|---|---:|---:|---:|
| 1 | 564 | 1.550 | 4.498 |
| 2 | 638 | 1.776 | 4.272 |
| 3 | 583 | 1.596 | 4.452 |

Anche qui non ci sono `full_prefetch_hits`: con Recall@6 circa 0,506 è normale
che il gruppo completo di sei expert non coincida spesso con la domanda reale.
Il riuso parziale è però sufficiente a mantenere uno speedup stabile e
positivo. Il confronto indica che, su questa macchina, dare priorità ai layer
più bassi all'interno di ciascun piccolo gruppo è più efficace della perfetta
equità FIFO.

## Variante sperimentata: quattro worker con scansione per layer modulo quattro

TEST_9 prova ora quattro worker SSD e conserva la politica migliore finora
misurata: ciascun worker seleziona il layer fisico numericamente più basso con
uno slot `QUEUED` nel proprio gruppo. Con 16 layer learned fisici da 3 a 18, la
partizione `layer % 4` è perfettamente bilanciata:

```text
worker 0 → layer 4, 8, 12, 16
worker 1 → layer 5, 9, 13, 17
worker 2 → layer 6, 10, 14, 18
worker 3 → layer 3, 7, 11, 15
```

Ogni worker deve quindi servire solo quattro layer, rispetto ai cinque o sei
della variante `output_10` a tre worker. Possono avvenire fino a quattro
letture SSD in parallelo, sempre in buffer pinned di slot distinti. Il mutex
della banca pinned protegge la selezione e gli stati degli slot, non rende
seriali le letture dall'SSD.

Durante questa run, il runner impostava internamente:

```text
DS4_TEST_9_PARTITION_WORKERS=4
DS4_TEST_9_WORKER_POLICY=layer_partition
```

Non devi aggiungere queste variabili in CLion. Il test misura il compromesso
tra una coda più corta, che può aumentare il riuso, e quattro accessi
contemporanei allo stesso SSD esterno, che possono rallentare ogni lettura.
La verifica a quattro worker ha creato nuovamente la directory `output_12`; i
risultati sono riportati nella sezione seguente.

## Risultati verificati — `output_12` (quattro worker modulo quattro)

La directory [`output_12`](output_12) usa 16 layer learned, due slot per
layer, tre coppie B/C alternate e quattro worker modulo quattro. È il confronto
diretto con `output_10`: stessa scansione per layer, ma quattro gruppi da
quattro layer invece di tre gruppi da cinque o sei layer.

### Speedup

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 104,569 s | 88,448 s | +15,42% |
| 2 | 109,790 s | 94,730 s | +13,72% |
| 3 | 110,021 s | 96,074 s | +12,68% |
| **Media** | **108,127 s** | **93,084 s** | **+13,94%** |

Le tre coppie restano tutte positive, quindi quattro worker funzionano
correttamente. Il risultato non supera però `output_10`, che con tre worker
mantiene il migliore speedup medio finora misurato: `+16,63%`.

### Quattro worker: coda più corta, SSD più conteso

| Indicatore | Tre worker, `output_10` | Quattro worker, `output_12` |
|---|---:|---:|
| Attesa media prima della lettura | 361–398 ms | **270–343 ms** |
| Lettura SSD media | **184–196 ms** | 239–254 ms |
| `prefetch_bank_full` | 0–4 | 1–4 |
| Expert riusati dalla RAM pinned | 2.322–2.471 | **2.476–2.550** |

I quattro worker riducono la coda e distribuiscono quasi perfettamente il
lavoro: ogni worker completa 245–252 letture per ripetizione. Anche i layer
alti ricevono molte più letture pronte; il layer 18, per esempio, arriva a
59–62 letture invece delle 19–38 di `output_10`.

Il limite diventa però l'SSD esterno: quattro letture concorrenti rendono una
singola lettura circa 25–35% più lenta. Il maggiore riuso pinned non compensa
interamente questa perdita di throughput I/O; perciò il tempo C e lo speedup
finale sono peggiori rispetto a tre worker.

### Conclusione dopo quattro worker

Su questa macchina il miglior compromesso osservato resta:

```text
3 worker modulo tre + scansione per layer più basso nel gruppo
→ output_10, speedup medio +16,63%
```

Quattro worker dimostrano che la banca pinned e la partizione dei layer
reggono correttamente il parallelismo aggiuntivo, ma indicano anche che il
collo di bottiglia è ormai la capacità dell'SSD esterno, non il numero di
worker o gli slot disponibili.

## Configurazione runtime del numero di worker

Il numero di worker SSD non è più fissato nel codice di TEST_9. Impostare:

```text
DS4_TEST_9_PREFETCH_NUM_WORKERS=1
```

oppure `2`, `3`, `4`, `5`, `6` o `7` fa dividere gli stessi layer fisici tra quel numero di
worker con la regola `layer % numero_worker`. Ogni worker continua a scegliere
il layer più basso disponibile nel proprio gruppo. Il runner salva sia il
numero nel campo `schedule.prefetch_num_workers` sia la politica completa nel
file `benchmark_summary.json`.

Il valore predefinito è `3`, perché `output_10` ha dato il miglior risultato
misurato: speedup medio `+16,63%`. Per esempio, per ripetere quella politica
nella configurazione CLion usare:

```text
DS4_TEST_9_PREFETCH_NUM_WORKERS=3
```

I valori fuori dall'intervallo 1–7 interrompono TEST_9 con un messaggio chiaro:
la struttura CUDA contiene spazio per al massimo sette worker.

## Risultati verificati — `output_13` (24 layer, tre worker)

La directory [`output_13`](output_13) mantiene i tre worker, due slot per
layer e la scansione per layer più basso che hanno prodotto `output_10`, ma
aumenta `DS4_TEST_9_PREFETCH_LAYERS` da 16 a 24. Il predictor prova quindi a
prefetchare i layer fisici 3–26: ogni worker deve ora servire otto layer,
anziché cinque o sei.

### Speedup

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 96,729 s | 89,564 s | +7,41% |
| 2 | 112,133 s | 97,988 s | +12,62% |
| 3 | 98,576 s | 94,420 s | +4,22% |
| **Media** | **102,480 s** | **93,991 s** | **+8,08%** |

Tutte le coppie rimangono positive, ma il risultato è molto inferiore a
`output_10`: con 16 layer e gli stessi tre worker lo speedup medio era
`+16,63%`. Aumentare la copertura a 24 layer non aggiunge quindi capacità I/O:
distribuisce la stessa capacità dei worker su più richieste.

### Evidenze nella coda e nella copertura dei layer

| Indicatore | 16 layer, `output_10` | 24 layer, `output_13` |
|---|---:|---:|
| Attesa media prima della lettura | **361–398 ms** | 457–565 ms |
| Lettura SSD media | **184–196 ms** | 201–218 ms |
| Expert riusati dalla RAM pinned | 2.322–2.471 | 2.392–2.524 |
| Expert letti dall'SSD alla domanda | **3.577–3.726** | 6.548–6.680 |
| `prefetch_bank_full` | 0–4 | 0–3 |

La banca pinned non è satura: `prefetch_bank_full` resta basso. Il limite è la
coda di lettura dell'SSD. Il numero di expert riusati resta quasi invariato,
ma deve essere distribuito su molte più domande: gli expert ancora caricati
dall'SSD alla domanda reale aumentano quasi del doppio.

I layer fisici 3–14 ricevono ancora circa 59–63 letture pronte per
ripetizione; i layer 18–26 ne ricevono invece solo 2–30. Per esempio, nella
prima ripetizione il layer 18 ha 25 letture pronte, il layer 21 solo 4 e i
layer 24–26 soltanto 2 ciascuno. Quelle letture tardive consumano capacità I/O
senza produrre un riuso proporzionato.

### Conclusione dopo l'aumento a 24 layer

Per questa macchina, SSD esterno e distanza due, il punto migliore osservato
resta 16 layer con tre worker:

```text
DS4_TEST_9_PREFETCH_LAYERS=16
DS4_TEST_9_PREFETCH_NUM_WORKERS=3
→ output_10, speedup medio +16,63%
```

L'esperimento `output_13` dimostra che aggiungere layer senza aumentare il
throughput utile dell'SSD riduce l'efficacia del prefetch. Un futuro aumento di
copertura dovrebbe quindi accompagnarsi a una sorgente I/O più veloce o a una
politica che interrompa le richieste ormai troppo lontane dalla domanda reale.

## Risultati verificati — `output_14` (24 layer, quattro worker)

La directory [`output_14`](output_14) mantiene 24 layer learned e due slot per
layer di `output_13`, ma aumenta `DS4_TEST_9_PREFETCH_NUM_WORKERS` da tre a
quattro. Ogni worker serve quindi sei layer fisici, con una partizione modulo
quattro perfettamente bilanciata.

### Speedup

| Ripetizione | B baseline | C learned prefetch | Speedup di C |
|---|---:|---:|---:|
| 1 | 104,486 s | 91,251 s | +12,67% |
| 2 | 116,791 s | 96,361 s | +17,49% |
| 3 | 107,590 s | 95,332 s | +11,39% |
| **Media** | **109,622 s** | **94,315 s** | **+13,85%** |

Con 24 layer, quattro worker migliorano chiaramente `output_13` a tre worker
(`+8,08%`), ma non raggiungono ancora `output_10` con 16 layer e tre worker
(`+16,63%`). Tutte e tre le coppie B/C restano positive.

### Effetto del quarto worker sul carico da 24 layer

| Indicatore | 24 layer, 3 worker `output_13` | 24 layer, 4 worker `output_14` |
|---|---:|---:|
| Speedup medio | +8,08% | **+13,85%** |
| Attesa media prima della lettura | 457–565 ms | 490–532 ms |
| Lettura SSD media | **201–218 ms** | 254–284 ms |
| Expert riusati dalla RAM pinned | 2.392–2.524 | 2.303–2.844 |
| Expert letti dall'SSD alla domanda | 6.548–6.680 | **6.228–6.769** |

Il quarto worker non riduce in modo netto il tempo medio di attesa: la coda
resta dell'ordine di mezzo secondo. Riduce però il numero di layer che ciascun
worker deve servire da otto a sei e porta più letture ai layer intermedi e
alti. Nella terza ripetizione, per esempio, i layer 19–22 ricevono 31–36
letture pronte, mentre con tre worker arrivavano tipicamente a 12–13.

Il prezzo resta la contesa dell'SSD esterno: quattro letture concorrenti
alzano il tempo medio della singola lettura a 254–284 ms. Il guadagno di
copertura compensa parte del costo e migliora `output_13`, ma non abbastanza da
superare il punto ottimale di 16 layer e tre worker.

### Conclusione aggiornata

I risultati confermano due cose:

1. Per 24 layer, quattro worker sono preferibili a tre worker.
2. Per il miglior speedup complessivo su questa macchina, resta preferibile
   prefetchare solo 16 layer con tre worker.

```text
DS4_TEST_9_PREFETCH_LAYERS=16
DS4_TEST_9_PREFETCH_NUM_WORKERS=3
→ output_10, speedup medio +16,63%
```
