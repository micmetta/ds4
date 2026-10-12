# TEST_15 — cache RAM pinned condivisa per `(layer, expert)`

TEST_15 nasce dalla misura dell'overlap fra i due slot di TEST_14 e ora ne
implementa la soluzione B: una cache condivisa di expert in RAM pinned,
indicizzata da `(layer fisico, expert)`. Il percorso record di TEST_14 resta
inalterato; questa nuova politica e' attiva soltanto quando il runner di
TEST_15 imposta `DS4_TEST_15_SHARED_EXPERT_CACHE=1`.

## Riepilogo degli esperimenti e record operativo attuale

TEST_15 ha verificato che la cache condivisa B.1 risolve la saturazione della
RAM pinned e permette di riusare i buffer tra layer; non ha pero' superato il
percorso a **slot batch Top-6**. Il record operativo resta quindi
**TEST_14 `output_2`**: modello di prefetch **Build 7 Top-4**, Top-6,
due slot per layer e **sei worker** (`layer % 6`). In quella configurazione
il batching legge e pubblica insieme il gruppo primario di sei expert,
evitando il costo aggiuntivo di gestione per-expert del percorso condiviso.

| Esperimento | Approccio | Worker | B medio | C medio | Speedup medio |
|---|---|---:|---:|---:|---:|
| TEST_14 `output_2` | slot batch Top-6 Build 7 | 6 | 103,170 s | **72,588 s** | **+29,65%** |
| TEST_15 `output_6` | cache condivisa, Top-6, remap corretto | 8 | 100,630 s | 70,938 s | +29,39% |
| TEST_15 `output_7` | cache condivisa, Top-6 | 6 | 106,996 s | 76,507 s | +28,62% |
| TEST_15 `output_12` | cache condivisa B.1, Top-10 | 8 | 102,706 s | 79,306 s | +22,73% |
| TEST_15 `output_13` | cache condivisa B.1, Top-10 | 10 | 97,972 s | 77,690 s | +20,55% |
| TEST_15 `output_16` | Top-10, primi 6 prioritari | 6 | 105,525 s | 77,826 s | +26,25% |
| TEST_15 `output_17` | slot batch Top-6 + cache tail 7--10 | 6 | 102,738 s | 110,263 s | -6,82% |

La cache B.1 e la priorita' Top-6 sono miglioramenti corretti e utili per
misurare il riuso oltre il gruppo primario, ma il record appartiene ancora
all'architettura piu' semplice: prefetch batch degli stessi sei expert che il
forward MoE puo' consumare. Le sezioni seguenti documentano anche i run
negativi e diagnostici, per rendere riproducibile questa conclusione.

## Percorso ibrido attuale: slot Top-6 primario + cache tail

L'esecuzione predefinita del runner attiva anche
`DS4_TEST_15_HYBRID_GROUP_PRIMARY=1`. Non sostituisce il percorso record con
la cache per-expert: preserva lo **slot batch Top-6** del TEST_14 come primo
livello e usa la cache condivisa solo per i ranghi aggiuntivi `7..K` del
predictor (con `K = DS4_TEST_14_PREFETCH_TOP_K`).

Per ogni previsione il thread asincrono accoda una sola lettura batch degli
expert 1--6 nello slot del gruppo. Gli expert 7--K sono inseriti come entry
secondarie della cache per `(layer, expert)`: non duplicano mai i byte già
richiesti dal gruppo. Al forward reale il decode prova prima lo slot Top-6;
se trova un gruppo `READY` (anche parziale), riusa quel percorso, che già
gestisce gli expert mancanti con il fallback ordinario. Solo se non c'è alcun
gruppo utilizzabile per quel `(token, layer)` consulta la cache tail: un suo
`READY` può allora evitare una lettura on-demand.

Questo è deliberatamente un ibrido conservativo: evita sia la granularità
I/O per-expert sui primi sei sia le letture duplicate. Il contributo della
cache tail dipende però da quanti expert classificati dal modello fra 7 e K
diventino realmente necessari proprio quando manca lo slot primario.

La configurazione di partenza resta quella record di TEST_14:

```text
checkpoint Build 7 Top-4, prefetch Top-6
layer fisici nella finestra worker: 3--42
layer con predizione learned:       5--42
slot futuri per layer:              2
worker I/O:                         6 (layer % 6)
```

Per ogni layer la cache dispone al massimo di `2 × Top-K` entry: con Top-6
sono quindi dodici expert residenti. Ogni entry ha buffer di RAM pinned propri
per `gate/up/down`, stato `QUEUED`, `READING`, `READY` o `FAILED`, un
`lease_count` durante il trasferimento alla VRAM e un `future_refcount` finche'
richieste future la referenziano. La sincronizzazione e' partizionata per
`layer % numero_worker`: il thread di decode, il thread asincrono di prefetch
e il rispettivo worker di I/O condividono soltanto mutex, condition variable e
coda dello shard del layer interessato.

Nella modalita' predefinita il runner esegue A una volta e poi B/C alternati:
la fase C misura end-to-end la cache condivisa contro la baseline B. La
modalita' `--mode overlap-measurement` conserva invece il profiling storico,
con tre sole fasi C e senza confronto di speedup.

## Implementazione della soluzione B

Quando il servizio modello prefetch Build 7 restituisce il Top-6 per un layer
futuro, il thread asincrono cerca ogni chiave `(layer, expert)` nella cache
dello shard. Un expert `READY` viene semplicemente referenziato; uno `QUEUED`
o `READING` riusa la stessa lettura in corso; soltanto una chiave assente o
`FAILED` viene accodata al worker proprietario. Non sono quindi create letture
SSD duplicate per lo stesso expert.

Il worker non inoltra però più automaticamente una chiamata bridge per ogni
entry. Sotto il mutex del proprio shard preleva una sequenza contigua di entry
`QUEUED` appartenenti allo stesso layer (fino a `Top-K`), le porta tutte a
`READING` e rilascia il lock. Invia quindi una sola richiesta batch al bridge
Windows; al ritorno distribuisce i byte di ciascun expert nella rispettiva
entry persistente `(layer, expert)` e le pubblica come `READY`. Questo conserva
la semantica della cache condivisa, ma recupera la granularità I/O Top-6 del
percorso record senza riaprire letture duplicate.

Quando il forward reale raggiunge il layer, il thread di decode cerca i sei
expert richiesti dal router reale: per ogni `READY` acquisisce temporaneamente
un lease e copia la RAM pinned in VRAM; gli expert non pronti o non predetti
restano coperti dal fallback on-demand gia' esistente. Il gruppo eseguito puo'
percio' essere misto: expert riusati dalla cache e expert caricati normalmente.

Se serve capacita', il gestore sostituisce sotto il lock dello shard un'entry
riutilizzabile direttamente con la nuova chiave `QUEUED`, riusandone i buffer
di RAM pinned. Le entry `FAILED` hanno priorita'; altrimenti sono eleggibili
solo quelle con `lease_count == 0` e `future_refcount == 0`, scegliendo la meno
utile secondo frequenza con aging e ultimo utilizzo. Nessun altro worker deve
competere per lo stesso shard.

## Cosa misura

Quando il thread asincrono di prefetch DS4 porta una nuova richiesta nello stato
`QUEUED`, la trace conserva in modo read-only anche gli expert e lo stato
dell'altro slot del layer. Per ogni coppia il report calcola:

- l'intersezione fra i due Top-6;
- lo stato dello slot che contiene l'eventuale overlap;
- quanti expert e quanti byte sarebbero riusabili **subito**, cioe' soltanto
  quando lo slot fratello e' `READY` nel momento della nuova richiesta.

Questo e' esattamente il dato necessario prima di scegliere fra due evoluzioni:
comporre il nuovo slot copiando gli expert gia' `READY`, oppure introdurre una
cache condivisa per `(layer, expert)`. Il report non afferma che quei byte
produrrebbero automaticamente uno speedup: la composizione, i lock e il costo
delle copie restano da implementare e misurare in un esperimento successivo.

## Misura che ha motivato la cache — `output`

L'esecuzione completata ha mantenuto invariata la configurazione record di
`TEST_14/output_2`: checkpoint Build 7 Top-4, Top-6, due slot futuri per layer,
sei worker e tre sole ripetizioni C. La fase A è stata registrata una volta
(`105,948 s`) esclusivamente per produrre l'oracolo; non entra in un confronto
di speedup, perché TEST_15 non esegue la baseline B.

| Misura aggregata sulle 3 fasi C | Valore |
|---|---:|
| Richieste learned analizzate | 7.174 |
| Osservazioni dello slot fratello con lista expert disponibile | 5.027 |
| Coppie con almeno un expert in comune | 4.721 |
| Expert nell'overlap grezzo | 12.653 |
| Coppie il cui slot fratello era già `READY` | 162 |
| Expert immediatamente riusabili perché già `READY` | 414 |
| Overlap `READY` per richiesta learned | 0,0577 expert |
| Byte richiesti dalle letture candidate | 304,661 GB |
| Byte immediatamente riusabili con composizione | 2,930 GB |
| Frazione di byte immediatamente riusabili | **0,9618%** |

Lo slot fratello risultava più spesso `CONSUMING` (4.380 osservazioni,
87,13%), poi `QUEUED` (422, 8,40%) e solo raramente `READY` (225, 4,48%).
L'overlap predetto esiste dunque frequentemente, ma quasi sempre i suoi byte
sono ancora legati al consumo della precedente richiesta oppure a una lettura
in corso: copiare soltanto da uno slot fratello `READY` avrebbe un potenziale
molto ridotto.

Questo risultato non esclude la cache per-expert della soluzione B. Al
contrario, ne evidenzia il vantaggio rispetto alla composizione slot A → slot
B: un expert rimane referenziabile anche mentre la richiesta precedente e' in
uso. La cache descritta sopra estende quindi il riuso oltre il solo istante in
cui lo slot fratello risulta `READY`.

Le tre ripetizioni hanno prodotto rispettivamente 138 expert `READY` in
overlap e circa 0,977 GB di byte immediatamente riusabili ciascuna: il risultato
aggregato è quindi coerente tra le esecuzioni. Il report completo e riproducibile
è [`output/overlap_measurement_report.json`](output/overlap_measurement_report.json).

## Primo benchmark end-to-end della soluzione B — `output_2`

La prima esecuzione completa della cache condivisa ha mantenuto il checkpoint
Build 7 Top-4, Top-6, due finestre di capacità per layer e sei shard
`layer % 6`. L'oracolo A è stato prodotto una volta (`106,784 s`); B e C sono
state alternate tre volte. C completa correttamente tutte le ripetizioni, ma
in questa prima misura non pareggia il record storico di TEST_14.

| Ripetizione | B baseline | C cache condivisa | Speedup di C |
|---|---:|---:|---:|
| 1 | 98,291 s | 91,674 s | **+6,73%** |
| 2 | 108,011 s | 112,298 s | -3,97% |
| 3 | 98,625 s | 102,558 s | -3,99% |
| Media | **101,642 s** | 102,177 s | **-0,41%** |

Per confronto, il record ancora in vigore è TEST_14 `output_2`: `72,588 s`
in C contro `103,170 s` in B, cioè **+29,65%**. La variazione fra le tre
ripetizioni di TEST_15 è ampia: la prima è più veloce della baseline, mentre
le altre due no. Perciò questo benchmark dimostra che il percorso è corretto,
ma non attribuisce ancora la perdita a una causa specifica: la telemetria
generica ereditata dalla banca a slot non distingue in modo affidabile cache
hit per expert, letture duplicate evitate, attese in coda, saturazione o costo
dei trasferimenti RAM pinned → VRAM.

Il report completo è
[`output_2/benchmark_summary.json`](output_2/benchmark_summary.json). I suoi
contatori legacy restano utili come controllo di coerenza, ma non vanno usati
per spiegare la cache per-expert.

## Benchmark con telemetria per-expert — `output_3`

La stessa configurazione è stata ripetuta senza modifiche: Build 7 Top-4,
Top-6, capacità `2 x Top-6 = 12` entry per layer e sei shard `layer % 6`.
Questa è la prima run che include la telemetria specifica della cache
condivisa. Il risultato medio torna positivo, ma resta molto distante dal
record TEST_14 `output_2`.

| Ripetizione | B baseline | C cache condivisa | Speedup di C |
|---|---:|---:|---:|
| 1 | 96,480 s | 92,929 s | **+3,68%** |
| 2 | 105,088 s | 102,346 s | **+2,61%** |
| 3 | 97,341 s | 98,895 s | -1,60% |
| Media | **99,636 s** | **98,057 s** | **+1,56%** |

La cache condivisa è quindi funzionale e due coppie su tre sono più veloci
della rispettiva baseline, ma il guadagno non è ancora stabile né vicino al
record di `+29,65%`. Il report completo è
[`output_3/benchmark_summary.json`](output_3/benchmark_summary.json); ogni
fase C contiene anche il proprio `shared_expert_cache_metrics.json`.

### Cosa mostrano le nuove metriche

| Segnale | Ripetizione 1 | Ripetizione 2 | Ripetizione 3 | Lettura |
|---|---:|---:|---:|---|
| Hit `READY` al decode | 22,39% | 18,97% | 22,30% | Solo circa un quinto degli expert realmente richiesti è già in RAM pinned. |
| Miss assenti al decode | 12.227 | 13.131 | 12.228 | È il motivo dominante del fallback on-demand. |
| Attesa media in coda / job | 3.324 ms | 235 ms | 3.164 ms | Nelle run 1 e 3 i job attendono molto più della sola lettura SSD. |
| Durata media lettura / job | 103 ms | 86 ms | 97 ms | L'I/O è relativamente breve rispetto al backlog osservato. |
| Fallback di capacità | 43 | 0 | 52 | La capacità da 12 entry/layer aiuta, ma talvolta non basta. |
| Entry `READY` sostituite | 324 | 162 | 324 | Una parte dei byte già letti viene comunque rimpiazzata prima di produrre riuso. |
| Attesa totale sui lock | 1,10 ms | 1,03 ms | 1,33 ms | Non è una causa materiale del mancato speedup. |

Il confronto fra byte uploadati conferma il punto centrale: il percorso cache
trasferisce comunque `21,8–25,7 GB` da RAM pinned, ma il fallback on-demand
trasferisce ancora `56,0–57,6 GB`. Circa il 69–72% dei byte trasferiti verso
la GPU resta quindi nel percorso senza prefetch.

### Telemetria estesa per isolare il collo di bottiglia

La prossima esecuzione conserva le misure precedenti e aggiunge: composizione
dei gruppi del decode (`full`, `partial`, `zero` hit), profondità media e massima
della coda di ciascuno shard, chiamate e byte per richiesta al bridge Windows,
numero di expert caricati dalla cache rispetto al fallback, ciclo di vita delle
entry (`queued → ready → primo uso`) e rimpiazzi di entry `READY` mai usate.
Il report distingue inoltre il primo uso sul token per cui l'expert era stato
predetto dal primo uso su un token diverso, e fotografa le entry ancora residenti
alla chiusura. Insieme permettono di separare quattro cause: copertura predittiva
insufficiente, backlog di uno shard, granularità a un expert del bridge e byte
letti ma rimpiazzati prima di diventare utili.

TEST_15 accetta ora anche `DS4_TEST_15_PREFETCH_NUM_WORKERS=8` come confronto
di scalabilità, mantenendo Top-6 e capacità `2 × Top-6` per layer. Il run a otto
worker va confrontato con quello a sei attraverso questi stessi contatori: il
risultato è utile soltanto se riduce il backlog dello shard più carico senza
aumentare troppo le chiamate piccole al bridge Windows.

L'altro segnale rilevante è lo squilibrio fra shard: nelle run 1 e 3 lo shard
5 esegue rispettivamente 265 job, contro 99–107 degli shard 0–4. Gli shard
riducendo la contesa sul lock sono corretti, ma la ripartizione `layer % 6`
non rende necessariamente uniforme il lavoro I/O reale. Le metriche indicano
quindi, in ordine di priorità investigativa: aumentare la copertura `READY`
utile, ridurre il backlog del worker/shard 5 e poi verificare se la capacità
per layer e la politica di sostituzione causino ulteriori perdite. Non emerge
invece evidenza che i mutex degli shard siano il collo di bottiglia.

## Scalabilità a otto shard — `output_4`

Questa run mantiene Build 7 Top-4, Top-6 e capacità `2 x Top-6 = 12` entry
per layer, ma porta i worker I/O da sei a **otto**, con assegnazione
`layer % 8`. Il confronto B/C è completo: tutte e tre le coppie risultano
più veloci con la cache condivisa.

| Ripetizione | B baseline | C cache condivisa, 8 worker | Speedup di C |
|---|---:|---:|---:|
| 1 | 100,397 s | 88,778 s | **+11,57%** |
| 2 | 108,216 s | 100,104 s | **+7,50%** |
| 3 | 100,596 s | 96,061 s | **+4,51%** |
| Media | **103,069 s** | **94,981 s** | **+7,86%** |

Rispetto a `output_3` a sei worker (`+1,56%`), otto shard migliorano
nettamente il risultato e rendono favorevoli tutte le ripetizioni. Il record
di TEST_14 `output_2` resta tuttavia superiore: `+29,65%`, con `72,588 s` di
media nella fase C. Il report completo è
[`output_4/benchmark_summary.json`](output_4/benchmark_summary.json).

### Diagnosi: il lock non è il limite, la copertura utile sì

| Segnale | Rep. 1 | Rep. 2 | Rep. 3 | Interpretazione |
|---|---:|---:|---:|---|
| Hit `READY` al decode | 20,77% | 18,97% | 21,59% | Solo circa un quinto degli expert reali è già nella cache. |
| Gruppi con tutti e 6 gli expert `READY` | 0 | 2 | 2 | Quasi ogni layer richiede ancora il fallback per almeno un expert. |
| Gruppi a riuso parziale | 1.915 | 1.830 | 1.907 | Il riuso esiste, ma non elimina il percorso on-demand dominante. |
| Gruppi senza hit cache | 784 | 869 | 792 | Una quota rilevante non riceve alcun beneficio dalla cache. |
| Attesa media in coda / job | 2.413 ms | 190 ms | 2.224 ms | Il backlog è importante nelle repliche 1 e 3, ma non è l'unica causa. |
| Lettura media SSD / job | 117 ms | 83 ms | 105 ms | La lettura è molto più corta dell'attesa di coda nelle repliche con backlog. |
| Profondità massima della coda | 49 | 9 | 51 | Otto worker riducono il problema solo nella replica 2. |
| Entry `READY` sostituite senza uso | 248 | 139 | 303 | Molti byte già letti vengono rimpiazzati prima di servire un forward. |
| Entry `READY` mai usate ancora residenti alla fine | 181 | 185 | 173 | La cache conserva anche dati che non maturano in riuso utile durante la run. |

L'attesa totale sui lock di planner, worker e decode resta nell'ordine di
frazioni di millisecondo per ripetizione: l'aumento a otto shard ha quindi
confermato che la contesa sui mutex non è il collo di bottiglia principale.
Nemmeno la sola coda lo spiega: nella replica 2 la coda è quasi scarica, ma
l'hit-rate `READY` è il più basso e il tempo C resta alto.

### Granularità del bridge e ciclo di vita delle entry

In `output_4` il bridge Windows riceveva ancora **una chiamata per singolo
expert**: 723, 612 e 780 chiamate nelle tre repliche, con esattamente `1,0`
expert per chiamata. Questo era il divario strutturale rispetto alla
configurazione record a slot, che poteva leggere un gruppo Top-6 nella medesima
operazione. La cache per-expert evitava letture duplicate, ma introduceva molte
richieste piccole verso il bridge. L'implementazione corrente coalesca ora le
entry `QUEUED` contigue dello stesso layer: la prossima run deve quindi mostrare
meno chiamate e un valore `bridge.experts_per_call` superiore a 1 quando il
Top-6 genera più miss da leggere.

Inoltre, dal momento `READY` al primo riuso reale passano in media circa
`9,9–13,3 s`. Solo una minoranza dei primi usi avviene sul token target
originario; la maggior parte avviene su richieste future diverse. Questo
conferma il valore concettuale della cache condivisa, ma mostra anche che la
politica corrente di priorità/eviction non conserva sempre gli expert nel
momento in cui servono davvero.

Con il batching ora implementato, la prossima priorità tecnica resta rivedere
la priorità delle richieste per imminenza del token target e la sostituzione
delle entry mai usate. Queste modifiche agiscono sulla copertura `READY`, il
secondo segnale dominante emerso da `output_4` dopo la granularità I/O.

## Letture bridge coalescenti — `output_5`

`output_5` ripete la configurazione a otto shard di `output_4`, ma abilita il
batching nel worker: le entry `QUEUED` contigue dello stesso layer vengono
lette dal bridge Windows in una singola richiesta e poi distribuite nelle
rispettive entry persistenti della cache condivisa. Le tre coppie B/C restano
favorevoli alla cache, ma il guadagno medio end-to-end è **+2,80%**, inferiore
al `+7,86%` di `output_4`.

| Ripetizione | B baseline | C cache condivisa con batching | Speedup di C |
|---|---:|---:|---:|
| 1 | 95,209 s | 92,700 s | **+2,71%** |
| 2 | 104,542 s | 100,278 s | **+4,25%** |
| 3 | 97,949 s | 96,299 s | **+1,71%** |
| Media | **99,233 s** | **96,426 s** | **+2,80%** |

Il report completo è
[`output_5/benchmark_summary.json`](output_5/benchmark_summary.json). La
variazione della baseline tra run separate impone cautela nel confrontare i
soli tempi assoluti con `output_4`; resta però significativo che, dentro
`output_5`, tutte le coppie appaiate B/C siano più veloci con il prefetch.

### Il batching ha funzionato

| Ripetizione | Chiamate bridge | Expert richiesti | Expert per chiamata | Lettura media per expert |
|---|---:|---:|---:|---:|
| 1 | 180 | 715 | 3,97 | 78,15 ms |
| 2 | 133 | 612 | 4,60 | 35,94 ms |
| 3 | 186 | 732 | 3,94 | 70,92 ms |
| Totale / media pesata | **499** | **2.059** | **4,13** | — |

Prima del batching, `output_4` aveva effettuato 723, 612 e 780 chiamate, con
un solo expert per chiamata. Il nuovo percorso riduce quindi drasticamente le
richieste piccole al bridge e dimostra che il worker raggruppa davvero gli
expert dello stesso layer. Il costo di lettura ammortizzato per expert scende
anch'esso in tutte le repliche.

### Perché il record non è ancora recuperato

La copertura `READY` al decode resta pressoché invariata: `20,81%`, `18,97%`
e `21,46%`; i gruppi interamente coperti sono solo `1`, `2` e `1`, mentre
quelli parziali sono `1.893`, `1.830` e `1.926`. Il batching ha quindi rimosso
il costo I/O più evidente, ma non aumenta da solo il numero di expert corretti
che sopravvivono nella cache fino al loro forward reale. Rimangono centrali la
priorità per imminenza del token target e la sostituzione delle entry `READY`
mai usate, prima di attribuire la differenza residua a ulteriori worker o alla
sola sincronizzazione.

## Correzione del remap GPU compatto — `output_6`

Durante l'analisi di `output_5` è stata identificata una regressione di
**correttezza** nel percorso della cache condivisa, distinta dalla qualità del
checkpoint Build 7. I buffer GPU temporanei contengono soltanto i sei expert
del router reale in ordine compatto. Per esempio:

```text
posizione nel buffer GPU:       0    1    2    3   4    5
expert reale corrispondente:  166    9  118  160  75  225
```

La cache usava correttamente gli ID reali per cercare le entry
`(layer, expert)` in RAM pinned, ma passava per errore quegli stessi ID globali
al kernel CUDA. Il kernel, invece, deve indicizzare il buffer compatto con il
remap locale `[0, 1, 2, 3, 4, 5]`. Un ID globale come `166` riferito a una
tabella di sei posizioni causava una lettura fuori dalla tabella dei pesi
compatti e degradava gli hidden dei layer successivi. Il servizio Build 7
finiva per restituire frequentemente il Top-6 degenerato `[0, 1, 2, 3, 4, 5]`.

La correzione conserva gli ID reali nella cache e sostituisce soltanto il
vettore consegnato al kernel con gli indici locali compatti, usando la stessa
logica già adottata dal percorso a slot di TEST_14. `output_6` ripete la
configurazione con **otto worker I/O / otto shard (`layer % 8`)**, Top-6,
capacità `2 × Top-6` per layer e batching del bridge dopo questa modifica.
Il record TEST_14 `output_2` usava invece sei worker: il confronto misura
quindi anche la soluzione B nella sua configurazione concorrente a otto
worker, non soltanto il remap corretto.

| Metrica | TEST_14 `output_2` record | TEST_15 `output_6` corretto |
|---|---:|---:|
| Worker I/O | 6 (`layer % 6`) | **8 (`layer % 8`)** |
| Recall@6 online del predictor | 0,516624 | **0,516624** |
| B baseline medio | 103,170 s | 100,630 s |
| C learned-prefetch medio | 72,588 s | **70,938 s** |
| Tempo risparmiato medio | 30,582 s | 29,692 s |
| Speedup medio | **+29,65%** | **+29,39%** |
| Expert riusati da RAM pinned, media | ~7.170 | **~8.743** |
| Expert on-demand da SSD, media | ~7.950 | **~6.377** |

| Ripetizione | B baseline | C cache condivisa corretta | Speedup di C |
|---|---:|---:|---:|
| 1 | 94,529 s | 69,119 s | **+26,88%** |
| 2 | 107,076 s | 72,299 s | **+32,48%** |
| 3 | 100,284 s | 71,396 s | **+28,81%** |
| Media | **100,630 s** | **70,938 s** | **+29,39%** |

Il fix ripristina esattamente la Recall@6 online del record: il modello
Build 7 e i suoi input sono quindi corretti. La cache condivisa raggiunge uno
speedup praticamente sovrapponibile al record storico (scarto di `0,26` punti
percentuali) e contemporaneamente riusa più expert dalla RAM pinned. Rimane
un piccolo costo strutturale: la cache per-expert ha effettuato circa 2.244
chiamate bridge per ripetizione, con circa 2,57 expert per chiamata; il
percorso a slot del record trasferiva invece gruppi Top-6 già composti. Il
report completo è [`output_6/benchmark_summary.json`](output_6/benchmark_summary.json),
con una telemetria per ripetizione in
`output_6/repetition_*/shared_expert_cache_metrics.json`.

## Confronto del numero di worker dopo il fix — `output_7`

`output_7` ripete esattamente la cache condivisa corretta di `output_6`, ma
torna a **sei worker / sei shard (`layer % 6`)**. Restano invariati checkpoint
Build 7 Top-4, Top-6, due finestre logiche per layer (`2 × Top-6` entry),
batching del bridge e tre coppie B/C alternate. Il modello produce quindi la
stessa Recall@6 online (`0,516624`): la differenza misura soltanto la
concorrenza I/O e la variabilità end-to-end della macchina.

| Metrica | `output_6` — 8 worker | `output_7` — 6 worker |
|---|---:|---:|
| Recall@6 online del predictor | 0,516624 | 0,516624 |
| B baseline medio | **100,630 s** | 106,996 s |
| C cache condivisa medio | **70,938 s** | 76,507 s |
| Tempo risparmiato medio | 29,692 s | **30,489 s** |
| Speedup medio | **+29,39%** | +28,62% |
| Deviazione standard dello speedup | 2,84 pp | 8,15 pp |
| Expert riusati da RAM pinned, media | ~8.743 | ~8.724 |
| Expert on-demand, media | ~6.377 | ~6.396 |
| Chiamate bridge, media | ~2.244 | ~2.246 |
| Expert per chiamata bridge | ~2,57 | ~2,58 |

| Ripetizione | B baseline | C cache condivisa, 6 worker | Speedup di C |
|---|---:|---:|---:|
| 1 | 104,959 s | **65,838 s** | **+37,27%** |
| 2 | 109,903 s | 86,733 s | +21,08% |
| 3 | 106,126 s | 76,951 s | +27,49% |
| Media | **106,996 s** | **76,507 s** | **+28,62%** |

Con sei worker il riuso dalla RAM pinned e il numero di chiamate bridge sono
praticamente identici al caso a otto worker: non emerge quindi un problema di
qualità del predictor né di capacità della cache. Il risultato è però più
variabile, soprattutto per la fase C, e lo speedup medio scende di `0,77`
punti percentuali. Per la cache condivisa la configurazione preferita resta
perciò **otto worker**, pur senza superare il record a slot di TEST_14
(`+29,65%`). I report sono
[`output_7/benchmark_summary.json`](output_7/benchmark_summary.json) e
`output_7/repetition_*/shared_expert_cache_metrics.json`.

## Top-10 con cache condivisa — `output_8`

`output_8` mantiene la configurazione condivisa di `output_7` — **sei worker
I/O / sei shard**, due finestre per layer e batching per layer del bridge — ma
richiede al modello Build 7 i primi **dieci** expert predetti anziché i primi
sei. La capacità della cache viene quindi portata coerentemente da
`2 × Top-6 = 12` a `2 × Top-10 = 20` entry per layer. L'obiettivo era
sfruttare la Recall@10, che è maggiore della Recall@6, senza modificare il
router reale: il forward continua infatti a eseguire sempre e soltanto i sei
expert scelti dal router DS4.

| Metrica | `output_7` — Top-6, 6 worker | `output_8` — Top-10, 6 worker |
|---|---:|---:|
| Recall@6 online | 0,516624 | 0,516624 |
| Recall@10 online | 0,516624 | **0,624363** |
| Entry cache per layer | 12 | **20** |
| B baseline medio | 106,996 s | **105,737 s** |
| C cache condivisa medio | **76,507 s** | 117,775 s |
| Tempo risparmiato medio | **30,489 s** | **−12,038 s** |
| Speedup medio | **+28,62%** | **−11,38%** |
| Coppie B/C vinte da C | 3 / 3 | 0 / 3 |
| Expert riusati dalla RAM pinned, media | ~8.724 | ~4.294 |
| Expert on-demand da SSD, media | ~6.396 | ~10.826 |
| Hit `READY` del decode, media | ~53,83% | ~26,50% |
| Chiamate bridge, media | ~2.246 | ~2.392 |
| Expert per chiamata bridge | ~2,58 | ~6,98 |

| Ripetizione | B baseline | C cache condivisa Top-10 | Speedup di C |
|---|---:|---:|---:|
| 1 | 103,670 s | 113,663 s | −9,64% |
| 2 | 110,344 s | 123,366 s | −11,80% |
| 3 | 103,199 s | 116,295 s | −12,69% |
| Media | **105,737 s** | **117,775 s** | **−11,38%** |

L'esperimento conferma la premessa di qualità: il Top-10 contiene più expert
reali del Top-6. Non basta però che un expert sia predetto correttamente: deve
arrivare nello stato `READY` prima del forward che lo richiede. Qui avviene il
contrario. Il batching è effettivamente più ampio (quasi sette expert per
chiamata bridge, contro circa 2,6), ma ogni ripetizione crea circa 16,6 mila
job e soltanto circa 600–714 letture diventano `READY`; circa 15,9–16,2 mila
job terminano `FAILED`. La copertura utile del decode si dimezza e il fallback
on-demand cresce di circa 4,4 mila expert rispetto al Top-6.

Il collo di bottiglia osservato non è quindi il remap GPU, né la Recall@10,
né il batching in sé. È la pressione I/O introdotta dai dieci candidati per
previsione: la cache e il bridge non riescono a rendere pronti abbastanza
expert entro la loro scadenza pratica. Per questa implementazione la
configurazione da mantenere come riferimento è quindi ancora **Top-6**.
Il report completo è
[`output_8/benchmark_summary.json`](output_8/benchmark_summary.json), con la
telemetria per ripetizione in
`output_8/repetition_*/shared_expert_cache_metrics.json`.

## Top-10 con otto worker — `output_9`

`output_9` ripete il test Top-10 con **otto worker I/O / otto shard**
(`layer % 8`) e capacità `2 × Top-10 = 20` entry per layer. Lo scopo era
verificare se gli shard aggiuntivi potessero aumentare il parallelismo delle
letture e recuperare lo svantaggio osservato in `output_8` con sei worker.

| Ripetizione | B baseline | C Top-10, 8 worker | Speedup di C | `READY` | `FAILED` | Hit `READY` |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 97,104 s | 114,016 s | −17,42% | 665 | 16.286 | 25,16% |
| 2 | 98,233 s | 109,363 s | −11,33% | 662 | 16.012 | 27,64% |
| 3 | 104,994 s | 116,645 s | −11,10% | 627 | 16.423 | 24,28% |
| Media | **100,110 s** | **113,341 s** | **−13,28%** | ~651 | ~16.240 | ~25,69% |

Il passaggio da sei a otto worker riduce leggermente il rallentamento rispetto
a `output_8`, ma non rende il prefetch vantaggioso. Il bridge riceve circa
2.392 batch per ripetizione, con circa sette expert per batch, mentre soltanto
circa 627–665 entry raggiungono `READY`. Questo run ha mostrato con chiarezza
il sintomo — moltissimi `FAILED` e bassa copertura `READY` — ma non ne poteva
ancora attribuire la causa: i contatori separati per bridge, allocazione pinned
persistente e invariante di scatter sono stati aggiunti solo successivamente.

Il report completo è
[`output_9/benchmark_summary.json`](output_9/benchmark_summary.json), con la
telemetria allora disponibile in
`output_9/repetition_*/shared_expert_cache_metrics.json`.

## Diagnosi dei `FAILED` Top-10 — `output_10`

`output_10` ripete il caso più gravoso della cache condivisa: **Top-10**, due
finestre logiche per layer (`2 × 10 = 20` entry), **otto worker I/O / otto
shard** (`layer % 8`). A differenza dei run precedenti, il report separa
esplicitamente le tre cause possibili di una lettura non pubblicata come
`READY`:

- `bridge_failed`: il bridge Windows non ha completato la lettura richiesta;
- `persistent_pinned_allocation_failed`: il bridge ha letto i byte, ma non è
  stato possibile allocare il buffer persistente di RAM pinned dell'entry
  condivisa;
- `scatter_invariant_failed`: incoerenza interna tra batch ricevuto e entry
  della cache da aggiornare.

| Ripetizione | B baseline | C Top-10, 8 worker | Speedup di C | `READY` | `FAILED` |
|---|---:|---:|---:|---:|---:|
| 1 | 95,313 s | 115,781 s | −21,48% | 544 | 16.793 |
| 2 | 98,813 s | 116,253 s | −17,65% | 667 | 16.073 |
| 3 | 101,805 s | 115,343 s | −13,30% | 652 | 16.121 |
| Media | **98,643 s** | **115,792 s** | **−17,47%** | ~621 | ~16.329 |

| Causa dei `FAILED` | Ripetizione 1 | Ripetizione 2 | Ripetizione 3 |
|---|---:|---:|---:|
| `bridge_failed` | 0 | 0 | 0 |
| `persistent_pinned_allocation_failed` | **16.793** | **16.073** | **16.121** |
| `scatter_invariant_failed` | 0 | 0 | 0 |

La diagnosi esclude quindi sia il bridge Windows sia un errore di
coerenza/scatter della cache. Il bridge continua a lavorare: in ciascuna
ripetizione esegue circa 2.392 chiamate, con circa sette expert per chiamata.
Il fallimento avviene **dopo** la lettura, quando il worker prova a rendere
persistente il singolo expert nella RAM pinned della cache condivisa.

Con Top-10 la configurazione può richiedere fino a `40 layer MoE × 20 entry ×
~6,75 MiB ≈ 5,27 GiB` di buffer pinned persistenti. La run riesce a mantenere
solo circa 544–667 entry `READY` prima che le ulteriori allocazioni falliscano;
la soglia empirica osservata è quindi nell'ordine di 3,6–4,4 GiB, con una
piccola variabilità dovuta all'ordine e alla pressione delle allocazioni. Le
entry fallite vengono continuamente rimpiazzate e ritentate, ma non diventano
mai riusabili dal decode: aumentano così i miss e gli upload/on-demand.

Il problema non è quindi la qualità del modello Build 7 né la sincronizzazione
fra gli shard. La cache condivisa deve diventare **budgetata**: un pool di
buffer RAM pinned con capacità esplicita, più una politica di ammissione ed
eviction che non avvii una lettura se non può conservare il risultato. Fino a
quel cambiamento, Top-6 resta la configurazione operativa preferibile.

I dati completi sono in
[`output_10/benchmark_summary.json`](output_10/benchmark_summary.json) e nei
rispettivi `output_10/repetition_*/shared_expert_cache_metrics.json`.

## Conferma della saturazione prima di B.1 — `output_11`

`output_11` ripete Top-10 con **otto worker**, prima della riallocazione
fisica sicura B.1. Include la telemetria delle entry candidate alla
riallocazione, ma il runtime non puo' ancora sostituire il buffer pinned di un
donor sicuro con quello di un nuovo expert.

| Ripetizione | B baseline | C cache condivisa Top-10 | Speedup di C |
|---|---:|---:|---:|
| 1 | 95,474 s | 115,108 s | -20,56% |
| 2 | 100,747 s | 121,886 s | -20,98% |
| 3 | 98,484 s | 126,852 s | -28,80% |
| Media | **98,235 s** | **121,282 s** | **-23,45%** |

Questo e' il run diagnostico che ha quantificato la capacita' pinned gia'
occupata ma non ancora riusabile dall'implementazione precedente: entry
`READY` senza `future_refcount` e senza lease potevano essere donor sicuri,
ma non venivano sostituite. Tale evidenza ha motivato B.1; il risultato
positivo di `output_12` non e' quindi un cambiamento del modello di prefetch,
ma l'effetto della riallocazione fisica dei buffer gia' disponibili.

I dati completi sono in
[`output_11/benchmark_summary.json`](output_11/benchmark_summary.json).

## Riallocazione fisica sicura B.1 — `output_12`

`output_12` verifica la correzione B.1 della cache condivisa. Quando un nuovo
expert non dispone di un buffer persistente, il suo layer può ricevere il
buffer fisico di una entry `READY` che non ha più né riferimenti futuri né
lease attivi. La configurazione resta il caso più gravoso: **Top-10**, due
finestre logiche per layer (`2 × 10 = 20` entry), **otto worker I/O / otto
shard**.

| Ripetizione | B baseline | C cache condivisa B.1 | Speedup di C |
|---|---:|---:|---:|
| 1 | 101,158 s | 77,685 s | +23,21% |
| 2 | 106,446 s | 79,501 s | +25,31% |
| 3 | 100,514 s | 80,733 s | +19,68% |
| Media | **102,706 s** | **79,306 s** | **+22,73%** |

| Telemetria cache B.1, media per ripetizione | Valore |
|---|---:|
| `persistent_pinned_allocation_failed` | **0** |
| `bridge_failed` | **0** |
| `scatter_invariant_failed` | **0** |
| Riallocazioni fisiche sicure tra layer | ~13.020 |
| Byte riallocati tra layer | ~85,8 GiB |
| Entry `READY` riusate dal decode | ~8.693 |
| Hit `READY` del decode | ~53,65% |
| Expert on-demand rimanenti | ~4.931 |
| Expert per chiamata bridge | ~5,74 |

La correzione elimina il limite che aveva reso `output_11` negativo
(-23,45%): non si osserva più alcun `FAILED` né per il bridge né per
l'allocazione pinned persistente. Le circa tredicimila riallocazioni per run
provengono tutte da donor `READY` sicuri; nessun donor `FAILED` è stato
necessario. La capacità pinned già allocata viene quindi davvero riutilizzata
tra layer senza sottrarre buffer al decode.

Il risultato è positivo in tutte le ripetizioni, ma non supera ancora il record
a slot di TEST_14 `output_2` (+29,65%). Rimane il costo del percorso
per-expert: circa 2.387 chiamate bridge per run e un'attesa media in coda di
circa 155 ms per job. B.1 elimina la saturazione, non l'overhead residuo di
pianificazione e I/O della cache condivisa.

I report completi sono
[`output_12/benchmark_summary.json`](output_12/benchmark_summary.json) e
`output_12/repetition_*/shared_expert_cache_metrics.json`.

## Scalabilità Top-10 con dieci worker — `output_13`

`output_13` mantiene cache condivisa B.1, capacità logica `2 × Top-10` per
layer e batching del bridge per layer, ma porta i reader a **dieci worker I/O /
dieci shard**. L'obiettivo è verificare se la minore attesa in coda riesca a
trasformarsi in uno speedup end-to-end maggiore rispetto agli otto worker di
`output_12`.

| Ripetizione | B baseline | C cache condivisa B.1 | Speedup di C |
|---|---:|---:|---:|
| 1 | 93,635 s | 77,706 s | +17,01% |
| 2 | 103,308 s | 76,835 s | +25,63% |
| 3 | 96,974 s | 78,528 s | +19,02% |
| Media | **97,972 s** | **77,690 s** | **+20,55%** |

| Telemetria media | `output_12`: 8 worker | `output_13`: 10 worker |
|---|---:|---:|
| Hit `READY` del decode | ~53,65% | **~53,82%** |
| Attesa in coda per expert | ~155 ms | **~94 ms** |
| Lettura per expert | ~27,4 ms | ~28,8 ms |
| Expert per chiamata bridge | ~5,74 | **~5,80** |
| `persistent_pinned_allocation_failed` | 0 | **0** |
| `bridge_failed` | 0 | **0** |

Dieci worker riducono realmente il backlog del relativo shard e il tempo C
assoluto passa da **79,306 s** a **77,690 s**. Il confronto percentuale non
migliora però perché in questa run anche la baseline è sensibilmente più veloce
(97,972 s contro 102,706 s); le sole tre ripetizioni non rendono confrontabile
quella variabilità come un guadagno architetturale certo.

Un punto importante: la cache **non** esegue una chiamata bridge per expert.
Ogni worker estrae dalla propria coda la sequenza FIFO contigua di entry
`QUEUED` dello stesso layer, fino a Top-K, e invia un unico batch al bridge
Windows; soltanto dopo disperde i dati ricevuti nelle entry persistenti per
expert. In `output_13`, 13.866 expert sono stati serviti da 2.391 chiamate,
pari a **5,80 expert per chiamata**. Perciò il prossimo collo di bottiglia non
si risolve duplicando il batching: restano la copertura `READY` (~54%), gli
upload/fallback degli expert mancanti e il costo intrinseco della cache
per-expert.

I dati completi sono in
[`output_13/benchmark_summary.json`](output_13/benchmark_summary.json) e in
`output_13/repetition_*/shared_expert_cache_metrics.json`.

## Top-20 dopo B.1 — `output_14` e `output_15`

Dopo avere eliminato i fallimenti di allocazione persistente con B.1, sono
stati provati i primi **20** expert predetti. Il forward reale resta sempre a
sei expert: Top-20 aumenta solo il lavoro speculativo e la capacita' logica a
`2 x 20 = 40` entry per layer. L'obiettivo era sfruttare la Recall@20 piu'
alta senza reintrodurre la saturazione della RAM pinned.

| Esperimento | Worker | B medio | C medio | Speedup medio |
|---|---:|---:|---:|---:|
| `output_14` | 6 | 102,541 s | 100,043 s | **+2,46%** |
| `output_15` | 8 | 99,159 s | 99,101 s | **+0,05%** |

| Ripetizione | `output_14`: C Top-20, 6 worker | `output_15`: C Top-20, 8 worker |
|---|---:|---:|
| 1 | 98,195 s (+3,36%) | 96,871 s (-0,75%) |
| 2 | 96,639 s (+5,03%) | 104,510 s (+0,19%) |
| 3 | 105,294 s (-1,00%) | 95,922 s (+0,72%) |

B.1 continua a prevenire la saturazione, ma Top-20 non migliora il record:
gli expert aggiuntivi richiedono piu' lavoro di pianificazione, I/O e
pubblicazione rispetto ai sei expert consumabili dal forward. Otto worker non
risolvono il limite, perche' il costo dominante non e' piu' la sola attesa
dello shard. Il risultato rafforza la scelta di privilegiare il gruppo Top-6
del percorso record, non di allargare indiscriminatamente Top-K.

I report completi sono
[`output_14/benchmark_summary.json`](output_14/benchmark_summary.json) e
[`output_15/benchmark_summary.json`](output_15/benchmark_summary.json).

## Top-10 con percorso critico Top-6 prioritario — `output_16`

`output_16` verifica un'ammissione a due priorita' mantenendo **Top-10**,
cache B.1, sei worker e capacita' logica `2 x Top-10` per layer. I primi sei
rank, ossia il massimo numero di expert che il forward MoE puo' realmente
consumare, sono accodati subito come **primari**; i rank 7--10 restano nella
cache come tail `QUEUED` e vengono consegnati al worker soltanto quando la
coda primaria dello shard e' vuota. Un expert tail promosso fra i primi sei
in una previsione successiva diventa primario senza una nuova lettura.

| Ripetizione | B baseline | C Top-10, Top-6 prioritario | Speedup di C |
|---|---:|---:|---:|
| 1 | 105,006 s | 77,948 s | +25,77% |
| 2 | 106,712 s | 79,649 s | +25,36% |
| 3 | 104,857 s | 75,881 s | +27,63% |
| Media | **105,525 s** | **77,826 s** | **+26,25%** |

| Indicatore | `output_16` |
|---|---:|
| `primary_top_k` | 6 |
| Accodamenti primari | ~7.149 per ripetizione |
| Tail rilasciati in modo differito | ~4.110 per ripetizione |
| Hit `READY` del decode | ~49,1% |
| `FAILED` residenti | 0 |
| `capacity_fallbacks` | ~1.224 per ripetizione |

La policy elimina l'interferenza diretta del tail sulla coda del gruppo che
puo' servire il forward, e non introduce fallimenti bridge o di RAM pinned.
Non supera pero' il record TEST_14 `output_2` (**+29,65%**): la copertura
`READY` scende a circa 49%, quindi oltre meta' degli expert reali richiede
ancora il fallback on-demand. Il tail Top-10 viene preservato per riuso
opportunistico, ma non e' pronto abbastanza spesso da compensare il costo
residuo della cache per-expert rispetto al vecchio percorso Top-6 a gruppi.

I contatori che consentono di verificare l'ammissione sono in
`output_16/repetition_*/shared_expert_cache_metrics.json`, nelle sezioni
`cache` e `planner`, e sono inclusi anche nel
[`output_16/benchmark_summary.json`](output_16/benchmark_summary.json).

## Percorso ibrido Top-6 + tail Top-10 — `output_17`

`output_17` e' il primo benchmark del percorso ibrido corrente. Mantiene
**Top-10**, sei worker e due finestre logiche per layer (`2 x Top-10`), ma
non usa piu' la cache per-expert come percorso principale. Per ogni richiesta:

- i rank **1--6** sono letti nello slot batch Top-6 del percorso record di
  TEST_14;
- solo i rank **7--10** entrano nella cache RAM pinned condivisa come tail;
- al decode si consulta prima lo slot Top-6; la cache tail viene consultata
  solo quando non e' disponibile alcun gruppo utilizzabile.

L'obiettivo era conservare la granularita' I/O favorevole del record sui sei
expert che il forward puo' realmente consumare, aggiungendo il riuso
opportunistico dei quattro candidati ulteriori senza letture duplicate.

| Ripetizione | B baseline | C ibrido Top-10 | Speedup di C |
|---|---:|---:|---:|
| 1 | 103,614 s | 107,002 s | -3,17% |
| 2 | 106,092 s | 114,108 s | -7,02% |
| 3 | 98,507 s | 109,679 s | -10,19% |
| Media | **102,738 s** | **110,263 s** | **-6,82%** |

La run e' quindi corretta funzionalmente, ma e' una regressione e non un nuovo
record. La qualita' online e' quella attesa dal modello Build 7: Recall@6
`0,516624` e Recall@10 `0,624363`. Il problema e' temporale e di costo
operativo, non il checkpoint o il remap GPU: vengono riusati in media circa
`6.670` expert dalla RAM pinned, ma ne servono ancora circa `8.450` da SSD ai
layer predetti; i gruppi completamente coperti sono soltanto uno per
ripetizione.

La telemetria mostra inoltre circa `55 ms` di attesa media prima che un worker
inizi una lettura, circa `146 ms` di lettura bridge e circa `201 ms` da
`QUEUED` a `READY`. Il tail aggiunge strutture e lavoro di cache, ma nella
misura attuale non produce abbastanza hit pronti da compensare quel costo. Il
record operativo resta pertanto il percorso a slot Top-6 di TEST_14
`output_2`; `output_17` resta utile come diagnosi del limite dell'ibrido.

I dati completi sono in
[`output_17/benchmark_summary.json`](output_17/benchmark_summary.json) e nei
file `pinned_ram_policy_audit.json` delle tre ripetizioni.

## Telemetria diagnostica della cache condivisa

Dalla run `output_3`, ogni fase C salva inoltre
`shared_expert_cache_metrics.json` nella sua directory di ripetizione e il suo
contenuto è incluso anche in `benchmark_summary.json`. La run `output_2` è
anteriore a questa telemetria e rimane quindi confrontabile solo sul tempo
end-to-end.

Le metriche aggiunte separano le cause che possono impedire di raggiungere il
record:

- **copertura utile:** expert `READY` trovati dal decode, miss distinti in
  `QUEUED`, `READING`, `FAILED` e assenti; il relativo hit-rate è calcolato
  sul totale degli expert realmente richiesti dal router;
- **lavoro del planner:** richieste e expert predetti, riusi di entry già
  `READY`, `QUEUED` o `READING`, retry dopo errore, accodamenti e richieste
  lasciate al fallback per mancanza di capacità;
- **capacità e I/O:** entry `FAILED` e `READY` sostituite, job del worker,
  letture riuscite/fallite, byte effettivamente letti dal bridge, attesa in
  coda e durata della lettura;
- **saturazione della RAM pinned:** per ogni batch worker che termina con
  `persistent_pinned_allocation_failed` viene acquisita una fotografia
  osservativa della capacità fisicamente già allocata: numero e byte delle
  entry residenti, quota `READY` immediatamente recuperabile
  (`future_refcount = 0` e `lease_count = 0`), quota ancora protetta da un
  riferimento futuro o da un upload GPU, ed eventuali entry `FAILED` che
  trattengono ancora buffer. Il report salva primo snapshot, media e massimo
  dei byte recuperabili: sono i dati necessari per capire se un futuro pool
  pinned globale potrebbe evitare i fallimenti senza sottrarre buffer ancora
  utili al decode. Nello stesso snapshot iniziale il report elenca tutte le
  coppie `(layer, expert)` candidate, con byte, `future_refcount`,
  `lease_count`, numero/ordine degli usi e il motivo: `READY` non referenziata
  e senza lease, oppure entry `FAILED` che trattiene ancora un buffer pinned;
- **costo GPU:** byte e tempo separati per upload da RAM pinned e per fallback
  on-demand, così da vedere se il riuso evita davvero lavoro oppure introduce
  molte piccole copie sincrone;
- **concorrenza e bilanciamento:** tempo di attesa sui lock di planner,
  decode e worker, più gli stessi contatori principali per ciascuno shard
  configurato. Questo rende visibile sia una contesa inattesa sia un worker più
  carico degli altri.

Questi dati permetteranno di distinguere quattro scenari: predizioni poco
coincidenti con il router reale; expert corretti ma non ancora `READY`; cache
troppo piccola o soggetta a sostituzioni; oppure overhead di I/O, upload o
sincronizzazione che supera il lavoro evitato.

## Esecuzione

TEST_15 riusa il bridge Windows di TEST_14. Avviarlo prima in PowerShell:

```powershell
$env:DS4_TEST_14_WINDOWS_MODEL='D:\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$env:DS4_TEST_14_WINDOWS_PORT='38114'
py '\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_14\windows_native\test_14_windows_reader_service.py'
```

Poi, in WSL:

```bash
cd /home/michele/ds4
source test_predictor/prefetch_predictor_pipeline/ML_pipeline/.venv/bin/activate
make test_15_runner

DS4_MODEL=/home/michele/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
DS4_TEST_15_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
DS4_TEST_15_CHECKPOINT=/home/michele/ds4/test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/neural_network/top4_patience10_l2_3e6_bce_v1/models/ffn.keras \
DS4_TEST_15_WINDOWS_HOST=172.23.208.1 \
DS4_TEST_15_WINDOWS_PORT=38114 \
DS4_TEST_15_PREFETCH_SLOTS_PER_LAYER=2 \
DS4_TEST_15_PREFETCH_STEP=1 \
DS4_TEST_15_PREFETCH_NUM_WORKERS=6 \
DS4_TEST_15_PREFETCH_TOP_K=6 \
DS4_TEST_15_REPETITIONS=3 \
python test_predictor/test_15/test_15_runner.py
```

Il runner crea `output*` senza sovrascrivere nulla. Nella modalita' benchmark
scrive `benchmark_summary.json` dopo A e dopo ogni coppia B/C; nella modalita'
storica aggiorna invece `overlap_measurement_report.json` dopo A e dopo ogni
fase C. Un'interruzione conserva quindi l'oracolo e tutte le ripetizioni gia'
completate. Il bridge Windows supporta batch coalescenti per layer fino a
Top-20. Il forward reale continua invece a usare esattamente sei expert:
Top-K cambia solo l'ampiezza della speculazione e della cache. Riavviare sia
il servizio Build 7 sia il bridge Windows dopo avere aggiornato gli script,
quindi ricompilare con `make test_15_runner` prima della prima esecuzione.

### Esperimento Top-20

Per valutare la maggiore copertura della Recall@20 senza cambiare il forward
reale, impostare `DS4_TEST_15_PREFETCH_TOP_K=20`. La cache mantiene fino a due
finestre speculative per layer (`2 x 20 = 40` entry logiche); il worker
coalesce le entry `QUEUED` contigue dello stesso layer in una singola chiamata
al bridge, fino a 20 expert. Gli expert realmente eseguiti dal router restano
sempre sei: il report deve quindi essere letto confrontando la copertura
`READY`, gli upload cached/on-demand, gli expert per chiamata bridge e il tempo
end-to-end, non come un forward MoE con venti expert.

Per rieseguire soltanto la misura storica dell'overlap:

```bash
python test_predictor/test_15/test_15_runner.py --mode overlap-measurement
```

### Ammissione prioritaria per Top-10

Con `Top-K=10` i dieci expert non hanno lo stesso valore operativo: il
forward MoE reale puo' consumarne al massimo sei. Per evitare che gli expert
nelle posizioni 7--10 ritardino i sei con probabilita' maggiore, TEST_15 ora
usa due priorita' nello stesso shard `layer % worker`:

- **primari:** le prime `DS4_TEST_15_PREFETCH_PRIMARY_K` predizioni
  (predefinito `6`) sono accodate subito al worker;
- **tail:** le posizioni successive restano `QUEUED` nella cache e vengono
  rilasciate al worker soltanto quando la coda primaria dello shard e' vuota.

Questo conserva il possibile riuso Top-10, la cache condivisa e il bridge
coalescente per layer, ma protegge il percorso che puo' ridurre direttamente
il fallback dei sei expert eseguiti dal router reale. La nuova telemetria in
`shared_expert_cache_metrics.json` espone `cache.primary_top_k` e, in
`planner`, `primary_queue_pushes`, `tail_queue_pushes` e
`deferred_tail_releases`.

Per il confronto diretto con il record usare esplicitamente:

```bash
DS4_TEST_15_PREFETCH_TOP_K=10 \
DS4_TEST_15_PREFETCH_PRIMARY_K=6 \
DS4_TEST_15_PREFETCH_NUM_WORKERS=6 \
python test_predictor/test_15/test_15_runner.py
```

L'ottimizzazione non garantisce da sola di superare il record: serve a
misurare separatamente se il limite residuo e' la copertura delle prime sei
predizioni, la disponibilita' temporale degli expert, oppure il costo I/O
anche dopo aver tolto al tail la priorita' sul percorso critico.

## Configurazione record attuale — riferimento da preservare

La migliore modalita' verificata fino ad oggi non e' una variante della cache
condivisa di TEST_15, ma il percorso batch originale rieseguito in
**TEST_14 `output_2`**. E' il riferimento obbligatorio per ogni esperimento
successivo: una nuova politica puo' essere considerata migliore soltanto se
supera questo confronto end-to-end, con lo stesso modello GGUF, prompt,
oracolo e alternanza B/C.

| Componente | Configurazione record |
|---|---|
| Predictor | miglior checkpoint Build 7 Top-4 disponibile al run |
| Expert richiesti al predictor | Top-6 |
| Expert realmente eseguiti dal forward | 6 |
| Layer fisici nella finestra learned | 3--42 (40 layer) |
| Layer con predizione learned | 5--42 |
| Slot batch futuri per layer | 2 |
| Worker I/O | 6, assegnati con `layer % 6` |
| Percorso dati | slot batch Top-6: bridge Windows legge `gate/up/down` direttamente nella RAM pinned dello slot, poi upload RAM pinned -> VRAM al forward |
| Media baseline B | 103,170 s |
| Media learned-prefetch C | **72,588 s** |
| Tempo medio risparmiato | **30,582 s** |
| Speedup medio | **+29,65%** |

Il motivo operativo del record e' che il gruppo speculativo e il gruppo
consumabile coincidono: il bridge trasferisce insieme i sei expert predetti,
il decode riusa quelli che coincidono con il router reale e completa soltanto
gli eventuali mancanti con il fallback on-demand. Le evoluzioni TEST_15 hanno
migliorato osservabilita', sincronizzazione e gestione della RAM pinned, ma
non hanno ancora compensato il loro costo aggiuntivo rispetto a questo percorso
batch Top-6. Per questo la configurazione sopra resta quella da usare per il
benchmark e per la generazione piu' veloce finche' un nuovo run non la supera.
