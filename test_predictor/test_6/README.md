# TEST_6 — Oracle prefetch con step controllabile

## Obiettivo

TEST_6 riprende il test A/B/C di TEST_5, ma separa chiaramente due parametri della pianificazione del prefetch da HDD esterno:

- `DS4_TEST_5_PREFETCH_INIT_DISTANCE`: quanti token dopo `t` viene richiesto il primo bersaglio;
- `DS4_TEST_5_PREFETCH_STEP`: ogni quanti token viene richiesto il bersaglio successivo.

Il nome `INIT_DISTANCE` indica soltanto l'anticipo del primo bersaglio. Non e lo step: lo step e controllato esclusivamente da `PREFETCH_STEP`.

Con `INIT_DISTANCE=8` e `STEP=8`, se il primo token di decode e `20`, il flusso e:

```text
durante token 20 -> richiede (token 28, layer 0..N-1)
durante token 21..27 -> non accoda un nuovo burst
durante token 28 -> richiede (token 36, layer 0..N-1)
durante token 29..35 -> non accoda un nuovo burst
```

Ogni burst resta quindi otto token in anticipo, ma l'HDD ha otto token di tempo senza nuove richieste speculative per completare la lettura. Il compromesso e voluto: il prefetch copre soltanto i token 28, 36, 44, ...; gli altri continuano a usare il demand load standard su NVMe interno.

## Run A, B e C

```text
test_6_runner
    |
    +-- Run A: processo DS4 separato
    |   registra il routing completo in routing_oracle.bin
    |
    +-- Run B: nuovo processo DS4
    |   replay completo dell'oracolo, senza worker prefetch
    |   baseline controllata
    |
    `-- Run C: nuovo processo DS4
        replay completo dell'oracolo
        worker: HDD esterno -> banco RAM pinned
        solo sui layer 0..N-1 e solo ai target rispettanti STEP
```

Run B e C usano gli stessi expert dell'oracolo, quindi qualunque differenza misurata dipende dal scheduling e dal prefetch, non da una divergenza del routing.

## Codice coinvolto

```text
test_predictor/test_6/test_6_runner.c
    definisce nome, directory output e variabili pubbliche di TEST_6
    |
    `-- include test_predictor/test_5/test_5_runner.c
        riusa l'orchestrazione A/B/C e la scrittura delle trace
        |
        `-- ds4.c: ds4_test4_prefetch_enqueue_after_moe(...)
            conserva il primo token di decode come fase del burst;
            accoda solo quando (pos - primo_token) % STEP == 0
        |
        `-- ds4_cuda.cu: worker del banco pinned
            legge il clone HDD con Direct I/O e prepara gate/up/down in RAM pinned
```

Il worker CUDA usa ancora `DS4_TEST_4_PREFETCH_*` come interfaccia interna. Il runner traduce automaticamente i parametri pubblici di TEST_6; non impostare manualmente le variabili interne.

## Configurazione CLion

Crea una configurazione **Native Application** chiamata, ad esempio, `TEST_6 Stepped External-Device Prefetch`:

| Campo | Valore |
|---|---|
| Target | `test_6_runner` |
| Executable | `/home/michele/ds4/test_predictor/test_6/test_6_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | vuoto |

Nelle variabili d'ambiente, in un'unica riga separata da `;`:

```text
DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_5_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_5_PREFETCH_LAYERS=3;DS4_TEST_5_PREFETCH_INIT_DISTANCE=8;DS4_TEST_5_PREFETCH_STEP=8
```

Valori iniziali consigliati: `LAYERS=3`, `INIT_DISTANCE=8`, `STEP=8`. I default sono: 1 layer, init distance 1 e step 1. `INIT_DISTANCE` e limitato a 8 perche il banco CUDA dispone di al massimo otto target futuri per layer; `STEP` puo arrivare a 1024.

Ogni esecuzione crea `output`, poi `output_2`, `output_3`, ecc. I file hanno la stessa struttura di TEST_5: `routing_oracle.bin`, trace JSONL di routing/prefetch per A/B/C, stdout/stderr e `summary.json`. Quest'ultimo salva anche `prefetch_init_distance` e `prefetch_step`.

## Diagnostica del collo di bottiglia

`summary.json` raccoglie automaticamente i dati che prima erano distribuiti
nelle JSONL. Non aggiunge sincronizzazioni nel decode: il runner legge i trace
solo dopo che ciascun processo DS4 e terminato.

```text
run_B
    baseline: tutti i gruppi passano dal demand load NVMe -> RAM -> VRAM

run_C
    demand load per gli altri gruppi
    + worker HDD -> RAM pinned per i gruppi predetti
    + hit RAM pinned -> VRAM per i gruppi pronti
    |
    `-- summary.json confronta i tre costi
```

I campi principali sono:

| Sezione | Cosa misura | Come interpretarla |
|---|---|---|
| `run_b` / `run_c` | `generation_tps`, tempo totale del processo (`wall_seconds`) e CPU user/system | Se C ha piu CPU o piu tempo wall di B, il worker o il suo I/O aggiungono costo globale. |
| `run_b.demand_*` | Byte e tempi del percorso normale | E la baseline reale da confrontare, non una stima teorica. |
| `run_c.demand_*` | Costo rimasto per i miss di C | Mostra quanta parte del decode continua comunque a dipendere dal NVMe. |
| `run_c.hit_*` | Costo degli hit RAM pinned -> VRAM | Deve essere inferiore al demand load; non include la lettura HDD gia conclusa dal worker. |
| `prefetch_worker.worker_read_ms_*` | Tempo HDD -> banco RAM pinned | Se cresce molto, la sorgente esterna e/o USB e il limite. |
| `prefetch_worker.queue_to_worker_ms_*` | Attesa prima che il worker inizi una richiesta | Se cresce, le richieste speculative arrivano piu velocemente di quanto il worker riesca a leggerle. |
| `prefetch_worker.queue_to_ready_ms_*` | Tempo completo coda + lettura fino a `ready` | Va confrontato con il tempo disponibile dato da `INIT_DISTANCE` e `STEP`. |
| `prefetch_worker.ready_bytes` | Dati realmente letti e conservati nel banco pinned | Quantifica il volume speculativo, anche se non tutto porta necessariamente a speedup netto. |
| `bank_full`, `lookup_loading`, `lookup_absent`, `stale_discarded`, `failed` | Motivo per cui uno slot non ha portato a un hit | Separano saturazione RAM, worker in ritardo, assenza intenzionale dovuta allo step, dati divenuti inutili ed errori. |
| `comparison` | Delta B/C, medie per hit/demand e `estimated_load_time_saved_by_hits_ms` | Il risparmio e solo il beneficio locale dei caricamenti: se C resta lenta, confrontarlo con wall/CPU/worker per trovare il costo che lo annulla. |

`estimated_load_time_saved_by_hits_ms` usa la differenza tra il demand load medio
di B e il costo medio di un hit di C, moltiplicata per gli hit. E un indicatore
utile, non un tempo totale garantito: i caricamenti CUDA e il compute non sono
necessariamente una sequenza completamente seriale.

## Interpretazione attesa

Un buon risultato non e necessariamente un numero assoluto maggiore di hit: con `STEP=8` sono pianificati meno token. Cerchiamo invece questi segnali:

- diminuzione di `prefetch_bank_full` e `prefetch_lookup_loading`;
- aumento della quota di `prefetch_hit` tra i soli target pianificati;
- minore `queue_to_ready_ms` rispetto al tempo disponibile;
- nessun peggioramento sistematico dei token/s di Run C rispetto a Run B.

Se il worker e pronto ma gli hit restano pochi, il limite e probabilmente la copertura intenzionalmente ridotta. Se invece i target pianificati restano `loading`, l'HDD USB e ancora troppo lento per quella combinazione di layer, distanza e step.

## Risultati osservati

I risultati seguenti usano il clone del modello sul disco rigido USB esterno.
Run B e Run C restano processi separati: il confronto e utile, ma una singola
misura di token/s contiene comunque una normale variabilita di I/O, GPU e
frequenza del sistema. Per questo il numero di hit, gli stati degli slot e i
tempi del worker sono piu informativi del solo token/s.

| Output | Layer | Init distance | Step | Run B | Run C | Hit / miss | Osservazione |
|---|---:|---:|---:|---:|---:|---:|---|
| `output` | 3 | 8 | 8 | 0,72 tok/s | 0,74 tok/s | 21 / 2688 | Lo step largo da tempo al worker: tutte le 21 richieste previste diventano hit, ma la copertura e volutamente molto piccola. |
| `output_2` | 10 | 8 | 8 | 0,80 tok/s | 0,74 tok/s | 70 / 2639 | Anche qui gli slot previsti arrivano pronti, ma i pochi hit non compensano sempre l'attivita del worker. |
| `output_3` | 10 | 8 | 3 | 0,79 tok/s | 0,82 tok/s | 129 / 2580 | Un run singolo apparentemente favorevole: piu burst generano piu hit, ma non e ancora una prova statistica di speedup stabile. |
| `output_4` | 10 | 8 | 3 | 0,81 tok/s | 0,75 tok/s | 51 / 2658 | Il trace diagnostico mostra in dettaglio perche la stessa configurazione puo peggiorare: worker seriale, coda lunga e banco saturo. |
| `output_5` | 10 | 8 | 3 | 0,74 tok/s | 0,79 tok/s | 190 / 2519 | Primo run con clone sul NVMe esterno USB4: tutte le richieste pianificate diventano hit, senza errori, coda bloccata o banco pieno. |
| `output_6` | 43 | 1 | 1 | 0,80 tok/s | 0,93 tok/s | 340 / 2369 | Primo upper bound che chiede `E_l(t+1)` per tutti i layer e per ogni token: gli hit sono concentrati nei layer iniziali; il worker seriale satura il banco per la maggior parte dei layer successivi. |

`output_3` e `output_4` hanno gli stessi parametri ma risultati diversi. Questo
non e una contraddizione: il disco rigido USB ha latenze variabili e il test
non isola ancora completamente la concorrenza fra CPU, GPU, USB e caching del
sistema. Proprio per questo TEST_6 salva ora la diagnostica estesa in
`output_4` e nelle esecuzioni successive.

### Diagnosi misurata in `output_4`

Il worker non fallisce: `failed=0`. Il limite e il tempo necessario per leggere
i gruppi di expert dall'HDD e il numero di richieste generate prima che il
worker riesca a smaltire quelle precedenti.

| Indicatore | Valore | Significato |
|---|---:|---|
| Dimensione di un gruppo MoE | 42,47 MiB | Sei expert completi dello stesso layer: blocchi gate, up e down. |
| Tempo medio di lettura worker | 1358 ms | L'HDD fornisce circa 31 MiB/s effettivi nel percorso Direct I/O del test. |
| Richieste accodate / avviate / pronte | 113 / 62 / 61 | Il worker singolo non riesce a iniziare tutte le richieste prima che alcune diventino inutili. |
| Attesa media fino all'avvio | 10,45 s | Le richieste attendono in coda molto piu del tempo disponibile per molti token target. |
| Attesa media fino a `ready` | 10,89 s | Coda piu lettura HDD: lo slot spesso non e pronto al momento del lookup. |
| `bank_full` | 77 | Gli slot di alcuni layer restano occupati da richieste in attesa o loading. |
| `lookup_loading` | 62 | Il token ha richiesto l'expert mentre il worker lo stava ancora leggendo. |
| Hit | 51 (1,883%) | Solo una piccola frazione dei 2709 eventi MoE evita il demand load. |
| Tempo medio demand B / hit C | 26,24 / 7,36 ms | Un hit e davvero piu rapido localmente. |
| Risparmio teorico da hit | 963 ms | Non basta a compensare il costo globale aggiunto dal worker. |

Gli hit di `output_4` riguardano solo layer 0, 1 e 2 (19, 19 e 13 hit).
Il worker arriva a iniziare 19 lavori per ciascuno dei layer 0--2, solo 5 per
il layer 3 e nessuno per i layer 4--9. I layer iniziali ricevono quindi gran
parte del servizio disponibile, mentre gli slot dei layer successivi si
riempiono e producono `bank_full` o `loading`.

Il costo osservato rende la conclusione precisa:

```text
beneficio locale: 51 hit * (26,24 - 7,36) ms ~= 0,96 s

costo osservato Run C rispetto a Run B:
  +8,83 s di wall time
  +2,80 s circa nei soli trasferimenti demand RAM -> VRAM
  +4,40 s circa di tempo CPU user + system
```

Il prefetch quindi funziona, ma in questa configurazione l'HDD e il worker
seriale costano piu del tempo che i pochi hit riescono a risparmiare.

### Risultati misurati in `output_5`: NVMe esterno USB4

`output_5` usa gli stessi parametri di `output_4` (`LAYERS=10`,
`INIT_DISTANCE=8`, `STEP=3`), ma il clone usato dal worker e sul NVMe WD
SN730 collegato tramite enclosure USB4. Il percorso demand di Run B e Run C
resta sul modello nel disco interno: cambia soltanto la sorgente del worker.

| Indicatore | Run B | Run C | Cosa indica |
|---|---:|---:|---|
| Generazione | 0,74 tok/s | 0,79 tok/s | In questa esecuzione C e piu veloce di circa **6,76%**. |
| Wall time | 102,251 s | 97,282 s | Run C termina 4,968 s prima. |
| Eventi demand | 2709 | 2519 | I 190 hit sostituiscono altrettanti demand load. |
| Hit / miss | — | 190 / 2519 | Il 7,014% dei 2709 eventi MoE totali e un hit. |
| Tempo medio demand di B | 28,499 ms | — | Costo medio del caricamento normale SSD interno -> RAM pinned -> VRAM. |
| Tempo medio hit di C | — | 7,692 ms | Costo del solo trasferimento gia pronto RAM pinned -> VRAM. |
| Risparmio locale stimato | — | 3953 ms | Differenza media demand/hit moltiplicata per 190 hit. |

Il 7,014% non misura l'accuratezza dell'oracolo: in TEST_6 il routing e
forzato dal riferimento di Run A, quindi i candidati sono corretti. E la
quota di eventi effettivamente **pianificata** con `STEP=3`: ci sono 19 burst
validi e 10 layer per burst, dunque `19 * 10 = 190` target. Tutti e 190
raggiungono `ready` e vengono poi consumati come `prefetch_hit`.

| Stato del worker / banco pinned | Valore | Interpretazione |
|---|---:|---|
| `enqueued` / `started` / `ready` | 190 / 190 / 190 | Nessuna richiesta resta indietro o fallisce. |
| `failed` | 0 | Il Direct I/O dal NVMe esterno funziona per tutte le richieste. |
| `bank_full` | 0 | Il banco pinned non si satura. |
| `lookup_loading` | 0 | Al lookup non esistono slot ancora in lettura. |
| `stale_discarded` | 0 | Nessun dato viene preparato troppo tardi. |
| `lookup_absent` | 440 | Non e un errore: sono eventi non coperti intenzionalmente da step, distanza o fine del decode. |
| Dati pronti nel banco | 8,07 GiB | I 190 gruppi MoE completi letti speculativamente. |

Il worker impiega in media **171,339 ms** per leggere un gruppo di circa
42,47 MiB e in media **855,207 ms** dalla coda allo stato `ready`. Nel test
con HDD lo stesso valore di lettura era circa 1358 ms e la coda arrivava a
oltre dieci secondi: qui il salto di prestazioni elimina la saturazione che
impediva ai layer successivi di essere serviti.

Questo e il primo segnale sperimentale netto che il prefetch puo produrre uno
speedup end-to-end quando la sorgente del worker e abbastanza veloce e
indipendente dal disco demand. Resta comunque una singola esecuzione: il passo
successivo corretto e ripetere la stessa configurazione piu volte e poi
variare un parametro alla volta (`LAYERS`, `INIT_DISTANCE` o `STEP`).

### Risultati misurati in `output_6`: tutti i layer, token successivo

`output_6` usa il clone NVMe esterno USB4 come `output_5`, ma cambia la
pianificazione nel caso che rappresenta l'upper bound piu vicino al predictor
finale:

```text
DS4_TEST_5_PREFETCH_LAYERS=43
DS4_TEST_5_PREFETCH_INIT_DISTANCE=1
DS4_TEST_5_PREFETCH_STEP=1
```

Quindi, dopo che il MoE del layer `l` del token `t` e stato accodato alla GPU,
il worker riceve l'oracolo di `E_l(t+1)`. La richiesta e fatta per ogni layer
`0..42` e per ogni token di decode che ha un successivo. Questo non e ancora
un predictor runtime: l'oracolo conosce sempre gli expert corretti e non ha il
costo di calcolo di una predizione. Misura pero in modo fedele la finestra
temporale, il banco pinned, la coda e il fallback che avrebbe il sistema
finale con predizioni perfette.

| Indicatore | Run B | Run C | Cosa indica |
|---|---:|---:|---|
| Generazione | 0,80 tok/s | 0,93 tok/s | In questa esecuzione C e piu veloce di circa **16,25%**. |
| Wall time | 95,538 s | 84,476 s | Run C termina 11,062 s prima. |
| Demand load | 2709 | 2369 | I 340 hit sostituiscono altrettanti demand load. |
| Hit / eventi MoE | -- | 340 / 2709 (12,551%) | Copertura effettiva del caso `t -> t+1` su tutti i layer. |
| Tempo medio demand B | 26,623 ms | -- | Caricamento normale NVMe interno -> RAM pinned -> VRAM. |
| Tempo medio hit C | -- | 5,837 ms | Solo trasferimento dal banco pinned gia pronto alla VRAM. |
| Risparmio locale stimato | -- | 7067 ms | Differenza media demand/hit moltiplicata per i 340 hit. |

Run A, B e C terminano tutte con exit code `0`. Inoltre `run_B.stdout.txt` e
`run_C.stdout.txt` hanno lo stesso SHA-256, cosi come
`run_B.routing.jsonl` e `run_C.routing.jsonl`. Il miglioramento osservato non
deriva quindi da token, routing o expert diversi: B e C hanno eseguito lo
stesso percorso MoE dell'oracolo e differiscono soltanto nel caricamento.

Il risultato non implica pero che il worker possa coprire tutti i 43 layer.
Le richieste teoriche sono `62 token con successore * 43 layer = 2666`. Il
worker ne accetta 443 e ne porta a `ready` 412; le altre 2223 trovano lo slot
del proprio layer gia occupato e vengono registrate come `prefetch_bank_full`.
Non e un errore: quel gruppo ricade nel demand load normale.

| Stato worker / banco pinned | Valore | Interpretazione |
|---|---:|---|
| `enqueued` | 2666 | Tutte le richieste teoriche `E_l(t+1)` sono state emesse dal core. |
| `queued` / `started` / `ready` | 443 / 413 / 412 | Il worker singolo ha potuto servire solo una parte della domanda. |
| `bank_full` | 2223 | Principale limite: per molti layer lo slot del token precedente e ancora loading o ready quando arriva la nuova richiesta. |
| `lookup_loading` | 103 | Il gruppo corretto esiste, ma la lettura non e finita al suo consumo. |
| `stale_discarded` | 66 | Il worker ha completato gruppi dopo che il token bersaglio era gia passato. |
| `failed` / `lookup_expert_mismatch` | 0 / 0 | Nessun errore Direct I/O e nessun uso di expert non corrispondenti. |

La distribuzione degli hit mostra il criterio di servizio attuale del worker.
Esso esamina gli slot in ordine di layer, quindi completa prima i layer
iniziali:

| Layer | Hit |
|---|---:|
| 0 | 59 |
| 1 | 59 |
| 2 | 59 |
| 3 | 57 |
| 4 | 53 |
| 5 | 33 |
| 6 | 16 |
| 7 | 3 |
| 8 | 1 |
| 9--42 | 0 |

La lettura fisica resta veloce, in media 166,355 ms per gruppo da circa
42,47 MiB. Il limite non e quindi una regressione del NVMe USB4, ma la
capacita complessiva: a ogni token arrivano 43 gruppi mentre il worker ne
legge uno alla volta. L'attesa media enqueue -> ready sale a 1462,640 ms e il
trace contiene anche richieste iniziali dei layer alti diventate `ready` dopo
oltre 68 s, ormai inutili. La politica corrente non puo quindi mantenere una
copertura uniforme dei 43 layer con un solo worker e un solo slot futuro per
layer.

`output_6` e comunque un risultato importante: dimostra che il disegno
`E_l(t) -> E_l(t+1)` e corretto, sicuro e utile per la porzione che il worker
riesce a servire. Il +16,25% e un singolo campione, non ancora una stima
statistica. I prossimi confronti devono ripetere questa configurazione e poi
misurare finestre progressive (`LAYERS=1, 2, 3, ...`) con
`INIT_DISTANCE=1` e `STEP=1`, prima di sostituire l'oracolo con il predictor
reale.

## Cosa potrebbe cambiare con un SSD NVMe esterno USB4

L'esperimento NVMe non e piu soltanto teorico: `output_5` usa il WD PC SN730
da 512 GB in un enclosure USB4. Windows ha misurato circa 2559 MB/s di lettura
sequenziale e 417 MB/s di lettura casuale, contro circa 83 MB/s e 1 MB/s del
vecchio HDD USB. La velocita osservata dal worker e naturalmente minore del
benchmark di Windows, perche ogni gruppo comporta tre letture Direct I/O,
gestione della coda e copia nei buffer pinned; resta pero sufficiente per
portare tutti i target pianificati a `ready`.

| Sorgente del worker | Ordine di grandezza per 42,47 MiB | Effetto atteso |
|---|---:|---|
| HDD USB | ~1358 ms osservati | Coda lunga, slot `loading`, `bank_full`, pochi hit. |
| NVMe esterno USB4 | ~171 ms osservati | Tutti i 190 target pianificati pronti; `bank_full=0` e `lookup_loading=0`. |

Con dieci layer, un burst e circa `10 * 42,47 MiB = 425 MiB`. L'HDD impiegava
all'incirca tredici secondi per leggerlo serialmente; il NVMe USB4 elimina
questo collo di bottiglia e rende possibile un overlap utile. Non garantisce
uno speedup proporzionale: restano il trasferimento RAM pinned -> VRAM, il copy
engine CUDA, la copertura limitata dallo step e la politica del worker.

La metrica decisiva resta la diminuzione di `worker_read_ms_average`,
`queue_to_ready_ms_average`, `lookup_loading` e `bank_full`, insieme a una
quota di hit sufficiente a superare l'overhead del worker. In `output_5` i
quattro segnali sono tutti favorevoli; ora occorre verificare la ripetibilita
del +6,76% e cercare una copertura maggiore senza ricreare una coda.

## Modello neurale reale che segue TEST_6

L'oracolo di `output_6` conosceva esattamente `E_l(t+1)` e quindi non misura
l'errore o il costo di un predictor. Dopo TEST_6 è stato addestrato e valutato
il modello neurale Build 5 `build5_fixed_nnz_hyperband`.

| Misura offline del predictor reale | Risultato |
|---|---:|
| Migliore validation Recall@6 | 0,490853 |
| Recall@6 sul test set mai usato nel tuning | 0,494631 |
| Checkpoint | `trial_checkpoints/0024/best.weights.h5` |

Il valore 0,494631 significa che, in media, circa 2,97 dei 6 expert realmente
necessari sono presenti tra i 6 expert predetti. Non è confrontabile
direttamente con la copertura perfetta dell'oracolo e non implica da solo uno
speedup. Il test successivo, `test_7`, misura la pipeline completa: estrazione
feature durante DS4, inferenza CPU del modello e learned prefetch.

La prima esecuzione completa di TEST_7 (`test_7/output_8`) ha misurato Recall@6
online `0,467742` su 62 predizioni e una latenza media CPU di circa `13,4 ms`.
Il risultato iniziale di `98,27 s` contro `103,45 s` va interpretato con
cautela. Dopo avere implementato il riuso parziale expert-per-expert,
`test_7/output_9` ha confermato 62 riusi parziali su 63 prefetch learned: la
lettura SSD media del gruppo coinvolto è scesa da `19,48 ms` a `11,02 ms`, ma
il tempo totale C è stato `101,50 s` contro `100,96 s` della baseline B
(+0,54%). Non esiste quindi ancora uno speedup dimostrato; occorrono
ripetizioni B/C alternate e probabilmente una maggiore sovrapposizione tra
predictor CPU e calcolo GPU.
