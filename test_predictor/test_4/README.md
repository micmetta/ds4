# TEST_4 — Oracle SSD → RAM pinned prefetch

## Glossario

| Termine | Significato semplice |
|---|---|
| **Expert / layer MoE** | Un layer MoE contiene molte piccole FFN, chiamate expert. Il router ne sceglie sei per elaborare un token. |
| **Router** | Parte del modello che calcola quali expert usare e il peso del contributo di ciascuno. |
| `E_l(t)` | I sei expert usati al token `t` nel layer `l`. `E_l(t+1)` e il gruppo che servira allo stesso layer per il token successivo. |
| **Oracle** | Registro creato da Run A. In TEST_4 conosce in anticipo gli expert del percorso di riferimento; sostituisce temporaneamente un vero predictor ML. |
| **Replay** | Run B o C esegue gli ID e i pesi registrati dall'oracle, cosi entrambe percorrono esattamente lo stesso cammino MoE. |
| **SSD / demand load** | L'SSD conserva i pesi completi. Un demand load e il caricamento normale avviato solo quando gli expert sono gia necessari. |
| **RAM pinned** | Area di RAM bloccata fisicamente: il sistema operativo non la puo spostare o scambiare finche CUDA la usa. Questo rende piu affidabile il trasferimento RAM -> VRAM. |
| **VRAM** | Memoria della GPU, dove gli expert devono trovarsi prima di essere eseguiti dai kernel CUDA. |
| **Prefetch** | Caricamento anticipato di dati che probabilmente serviranno poco dopo. In TEST_4 porta i pesi dall'SSD a un banco RAM pinned. |
| **Hit / miss** | Un hit significa che il gruppo futuro e pronto nel banco pinned; un miss usa il normale demand load. |
| **Worker** | Thread separato che esegue letture SSD verso il banco pinned mentre la GPU continua il lavoro corrente. |
| **Direct I/O** | Lettura SSD che evita la page cache Linux; riduce il rischio di misurare solo dati rimasti nella cache RAM del sistema operativo. |
| **JSONL** | Un file testuale con un oggetto JSON per riga: ogni riga rappresenta un evento e puo essere analizzata facilmente. |
| **token/s** | Velocita di generazione autoregressiva, cioe quanti token di risposta vengono prodotti in un secondo. |

## Obiettivo

TEST_4 sara il primo esperimento che prova realmente a nascondere la lettura SSD degli expert senza richiedere un secondo grande banco in VRAM.

TEST_4 usera tre processi DS4 separati. Run A registra il routing reale in un
oracolo. Run B riproduce forzatamente quel routing ma non esegue alcun
prefetch: e la baseline controllata. Run C riproduce lo stesso routing forzato
di Run A e abilita il prefetch SSD -> RAM pinned: e il caso da confrontare con
Run B.

~~~text
Domanda del test:

Se E_l(t+1) e gia nella RAM pinned quando arriva il token t+1,
quanta latenza SSD puo essere rimossa dal percorso critico?
~~~

## Esecuzione in CLion

Usa una configurazione Native Application chiamata, per esempio,
**TEST_4 Oracle Pinned-RAM Prefetch**, con:

| Campo | Valore |
|---|---|
| Target | `test_4_runner` |
| Executable | `test_predictor/test_4/test_4_runner` |
| Working directory | `/home/michele/ds4/` |
| Environment variables | `DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_4_PREFETCH_LAYERS=5` |

Il target compila il piccolo runner e ricompila `ds4` in modo seriale. Il runner
esegue automaticamente **Run A**, poi **Run B**, poi **Run C**, sempre come tre
processi distinti. Ogni esecuzione crea `output/`, poi `output_2/`, e cosi via,
in `test_predictor/test_4/`.

`DS4_TEST_4_PREFETCH_LAYERS` controlla la dimensione della finestra usata
**solo da Run C**. Il valore predefinito e `5`: vengono quindi preparati i
layer `0..4`. Con valore `N`, Run C prepara i layer `0..N-1`; sono accettati
valori da `0` a `43`. Il valore `0` disabilita la finestra di prefetch, mentre
un valore maggiore di `43` viene limitato a `43`. I buffer RAM pinned non sono
preallocati tutti insieme: ogni layer occupa RAM soltanto quando riceve la sua
prima richiesta dal worker.

## Perche questo test viene dopo TEST_3

| Misura TEST_3 | Risultato |
|---|---:|
| VRAM CUDA libera | 0 MiB |
| RAM Linux/WSL disponibile | ~9,7 GiB, stabile durante il decode |
| Pool pinned DS4 esistente | ~256 MiB |
| Gruppo di sei expert per layer | ~40,5 MiB |

Non possiamo aggiungere un banco VRAM. Possiamo invece usare un banco host pinned separato, con capacita iniziale di 256 MiB. Il valore predefinito della finestra e cinque gruppi futuri, circa 202,5 MiB; `DS4_TEST_4_PREFETCH_LAYERS` permette di variare il numero di layer per gli esperimenti successivi.

## Correttezza del confronto controllato

~~~text
router di Run B/C calcola la propria scelta originale
    |
    `--> TEST_4 la registra soltanto a fini diagnostici
            |
            `--> oracle replay sostituisce expert e pesi con E_l^A(t)
                    |
                    +--> Run B: demand load normale
                    |
                    `--> Run C: prova il banco pinned; hit o fallback normale
~~~

In TEST_4 il replay forzato e intenzionale: serve a fare in modo che Run B e
Run C eseguano lo stesso percorso di expert e abbiano una baseline confrontabile
al 100%. Questo e un esperimento diagnostico, non il comportamento finale del
predictor: nella versione finale il router restera autoritativo e il predictor
potra influire soltanto sul prefetch.

Un prefetch non pronto in Run C puo solo causare un fallback al demand load
normale. Non cambia gli expert e i pesi **effettivamente eseguiti**, che in
questo test sono gia fissati dall'oracolo di Run A.

## Flusso del test

### Ruolo di Run A, Run B e Run C

Le due run non sono due benchmark indipendenti da confrontare soltanto in
token/s. Hanno ruoli complementari.

| Run | Ruolo | Cosa produce/consuma | Cosa **non** fa |
|---|---|---|---|
| **Run A** | Costruisce la conoscenza futura perfetta del test. Esegue normalmente il prompt e il decode, e registra per ogni `(token, layer)` i sei expert realmente scelti. | Produce `routing_oracle.bin`: per esempio contiene `E_l^A(t+1)`, cioe gli expert che nel percorso di riferimento sono stati usati dal layer `l` al token successivo. | Non esegue prefetch e non misura lo speedup: serve a costruire il riferimento, non a dimostrare un miglioramento. |
| **Run B** | Baseline controllata. Calcola e registra il proprio routing originale, poi riproduce expert e pesi di Run A per l'esecuzione. | Consuma `routing_oracle.bin`; misura tempo, demand load, byte e output senza worker o banco prefetch. | Non avvia letture speculative SSD -> RAM pinned. |
| **Run C** | Caso sperimentale. Segue lo stesso replay forzato di Run B e, durante il MoE corrente, avvia il prefetch oracle del token successivo. | Consuma `routing_oracle.bin`, banco pinned e worker; registra hit, miss, attesa residua e tempo di decode. | Non usa il routing originale calcolato localmente per decidere gli expert effettivi. |

La distinzione fondamentale e quindi questa:

~~~text
Run A: E_l^A(t) = percorso MoE di riferimento salvato nell'oracolo

Run B: registra E_l^B,original(t), poi esegue E_l^A(t) senza prefetch

Run C: registra E_l^C,original(t), poi esegue E_l^A(t) con prefetch
~~~

Run B e Run C devono quindi avere, per costruzione, gli stessi expert, gli
stessi pesi router e idealmente gli stessi token. La differenza di tempo fra B
e C puo cosi essere attribuita al prefetch. I rispettivi routing originali sono
comunque salvati per mostrare cosa avrebbero scelto senza il replay.

Per ottenere davvero questa garanzia anche nei layer `is_hash=true`, TEST_4
dovra estendere il formato oracle di TEST_1: registrera e riprodurra gli ID e
i pesi di **tutti** i layer MoE, non soltanto dei layer top-k. Questo e
necessario perché TEST_1 attuale esclude i layer hash; se i token divergono,
essi possono ancora scegliere expert diversi.

### Differenza piccola ma importante rispetto a TEST_1

TEST_1 ha dimostrato che il replay dell'oracolo funziona per i layer MoE
top-k. I suoi ultimi esperimenti hanno anche raggiunto il 100% di match, ma
questo e accaduto perche Run A e Run B hanno generato spontaneamente la stessa
sequenza di token fino alla fine. In quella situazione i primi tre layer
`is_hash=true` ricevono lo stesso `token_id` e scelgono naturalmente gli stessi
expert, pur non essendo nel file oracle di TEST_1.

TEST_4 non deve dipendere da questa condizione favorevole. Estende quindi
l'oracolo per registrare e riprodurre anche nei layer hash sia gli ID sia i
pesi dei sei expert. Di conseguenza Run B e Run C eseguono il percorso di Run A
per **tutti** i layer MoE, anche se il loro routing locale o i loro token
avrebbero iniziato a divergere. Questa e una differenza minima nel meccanismo
di replay, ma essenziale per attribuire un'eventuale differenza di tempo al
prefetch e non al fatto che le run abbiano eseguito expert diversi.

~~~text
test_predictor/test_4/test_4_runner.c
    |
    +--> Run A, processo DS4 separato: costruisce il riferimento
    |      |
    |      +--> ds4.c: ds4_routing_oracle_*()
    |      |      registra E_l^A(t) per ogni token/layer
    |      `--> routing_oracle.bin
    |
    +--> Run B, nuovo processo: baseline controllata senza prefetch
    |      |
    |      +--> calcola e registra il routing originale di Run B
    |      +--> replay oracle: esegue gli expert e i pesi di Run A
    |      `--> demand load normale SSD -> RAM pinned -> VRAM
    |
    `--> Run C, nuovo processo: misura il prefetch controllato
           |
           |  PERCORSO CRITICO: token t, layer l
           |  -----------------------------------
           +--> ds4.c: metal_graph_decode_cuda_selected_load(...)
           |      router locale calcola e registra E_l^C,original(t)
           |      replay oracle imposta E_l^A(t), gli expert da eseguire
           |
           +--> ds4_cuda.cu: cuda_stream_selected_cache_begin_load(...)
           |      demand corrente: SSD -> RAM pinned staging -> VRAM
           |
           +--> ds4_gpu_routed_moe_one_tensor(...)
           |      accoda alla GPU l'esecuzione degli expert E_l^A(t)
           |
           +--> dopo l'accodamento del MoE:
           |      la CPU ha appena inviato allo stream CUDA i kernel di E_l^A(t)
           |      e non attende qui il loro completamento; mentre la GPU li esegue,
           |      la CPU puo quindi avviare il lavoro del worker SSD
           |
           |      l'oracolo fornisce il candidato futuro E_l^A(t+1):
           |      i sei ID che Run A ha usato per il layer l del prossimo token
           |
           |      la CPU crea una richiesta separata dal lavoro CUDA:
           |      prefetch_request { target_token=t+1, layer=l, expert_ids=E_l^A(t+1) }
           |      e la inserisce nella coda del worker SSD
           |
           |      questa richiesta e il punto di consegna che apre l'overlap:
           |      GPU continua E_l^A(t)  ||  worker prepara E_l^A(t+1)
           |
           +================= ATTIVITA' IN PARALLELO =================+
           |                                                           |
           |  GPU / decode                                             |  worker TEST_4
           |  -------------                                             |  --------------------------
           |  la GPU esegue E_l^A(t)                                    |  riceve il candidato E_l^A(t+1)
           |  e Run C prosegue verso i layer successivi                 |          |
           |                                                           |          v
           |                                                           |  **BANCO PINNED PREFETCH**
           |                                                           |  dizionario: (t+1, l) -> slot
           |                                                           |  slot: empty -> loading
           |                                                           |          |
           |                                                           |          v
           |                                                           |  Direct I/O: SSD -> gate/up/down
           |                                                           |                 nei buffer RAM pinned dello slot
           |                                                           |          |
           |                                                           |          v
           |                                                           |  slot: ready / failed
           |                                                           |
           +===========================================================+
           |
           `--> PERCORSO CRITICO: token t+1, layer l
                  |
                  +--> **LOOKUP BANCO PINNED PREFETCH**
                  |      cerca nel dizionario la chiave (t+1, l)
                  |
                  +--> hit: slot ready con gli ID oracle E_l^A(t+1)
                  |      usa gate/up/down gia in RAM pinned
                  |      -> RAM pinned -> banco domanda VRAM -> esecuzione
                  |
                  `--> miss: slot assente, loading, failed o IDs diversi
                         -> demand load normale: SSD -> RAM pinned -> VRAM
~~~

## Banco pinned prefetch

Il banco sara separato dal pool pinned che il demand load usa gia. Ogni slot e indicizzato da `(target_token, layer)` e contiene: `selected_expert_ids[6]`, pesi `gate`, `up`, `down`, dimensioni dei blocchi e stato `empty | loading | ready | consumed | failed`.

La chiave include token e layer perche durante il token `t` il worker prepara
molti gruppi per `t+1`; un solo buffer globale sarebbe sovrascritto prima del
consumo. La finestra ha un numero configurabile di slot, uno per ciascun
layer attivo. Con il valore predefinito sono i layer `0..4`. Se uno slot e ancora in lettura, il decode non aspetta: al
consumo registrera un miss e usera il demand load normale.

### Struttura dati: dizionario del banco pinned

Useremo un piccolo dizionario con una voce per ciascun layer della finestra,
interamente in RAM normale. La chiave univoca `(target_token, layer)` identifica direttamente
lo slot del banco pinned che contiene le informazioni per quello specifico
token e layer. Il dizionario serve solo a localizzare lo slot; i pesi veri e
propri restano in una zona separata della RAM host chiamata **RAM pinned**.

La RAM pinned e memoria che CUDA blocca temporaneamente: il sistema operativo
non puo spostarne le pagine su disco o ricollocarle mentre sono registrate. Per
questo CUDA puo trasferire in modo diretto, efficiente e prevedibile i buffer
`gate`, `up` e `down` dalla RAM pinned alla VRAM. Il worker di TEST_4 legge
prima gli expert dall'SSD in questi buffer; quando il router conferma un hit,
DS4 puo poi effettuare soltanto il tratto RAM pinned -> VRAM, senza ripetere la
lettura SSD.

~~~text
chiave: (target_token, layer)
                     |
                     v
dizionario_pinned[(target_token, layer)]
                     |
                     v
slot del banco pinned
    - chiave: target_token, layer
    - expert_ids[6]
    - gate_bytes_size, up_bytes_size, down_bytes_size
    - gate_buffer, up_buffer, down_buffer       [RAM pinned]
    - stato: empty | loading | ready | consumed | failed
    - tempi: enqueue, inizio I/O, ready, consumo
~~~

### Significato dei campi di uno slot

| Campo | Tipo concettuale | Significato |
|---|---|---|
| `target_token` | intero | Token futuro per cui lo slot e stato preparato. Se il worker parte durante il token `t`, vale `t+1`. |
| `layer` | intero | Layer MoE a cui appartiene il gruppo prefetched. In questo modo lo slot di layer 0 non puo essere usato per layer 1. |
| `(target_token, layer)` | chiave del dizionario | Identificatore usato dal decode per trovare rapidamente lo slot giusto quando arriva quel token a quel layer. |
| `expert_ids[6]` | sei interi | I sei expert candidati prodotti dall'oracolo per quella chiave. In Run C sono confrontati con i sei ID appena prodotti dal router locale, prima che il replay imponga il percorso controllato di Run A. |
| `gate_buffer` | puntatore | Indirizzo iniziale del blocco `gate` nella RAM pinned dello slot. Il blocco contiene in sequenza le matrici gate dei sei expert. |
| `up_buffer` | puntatore | Indirizzo iniziale del blocco `up` nella RAM pinned dello slot. Contiene le matrici up dei sei expert. |
| `down_buffer` | puntatore | Indirizzo iniziale del blocco `down` nella RAM pinned dello slot. Contiene le matrici down dei sei expert. |
| `gate_bytes_size` | numero di byte | Dimensione totale del blocco indicato da `gate_buffer`, quindi dei pesi gate dei sei expert presenti nello slot. |
| `up_bytes_size` | numero di byte | Dimensione totale del blocco indicato da `up_buffer`. |
| `down_bytes_size` | numero di byte | Dimensione totale del blocco indicato da `down_buffer`. |
| `state` | stato | Ciclo di vita: `empty` libero; `loading` in lettura SSD; `ready` pronto per il decode; `consumed` trasferito o scartato dopo il controllo; `failed` lettura non riuscita. |
| timestamp di `enqueue`, `start`, `ready`, `consume` | tempi | Permettono di misurare quanto il worker ha letto in anticipo e quale attesa residua resta al momento del consumo. |

I tre `*_buffer` sono quindi puntatori, mentre i tre `*_bytes_size` sono
dimensioni. Per esempio, `gate_buffer` indica il primo byte del blocco gate e
`gate_bytes_size` indica quanti byte compongono l'intero blocco gate da
trasferire. Nel primo TEST_4 `gate_bytes_size` e **gia la dimensione totale
dei sei expert gate**, non la dimensione di un singolo expert.

Il trasferimento gate usera quindi questo calcolo semplice:

~~~text
indirizzo iniziale = gate_buffer
byte da trasferire = gate_bytes_size
intervallo copiato  = [gate_buffer, gate_buffer + gate_bytes_size)
~~~

`gate_buffer + gate_bytes_size` e l'indirizzo immediatamente successivo
all'ultimo byte copiato; non e necessario calcolare
`gate_buffer + 6 * gate_bytes_size`, perche questo copierebbe sei volte il
blocco completo.

Ogni blocco raggruppa i sei expert selezionati. Se la porzione gate di un singolo
expert ha dimensione `gate_expert_bytes_size`, l'inizio della sua matrice nel
blocco e:

~~~text
gate_buffer + indice_expert * gate_expert_bytes_size
~~~

In questa notazione, se tutti i sei expert hanno la stessa dimensione,
`gate_bytes_size = 6 * gate_expert_bytes_size`. Questa formula serve soltanto
per capire la posizione interna della matrice di un singolo expert; non sara
usata per il trasferimento compatto del primo TEST_4.

Il primo TEST_4 non usera questo offset per spedire 18 matrici separate. Poiche
DS4 esegue gia il layer con il gruppo completo di sei expert, al `prefetch_hit`
trasferira i tre blocchi contigui completi:

~~~text
gate_buffer  (gate_bytes_size byte)  -> banco domanda VRAM
up_buffer    (up_bytes_size byte)    -> banco domanda VRAM
down_buffer  (down_bytes_size byte)  -> banco domanda VRAM
~~~

Sono tre trasferimenti, non `6 gate + 6 up + 6 down` trasferimenti individuali.
Questo mantiene lo stesso layout del loader DS4 corrente, riduce chiamate CUDA e
gestione degli offset, e rende il confronto con il demand load piu affidabile.

La ricerca effettuata quando il router reale richiede `E_l(t)` sara quindi:

~~~text
(token t, layer l)
    -> cerca la chiave (t, l) nel dizionario
    -> slot assente/non-ready/IDs diversi: miss e demand load normale
    -> slot ready e IDs uguali: hit, usa i suoi buffer pinned per RAM -> VRAM
~~~

### Finestra configurabile e layer prefetched

Run A registra nell'oracolo il routing di **tutti** i token di decode e di
**tutti** i layer MoE. Non conserva i pesi degli expert in RAM: salva solo gli
ID selezionati, per esempio `E_0^A(t+1)`, `E_1^A(t+1)`, fino a
`E_42^A(t+1)`, nel file `routing_oracle.bin`.

Man mano durante l'esecuzione del Run C per il token `t`, il banco pinned
memorizza uno slot per ogni layer della finestra. Con il valore predefinito
`DS4_TEST_4_PREFETCH_LAYERS=5`, durante la computazione del token `t` il worker
riceve queste cinque richieste future:

~~~text
dopo E_0^A(t)  -> Run C richiede al worker la memorizzazione di (t+1, layer 0) nel banco_pinned in RAM.
dopo E_1^A(t)  -> Run C richiede al worker la memorizzazione di (t+1, layer 1) nel banco_pinned in RAM.
dopo E_2^A(t)  -> Run C richiede al worker la memorizzazione di (t+1, layer 2) nel banco_pinned in RAM.
dopo E_3^A(t)  -> Run C richiede al worker la memorizzazione di (t+1, layer 3) nel banco_pinned in RAM.
dopo E_4^A(t)  -> Run C richiede al worker la memorizzazione di (t+1, layer 4) nel banco_pinned in RAM.
~~~

Se le letture terminano in tempo, poco prima dell'inizio di `t+1` il banco ha
questo significato:

~~~text
banco_prefetch_pinned = {
  (t+1, layer 0): slot pronto oppure in caricamento,
  (t+1, layer 1): slot pronto oppure in caricamento,
  (t+1, layer 2): slot pronto oppure in caricamento,
  (t+1, layer 3): slot pronto oppure in caricamento,
  (t+1, layer 4): slot pronto oppure in caricamento
}
~~~

Nel primo TEST_4 i layer `5..42` del token futuro non vengono prefetched e
seguono il demand load DS4 normale. Non possiamo riutilizzare subito lo slot di
`(t+1, layer 0)` per `(t+1, layer 5)`: il primo slot deve restare disponibile
fino a quando Run C raggiunge layer 0 del token `t+1`. Sostituirlo prima
ci farebbe perdere i dati prefetchati e quindi annullerebbe proprio il prefetch che vogliamo misurare.

Gli slot tornano liberi progressivamente nel token successivo. Per ciascun
layer la sequenza e sempre: **consumo dello slot corrente -> slot libero ->
nuova richiesta al worker per il token successivo**.

~~~text
TOKEN t+1, LAYER 0
    |
    +--> router locale produce E_0^C,original(t+1); il replay esegue E_0^A(t+1)
    |
    +--> lookup e consumo/scarto dello slot (t+1, layer 0)
    |      -> slot 0 torna libero
    |
    `--> dopo E_0^A(t+1), Run C richiede al worker:
           memorizza (t+2, layer 0) nel banco_pinned in RAM, nello slot 0

TOKEN t+1, LAYER 1
    |
    +--> router locale produce E_1^C,original(t+1); il replay esegue E_1^A(t+1)
    |
    +--> lookup e consumo/scarto dello slot (t+1, layer 1)
    |      -> slot 1 torna libero
    |
    `--> dopo E_1^A(t+1), Run C richiede al worker:
           memorizza (t+2, layer 1) nel banco_pinned in RAM, nello slot 1

TOKEN t+1, LAYER 2
    |
    +--> router locale produce E_2^C,original(t+1); il replay esegue E_2^A(t+1)
    |
    +--> lookup e consumo/scarto dello slot (t+1, layer 2)
    |      -> slot 2 torna libero
    |
    `--> dopo E_2^A(t+1), Run C richiede al worker:
           memorizza (t+2, layer 2) nel banco_pinned in RAM, nello slot 2

TOKEN t+1, LAYER 3
    |
    +--> router locale produce E_3^C,original(t+1); il replay esegue E_3^A(t+1)
    |
    +--> lookup e consumo/scarto dello slot (t+1, layer 3)
    |      -> slot 3 torna libero
    |
    `--> dopo E_3^A(t+1), Run C richiede al worker:
           memorizza (t+2, layer 3) nel banco_pinned in RAM, nello slot 3

TOKEN t+1, LAYER 4
    |
    +--> router locale produce E_4^C,original(t+1); il replay esegue E_4^A(t+1)
    |
    +--> lookup e consumo/scarto dello slot (t+1, layer 4)
    |      -> slot 4 torna libero
    |
    `--> dopo E_4^A(t+1), Run C richiede al worker:
           memorizza (t+2, layer 4) nel banco_pinned in RAM, nello slot 4
~~~

Il dizionario mantiene esplicita la relazione univoca
`(target_token, layer) -> slot`: il decode trova lo slot senza cercare i pesi
nei buffer sbagliati e senza creare nuove strutture durante il decode. La stessa
struttura resta valida quando la finestra viene aumentata oltre il valore
predefinito di cinque gruppi.
Un mutex breve protegge il passaggio `loading -> ready` tra worker e thread di
decode; il worker non tocca mai uno slot gia in uso dal consumer.

## Perche e possibile creare overlap

~~~text
token t, layer l     -> avvia prefetch E_l(t+1)
token t, layer l+1   -> GPU continua il lavoro corrente
...
token t+1, layer l   -> consuma E_l(t+1)
~~~

Ogni gruppo ha approssimativamente un token di anticipo. TEST_2 ha misurato circa 19--20 ms di lettura SSD e 7--9 ms di trasferimento RAM -> VRAM per gruppo. TEST_4 vuole sovrapporre soprattutto la lettura SSD; il trasferimento finale verso VRAM rimane necessario.

## Misure previste

| Evento/campo | Significato |
|---|---|
| `prefetch_enqueue` | Richiesta `(t+1, l)` accodata al worker di Run C. |
| `prefetch_hit` | Lo slot con chiave `(t+1,l)` era pronto e i sei ID coincidevano: sono stati trasferiti soltanto i tre blocchi RAM pinned -> VRAM. |
| `prefetch_miss_demand` | Slot assente, ancora in lettura o non pronto: DS4 ha usato il demand load normale SSD -> RAM pinned -> VRAM. |
| `bytes` | Byte trasferiti verso la VRAM per quel gruppo; su hit non include una nuova lettura SSD. |
| `ssd_read_ms` | Tempo SSD sul percorso critico. Su `prefetch_hit` deve essere zero, poiche la lettura e stata svolta dal worker prima del consumo. |
| `transfer_from_RAM_to_VRAM_ms` | Tempo del trasferimento RAM pinned -> VRAM e della sua sincronizzazione. |
| `total_load_event_ms` | Tempo totale del loader osservato dal decode per quel gruppo. |
| `run_B_generation_tps`, `run_C_generation_tps` | Token/s di B e C, estratti dai rispettivi stderr e riportati nel `summary.json`. |

Il `summary.json` confronta B e C. `run_A.routing.jsonl` conserva il percorso
di riferimento, mentre `run_B.routing.jsonl` e `run_C.routing.jsonl`
conservano anche il routing originale locale e quello forzato dall'oracolo.
Uno speedup sara valido soltanto se l'output resta
equivalente.

## Cosa non e ancora incluso

- predictor ML;
- banco prefetch in VRAM;
- modifica della cache expert corrente;
- sostituzione del router reale;
- predictor ML al posto dell'oracolo.

TEST_4 misurera quindi un upper bound controllato del prefetch host-side. Se riduce l'attesa SSD senza rallentare il decode, potremo aumentare la finestra o sostituire l'oracolo con un predictor reale.

## Nota sui primi layer `is_hash=true`

I primi tre layer MoE del modello sono marcati `is_hash=true`. In questi layer
DS4 non sceglie i sei expert con il normale top-k del router: la funzione
`layer_hash_selected_experts()` in `ds4.c` legge gli ID da una tabella
`ffn_gate_tid2eid`, indicizzata dal token ID corrente.

In pratica il modello fornisce per questi layer una regola del tipo:

~~~text
token_id corrente
       |
       v
ffn_gate_tid2eid[token_id]
       |
       v
sei ID expert da eseguire
~~~

Il primo `token_id corrente` non e un numero fisso: dipende dal prompt dopo
la tokenizzazione. Nel decode della risposta, il flusso e il seguente:

~~~text
prompt testuale
    -> tokenizer
    -> [token_id prompt 0, ..., token_id prompt finale]
    -> prefill del prompt
    -> logits finali del prefill
    -> sampling con temp=0
    -> primo token generato y_0
    -> decode di y_0
    -> nei layer is_hash: ffn_gate_tid2eid[y_0] -> sei expert
~~~

Quindi, nel primo passo di decode della risposta, il `token_id corrente` dei
layer hash e normalmente il primo token generato `y_0`. Durante il prefill,
invece, ciascun token del prompt e a sua volta un token corrente e puo usare la
stessa tabella di selezione hash.

Il codice DS4 definisce esplicitamente questi come “early DS4 layers” con
token-id hash routing. La repository non documenta invece la motivazione di
training originale degli autori del modello; non possiamo quindi affermare con
certezza *perche* DeepSeek abbia scelto questa architettura. Possiamo però
descrivere i vantaggi pratici che essa comporta nell'inferenza.

| Vantaggio pratico | Perche |
|---|---|
| Selezione riproducibile a parita di token ID | Gli ID non dipendono dal confronto top-k fra score quasi uguali: DS4 legge una riga prefissata della tabella. |
| Selezione piu semplice | Per individuare i sei expert non serve ordinare o confrontare i 256 score del router. |
| Candidato di prefetch piu prevedibile una volta noto il token ID | Conoscendo il token di input del layer hash, gli expert sono ottenibili con una lookup nella tabella. |
| Meno sensibilita al non-determinismo del routing | Piccole differenze floating-point nei router score non cambiano gli ID hash, perché non sono gli score a decidere la selezione. |

Il layer continua comunque a calcolare i pesi di combinazione degli expert a
partire dal hidden state: cambiano solo gli ID selezionati, non il fatto che i
sei contributi debbano essere pesati e combinati. Inoltre, durante il token
`t` il token ID futuro `t+1` non è ancora noto finché il modello non lo ha
generato: anche per i layer hash, quindi, TEST_4 usa l'oracolo per simulare la
conoscenza in anticipo di `E_l(t+1)`.

Questo non elimina il costo di streaming: anche questi sei expert hanno pesi
`gate`, `up` e `down` e devono seguire il normale percorso:

~~~text
SSD -> RAM pinned -> VRAM -> esecuzione MoE
~~~

Di conseguenza ha senso includere i layer `0..4` nella finestra predefinita di
TEST_4:

| Layer | Come sono scelti gli expert | Utilita nel TEST_4 |
|---|---|---|
| `0..2` (`is_hash=true`) | Tabella deterministica token-ID -> sei expert | Verificano worker, banco pinned, lookup e riduzione dell'attesa SSD. |
| `3..4` (`is_hash=false`) | Score del router e top-k | Verificano lo stesso meccanismo nel caso che sara rappresentativo del futuro predictor ML. |

I layer hash sono quindi utili per misurare lo speedup I/O complessivo e per
validare il flusso corretto del banco pinned. Saranno invece meno indicativi
quando sostituiremo l'oracolo con un predictor ML: la loro selezione deriva da
una tabella legata al token ID, non dal ranking dei router score.

## Implementazione completata e avvio del test

TEST_4 e implementato.

In sintesi, il test esegue questo esperimento controllato:

- **Run A** crea un oracle completo, inclusi i layer `is_hash=true`.
- **Run B** riproduce l'oracle senza prefetch: e la baseline controllata.
- **Run C** riproduce lo stesso oracle e usa un worker SSD piu un banco RAM
  pinned separato per i layer della finestra configurata (per default `0..4`).
- Se il gruppo prefetched e pronto, il percorso e `RAM pinned -> VRAM`.
- Se non e pronto, DS4 segue il fallback invariato
  `SSD -> RAM pinned -> VRAM`.
- Il routing locale di B e C viene comunque registrato a fini diagnostici. Nel
  test, il replay sostituisce ID e pesi con quelli di Run A per mantenere B e C
  confrontabili; non e il comportamento previsto per il predictor finale.

### File creati o modificati

| File | Funzioni/parti principali | Perche e stato toccato |
|---|---|---|
| `test_predictor/test_4/test_4_runner.c` | `run_ds4()`, `make_output()`, `write_summary()` | Avvia automaticamente Run A, Run B e Run C come processi DS4 distinti; crea output incrementali e produce `summary.json`. |
| `test_predictor/test_4/README.md` | Documentazione di TEST_4 | Descrive il flusso controllato, il banco pinned, i limiti della prima finestra e le misure attese. |
| `test_predictor/test_4/.gitignore` | Output e binario del runner | Evita di versionare risultati sperimentali e binari locali. |
| `Makefile` | Target `test_4_runner` e `test-predictor-4` | Compila il runner e ricompila `ds4` in modo seriale prima dell'esecuzione. |
| `ds4.c` | `metal_graph_decode_set_hash_selected_override()`, `metal_graph_decode_cpu_router()`, `ds4_test4_prefetch_enqueue_after_moe()` | Estende l'oracle anche ai layer hash; prova un hit nel banco pinned prima del demand load; accoda il candidato `(t+1, layer)` subito dopo che il MoE corrente e stato accodato alla GPU. |
| `ds4_cuda.cu` | `cuda_test4_prefetch_worker()`, `cuda_test4_read_slot()`, `cuda_test4_upload_slot()`, `ds4_gpu_test4_prefetch_enqueue()`, `ds4_gpu_test4_prefetch_try_selected_load()` | Implementa il banco host-pinned separato, il worker SSD, l'upload RAM pinned -> VRAM e il fallback non bloccante. Il cleanup attende anche il worker prima di liberare le risorse CUDA. |
| `ds4_gpu.h` | API `ds4_gpu_test4_prefetch_enqueue()` e `ds4_gpu_test4_prefetch_try_selected_load()` | Espone a `ds4.c` solo le due operazioni necessarie del banco TEST_4: accodare un prefetch e tentare un hit. |

Le modifiche al core sono attive solo quando Run C imposta
`DS4_TEST_4_PREFETCH=1`. Run B non abilita questa variabile e quindi mantiene
il demand load normale.

### Configurazione CLion da creare

Crea una configurazione **Native Application** con nome, per esempio,
**TEST_4 Oracle Pinned-RAM Prefetch**:

| Campo CLion | Valore |
|---|---|
| Target | `test_4_runner` |
| Executable | `test_predictor/test_4/test_4_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | nessuno |
| Environment variables | `DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |

Il target compila il runner e ricompila `ds4` in modo seriale. Il runner avvia
automaticamente **Run A**, poi **Run B**, poi **Run C**, sempre come tre
processi distinti. Ogni esecuzione crea `output/`, poi `output_2/`, e cosi via,
in `test_predictor/test_4/`.

In ciascuna cartella di output troverai almeno:

- `routing_oracle.bin`, creato da Run A;
- stdout e stderr separati per Run A, B e C;
- `run_A.routing.jsonl`, `run_B.routing.jsonl` e `run_C.routing.jsonl`;
- `run_A.prefetch.jsonl`, `run_B.prefetch.jsonl` e `run_C.prefetch.jsonl`;
- `summary.json`, con exit code, token/s e conteggi di hit, miss e richieste di
  prefetch.

La compilazione statica di `ds4.c`, `ds4_cuda.cu` con architettura `sm_89` e
del runner e stata verificata. L'inferenza sul modello non e stata eseguita
durante l'implementazione: va lanciata dalla configurazione qui descritta.

## Test eseguiti

La prima esecuzione completa e salvata in `test_predictor/test_4/output/`.
Run A, Run B e Run C sono terminati tutti con exit code `0`: quindi il test ha
completato i tre processi senza errori. Questa sezione spiega prima i file
prodotti, poi interpreta esclusivamente i risultati della cartella `output/`
attuale.

### File nella cartella `output/`

| File | Chi lo produce | Contenuto e scopo |
|---|---|---|
| `routing_oracle.bin` | Run A | Registro binario compatto del percorso MoE di riferimento. Contiene, per ogni coppia `(token_index, layer)`, gli ID degli expert e i relativi pesi da riprodurre. Run B e Run C lo leggono. Non e pensato per la lettura manuale. |
| `run_A.stdout.txt` | Run A | Il testo generato dal modello durante la creazione dell'oracolo. |
| `run_B.stdout.txt`, `run_C.stdout.txt` | Run B, Run C | I testi generati dalla baseline e dal caso con prefetch. Devono coincidere se il replay e corretto. |
| `run_A.stderr.txt`, `run_B.stderr.txt`, `run_C.stderr.txt` | DS4 | Log tecnici: inizializzazione CUDA, SSD streaming, cache, prefill e velocita di generazione. |
| `run_A.routing.jsonl` | DS4 / Run A | Trace del routing reale che costruisce l'oracolo di riferimento. |
| `run_B.routing.jsonl`, `run_C.routing.jsonl` | DS4 / Run B-C | Trace del routing locale e degli expert/pesi effettivamente eseguiti dopo il replay dell'oracolo. |
| `run_A.prefetch.jsonl` | DS4 / Run A | Trace dei demand load del percorso di riferimento, senza replay e senza prefetch. |
| `run_B.prefetch.jsonl`, `run_C.prefetch.jsonl` | DS4 / Run B-C | Trace del caricamento dei gruppi di expert: demand load, richiesta di prefetch e hit del banco pinned. |
| `summary.json` | Runner TEST_4 | Riassunto compatto di exit code, token/s e contatori B/C. |

### `summary.json`

| Campo | Significato |
|---|---|
| `test` | Nome dell'esperimento che ha generato il file. |
| `timestamp_utc` | Data e ora UTC in cui il runner ha scritto il riepilogo. |
| `run_a_exit_code`, `run_b_exit_code`, `run_c_exit_code` | Codice di uscita dei tre processi: `0` significa completato correttamente. |
| `run_b_generation_tps`, `run_c_generation_tps` | Velocita di decode estratta dalla riga `generation: ... t/s` dei rispettivi `stderr`. |
| `run_c_prefetch_layers` | Numero di layer nella finestra del prefetch di Run C. `5` significa layer `0..4`; il valore deriva da `DS4_TEST_4_PREFETCH_LAYERS`. |
| `run_b_prefetch_hits` | Hit in B: deve essere `0`, perche B non abilita il prefetch. |
| `run_c_prefetch_hits` | Gruppi serviti dal banco pinned in C, senza una nuova lettura SSD nel momento della richiesta. |
| `run_c_prefetch_miss_demand` | Gruppi di C non pronti nel banco e quindi caricati con il demand load normale. |
| `run_c_prefetch_enqueues` | Richieste future accodate al worker del prefetch. Non implica da sola che il caricamento fosse gia pronto. |

Il riepilogo della cartella `output/` attuale e il seguente:

| Misura | Run B: replay senza prefetch | Run C: replay + prefetch pinned |
|---|---:|---:|
| Exit code | 0 | 0 |
| Velocita di generazione | 0,73 token/s | 0,87 token/s |
| Prefetch hit | 0 | 310 |
| Prefetch miss / demand load | 2.709 demand load | 2.399 |
| Richieste accodate | 0 | 310 |

Il runner legge i token/s dalla riga `generation: ... t/s` presente nei file
`stderr`, che e il punto in cui DS4 riporta la velocita di decode.

### `run_*.stdout.txt` e `run_*.stderr.txt`

I file `stdout` non hanno campi JSON: contengono direttamente la risposta
testuale. La loro uguaglianza tra B e C e un controllo semplice ma importante:
il prefetch non deve cambiare il testo generato.

I file `stderr` sono log umani di DS4 e non un formato a campi rigidi. Le righe
piu utili per TEST_4 sono queste:

| Voce nel log | Significato |
|---|---|
| `CUDA backend initialized ...` | GPU CUDA effettivamente selezionata. |
| `CUDA model direct I/O enabled ...` | Le letture del modello usano Direct I/O, quindi non dipendono dalla page cache Linux. |
| `cuda SSD streaming ... expert cache ...` | Budget della cache degli expert in VRAM e numero di expert che DS4 puo gestire nel budget. |
| `prefill: X t/s` | Velocita dell'elaborazione iniziale del prompt. Non e il dato principale del prefetch token-a-token. |
| `generation: X t/s` | Velocita del decode autoregressivo usata nel confronto B/C. |

| Run | Prefill | Generazione | Commento |
|---|---:|---:|---|
| A | 1,33 token/s | 0,73 token/s | Crea il percorso di riferimento e le sue trace; non e la baseline di speedup. |
| B | 1,28 token/s | 0,73 token/s | Baseline controllata: replay, nessun worker pinned. |
| C | 1,40 token/s | 0,87 token/s | Replay identico a B con worker SSD -> RAM pinned sui layer `0..4`. |

Il confronto corretto e **Run B -> Run C**: entrambe leggono lo stesso oracolo,
riproducono gli stessi expert e gli stessi pesi, e generano lo stesso testo.
L'unica differenza intenzionale e il worker SSD -> RAM pinned attivo in C.
Il passaggio da 0,73 a 0,87 token/s e quindi circa **+19,2%** in questa singola
esecuzione.

Run A non e una baseline di speedup, anche se qui raggiunge 0,73 token/s. Usa
il demand load standard `SSD -> RAM pinned -> VRAM`, ma ha un ruolo diverso:
costruisce l'oracolo invece di leggerlo e non gestisce il replay per ogni
`(token, layer)`. Inoltre non ha il worker SSD concorrente di Run C. In C gli
hit evitano la lettura SSD per i layer `0..4`, ma il worker puo competere per
l'I/O con i demand load dei layer rimanenti `5..42`.

Per **replay** si intende la riapplicazione della scelta registrata da Run A.
Per esempio:

~~~text
Run A, token 25, layer 7
    -> salva nell'oracolo gli expert [12, 41, 73, 101, 180, 229]
    -> salva anche i sei router_weights corrispondenti

Run B/C, token 25, layer 7
    -> calcolano comunque il router locale, soltanto per diagnostica
    -> leggono dall'oracolo la scelta di Run A
    -> eseguono [12, 41, 73, 101, 180, 229]
       con i router_weights registrati da Run A
~~~

Questa lettura e applicazione dell'oracolo avviene per ogni `(token, layer)`
in B e C; Run A fa invece soltanto la registrazione. E un piccolo costo
aggiuntivo necessario per rendere B e C perfettamente confrontabili.

In termini pratici, per ciascun layer B e C compiono questi passaggi in piu
rispetto a un normale percorso che userebbe direttamente il proprio router:

~~~text
router locale B/C calcola ID e pesi locali
    -> gli ID/pesi locali vengono conservati nella trace diagnostica
    -> lookup dell'entry (token, layer) in routing_oracle.bin
    -> copia degli ID e dei pesi di Run A nella selezione effettiva
    -> caricamento ed esecuzione degli expert di Run A
~~~

Questo lavoro puo contribuire al fatto che Run B/C risultino piu lenti di Run
A. Non e pero corretto affermare che sia *soprattutto* questa la causa: nel
nostro caso il demand load di un solo gruppo di expert richiede in media circa
29 ms, mentre il replay e una lookup e una piccola copia di sei ID e sei pesi.
Anche Run A scrive ora le trace JSONL, quindi il loro costo e presente nei tre
processi. Il confronto utile rimane dunque B contro C, perche entrambi pagano
lo stesso costo di router locale, replay e tracing; C differisce da B soltanto
per il prefetch pinned.

Il risultato e quindi un segnale incoraggiante, non una stima statistica
definitiva: serviranno piu run per separare il vantaggio del prefetch dalla
normale variabilita di I/O e GPU.

### `run_A.routing.jsonl`, `run_B.routing.jsonl` e `run_C.routing.jsonl`

Ogni riga descrive il routing di un layer durante il decode. In Run A gli
expert effettivi sono quelli realmente decisi nel percorso di riferimento. In
Run B e C sono quelli dell'oracolo, cosi il confronto resta controllato.

| Campo | Significato |
|---|---|
| `phase` | Fase dell'inferenza; qui e `decode`, non prefill. |
| `token_index` | Posizione del token nel contesto DS4. Nel test va da 20 a 82. |
| `token_id` | ID numerico del token che il modello sta elaborando. |
| `layer` | Layer MoE interessato, da 0 a 42. |
| `is_hash` | `true` nei layer con selezione ID tramite tabella token-ID; `false` per il normale top-k del router. |
| `experts` | I sei expert effettivamente eseguiti. In B/C sono quelli dell'oracolo di Run A. |
| `router_weights` | I sei pesi associati agli expert effettivi. Sono necessari per combinare correttamente le loro uscite. |
| `original_experts` | I sei expert che il router locale di B/C avrebbe scelto prima che il replay li sostituisse. Campo diagnostico. |
| `original_router_weights` | Pesi che il router locale avrebbe assegnato alla sua scelta originale. Campo diagnostico. |
| `top_candidates` | Otto migliori candidati del router nei layer top-k: per ciascuno salva ID e valori di score utili alla diagnosi. Nei layer `is_hash=true` e `null`, perche non esiste quel ranking. |
| `cutoff_margin` | Distanza numerica fra l'ultimo expert selezionato e il primo escluso dal top-k. Nei layer `is_hash=true` e `null`. |

| Controllo di correttezza | Risultato nell'output |
|---|---|
| Eventi di decode in A, B e C | 2.709 ciascuno = 63 token x 43 layer |
| Expert e pesi effettivi A/B/C | Identici in tutti i 2.709 eventi |
| Trace routing B/C | Identici byte per byte |
| Testo stdout B/C | Identico |

Questo e il controllo piu importante: B e C hanno eseguito esattamente gli
stessi expert, con gli stessi pesi. La differenza di tempo puo quindi essere
attribuita al percorso di caricamento, non a un percorso MoE diverso.

### `run_A.prefetch.jsonl`, `run_B.prefetch.jsonl` e `run_C.prefetch.jsonl`

Ogni riga rappresenta un evento di caricamento per un gruppo di sei expert
(un blocco `gate`, uno `up` e uno `down`). Nel modello del test un gruppo vale
`42.467.328` byte, circa **40,5 MiB**.

| Campo | Significato |
|---|---|
| `phase` | Fase dell'inferenza: qui sempre `decode`. |
| `event` | Tipo di evento: `prefetch_enqueue`, `prefetch_hit` o `prefetch_miss_demand`. |
| `token_index`, `layer` | Gruppo richiesto per quel token e layer. |
| `experts` | Sei ID del gruppo effettivamente richiesto/eseguito. |
| `bytes` | Numero di byte del gruppo. Vale `0` per la sola accodatura, che non trasferisce ancora dati. |
| `ssd_read_ms` | Tempo di lettura SSD osservato nell'evento. E `0` per un hit, perche i pesi sono gia in RAM pinned. |
| `transfer_from_RAM_to_VRAM_ms` | Tempo per portare i dati dalla RAM pinned alla VRAM e attendere che siano utilizzabili. |
| `total_load_event_ms` | Tempo totale dell'evento. In un demand load include `ssd_read_ms` piu `transfer_from_RAM_to_VRAM_ms`; in un prefetch hit coincide quasi con il solo `transfer_from_RAM_to_VRAM_ms`, perche la lettura SSD era gia stata completata dal worker. |

| Evento | Run A | Run B | Run C | Cosa indica |
|---|---:|---:|---:|---|
| `prefetch_enqueue` | 0 | 0 | 310 | Candidati futuri inviati al worker. |
| `prefetch_hit` | 0 | 0 | 310 | Gruppi gia pronti in RAM pinned e riusati senza SSD sul percorso critico. |
| `prefetch_miss_demand` | 2.709 | 2.709 | 2.399 | Demand load normale; in A/B e previsto per tutti i gruppi. |

I 310 hit di C sono distribuiti in modo regolare: **62 hit per ciascuno dei
layer 0, 1, 2, 3 e 4**. Il primo token di decode non puo avere un gruppo
precedente prefetched; per i 62 token successivi ogni layer della finestra
trova il proprio gruppo futuro pronto. I layer 5..42 restano volutamente sul
percorso normale in questa prima versione del test.

| Percorso misurato | Eventi | Lettura SSD media | Trasferimento RAM -> VRAM medio | Tempo totale medio |
|---|---:|---:|---:|---:|
| A, demand load di riferimento | 2.709 | 19,536 ms | 9,485 ms | 29,037 ms |
| B, demand load normale | 2.709 | 19,380 ms | 9,229 ms | 28,629 ms |
| C, miss con demand load | 2.399 | 20,896 ms | 6,465 ms | 27,381 ms |
| C, hit dal banco pinned | 310 | 0,000 ms | 5,606 ms | 5,606 ms |

Un hit evita in media circa **23,0 ms** del percorso critico rispetto al demand
load di B. In questa run anche i miss di C non sono peggiori della baseline,
ma non va interpretato come una garanzia: I/O SSD e GPU variano da run a run.
Un predictor futuro dovra quindi mantenere buona precisione: un hit vale molto,
ma prefetch errati o troppo aggressivi possono ancora competere con le letture
necessarie immediatamente.

### Conclusione del primo TEST_4

Il banco pinned e il worker funzionano nel caso controllato: i pesi futuri dei
layer `0..4` vengono letti in anticipo, rimangono nella RAM pinned e poi sono
copiati in VRAM senza una nuova lettura SSD. Il test conserva la correttezza
del modello, perche B e C usano lo stesso oracle completo e producono trace e
testo identici.

Il prossimo passo non e ancora estendere subito la finestra a tutti i 43
layer. Conviene prima ripetere B/C piu volte e osservare media e variabilita di
token/s, hit, miss e tempi SSD. Solo dopo potremo aumentare con cautela il
numero di layer prefetched o sostituire l'oracolo con un predictor reale.

### Esperimenti con finestra a dieci layer: `output_3` e `output_4`

Nei due esperimenti seguenti la configurazione contiene:

```text
DS4_TEST_4_PREFETCH_LAYERS=10
```

Quindi Run C prova a preparare in RAM pinned gli expert dei layer `0..9` del
token successivo. Per ogni coppia di token consecutivi ci sono dieci candidati
futuri: con 63 token di decode, l'ultimo non ha un token successivo e il numero
atteso di accodature e quindi `62 x 10 = 620`.

| Output | Data UTC | Layer prefetched | Accodature C | Hit C | Hit sugli accodati | Miss demand C |
|---|---|---:|---:|---:|---:|---:|
| `output_3` | 2026-08-22 10:02:10 | `0..9` | 620 | 567 | 91,5% | 2.142 |
| `output_4` | 2026-08-22 10:27:48 | `0..9` | 620 | 583 | 94,0% | 2.126 |

Le 620 accodature osservate in entrambi gli output sono corrette. Gli hit non
sono 620 perche il worker puo non terminare la lettura prima che il gruppo sia
richiesto: in quel caso il decode usa il normale demand load. In `output_4` il
worker ha completato 16 gruppi in piu rispetto a `output_3`, per questo i miss
sono 16 in meno.

| Output | Run B, generazione | Run C, generazione | Variazione B -> C | Esito di correttezza |
|---|---:|---:|---:|---|
| `output_3` | 0,73 token/s | 0,75 token/s | +2,7% | exit code A/B/C = 0; stdout e routing B/C identici |
| `output_4` | 0,72 token/s | 0,86 token/s | +19,4% | exit code A/B/C = 0; stdout e routing B/C identici |

I due risultati sono coerenti con un prefetch che funziona, ma mostrano anche
che una singola run non basta per stimare lo speedup. `output_4` e un risultato
molto favorevole; `output_3` mostra invece un miglioramento piccolo. In
entrambi i casi B e C seguono lo stesso percorso MoE forzato dall'oracolo, per
cui la differenza non deriva da token o expert diversi.

| Output | Percorso | Eventi | Lettura SSD media | Tempo totale medio |
|---|---|---:|---:|---:|
| `output_3` | B, demand load | 2.709 | 19,643 ms | 28,814 ms |
| `output_3` | C, miss demand | 2.142 | 27,921 ms | 34,967 ms |
| `output_3` | C, hit pinned | 567 | 0,000 ms | 5,857 ms |
| `output_4` | B, demand load | 2.709 | 19,787 ms | 29,036 ms |
| `output_4` | C, miss demand | 2.126 | 25,143 ms | 31,023 ms |
| `output_4` | C, hit pinned | 583 | 0,000 ms | 5,034 ms |

#### Perche il worker puo rallentare alcuni miss: competizione I/O

Run C ha due letture che usano lo stesso SSD e lo stesso controller:

```text
worker prefetch:  SSD -> RAM pinned per E_l(t+1), gruppo futuro
                         ||
demand load:      SSD -> RAM pinned -> VRAM per E_l(t), gruppo necessario ora
```

Il worker legge in anticipo un gruppo di circa 40,5 MiB. Nello stesso intervallo
DS4 puo dover leggere con urgenza un altro gruppo per il token e il layer
correnti. Le due richieste non hanno due SSD distinti: entrano nella stessa
coda del dispositivo e condividono banda, controller e latenza. Se una lettura
prefetch e gia in corso, la richiesta urgente non puo cancellarla
istantaneamente; deve quindi attendere o alternarsi con essa.

Questo effetto e visibile nei dati: i miss di C leggono mediamente da SSD
25,143--27,921 ms, contro circa 19,7 ms dei demand load di B. Con la finestra
di dieci layer il worker resta attivo piu a lungo e trasferisce piu dati
speculativi: aumenta gli hit potenziali, ma aumenta anche la probabilita di
ritardare un miss necessario immediatamente.

Il compromesso e quindi il seguente:

```text
finestra troppo piccola  -> pochi gruppi pronti, pochi hit
finestra troppo grande   -> piu hit, ma piu pressione e contesa sullo SSD
finestra ben calibrata   -> abbastanza hit senza rallentare il demand load
```

Il prossimo esperimento utile e ripetere ogni valore della finestra, ad
esempio `3`, `5`, `7` e `10`, piu volte. Per ciascun valore confronteremo la
mediana di token/s, la percentuale di hit e il tempo dei miss demand. Solo
questa serie permette di scegliere una finestra fissa ragionevole o di
motivare un worker con priorita/adattamento quando il demand load e sotto
pressione.
