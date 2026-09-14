# TEST_8 — Benchmark ripetuto del learned prefetch asincrono

## Obiettivo

TEST_8 misura con maggiore affidabilita' l'effettivo beneficio temporale del
prefetch learned introdotto e validato in TEST_7. Non modifica il modello di
rete neurale, il predictor CPU, il protocollo socket o il worker pinned-RAM:
riusa esattamente la pipeline asincrona di TEST_7.

Le misure TEST_8 gia' presenti usano il checkpoint Hyperband `0024`, con
validation Recall@6 `0,490852594`. Il nuovo migliore modello Build 5 ha
raggiunto validation Recall@6 **`0,502358287`** e Recall@6 test
**`0,506056192`** dopo il warm-start del trial 0024. Non e' ancora stato
inserito nel servizio runtime. I benchmark attuali non devono quindi essere
attribuiti al nuovo record: servirà una successiva integrazione e una nuova
serie di misure per valutarne l'impatto sul prefetch.

Il problema del confronto A/B/S/C singolo e' che le fasi eseguite piu' tardi
possono beneficiare della page cache Linux: porzioni del GGUF lette in una
fase precedente possono restare in RAM del sistema operativo. La VRAM e la
RAM pinned vengono invece ricreate a ogni processo DS4. Alternare B e C non
elimina completamente la page cache, ma evita che C sia sempre eseguita dopo
il baseline e rende il confronto meno sbilanciato.

## Le due fasi di TEST_8

All'inizio TEST_8 esegue una sola fase A per registrare `routing_oracle.bin`.
Ogni successiva fase usa quindi lo stesso routing reale, rendendo direttamente
confrontabili tempi e richieste di expert.

Poi esegue `N` coppie di fasi:

| Ripetizione | Ordine | Scopo |
|---:|---|---|
| 1 | B → C | Prima misura con baseline eseguito per primo. |
| 2 | C → B | Compensa il possibile vantaggio della cache della fase eseguita dopo. |
| 3 | B → C | Completa il primo benchmark standard da tre coppie. |

- **B — replay baseline:** DS4 riproduce l'oracolo senza predictor CPU e senza
  prefetch learned.
- **C — learned prefetch asincrono:** il predictor CPU Build 5 prepara una
  sestina per il token successivo in ciascuno dei layer MoE configurati da
  `DS4_TEST_8_PREFETCH_LAYERS`; il thread GPU non aspetta la risposta. Gli
  expert predetti vengono caricati nella RAM pinned e, alla domanda reale,
  DS4 riusa quelli corretti e legge da SSD solo gli altri.

## Output e affidabilita' dei risultati

Ogni avvio crea `output`, `output_2`, e cosi' via. All'interno della directory:

- `routing_oracle.bin`: routing di riferimento registrato una volta;
- `repetition_01_B_then_C`, `repetition_02_C_then_B`, ...: log, trace e
  predizioni delle due fasi di quella ripetizione;
- `benchmark_summary.json`: report progressivo e finale.

Il report viene riscritto in modo atomico dopo ogni coppia B/C completata. Se
si interrompe con `Ctrl+C`, le coppie completate restano analizzabili con
`status: "interrupted"`; al termine normale il valore diventa `"completed"`.

`statistics` contiene media, mediana, minimo, massimo e deviazione standard
campionaria di B, C, della differenza `C - B` e dello speedup percentuale
`(B - C) / B x 100`. Un valore positivo di `speedup_percent` indica che C e'
piu' veloce del baseline. `pairs_faster_with_prefetch` dice in quante coppie
la C corrispondente ha battuto la B.

## Primo risultato misurato: `output`

Il primo benchmark completo ha terminato tutte e tre le coppie. I risultati
accoppiati sono:

| Coppia | Ordine reale | B baseline | C learned prefetch | C - B | Speedup di C |
|---:|---|---:|---:|---:|---:|
| 1 | B → C | 101,60 s | 98,04 s | -3,56 s | +3,50% |
| 2 | C → B | 99,17 s | 109,26 s | +10,09 s | -10,17% |
| 3 | B → C | 96,91 s | 102,62 s | +5,72 s | -5,90% |
| Media | — | 99,23 s | 103,31 s | +4,08 s | **-4,19%** |

Solo una coppia su tre ha ottenuto un miglioramento (`pairs_faster_with_prefetch
= 1`), mentre due sono state piu' lente. La deviazione standard campionaria
della differenza C - B e' 6,97 s: molto grande rispetto alla media di 4,08 s.

Conclusione al momento: il +9,28% visto nella singola `output_11` di TEST_7
non e' confermato dal benchmark ripetuto. TEST_8 non dimostra per ora uno
speedup stabile del learned prefetch a un solo layer; la stima osservata e'
anzi una C mediamente piu' lenta di circa il 4,19%. Il riuso parziale degli
expert resta reale, ma il suo vantaggio non compensa ancora con affidabilita'
la latenza, la tempistica delle richieste e la variabilita' del sistema.

## Secondo benchmark: GGUF di prefetch su SSD esterno (`output_2`)

In `output_2` il modello principale e' rimasto nel percorso locale
`/home/michele/ds4/gguf/...`, mentre il worker di prefetch ha letto la sua
copia del GGUF dall'unita' esterna `/mnt/d/ds4/gguf/...`. Lo scopo era evitare
che letture speculative e letture a domanda competessero sullo stesso percorso
di archiviazione.

| Coppia | Ordine reale | B baseline | C learned prefetch | C - B | Speedup di C |
|---:|---|---:|---:|---:|---:|
| 1 | B → C | 101,25 s | 111,99 s | +10,74 s | -10,61% |
| 2 | C → B | 101,75 s | 108,29 s | +6,54 s | -6,43% |
| 3 | B → C | 109,78 s | 100,80 s | -8,98 s | +8,18% |
| Media | — | 104,26 s | 107,03 s | +2,77 s | **-2,95%** |

Le tre C hanno mantenuto la stessa Recall@6 online (`0,467742`) e hanno
riusato rispettivamente 102, 113 e 110 expert dalla RAM pinned. Quindi il
prefetch continua a essere eseguito e il modello continua a fare le stesse
predizioni. Tuttavia due coppie su tre restano piu' lente e la deviazione
standard della differenza C - B e' `10,39 s`.

Il passaggio all'SSD esterno riduce la stima negativa media da -4,19% a
-2,95%, ma con sole tre coppie e questa varianza non permette di attribuire
un miglioramento al cambio di disco. In particolare, attraverso WSL l'unita'
Windows e' accessibile mediante il mount `/mnt/d`, che puo' introdurre un
costo diverso rispetto alle letture sul filesystem Linux locale. Il risultato
corretto e' quindi: la separazione dei percorsi e' stata verificata, ma non
ha ancora prodotto uno speedup stabile misurabile a un solo layer.

## Configurazione CLion

Crea una **Native Application** con:

- **Target:** `test-predictor-8`
- **Executable:** `test_8_runner`
- **Working directory:** `/home/michele/ds4`

Nelle variabili d'ambiente, tutte sulla stessa riga e separate da `;`, usa:

```text
DS4_MODEL=/home/michele/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_8_PREFETCH_MODEL=/home/michele/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_8_PREFETCH_LAYERS=1;DS4_TEST_8_PREFETCH_INIT_DISTANCE=1;DS4_TEST_8_PREFETCH_STEP=1;DS4_TEST_8_REPETITIONS=3
```

`DS4_TEST_8_PREFETCH_LAYERS=1` significa che il predictor e' usato soltanto
per il primo layer MoE fisico, il layer 3. Per un primo benchmark non cambiare
questi valori: lo scopo di TEST_8 e' misurare in modo stabile la configurazione
gia' verificata in TEST_7.

Internamente TEST_8 passa questo stesso numero anche a
`DS4_TEST_7_PREFETCH_LAYERS` nel processo DS4. Il nome interno resta quello
di TEST_7 perche' il client predictor asincrono e' stato implementato li';
non occorre aggiungere manualmente una seconda variabile nella configurazione
CLion di TEST_8.

## Primo benchmark valido a quattro layer: `output_4`

`output_4` usa `DS4_TEST_8_PREFETCH_LAYERS=4`: il predictor asincrono lavora
per i layer MoE fisici 3–6 e il GGUF delle letture speculative si trova
sull'SSD esterno montato in `/mnt/d`. Ogni fase C ha prodotto 248 predizioni
valutabili e 252 enqueue learned, cioe' circa quattro richieste per ognuno dei
63 token/layer di confine della generazione. Questo conferma che la
configurazione a quattro layer e' stata applicata correttamente.

| Coppia | Ordine reale | B baseline | C learned prefetch | C - B | Speedup di C |
|---:|---|---:|---:|---:|---:|
| 1 | B → C | 102,86 s | 105,18 s | +2,32 s | -2,26% |
| 2 | C → B | 96,55 s | 94,44 s | -2,11 s | +2,18% |
| 3 | B → C | 103,11 s | 97,79 s | -5,32 s | +5,16% |
| Media | — | 100,84 s | 99,14 s | -1,70 s | **+1,69%** |

In ciascuna C, fra 144 e 154 gruppi hanno riusato almeno un expert e sono
stati riusati rispettivamente 410, 419 e 399 expert dalla RAM pinned. I
contatori `expert riusati + expert letti da SSD` sono sempre 1.512, cioe'
`252 x 6`: ogni domanda e' quindi contabilizzata in modo coerente. Non ci
sono ancora hit completi 6/6, ma il riuso parziale e' frequente.

Il benchmark e' incoraggiante: due coppie su tre sono piu' veloci con C e la
media e' positiva. Tuttavia la deviazione standard dello speedup e' 3,73%,
piu' grande della media di 1,69%; con sole tre coppie non possiamo ancora
concludere che lo speedup sia stabile. Il risultato corretto e': quattro layer
rendono il learned prefetch molto piu' utile rispetto a un layer, ma occorrono
altre ripetizioni per stimare con fiducia il miglioramento medio.

## Benchmark a otto layer: `output_5`

`output_5` estende la stessa configurazione all'SSD esterno ai layer fisici
3–10 (`DS4_TEST_8_PREFETCH_LAYERS=8`). In ogni C sono presenti 496 predizioni
valutabili e 504 enqueue: il predictor e il worker hanno quindi coperto tutti
gli otto layer previsti, senza restare al default di un layer.

| Coppia | Ordine reale | B baseline | C learned prefetch | C - B | Speedup di C |
|---:|---|---:|---:|---:|---:|
| 1 | B → C | 99,73 s | 101,22 s | +1,49 s | -1,49% |
| 2 | C → B | 94,70 s | 92,10 s | -2,60 s | +2,75% |
| 3 | B → C | 103,10 s | 94,80 s | -8,30 s | +8,05% |
| Media | — | — | — | -3,14 s | **+3,10%** |

Le tre C hanno registrato rispettivamente 278, 290 e 263 gruppi con riuso
parziale e 754, 781 e 699 expert riusati dalla RAM pinned. Il valore medio di
3,10% e' superiore al 1,69% misurato con quattro layer; aumentare i layer
aumenta quindi effettivamente le opportunita' di riuso e, in questo benchmark,
anche il guadagno medio osservato.

Restano due limiti: due coppie su tre sono positive ma una e' negativa, e la
deviazione standard dello speedup e' 4,78%. Otto layer e' per ora la
configurazione migliore fra quelle misurate, ma il numero di ripetizioni e'
ancora troppo basso per stimare un guadagno stabile con alta confidenza.

## Come vengono riusati oggi gli expert predetti

L'implementazione corrente non usa una cache globale del tipo `expert ID →
RAM pinned` o `expert ID → VRAM`. Gli slot pinned sono associati alla coppia
**token futuro + layer**. Questo e' necessario perche' l'expert con ID, per
esempio, `42` del layer 3 contiene pesi diversi dall'expert `42` del layer 4.

Quando il predictor produce sei expert per il token successivo, il worker
controlla soltanto se esiste gia' una richiesta identica per lo stesso token
futuro e lo stesso layer. Negli altri casi si comporta cosi':

| Situazione al momento dell'enqueue | Comportamento attuale |
|---|---|
| Nessuno dei sei expert e' presente in uno slot pinned rilevante | Il worker legge tutti e sei dal GGUF di prefetch e li inserisce nello slot RAM pinned del token/layer futuro. |
| Alcuni o tutti gli expert sono gia' nello slot di un altro token o di un altro layer | Non vengono cercati o riusati durante l'enqueue; il worker puo' leggere di nuovo gli stessi expert in un nuovo slot. |
| Esiste gia' la stessa richiesta per lo stesso token futuro e layer | La richiesta non viene duplicata: se lo slot e' gia' in caricamento o pronto, l'enqueue termina senza una seconda lettura. |
| Alcuni o tutti gli expert erano gia' nella cache VRAM di una richiesta precedente | Non vengono riusati: la cache VRAM corrente contiene soltanto il gruppo compatto piu' recente e viene invalidata/sovrascritta per la richiesta successiva. |

Il riuso viene quindi valutato soltanto quando il router reale deve eseguire
il layer del token futuro:

```text
predizione pronta in RAM pinned per (token t+1, layer l)
                            |
                            v
router reale del token t+1 richiede sei expert al layer l
```

- se coincidono tutti e sei, il trace registra `prefetch_hit`: i sei expert
  vengono copiati dalla RAM pinned alla VRAM;
- se coincidono solo alcuni, registra `prefetch_partial_reuse`: gli expert
  comuni vengono copiati dalla RAM pinned alla VRAM e i mancanti vengono letti
  dal modello principale e caricati in VRAM;
- se non coincide nessuno, DS4 esegue il normale caricamento a domanda dei sei
  expert dal modello principale.

Questo spiega perche' `full_prefetch_hits` puo' restare zero anche con un
modello che indovina alcuni expert: la configurazione corrente trae beneficio
solo dagli expert pronti per la stessa richiesta token/layer e non conserva
gli expert singoli per riusarli in richieste future diverse.

### Nota su `output_3` con quattro layer

La prima esecuzione `output_3` configurata con
`DS4_TEST_8_PREFETCH_LAYERS=4` non e' valida come benchmark dei quattro
layer. In quella versione il runner configurava il worker pinned-RAM per
quattro layer, ma non passava il valore al client predictor interno di DS4,
che quindi rimaneva al valore predefinito di un layer. Lo si riconosce da sole
62 predizioni/enqueue invece delle circa 248 attese. Il runner e' stato
corretto: le run successive con `PREFETCH_LAYERS=4` invieranno le quattro
richieste per token ai layer fisici 3–6. Conserviamo `output_3` come traccia
diagnostica, ma non lo usiamo per valutare lo speedup a quattro layer.

## Tempo atteso

Con tre ripetizioni TEST_8 esegue una fase A e sei processi DS4 B/C. Il tempo
atteso e' quindi circa sette volte il tempo di una singola fase, piu' il
caricamento del servizio predictor prima di ogni C. TEST_8 non esegue la fase
shadow S: TEST_7 ha gia' misurato qualita' e latenza del predictor; qui serve
concentrare il tempo sulle coppie direttamente confrontabili B/C.

## Predizioni corrette ma completate troppo tardi

Una predizione corretta non garantisce automaticamente che il relativo expert
sia disponibile quando il router GPU ne ha bisogno. Per trasformarsi in un
riuso dalla RAM pinned, devono completarsi in tempo tutte queste operazioni:

```text
GPU esegue token t, layer l
        |
        v
richiesta asincrona al predictor CPU
        |
        v
predittore produce i sei expert stimati per token t+1, layer l
        |
        v
worker legge gli expert dal GGUF di prefetch e li copia nella RAM pinned
        |
        v
GPU arriva al token t+1, layer l e richiede il gruppo reale di expert
```

Il riuso e' possibile solo se l'ultima richiesta arriva **dopo** che il
predictor e il worker hanno completato i passaggi precedenti. Se la GPU
raggiunge il token `t+1`, layer `l` prima, DS4 deve comunque caricare gli
expert mancanti a domanda dal modello principale: la predizione puo' essere
semanticamente corretta, ma e' arrivata troppo tardi per accelerare quella
specifica richiesta.

Questo e' uno dei limiti osservati nelle run a otto layer. Il predictor ha una
Recall@6 online di circa 0,43, cioe' indovina mediamente circa 2,56 expert su
6 per predizione; tuttavia non tutti questi expert si trasformano in riusi
effettivi dalla RAM pinned. La differenza dipende sia dagli expert errati sia
dal ritardo della catena predictor CPU -> coda -> worker -> lettura dal GGUF
di prefetch -> RAM pinned.

L'inferenza del predictor e' gia' asincrona: la GPU non attende la risposta
CPU prima di proseguire. Cio' evita di bloccare direttamente il forward pass,
ma non elimina il requisito temporale: il lavoro asincrono deve comunque
finire prima della domanda reale. Per valutare con precisione quanto pesa
questo effetto, un miglioramento futuro utile e' registrare separatamente:

- predizioni consegnate al worker prima della domanda del router;
- predizioni consegnate troppo tardi;
- expert corretti che erano pronti nella RAM pinned;
- expert corretti ma non ancora pronti al momento della domanda.
