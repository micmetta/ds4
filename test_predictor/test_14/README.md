# TEST_14 — bridge Windows di TEST_13 con predictor Build 7

TEST_14 conserva il meccanismo che ha prodotto `TEST_13/output_9`: stesso
GGUF, stesso bridge Windows che legge gate/up/down direttamente nel buffer CUDA
pinned, due slot futuri per layer, sei worker DS4 e tre coppie B/C in ordine
alternato. L'unica sostituzione funzionale e' il predittore: Build 5 viene
rimpiazzato dal migliore checkpoint Top-4 di Build 7.

## Predictor usato

```text
checkpoint: build_dataset_7/ml/neural_network/
top4_patience10_l2_3e6_bce_v1/models/ffn.keras

state fixed-NNZ corrente: 2048
raw hidden l:              4096 -> Dense(64) -> ReLU
raw hidden l-1:            4096 -> Dense(64) -> ReLU
q transizione:             256D, residuale gated
fusione:                   2048 -> ReLU -> Dropout(0.3)
```

Il checkpoint e' valido per i layer fisici **5--42** (38 predizioni). I layer
3 e 4 restano sul mapping deterministico e **non** producono una richiesta
del prefetch learned di TEST_14: gli expert necessari al loro forward pass sono
quindi risolti dal normale percorso on-demand di DS4 (cache e, se necessario,
lettura SSD -> RAM -> VRAM). DS4 ne conserva comunque gli hidden necessari per
far usare a layer 5 sia `hidden(l)` sia `hidden(l-1)`.

L'orchestratore ereditato da TEST_9 mantiene internamente la finestra fisica
3--42, cosi' il worker puo' prefetchare anche l'ultimo layer. Le richieste al
predictor Build 7 partono pero' esclusivamente da 5 a 42.

## 1. Compilare DS4 in WSL

Il protocollo TEST_14 e' nel binario `ds4`; questa compilazione e' obbligatoria
prima della prima esecuzione o dopo ogni modifica a `ds4.c`.

```bash
cd /home/michele/ds4
make test_14_runner
```

## 2. Avviare il reader Windows

In PowerShell, in una finestra separata:

```powershell
$env:DS4_TEST_14_WINDOWS_MODEL='D:\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$env:DS4_TEST_14_WINDOWS_PORT='38114'
py '\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_14\windows_native\test_14_windows_reader_service.py'
```

Il servizio usa intenzionalmente lo stesso formato TCP di TEST_13, ma una porta
diversa (38114) per permettere di tenere separati i due esperimenti.

## 3. Avviare TEST_14 in WSL

In un terminale WSL distinto dal reader Windows:

```bash
cd /home/michele/ds4
source test_predictor/prefetch_predictor_pipeline/ML_pipeline/.venv/bin/activate

DS4_MODEL=/home/michele/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
DS4_TEST_14_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
DS4_TEST_14_CHECKPOINT=/home/michele/ds4/test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/neural_network/top4_patience10_l2_3e6_bce_v1/models/ffn.keras \
DS4_TEST_14_WINDOWS_HOST=172.23.208.1 \
DS4_TEST_14_WINDOWS_PORT=38114 \
DS4_TEST_14_PREFETCH_SLOTS_PER_LAYER=2 \
DS4_TEST_14_PREFETCH_STEP=1 \
DS4_TEST_14_PREFETCH_NUM_WORKERS=6 \
DS4_TEST_14_REPETITIONS=3 \
python test_predictor/test_14/test_14_runner.py
```

L'output e' salvato in `test_predictor/test_14/output*`. La fase A e la fase B
restano le baseline standard; soltanto C attiva `DS4_TEST_14_ENABLED=1`, il
protocollo runtime Build 7 e il bridge Windows.

Per un primo smoke test, sostituire temporaneamente `REPETITIONS=3` con `1` e
`PREFETCH_NUM_WORKERS=6` con `1`; mantenere tuttavia `SLOTS_PER_LAYER=2`, che
e' la configurazione di riferimento di `output_9`.

## Risultati verificati — `output_2` (40 layer, sei worker, Build 7)

La directory [`output_2`](output_2) e' il confronto diretto con
[`TEST_13/output_9`](../test_13/output_9): mantiene il medesimo modello GGUF,
il bridge Windows, 40 layer fisici nella finestra di prefetch (3--42), due slot
per layer, sei worker con politica `layer % 6` e tre coppie B/C alternate.
La sola differenza funzionale e' il predictor: Build 5 in TEST_13 viene
sostituito dal checkpoint Top-4 di Build 7, le cui predizioni learned iniziano
dal layer fisico 5.

### Recall@6 online sulla frase di benchmark

Le tre ripetizioni usano la stessa frase/traccia di generazione registrata
dalla fase A:

> Explain what a binary search is in one short paragraph.

La metrica seguente e' la Recall@6 misurata online sulle predizioni realmente
emesse nella fase C; non va confusa con la Recall@6 offline calcolata sui
dataset train, validation e test.

| Predictor | Predizioni con target nella frase | Expert target correttamente inclusi | Recall@6 online |
|---|---:|---:|---:|
| TEST_13 `output_9` — Build 5 | 2.480 | 6.953 / 14.880 | 0,467272 |
| TEST_14 `output_2` — Build 7 | 2.356 | 7.303 / 14.136 | **0,516624** |

Il valore e' identico nelle tre ripetizioni di ciascun test, poiche' la
traccia e le predizioni del modello sono deterministiche. Build 7 migliora la
Recall@6 online di **0,049353** (**+4,94 punti percentuali**). Le quantita'
assolute di predizioni differiscono perche' Build 7 non produce predizioni
learned per i layer di bootstrap 3 e 4.

| Ripetizione | B baseline | C bridge Windows + Build 7 | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---:|---:|---:|---:|---:|---:|
| 1 | 102,646 s | 71,977 s | +29,88% | 69,77 ms | 88,55 ms |
| 2 | 105,389 s | 75,070 s | +28,77% | 9,22 ms | 84,43 ms |
| 3 | 101,474 s | 70,717 s | +30,31% | 66,16 ms | 82,36 ms |
| **Media** | **103,170 s** | **72,588 s** | **+29,65%** | **48,38 ms** | **85,11 ms** |

Tutte e tre le coppie sono positive. Il tempo risparmiato medio e' **30,582 s**
per generazione; la deviazione standard dello speedup e' contenuta a **0,79
punti percentuali**, con un intervallo tra +28,77% e +30,31%.

| Misura media per fase C | TEST_13 `output_9` — Build 5 | TEST_14 `output_2` — Build 7 |
|---|---:|---:|
| Finestra fisica dell'orchestratore / worker | 3--42 (40) | 3--42 (40) |
| Layer con predizione learned | 3--42 (40) | 5--42 (38) |
| Slot per layer | 2 | 2 |
| Worker | 6 | 6 |
| Richieste learned accodate | 2.520 | 2.394 |
| Letture `READY` | ~2.498 | ~2.373 |
| Gruppi con riuso parziale | ~2.359 | ~2.242 |
| Riuso parziale / richieste learned | ~93,61% | ~93,66% |
| Expert riusati dalla RAM pinned | ~6.810 | **~7.170** |
| Expert riusati / richiesta learned | ~2,70 | **~3,00** |
| Attesa media prima della lettura | **~42,95 ms** | ~48,38 ms |
| Lettura / trasferimento medio | **~79,23 ms** | ~85,11 ms |
| B baseline medio | 104,300 s | **103,170 s** |
| C learned-prefetch medio | 75,959 s | **72,588 s** |
| Tempo risparmiato medio | 28,342 s | **30,582 s** |
| Speedup medio | +27,14% | **+29,65%** |

La riga `Layer con predizione learned` differisce per ragioni architetturali,
non per una diversa copertura del benchmark. Il predictor di Build 5 usa il
solo hidden raw corrente e puo' iniziare dal layer fisico 3, subito dopo i
layer 0--2 assegnati dal mapping deterministico. Build 7 usa invece
congiuntamente `hidden(l)` e `hidden(l-1)`: i layer 3 e 4 non ricevono quindi
una richiesta learned, ma sono eseguiti tramite il percorso on-demand e i loro
hidden vengono calcolati e conservati. Dal layer 5 in poi la coppia di
contesto causale e' disponibile e il predictor Top-4 puo' emettere le sue
predizioni learned.

In entrambi i test l'orchestratore mantiene la stessa finestra fisica 3--42.
In TEST_14 questo non implica una richiesta proattiva per i layer 3 e 4: la
distinzione indica quali layer ricevono una previsione dalla rete e, quindi,
una richiesta al prefetch asincrono.

I valori assoluti inferiori di `READY` e di `Gruppi con riuso parziale` in
TEST_14 non indicano un peggioramento della pipeline: Build 7 accoda **2.394**
richieste learned invece delle **2.520** di Build 5, precisamente perche' non
prefetcha i due layer di bootstrap 3 e 4. Normalizzando per il numero di
richieste, la quota di riuso parziale e' sostanzialmente identica (~93,66%
contro ~93,61%); Build 7 riusa invece piu' expert corretti per richiesta
(~3,00 contro ~2,70). E' questa maggiore densita' di expert utili nella RAM
pinned che compensa le due richieste mancanti per token e contribuisce al nuovo
record di speedup.

### Come leggere il confronto per gruppi di metriche

#### 1. Qualita' predittiva e riuso dalla RAM pinned

Qui TEST_14 supera TEST_13. La Recall@6 online sale da 0,467272 a 0,516624 e,
di conseguenza, gli expert proposti dal predictor hanno una maggiore
probabilita' di essere proprio quelli richiesti durante il forward.

| Indicatore | TEST_14 rispetto a TEST_13 | Significato |
|---|---:|---|
| Expert riusati dalla RAM pinned | +~360 per fase C | Build 7 trova piu' spesso expert utili gia' pronti nel buffer pinned. |
| Expert riusati / richiesta learned | 3,00 vs 2,70 | Ogni richiesta Build 7 porta in media ~0,30 expert utile in piu'. |
| Riuso parziale / richieste learned | 93,66% vs 93,61% | La capacita' del meccanismo asincrono di rendere disponibile almeno una parte del gruppo e' sostanzialmente invariata. |

I `READY` e i gruppi a riuso parziale sono inferiori solo in valore assoluto
perche' mancano 126 richieste learned relative ai due layer di bootstrap; non
sono una misura della precisione del modello. Il segnale che distingue i due
predictor e' la maggiore densita' di expert corretti riusati per richiesta.

#### 2. Consegna I/O: coda, lettura e trasferimento

Qui TEST_14 non supera TEST_13: l'attesa media prima della lettura cresce da
42,95 a 48,38 ms (+5,43 ms), mentre lettura e trasferimento crescono da 79,23
a 85,11 ms (+5,88 ms). Queste due misure appartengono al bridge Windows,
all'SSD, alla rete locale e allo scheduling dei worker; non misurano la qualita'
del predictor.

La configurazione del bridge e' la stessa, ma i due benchmark sono esecuzioni
separate: cache del sistema operativo, posizione fisica degli expert letti e
interleaving temporale dei worker possono cambiare le latenze. Non e' quindi
corretto attribuire il lieve peggioramento I/O a Build 7 senza ripetere i due
predictor nello stesso blocco sperimentale. Il punto importante e' che Build 7
ottiene il record **nonostante** queste latenze I/O leggermente peggiori.

#### 3. Risultato end-to-end della generazione

| Fase | TEST_13 `output_9` — Build 5 | TEST_14 `output_2` — Build 7 | C Build 7 vs C Build 5 |
|---|---:|---:|---:|
| A — oracolo di routing, una sola esecuzione | 100,838 s | 104,573 s | +3,735 s |
| B — baseline replay, media di 3 | 104,300 s | 103,170 s | **-1,130 s** |
| C — learned prefetch, media di 3 | 75,959 s | **72,588 s** | **-3,371 s** |

La fase A serve soltanto a registrare una volta l'oracolo di routing condiviso
da B e C nello stesso test. Non e' ripetuta e non e' appaiata a C: per questo
non viene usata per calcolare lo speedup. La sua differenza fra TEST_13 e
TEST_14 riflette il normale rumore fra generazioni separate. La tabella mostra
comunque il tempo totale dell'intero protocollo: Build 7 riduce il tempo medio
della fase C a **72,588 s**, cioe' **3,371 s** sotto la fase C del precedente
record Build 5.

Il confronto decisivo e' la coppia B/C all'interno di ogni ripetizione: B non
attiva alcun predictor, mentre C attiva il prefetch. Il baseline medio di
TEST_14 (103,170 s) e' 1,130 s piu' basso di quello di TEST_13, ma questa non
e' una proprieta' di Build 7: e' normale variabilita' tra benchmark separati.

Il dato robusto e' il vantaggio appaiato C rispetto a B: TEST_14 risparmia in
media **30,582 s**, contro 28,342 s di TEST_13 (+2,240 s), e porta lo speedup
dal **+27,14%** al **+29,65%**. In sintesi, il guadagno predittivo di Build 7
e' abbastanza grande da compensare sia i due layer non prefetched sia una
consegna I/O leggermente meno favorevole.

Il nuovo checkpoint raggiunge quindi un guadagno di **2,52 punti percentuali**
sul precedente record verificato. Il miglioramento non dipende da una maggiore
concorrenza I/O: numero di worker, slot, finestra di layer, bridge Windows e
sequenza delle tre coppie B/C sono invariati. TEST_14 riusa in media circa 360
expert corretti in piu' dalla RAM pinned per fase C, pur con latenze I/O
leggermente maggiori; il vantaggio netto e' quindi coerente con una qualita'
predittiva superiore del modello Build 7 nel percorso online.

Il report riproducibile della run e'
[`output_2/benchmark_summary.json`](output_2/benchmark_summary.json). Il record
precedente e' documentato in
[`TEST_13/output_9/benchmark_summary.json`](../test_13/output_9/benchmark_summary.json).
