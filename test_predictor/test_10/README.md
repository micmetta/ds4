# TEST_10 — Benchmark isolato delle letture del GGUF esterno da WSL

## Obiettivo

TEST_9 misura il prefetch nel sistema completo: predictor CPU, thread di
decoding DS4, GPU, worker SSD, RAM pinned e VRAM. I suoi tempi sono quindi
utili per misurare lo speedup, ma non possono dire con precisione quanta banda
del disco esterno sia disponibile alla sola lettura dei pesi.

TEST_10 rimuove intenzionalmente tutte le altre componenti. Non avvia DS4,
CUDA, TensorFlow né il predictor. Legge solo il GGUF esterno dal processo
Python Linux in WSL tramite `O_DIRECT`.

Il confronto permette di separare:

```text
prestazioni SSD + enclosure + porta + cavo + Windows + WSL/DrvFS
                da
overhead aggiuntivo di DS4, scheduling, predictor e GPU.
```

## Forma di una richiesta letta

Nel GGUF DeepSeek V4 Flash usato dai test, il worker DS4 precarica sei expert.
Per ogni expert legge tre blocchi: gate, up e down. TEST_10 apre l'header
GGUF e ricava automaticamente offset e dimensione reali dei tensor. Con il
GGUF corrente, i valori sono:

| Blocco per expert | Byte | MiB |
|---|---:|---:|
| gate | 2.162.688 | 2,0625 |
| up | 2.162.688 | 2,0625 |
| down | 2.752.512 | 2,6250 |
| **un expert** | **7.077.888** | **6,75** |
| **sei expert / una richiesta DS4** | **42.467.328** | **40,5** |

Ogni richiesta TEST_10 effettua quindi **18 letture direct I/O**: gate, up e
down per ciascuno dei sei expert. Le dimensioni sono quelle osservate nelle
trace del prefetch DS4.

TEST_10 può leggere una trace storica di TEST_9 e riusare esattamente gli stessi
layer e sei expert che il predictor aveva richiesto. Riproduce quindi numero,
offset, dimensione, allineamento e ordine delle letture del worker DS4. Se non
si passa la trace, sceglie layer ed expert in modo deterministico: il benchmark
rimane utile per la banda, ma è meno aderente al routing reale.

## Cosa misura

Per ogni numero di worker configurato, TEST_10 salva in
`io_benchmark_summary.json`:

- byte totali letti;
- tempo wall dell'intera configurazione;
- throughput aggregato **misurato** in MiB/s;
- tempo di una richiesta da 40,5 MiB;
- tempo di una singola lettura gate/up/down;
- risultati separati per worker.

Il throughput qui è più affidabile della stima usata in precedenza nei TEST_9:

```text
throughput aggregato = byte effettivamente letti / tempo wall misurato
```

I worker partono con una barriera comune e leggono richieste distinte. Le
letture sono `O_DIRECT`, dunque TEST_10 non misura la cache delle pagine Linux.
Il controller e il firmware dell'SSD possono comunque avere le proprie cache.

## Configurazione CLion

Crea una configurazione **Native Application** simile a TEST_9:

| Campo | Valore |
|---|---|
| Name | `TEST_10` |
| Target | `test-predictor-10` |
| Executable | `test_10_runner` |
| Working directory | `/home/michele/ds4/` |

Per il primo benchmark usa queste variabili d'ambiente, tutte su una sola riga
e separate da `;`:

```text
DS4_TEST_10_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf;DS4_TEST_10_TRACE=/home/michele/ds4/test_predictor/test_9/output_10/repetition_01_B_then_C/run_C.prefetch.jsonl;DS4_TEST_10_GROUPS_PER_WORKER=12;DS4_TEST_10_WARMUP_GROUPS_PER_WORKER=2;DS4_TEST_10_WORKER_COUNTS=1,2,3,4
```

`DS4_TEST_10_MODEL` deve essere il GGUF sull'SSD esterno montato in WSL.
`DS4_TEST_10_TRACE` è consigliata: è la trace della migliore configurazione
TEST_9 (`output_10`, 16 layer e tre worker) e fornisce richieste realistiche.
I due gruppi di warm-up per worker non entrano in tempi né throughput finali.
Prima della run verificare:

```bash
mountpoint /mnt/d
ls -lh /mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Se `/mnt/d` non è montato:

```bash
sudo mount -t drvfs D: /mnt/d
```

## Interpretazione prevista

Confrontiamo il throughput misurato in TEST_10 con:

- CrystalDiskMark Windows: 3.110,75 MB/s sequenziale Q8T1 sul disco `D:`;
- le letture del worker DS4 in TEST_9;
- lo speedup end-to-end di TEST_9.

Se TEST_10 resta molto sotto CrystalDiskMark, il percorso WSL/DrvFS e le
letture sparse/direct I/O contribuiscono in modo importante al limite. Se
TEST_10 è molto più veloce di TEST_9, il limite aggiuntivo è nel sistema DS4:
scheduling dei worker, lead time insufficiente, richieste stale e trasferimenti
RAM pinned → VRAM.

I file `output*` sono risultati locali e non vengono versionati.

## Risultati verificati — `output_4`

Configurazione eseguita:

```text
GGUF sorgente: /mnt/d/ds4/gguf/...0731.gguf
trace reale: TEST_9 output_10, prima ripetizione C
12 richieste misurate per worker
2 richieste di warm-up per worker, escluse dalla misura
una richiesta = 6 expert = 18 letture O_DIRECT = 40,5 MiB
```

| Worker SSD | Dati letti | Tempo wall | Throughput aggregato misurato | Tempo medio di una richiesta per worker |
|---:|---:|---:|---:|---:|
| 1 | 486 MiB | 2,179 s | **223,00 MiB/s** | 181,6 ms |
| 2 | 972 MiB | 1,929 s | **503,98 MiB/s** | 138,4 ms |
| 3 | 1.458 MiB | 2,048 s | **711,77 MiB/s** | 160,4 ms |
| 4 | 1.944 MiB | 2,663 s | **730,12 MiB/s** | 213,4 ms |

Il quarto worker aggiunge soltanto `+2,58%` di throughput rispetto a tre
worker (`730,12` contro `711,77 MiB/s`), ma aumenta il tempo medio necessario
per completare una richiesta da 40,5 MiB da `160,4` a `213,4 ms` (`+33%`).
Per il prefetch, dove conta che una richiesta termini prima del layer GPU,
**tre worker sono quindi il punto operativo migliore**.

### Cosa dimostra questo esperimento

CrystalDiskMark in Windows ha misurato `3.110,75 MB/s` nella lettura
sequenziale Q8T1. TEST_10 misura invece circa `711,77 MiB/s` con tre worker,
cioè circa `746 MB/s` nella stessa unità decimale: approssimativamente il 24%
del picco sequenziale Windows.

Questo non dimostra che l'SSD esterno sia lento o difettoso: CrystalDiskMark
conferma che SSD, enclosure, cavo e porta possono superare 3 GB/s. Dimostra
però che il carico reale del progetto — 18 letture sparse `O_DIRECT` per
richiesta, da un processo Linux che accede a un volume Windows tramite
`/mnt/d` — si ferma intorno a 0,7 GiB/s e non scala utilmente oltre tre
worker.

Il risultato è coerente con TEST_9: la configurazione migliore end-to-end era
proprio 16 layer e tre worker (`output_10`, speedup medio `+16,63%`). Il limite
non è soltanto lo scheduling DS4, perché TEST_10 raggiunge una saturazione
simile senza predictor, CUDA, RAM pinned o GPU. Rimangono nel percorso misurato
WSL/DrvFS, `O_DIRECT`, l'access pattern sparse dei tensor e l'SSD esterno.

Il passo successivo, prima di acquistare un altro SSD, sarebbe ripetere questo
identico pattern di 18 letture da Windows nativo. Solo quel confronto può
separare quantitativamente il limite hardware dall'overhead WSL/DrvFS.

## Variante Windows nativa

La variante [windows_native/test_10_windows_native.py](windows_native/test_10_windows_native.py)
e' pronta per eseguire lo stesso test da PowerShell con il Python Windows.
Usa il percorso `D:\\...` e le API `CreateFileW` + `FILE_FLAG_NO_BUFFERING` +
`ReadFile`, mentre legge la trace TEST_9 prima di avviare il cronometro. Le
istruzioni di avvio e i criteri di confronto sono in
[windows_native/README.md](windows_native/README.md).

### Esito del confronto Windows nativo

Il confronto e' stato eseguito e ha confermato l'ipotesi: con tre worker, il
pattern reale ha misurato `711,77 MiB/s` da WSL (`/mnt/d`) contro
`1.867,50 MiB/s` da Windows nativo (`D:`), cioe' **2,62 volte** di throughput
in piu'. Con quattro worker il confronto e' `730,12` contro `2.036,67 MiB/s`.

L'SSD esterno non e' dunque il primo limite da sostituire: CrystalDiskMark e il
benchmark Windows nativo mostrano che il dispositivo funziona bene. Il limite
prioritario da affrontare e' l'accesso del processo Linux WSL al filesystem
Windows tramite DrvFS.
