# TEST_12 — Windows → CUDA pinned RAM in WSL

TEST_12 estende TEST_11: il servizio Windows legge gli expert dal GGUF esterno
e il client WSL li riceve direttamente in memoria allocata con `cudaMallocHost`.
Questo e' lo stesso tipo di memoria host pinned che DS4 usa nei propri slot di
prefetch. Non trasferisce ancora i dati in VRAM.

Avvia prima il servizio Windows di TEST_11, poi da WSL:

```bash
DS4_TEST_12_GROUPS_PER_WORKER=12 DS4_TEST_12_WARMUP_GROUPS_PER_WORKER=2 DS4_TEST_12_WORKER_COUNTS=1,2,3,4 \
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
python test_predictor/test_12/test_12_runner.py
```

Il report e' `test_predictor/test_12/output/windows_to_wsl_pinned_summary.json`.

## Risultati verificati — `output`

Configurazione: stesso GGUF e stessa trace reale di TEST_9 usati da TEST_11,
12 richieste misurate e 2 warm-up per worker. Ogni worker riceve ciascuna
richiesta da 40,5 MiB direttamente in un buffer ottenuto con `cudaMallocHost`.

| Flussi | Throughput end-to-end | Latenza media richiesta | p95 richiesta |
|---:|---:|---:|---:|
| 1 | **887,31 MiB/s** | 45,3 ms | 57,2 ms |
| 2 | **1.376,33 MiB/s** | 57,8 ms | 76,6 ms |
| 3 | **2.168,09 MiB/s** | 54,7 ms | 69,1 ms |
| 4 | **2.350,71 MiB/s** | 67,9 ms | 82,6 ms |

Con tre flussi, il percorso completo fino alla RAM pinned mantiene circa il
93% del throughput di TEST_11, che riceveva in RAM WSL normale
(`2.168` contro `2.337 MiB/s`). Il piccolo costo aggiuntivo della memoria
pinned e' quindi accettabile.

Il confronto piu' rilevante e' con TEST_10: tre worker che leggono il GGUF da
WSL tramite `/mnt/d` arrivavano a `711,77 MiB/s`; Windows -> WSL pinned RAM
arriva a `2.168,09 MiB/s`, cioe' **3,05 volte** tanto. Con quattro flussi il
rapporto e' `2.350,71 / 730,12 = 3,22×`.

La run a due flussi e' piu' bassa rispetto a TEST_11, mentre tre e quattro
restano vicini al risultato precedente. Le letture direct I/O, SSD e scheduler
possono variare da una run all'altra; la conclusione robusta e' che il percorso
Windows -> socket -> RAM pinned WSL conserva un vantaggio di circa tre volte
rispetto alla lettura DrvFS attuale.

## Decisione tecnica

L'esperimento completa la verifica dei due confini piu' rischiosi:

```text
GGUF Windows -> socket Windows/WSL -> buffer CUDA pinned WSL
```

Entrambi sono abbastanza veloci. Il prossimo lavoro non e' un altro benchmark
di banda, ma una prima integrazione ridotta nel prefetch DS4: una richiesta
asincrona per volta, ricevuta nel corretto slot pinned e poi caricata in VRAM.
