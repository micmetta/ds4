# TEST_11 — Micro-benchmark Windows → WSL via localhost

TEST_11 misura il costo che rimane se le letture del GGUF vengono spostate in
un processo Windows nativo, mentre DS4 resta in WSL. Il servizio Windows legge
gli stessi expert reali del TEST_9 da `D:\\...` con direct I/O e li invia via
TCP sulla rete locale privata Windows↔WSL; il client WSL riceve i 40,5 MiB di
ogni richiesta.

Non misura ancora CUDA o RAM pinned. Serve a decidere se una futura
integrazione Windows-reader → DS4/WSL ha abbastanza banda da valere il lavoro.

## Avvio

In PowerShell avvia il servizio e lascialo aperto:

```powershell
$env:DS4_TEST_11_WINDOWS_MODEL='D:\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$env:DS4_TEST_11_WINDOWS_TRACE='\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_9\output_10\repetition_01_B_then_C\run_C.prefetch.jsonl'
py '\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_11\windows_native\test_11_windows_reader_service.py'
```

In un terminale WSL separato esegui:

```bash
DS4_TEST_11_GROUPS_PER_WORKER=12 DS4_TEST_11_WARMUP_GROUPS_PER_WORKER=2 DS4_TEST_11_WORKER_COUNTS=1,2,3,4 \
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
python test_predictor/test_11/test_11_runner.py
```

Alla fine il client manda automaticamente il comando di stop al servizio
Windows. Il report WSL e' `test_predictor/test_11/output/windows_to_wsl_summary.json`.

Il client trova automaticamente l'indirizzo gateway Windows della VM WSL2.
Questo e' necessario su questa macchina, dove `127.0.0.1` da WSL non viene
inoltrato al servizio Windows. Se necessario puo' essere forzato con
`DS4_TEST_11_HOST`, ma normalmente non va impostato.

## Risultati verificati — `output_4`

Configurazione misurata:

```text
GGUF letto da Windows: D:\ds4\gguf\...0731.gguf
trace reale: TEST_9 output_10, prima ripetizione C
12 richieste misurate per worker; 2 warm-up esclusi
una richiesta: 6 expert, 18 letture, 40,5 MiB
```

| Flussi Windows → WSL | Throughput end-to-end | Lettura Windows media / richiesta | Invio Windows medio / richiesta |
|---:|---:|---:|---:|
| 1 | **978,70 MiB/s** | 23,7 ms | 37,7 ms |
| 2 | **1.893,48 MiB/s** | 24,6 ms | 39,2 ms |
| 3 | **2.337,05 MiB/s** | 31,2 ms | 48,0 ms |
| 4 | **2.552,94 MiB/s** | 41,3 ms | 62,0 ms |

Il throughput e' end-to-end: parte dall'invio della richiesta dal client WSL e
finisce quando il client ha ricevuto tutti i 40,5 MiB. Include quindi lettura
direct I/O Windows, invio TCP, attraversamento Windows↔WSL e copia nella RAM
normale WSL.

### Confronto con la lettura attuale da WSL

| Worker/flussi | TEST_10: WSL legge `/mnt/d` | TEST_11: Windows legge e invia a WSL | Rapporto TEST_11 / TEST_10 |
|---:|---:|---:|---:|
| 1 | 223,00 MiB/s | 978,70 MiB/s | **4,39×** |
| 2 | 503,98 MiB/s | 1.893,48 MiB/s | **3,76×** |
| 3 | 711,77 MiB/s | 2.337,05 MiB/s | **3,28×** |
| 4 | 730,12 MiB/s | 2.552,94 MiB/s | **3,50×** |

Questo elimina il dubbio principale: la comunicazione Windows→WSL non annulla
il vantaggio della lettura Windows nativa. Per tre worker, configurazione che
aveva dato il miglior TEST_9, rimangono circa `2.337 MiB/s` end-to-end invece
di `712 MiB/s` attraverso DrvFS.

I valori possono variare tra run per cache del controller SSD, temperatura e
attivita' di Windows. La conclusione robusta non e' il singolo picco di 2.553
MiB/s, ma il margine di circa tre-quattro volte rispetto al percorso WSL
attuale.

## Cosa manca prima dell'integrazione in DS4

TEST_11 non scrive ancora nei buffer pinned allocati da CUDA in WSL. Una futura
integrazione dovra' aggiungere la copia `socket -> buffer pinned` e collegare
richieste, stati TOO_LATE e telemetria al worker di prefetch. Tuttavia il test
dimostra che il nuovo confine IPC e' abbastanza veloce da rendere quel lavoro
tecnicamente sensato.
