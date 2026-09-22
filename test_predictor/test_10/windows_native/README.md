# TEST_10 Windows nativo — confronto diretto con WSL

## Perche' esiste

Il TEST_10 originale legge il GGUF esterno come processo Linux in WSL, dal
percorso `/mnt/d/...`. Questo test ripete **lo stesso piano di letture reali**,
ma viene eseguito dal Python di Windows e apre direttamente `D:\...`.

```text
TEST_10 WSL:       Python Linux -> /mnt/d -> DrvFS/WSL -> D:
TEST_10 Windows:   Python Windows -> ReadFile -> D:
```

Il confronto isola il costo introdotto dal confine WSL/DrvFS. Non misura DS4,
CUDA, predictor o trasferimento RAM pinned -> VRAM.

## Cosa resta identico a TEST_10 WSL

- lo stesso GGUF DeepSeek V4 Flash;
- la stessa trace `learned_prefetch_enqueue` di TEST_9;
- gli stessi layer e sei ID expert di ogni richiesta;
- 18 letture per richiesta: gate, up e down dei sei expert;
- 40,5 MiB per richiesta;
- 12 richieste misurate e 2 warm-up per worker;
- confronto con 1, 2, 3 e 4 worker concorrenti.

Il programma usa `CreateFileW` con `FILE_FLAG_NO_BUFFERING` e
`FILE_FLAG_OVERLAPPED`, buffer allocati con `VirtualAlloc` e `ReadFile` con
offset esplicito. Quindi evita la cache file di Windows, in modo analogo
all'uso di `O_DIRECT` nel TEST_10 WSL, e consente ai worker di avere letture in
volo contemporaneamente. La trace viene letta prima che inizi la misura: la sua
posizione non influenza throughput e latenza riportati.

Alcuni offset dei tensor GGUF non sono multipli di 4 KiB. Windows rifiuta una
lettura direct I/O non allineata; per ogni gate/up/down il programma legge
quindi il minimo intervallo **allineato** che lo contiene. Si tratta al massimo
di pochi KiB aggiuntivi per blocco e gli expert logici rimangono gli stessi.

## Avvio da PowerShell

Apri **PowerShell**, non il terminale WSL, e incolla una riga per volta:

```powershell
$env:DS4_TEST_10_WINDOWS_MODEL='D:\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$env:DS4_TEST_10_WINDOWS_TRACE='\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_9\output_10\repetition_01_B_then_C\run_C.prefetch.jsonl'
$env:DS4_TEST_10_WINDOWS_GROUPS_PER_WORKER='12'
$env:DS4_TEST_10_WINDOWS_WARMUP_GROUPS_PER_WORKER='2'
$env:DS4_TEST_10_WINDOWS_WORKER_COUNTS='1,2,3,4'
py '\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_10\windows_native\test_10_windows_native.py'
```

Il sorgente e la trace possono stare in `\\wsl.localhost\...`: vengono letti
prima del cronometro. Il file da benchmark e' invece `D:\...`, aperto dal
processo Windows con le API di Windows. Il report viene creato localmente in:

```text
C:\Users\miky9\Desktop\test_10_windows_native\output\io_benchmark_windows_summary.json
```

Puoi cambiare la cartella dei report con
`DS4_TEST_10_WINDOWS_OUTPUT_ROOT`, se necessario.

## Come interpretare il confronto

Confronta soprattutto il throughput aggregato dei **tre worker**, che era il
punto operativo migliore del TEST_10 WSL:

| Confronto | Cosa indica |
|---|---|
| Windows nativo circa uguale a WSL | Il limite e' soprattutto l'access pattern sparse/direct I/O e il dispositivo. |
| Windows nativo molto piu' veloce di WSL | DrvFS/WSL contribuisce in modo rilevante: spostare DS4 o il worker in Windows nativo potrebbe essere utile. |
| Entrambi molto sotto CrystalDiskMark | Il pattern di 18 letture sparse da 40,5 MiB e la concorrenza, non la sola banda sequenziale, sono il limite principale. |

CrystalDiskMark ha gia' mostrato `3.110,75 MB/s` in lettura sequenziale Q8T1.
Il confronto qui serve a capire quanta di quella differenza dipenda
specificamente da WSL.

## Risultati verificati — `output`

Configurazione misurata su Windows nativo:

```text
GGUF: D:\ds4\gguf\...0731.gguf
trace reale: TEST_9 output_10, prima ripetizione C
12 richieste misurate per worker
2 richieste di warm-up per worker, escluse dalla misura
una richiesta logica: 6 expert, 18 letture, 40,5 MiB
```

| Worker SSD | Dati logici letti | Tempo wall | Throughput aggregato misurato |
|---:|---:|---:|---:|
| 1 | 486 MiB | 0,293 s | **1.656,54 MiB/s** |
| 2 | 972 MiB | 0,519 s | **1.872,19 MiB/s** |
| 3 | 1.458 MiB | 0,781 s | **1.867,50 MiB/s** |
| 4 | 1.944 MiB | 0,954 s | **2.036,67 MiB/s** |

### Confronto diretto con TEST_10 WSL

| Worker | WSL (`/mnt/d`) | Windows nativo (`D:`) | Rapporto Windows / WSL |
|---:|---:|---:|---:|
| 1 | 223,00 MiB/s | 1.656,54 MiB/s | **7,43×** |
| 2 | 503,98 MiB/s | 1.872,19 MiB/s | **3,71×** |
| 3 | 711,77 MiB/s | 1.867,50 MiB/s | **2,62×** |
| 4 | 730,12 MiB/s | 2.036,67 MiB/s | **2,79×** |

Questo e' il confronto decisivo: SSD, enclosure, cavo e porta sono in grado
di fornire circa 1,8--1,99 GiB/s anche con il pattern reale di 18 letture sparse
per richiesta. Il medesimo pattern, quando e' eseguito da Linux su `/mnt/d`,
si ferma intorno a 0,7 GiB/s. Il limite dominante del prefetch attuale e'
quindi il percorso **WSL/DrvFS**, non l'SSD esterno da solo.

Il quarto worker aumenta leggermente il throughput Windows, ma il guadagno e'
modesto rispetto a due o tre worker. Questo conferma che aggiungere worker al
codice DS4 non elimina il limite WSL: per sfruttare la banda misurata qui
sarebbe necessario spostare il percorso di lettura del GGUF fuori da DrvFS,
oppure eseguire il worker/pre-fetch in un processo Windows nativo.
