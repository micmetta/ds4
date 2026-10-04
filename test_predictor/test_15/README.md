# TEST_15 — misura dell'overlap fra slot nella RAM pinned

TEST_15 non introduce ancora una cache per expert e non compone due slot. E'
un profiling separato della configurazione record di TEST_14:

```text
checkpoint Build 7 Top-4, prefetch Top-6
layer fisici nella finestra worker: 3--42
layer con predizione learned:       5--42
slot futuri per layer:              2
worker I/O:                         6 (layer % 6)
```

La fase A viene eseguita una sola volta solo per produrre l'oracolo di routing
necessario al replay; le tre misure successive sono **solo fasi C**. TEST_15
non confronta uno speedup con B: registra il potenziale riuso fra il Top-6 di
una nuova richiesta e gli expert gia' presenti nell'altro slot dello stesso
layer.

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

## Risultati ottenuti — `output`

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
contrario, evidenzia il suo vantaggio concettuale rispetto alla composizione
slot A → slot B: una cache condivisa può conservare e referenziare gli stessi
buffer anche mentre la richiesta precedente è in uso, purché `lease_count`,
`future_refcount`, quota `2 × Top-K` e sincronizzazione per shard siano
implementati correttamente. TEST_15 misura il potenziale iniziale; non prova
ancora uno speedup end-to-end della soluzione B.

Le tre ripetizioni hanno prodotto rispettivamente 138 expert `READY` in
overlap e circa 0,977 GB di byte immediatamente riusabili ciascuna: il risultato
aggregato è quindi coerente tra le esecuzioni. Il report completo e riproducibile
è [`output/overlap_measurement_report.json`](output/overlap_measurement_report.json).

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

Il runner crea `output*` senza sovrascrivere nulla. Aggiorna
atomicamente `overlap_measurement_report.json` dopo A e dopo ogni fase C:
un'interruzione conserva percio' l'oracolo e tutte le ripetizioni gia'
completate. Le trace includono gli expert degli slot fratelli soltanto con
questa versione di `ds4`; ricompilare con `make test_15_runner` prima della
prima esecuzione.
