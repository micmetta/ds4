# TEST_1 — oracle replay del routing MoE

TEST_1 esegue due processi DS4 separati. Run A usa il router normale e salva,
per ogni `(token_index, layer)` dei layer MoE con router top-k, i sei expert
selezionati e i sei pesi del router. Run B calcola comunque il proprio router, lo registra nel trace come
`original_experts` e `original_router_weights`, poi sostituisce i tensor CUDA
degli expert e dei pesi con i valori di Run A prima dello streaming SSD e
dell'esecuzione MoE.

Questo è un oracle replay diagnostico, non un predictor e non il percorso
finale di predictive prefetch. Il router viene volutamente sovrascritto solo
quando TEST_1 imposta le variabili d'ambiente dedicate.

## Esecuzione

Dalla root:

~~~sh
make test-predictor-1
~~~

## Configurazione IDE di riferimento

Nell'IDE JetBrains usato per la repository (configurazione **Native
Application**), crea una configurazione chiamata:

~~~text
TEST_1 Forced Routing Replay
~~~

Imposta questi campi:

| Campo | Valore |
|---|---|
| Target | `test_1_runner` |
| Executable | `/home/michele/ds4/test_predictor/test_1/test_1_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | vuoto |
| Environment variables | `DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Before launch | `Build` |

`test_1_runner` è un piccolo wrapper: imposta `DS4_TEST_1=1` e avvia il
runner condiviso di TEST_0. Il target ricompila il binario `ds4` CUDA e i due
runner prima dell'esecuzione.

I risultati sono salvati in `test_predictor/test_1/output/`, poi `output_2/`,
e così via. Ogni cartella contiene anche `routing_oracle.bin`, il dizionario
binario compatto prodotto da Run A e letto da Run B, e
`routing_oracle.json`, la sua versione leggibile.

Subito dopo Run A il runner stampa:

~~~text
TEST_1: starting Run A via shared TEST_0 runner
TEST_1: converting Run A routing oracle to readable JSON
TEST_1: routing oracle JSON written to .../routing_oracle.json (... entries)
TEST_1: starting Run B via shared TEST_0 runner with oracle replay
~~~

La conversione avviene nel runner, fuori dal processo DS4 e prima di Run B.
Il file `.bin` resta l'unico formato usato per il replay, quindi il JSON non
aggiunge parsing o I/O al percorso critico CUDA di Run B.

## Output

- `run_A.jsonl`, `run_B.jsonl`: trace del decode; in Run B `experts` e
  `router_weights` sono i valori effettivamente eseguiti, mentre
  `original_experts` e `original_router_weights` mostrano cosa avrebbe scelto
  il router di Run B senza oracle;
- `routing_oracle.bin`: dizionario binario compatto, usato esclusivamente dal
  replay di Run B;
- `routing_oracle.json`: stessa informazione in formato leggibile, generata
  una sola volta dopo Run A;
- `comparison.json`: confronto automatico, timestamp UTC ed esito PASS/FAIL;
- `run_A/run_B.stdout.txt` e `run_A/run_B.stderr.txt`: output e log dei due
  processi DS4.

I layer `is_hash=true` non producono pesi top-k e non vengono inseriti nel
dizionario oracle: la loro selezione dipende dal token elaborato.

## Modifiche alla repository originale

Sono state modificate due parti esistenti:

- `ds4.c`: aggiunge il trace di routing opt-in condiviso con TEST_0 e il replay
  oracle opt-in di TEST_1. Run A scrive un record binario con chiave
  `(token_index, layer)`, expert e pesi; Run B cerca il record, conserva la
  propria selezione originale nel trace e scrive expert/pesi di Run A nei
  tensor CUDA prima del cache lookup SSD. Senza
  `DS4_ROUTING_ORACLE_WRITE_PATH` o `DS4_ROUTING_ORACLE_REPLAY_PATH` il
  comportamento DS4 resta invariato.
- `Makefile`: aggiunge i target di build/esecuzione di TEST_0 e TEST_1.

Inoltre `test_predictor/test_0/test_0_runner.c` viene riutilizzato per il
confronto, la creazione delle cartelle e l'avvio dei due processi. In modalità
TEST_1 gli indica soltanto di usare la root `test_1/output` e di passare i
percorsi dell'oracolo ai figli DS4.

## Cosa misura

Il confronto principale usa gli expert effettivamente eseguiti da Run B, che
dovrebbero coincidere con Run A per costruzione. Il trace di Run B conserva
anche la scelta originale del suo router: permette di distinguere ciò che
avrebbe scelto Run B da ciò che l'oracolo ha forzato.

Forzare expert e pesi elimina una grande fonte di divergenza, ma non promette
matematicamente token identici: piccole differenze nei calcoli non-MoE possono
restare. Il risultato del test serve proprio a misurare quanto il routing è
responsabile della divergenza osservata in TEST_0.

## Risultati osservati

Le cartelle presenti contengono tutte 2709 eventi di routing e usano la
configurazione CUDA + SSD streaming di riferimento, con TF32 attivo e senza
`--quality`.

| Cartella | Data UTC | Token uguali | Routing ordinato uguale | Insieme expert uguale | Esito |
|---|---|---:|---:|---:|---|
| `output/` | 2026-08-18 16:05:28 | 92,06% | 99,45% | 99,45% | FAIL |
| `output_2/` | 2026-08-18 16:22:04 | 100,00% | 100,00% | 100,00% | PASS |
| `output_3/` | 2026-08-18 16:49:30 | 100,00% | 100,00% | 100,00% | PASS |
| `output_4/` | 2026-08-18 16:54:11 | 100,00% | 100,00% | 100,00% | PASS |

`output/` è utile perché mostra il limite del replay: il primo token diverso
è al token 78. Da quel momento cambiano anche i tre layer `is_hash=true`, che
non usano il router top-k registrato nell'oracolo; risultano quindi 15 mismatch
di routing (cinque token divergenti per tre layer hash). Nei 40 layer MoE
registrati nell'oracolo il replay ha continuato a imporre la scelta di Run A.

Le tre esecuzioni successive hanno invece ottenuto 100% su token, ordine e
insieme. Questo conferma che la pipeline Run A → oracolo → Run B funziona:
quando le piccole differenze numeriche residue non cambiano un token, Run B
riproduce completamente il percorso osservato in Run A. Non significa però
che il 100% sia garantito in ogni futura coppia di processi: `output/` mostra
che una divergenza numerica non-MoE può ancora propagarsi fino ai token e ai
layer hash.
