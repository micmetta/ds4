# TEST_0 — determinismo del routing MoE

TEST_0 verifica che due processi DS4 separati, eseguiti con la stessa
configurazione CUDA + SSD streaming e greedy decoding, producano gli stessi
token che entrano nel decode e gli stessi expert del router per ogni
(token_index, layer).

## Esecuzione

Dalla root della repository:

~~~sh
make test-predictor-0
~~~

Il target aggiorna l'eseguibile DS4 CUDA e compila il runner C, quindi lo
esegue. Il runner crea una nuova cartella di output, avvia Run A, attende la
sua terminazione, avvia Run B come nuovo processo e confronta i trace.

Variabili opzionali:

~~~sh
DS4_MODEL=/percorso/modello.gguf \
DS4_TEST_0_N=64 \
make test-predictor-0
~~~

Per eseguire l'esperimento con TF32 disattivato per entrambe le run CUDA:

~~~sh
DS4_TEST_0_NO_TF32=1 make test-predictor-0
~~~

Questa opzione imposta `DS4_CUDA_NO_TF32=1` solo nei due processi DS4 avviati
da TEST_0. Può ridurre differenze numeriche, ma può anche rallentare il test e
non garantisce da sola un routing perfettamente identico.

Per provare la modalità qualità di DS4 mantenendo TF32 attivo:

~~~sh
DS4_TEST_0_QUALITY=1 make test-predictor-0
~~~

Questa opzione aggiunge `--quality` a entrambe le run. Per isolare il suo
effetto, non impostare contemporaneamente `DS4_TEST_0_NO_TF32=1`.

Attualmente questa combinazione non è utilizzabile con CUDA + SSD streaming
per il modello del test: Run A fallisce già nel prefill con
`CUDA streaming selected experts are unavailable for layer 0`. `--quality`
disabilita il percorso degli expert selezionati che lo streaming CUDA richiede.
L'opzione resta nel runner per rendere esplicito il test e il suo limite, ma
non produce ancora un confronto valido.

## Configurazione IDE di riferimento

Nell'IDE JetBrains usato per la repository (configurazione **Native
Application**), crea una configurazione chiamata:

~~~text
TEST_0 Routing Determinism
~~~

Imposta questi campi:

| Campo | Valore |
|---|---|
| Target | `test_0_runner` |
| Executable | `/home/michele/ds4/test_predictor/test_0/test_0_runner` |
| Working directory | `/home/michele/ds4/` |
| Program arguments | vuoto |
| Environment variables | `DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Before launch | `Build` |

Il target `test_0_runner` aggiorna DS4 CUDA e compila il runner prima
dell'esecuzione. Le opzioni sperimentali si aggiungono alle variabili di
ambiente: `DS4_TEST_0_NO_TF32=1` oppure `DS4_TEST_0_QUALITY=1`.

Ad ogni esecuzione il runner crea la prima cartella libera: `output`, poi
`output_2`, `output_3` e così via. I risultati precedenti non vengono mai
sovrascritti.

Per confrontare trace già prodotti senza avviare nuove inferenze:

~~~sh
./test_predictor/test_0/test_0_runner --compare-only
~~~

Il comando senza argomenti confronta `output/`. Per una cartella successiva:

~~~sh
./test_predictor/test_0/test_0_runner --compare-only test_predictor/test_0/output_2
~~~

## Output

- output*/run_A.jsonl, output*/run_B.jsonl: un record per il routing decode;
- output*/run_A.stdout.txt, output*/run_B.stdout.txt: risposta testuale;
- output*/run_A.stderr.txt, output*/run_B.stderr.txt: log DS4;
- output*/comparison.json: timestamp UTC dell'esecuzione, confronto strutturato
  e dettagli dei mismatch.

`comparison.json` indica anche `cuda_tf32_disabled` e `quality_mode`, così è
possibile sapere con quali opzioni CUDA è stato prodotto quel risultato.

Ogni evento JSONL contiene il token che entra nel decoder, gli expert
autoritativi e, per i layer top-k, gli otto candidati con score più alto:

~~~json
{"phase":"decode","token_index":12,"token_id":12345,"layer":17,"is_hash":false,"experts":[12,41,73,101,180,229],"top_candidates":[[12,0.8,0.01,0.81],[41,0.7,0.01,0.71]],"cutoff_margin":0.000017}
~~~

token_index è la posizione assoluta nel contesto DS4. `prob` è il valore
router dopo `sqrt(softplus(logit))`; `selection_score` è `prob + bias` ed è il
valore usato dal top-k. `cutoff_margin` è la differenza tra il sesto e il
settimo candidato. Per hash routing i candidati sono `null`, perché gli ID
expert dipendono dalla tabella token→expert, non dagli score.

Il trace non include prefill. Il readback degli score è attivo solo quando
`DS4_ROUTING_TRACE_PATH` è impostata e serve esclusivamente alla diagnostica:
non va usato per benchmark di prestazioni.

Il runner è volutamente scritto in C e usa solo API POSIX semplici: fork,
execv, waitpid, file descriptor e un parser limitato al formato JSONL generato
da TEST_0.

## Come leggere i dettagli dei mismatch

Nel file `comparison.json` le tre liste seguenti sono vuote quando non esiste
quel tipo di differenza. Non sono errori del parser: descrivono differenze
misurate fra le due inferenze separate.

### `token_mismatches`

Rappresenta i punti in cui il token che entra nel decode non è uguale fra Run A
e Run B. È il segnale più importante perché, dal primo token diverso in poi,
le due inferenze ricevono input diversi e i mismatch di routing successivi
possono essere una conseguenza a cascata.

Ogni record contiene:

- `token_index`: posizione assoluta del token nel contesto DS4;
- `token_id_run_A`: ID del token entrato nel decode della Run A;
- `token_id_run_B`: ID del token entrato nel decode della Run B.

Un valore `null` per uno dei due ID indica che quell'evento token è presente
solo nell'altra run.

### `order_mismatch_details`

Rappresenta gli eventi in cui i sei expert selezionati non hanno lo stesso
ordine. Per esempio, `[10, 20, 30, 40, 50, 60]` e
`[10, 30, 20, 40, 50, 60]` hanno lo stesso insieme di expert, ma un ordine
diverso. Questa lista include anche i casi in cui cambia l'insieme, perché in
quel caso le due sequenze non possono essere identiche.

Ogni record contiene:

- `token_index`: posizione del token elaborato;
- `layer`: indice del layer MoE che ha eseguito il routing;
- `experts_run_A`: i sei expert scelti dalla Run A, nell'ordine del router;
- `experts_run_B`: i sei expert scelti dalla Run B, nello stesso formato;
- `router_scores_run_A`: riassunto degli score del router della Run A;
- `router_scores_run_B`: riassunto equivalente della Run B.

`experts_run_A` o `experts_run_B` può essere `null` quando l'evento manca del
tutto in una delle due run. I campi `router_scores_run_A` e
`router_scores_run_B` possono invece essere `null` per hash routing, dove gli
expert dipendono da una tabella token→expert e non da score del router.

Quando presente, ciascun oggetto `router_scores_run_*` contiene:

- `top_candidates`: gli otto candidati con score più alto. Ogni candidato ha
  `expert` (ID), `prob` (valore router dopo `sqrt(softplus(logit))`), `bias`
  (bias statico del modello) e `selection_score` (`prob + bias`, usato dal
  top-k);
- `cutoff_margin`: differenza tra il sesto e il settimo `selection_score`.
  Un valore piccolo segnala una scelta top-6 vicina alla soglia.

### `set_mismatch_details`

Rappresenta gli eventi in cui non cambia solo l'ordine: almeno un expert dei
sei è diverso tra Run A e Run B. È la lista usata per il criterio PASS/FAIL,
perché l'insieme `E_l(t)` degli expert non è identico.

Ogni record ha esattamente gli stessi campi e lo stesso significato descritti
per `order_mismatch_details`: `token_index`, `layer`, i due array `experts`,
e i due riassunti degli score. Un record di questa lista appare anche in
`order_mismatch_details`; il contrario non è necessariamente vero, perché un
semplice scambio di posizione non cambia l'insieme.

## Modifiche alla repository originale

TEST_0 richiede una strumentazione minima nel core, ma non modifica router,
logits, sampling, cache expert o SSD streaming.

- `ds4.c`: quando la variabile `DS4_ROUTING_TRACE_PATH` è impostata, salva un
  record JSONL per ciascun evento di decode. Il record contiene token, layer,
  expert realmente selezionati e, per i layer con router top-k, candidati e
  score diagnostici. Se la variabile non è impostata, DS4 segue il normale
  percorso vanilla e non apre alcun trace.
- `Makefile`: aggiunge il target `test_0_runner` e il comando
  `test-predictor-0`, così il runner C può essere compilato e avviato in modo
  ripetibile dall'IDE o dalla shell.

I file in `test_predictor/test_0/` appartengono al test: il runner gestisce i
due processi separati, le cartelle incrementali e il confronto offline.

## Risultati osservati

I risultati seguenti provengono dalle cartelle di output presenti al momento
della scrittura. Ogni run completa contiene 2709 eventi di routing.

| Cartella | Configurazione | Token uguali | Routing ordinato uguale | Insieme expert uguale | Primo mismatch routing |
|---|---|---:|---:|---:|---|
| `output/` | baseline | 100,00% | 93,47% | 97,23% | token 69, layer 7 |
| `output_2/` | TF32 disattivato | 100,00% | 71,24% | 86,08% | token 31, layer 4 |
| `output_3/` | baseline | 58,73% | 44,59% | 49,76% | token 38, layer 5 |
| `output_4/` | baseline | 100,00% | 78,07% | 89,66% | token 43, layer 3 |

Nessuna delle run complete soddisfa il criterio PASS. Il risultato migliore
finora è `output/`, ma anche lì sono presenti 75 mismatch dell'insieme degli
expert e 177 mismatch di ordine. Disattivare TF32 non ha migliorato il test:
ha anticipato la prima divergenza e ha ridotto sia il match ordinato sia il
match dell'insieme. La prova con `--quality` non ha generato un risultato
perché è incompatibile con SSD streaming nel percorso CUDA corrente.

## Interpretazione del non-determinismo

I pesi del modello e i bias del router restano uguali fra Run A e Run B. Le
differenze osservate sono nei valori numerici calcolati prima del router; per
expert con score vicini, una piccola variazione può cambiare l'ordine dei sei
expert o sostituire il sesto con il settimo. `router_scores_run_A` e
`router_scores_run_B` in `comparison.json` servono proprio a vedere questi
casi, mentre `cutoff_margin` mostra la distanza fra sesto e settimo candidato.

La causa generale è il modo in cui CUDA esegue in parallelo i calcoli del
Transformer. Il forward contiene molte moltiplicazioni matrice-vettore e
matrice-matrice (attention, FFN e router), eseguite in DS4 tramite cuBLAS e
kernel CUDA. Per sommare molti contributi, migliaia di thread lavorano in
parallelo e le somme parziali possono essere combinate in un ordine diverso
fra due processi. In aritmetica floating point la somma non è perfettamente
associativa: `(a + b) + c` può differire di pochi bit da `a + (b + c)`.

CUDA può inoltre scegliere implementazioni e ottimizzazioni diverse in base a
risorse temporanee, scheduling e stato del processo. TF32 e Tensor Core
privilegiano velocità rispetto alla precisione completa FP32; tuttavia il test
con TF32 disattivato ha mostrato che TF32 non è l'unica possibile fonte della
variazione. Anche le riduzioni parallele e gli altri kernel precedenti al
router possono contribuire.

Questo non dimostra ancora quale singola operazione produca il primo valore
diverso: per identificarla servirebbe tracciare gli output intermedi prima del
primo mismatch. SSD streaming e cache determinano quando i pesi sono
disponibili, ma non sono al momento la causa dimostrata delle differenze negli
score del router.

Quando il routing diverge, gli expert eseguiti possono produrre hidden state
leggermente diversi. La differenza può quindi propagarsi ai layer e token
successivi. In `output_3/`, per esempio, il primo mismatch di routing è al
token 38 e il primo token generato diverso è al token 51. Dopo quel punto le
due run non ricevono più lo stesso input token: le percentuali successive non
misurano più soltanto la stabilità del router, ma anche questa divergenza a
cascata.

TEST_0 dimostra quindi che il routing CUDA non è perfettamente riproducibile
fra processi separati nella configurazione corrente. Non è però un blocco per
il predictive prefetch: il futuro predictor dovrà influenzare solo prefetch,
cache e scheduling; il router reale resterà sempre autoritativo per gli expert
eseguiti e per l'output del modello.

## PASS / FAIL

Il test è PASS se:

- la sequenza (token_index, token_id) è identica;
- il numero di eventi routing è identico;
- per ogni evento, l'insieme degli expert è identico.

L'ordine dei sei expert è riportato e confrontato separatamente. Su una run
interamente deterministica ci si aspetta anche ordered_routing_match_percent
uguale a 100%; il requisito formale del test resta l'identità dell'insieme,
perché E_l(t) è un insieme.

## Collegamento con la tesi

Nella FFN classica descritta nella tesi, il token attraversa una sola
trasformazione feed-forward del layer. In DeepSeek V4 Flash quella posizione
del layer è MoE: il router sceglie sei FFN specializzate su 256. TEST_0 misura
la riproducibilità di quella scelta, cioè di E_l(t), senza cambiare la
computazione del Transformer.
