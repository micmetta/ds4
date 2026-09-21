Exit code: 0
Wall time: 0.7 seconds
Output:
# TEST_7 — Prefetch con il modello neurale Build 5

## Scopo

TEST_7 misura se il modello neurale Build 5 può essere usato durante
l'inferenza reale di DS4 per predire i sei expert del token successivo e
pre-caricarli dal secondo GGUF nell'area RAM pinned già usata da TEST_6.

Le run TEST_7 e TEST_8 finora registrate hanno usato il checkpoint Hyperband
`0024`:

- validation Recall@6: `0.49085259437561035`;
- test Recall@6 offline: `0.494631`;
- architettura: state fixed-NNZ 512D, raw hidden 4096→512 ReLU, token
  embedding 4096→64 ReLU, KV 512D diretto, fusione 512 ReLU, classificatore
  sigmoid a 256 expert.

Successivamente, il warm-start dello stesso trial ha raggiunto il nuovo record
Build 5 di **validation Recall@6 `0.502358287`**, salvato in:

```text
.../retrain_best_hyperband_trial_0024_EP1000_PAT5/models/ffn.keras
```

La valutazione finale una sola volta sul test set ha ottenuto Recall@6
**`0.506056192`**. Questo e' il miglior risultato offline Build 5 finora, ma
non e' ancora stato usato nelle run runtime TEST_7/8. Di conseguenza, i
risultati di speedup documentati qui restano correttamente riferiti al
checkpoint `0024` da `0.49085259437561035`, non al nuovo modello.

Questo risultato offline non è ancora uno speedup. TEST_7 serve proprio a
misurare il costo reale di estrarre le feature dalla GPU, di eseguire il
predictor sulla CPU e di usare le sue predizioni per il prefetch.

Il checkpoint rimane identico a quello valutato offline. Durante TEST_7 il
forward pass CPU usa `float32`: la CPU di questa macchina non accelera
efficacemente `float16` e TensorFlow altrimenti usa un fallback estremamente
lento. Questa scelta cambia solo il tipo numerico dei calcoli runtime, non i
pesi caricati né l'architettura del modello.

## Due fasi necessarie

### Fase 1 — Shadow mode

DS4 calcola router, hidden raw e KV per il token `t` e per il layer `l`.
Invia questi dati a un processo Python persistente sulla CPU, che riproduce la
stessa preparazione Build 5 e restituisce la previsione di `E_l(t+1)`.

In questa fase DS4 **non** inoltra la previsione al worker di prefetch. Quando
arriva il token successivo confronta invece i sei expert predetti con i sei
expert realmente scelti dal router e scrive un record JSONL. La fase shadow è
necessaria per misurare separatamente:

- Recall@6 online, sulle feature create durante la run reale;
- latenza del predictor CPU per richiesta;
- overhead di copia GPU→CPU e comunicazione locale;
- invarianza dell'output DS4, perché nessun expert viene precaricato grazie al
  modello.

### Fase 2 — Learned prefetch

Durante l'esecuzione del token `t` al layer MoE `l`, il predictor riceve le
feature di quel punto e produce la previsione `E_l(t+1)`: gli expert attesi
per il **token successivo** nello **stesso layer**. La previsione viene subito
inviata al worker SSD→RAM pinned già validato in TEST_6. Mentre DS4 continua il
calcolo del token `t`, il worker prova a leggere dal secondo GGUF e a rendere
pronti gli expert in RAM pinned. Quando DS4 raggiunge il token `t+1`, layer
`l`, riusa gli expert pronti e carica da SSD soltanto quelli ancora mancanti.

La fase misura il guadagno o la perdita netta rispetto al replay di controllo:
non assume che una Recall@6 elevata produca automaticamente uno speedup.

## A/B/S/C della run

Il runner Python crea una nuova directory `output`, `output_2`, … e svolge:

| Run | Funzione |
|---|---|
| A | Registra `routing_oracle.bin`, la sequenza di routing di riferimento. |
| B | Replay dell'oracolo senza predictor né prefetch: baseline comparabile. |
| S | Replay + predictor CPU in shadow mode. |
| C | Replay + predictor CPU + learned prefetch nel worker pinned-RAM. |

Il replay rende identico il routing A/B/S/C. Le differenze nei tempi derivano
quindi dall'integrazione del predictor e dal prefetch, non da expert diversi.

### Predictor CPU asincrono

La prima versione di TEST_7 inviava la richiesta al servizio TensorFlow e
aspettava la risposta prima di continuare l'inferenza GPU. La versione attuale
usa invece una coda locale a capacità fissa e un thread CPU dedicato:

```text
GPU/DS4: cattura feature di (token t, layer l)
        → copia la richiesta nella coda → continua subito la GPU

thread CPU: riceve la richiesta → esegue TensorFlow → riceve i 6 expert
        → accoda il prefetch SSD → RAM pinned per (token t+1, layer l)
```

Il thread GPU non attende l'inferenza TensorFlow. Se la coda fosse piena o il
servizio fallisse, TEST_7 interrompe la run con errore invece di accumulare
richieste senza limite. La fase shadow continua a misurare Recall@6 e latenza
del predictor; la fase C misura se il lavoro CPU può essere nascosto dietro il
calcolo GPU prima che servano gli expert del token successivo.

### Conteggio dei layer nella fase C

`DS4_TEST_7_PREFETCH_LAYERS=1` significa: predire e precaricare soltanto il
primo layer MoE del modello, che nell'esecuzione DS4 ha indice fisico `3`.
Internamente il runner configura il worker pinned-RAM con quattro indici fisici
(`0`–`3`): i primi tre non ricevono alcun prefetch; sono riservati perché il
worker storico numera i layer a partire da zero e deve poter accettare il
primo enqueue learned al layer `3`.

## Componenti

- `runtime_predictor_service.py`: processo Python persistente. Carica una sola
  volta checkpoint, embedding e artefatti di standardizzazione; riceve richieste
  binarie locali e restituisce sei ID expert.
- `test_7_runner.py`: avvia A/B/S/C, avvia e arresta il servizio per S e C,
  salva stdout/stderr, trace e `summary.json`.
- `ds4.c`: contiene un piccolo client Unix socket attivo esclusivamente con
  `DS4_TEST_7_MODE=shadow` oppure `prefetch`. TEST_6 non lo attiva e conserva
  il proprio comportamento.

## Output importanti

- `run_S.learned_predictions.jsonl`: previsione, target reale, Recall@6 e
  millisecondi del predictor per ogni coppia token/layer.
- `run_C.learned_predictions.jsonl`: stesso controllo online della fase 2.
- `run_C.prefetch.jsonl`: eventi del worker pinned-RAM, inclusi
  `learned_prefetch_enqueue`, hit e miss.
- `summary.json`: tempi wall-clock e token/s di A/B/S/C.
- `predictor_service.log`: caricamento e diagnostica del processo CPU.

## Guida completa a `summary.json`

La chiave `phases` contiene quattro esecuzioni dello stesso prompt. Il confronto
principale per misurare il beneficio del modello e' `C_learned_prefetch` contro
`B_replay_baseline`, **all'interno della stessa directory `output_N`**.

| Fase | Cosa misura |
|---|---|
| `A_oracle_reference` | Prima esecuzione: registra il routing realmente scelto dal router DS4. Produce l'oracolo usato dal replay successivo. Non e' il baseline di velocita'. |
| `B_replay_baseline` | Riesegue il routing dell'oracolo senza predictor CPU e senza learned prefetch. E' il riferimento temporale per C. |
| `S_shadow_predictor` | Esegue il predictor CPU con le feature reali, ma non chiede al worker di precaricare gli expert. Misura qualita' e costo del predictor senza il beneficio del prefetch. |
| `C_learned_prefetch` | Esegue il predictor CPU e usa le sue sei predizioni per precaricare gli expert del token successivo. E' la fase da confrontare con B. |

I campi comuni alle quattro fasi sono:

- `wall_seconds`: durata reale totale della fase, dal lancio di DS4 alla sua
  terminazione, espressa in secondi. Include inizializzazione, generazione e
  tutte le attese del processo.
- `generation_tokens_per_second`: token generati al secondo ricavati dal log
  standard di DS4. Nell'implementazione attuale il runner cerca la stringa
  `generation: <numero> t/s` nello stdout. Se quel formato non viene scritto
  nello stdout, come nelle run mostrate, il valore resta `0.0`: **non significa
  che DS4 abbia generato zero token**. Per il confronto attuale usare quindi
  `wall_seconds`.

Le fasi S e C includono inoltre i risultati del predictor:

- `predictions_with_target`: numero di predizioni per cui, al momento in cui
  il token successivo e' stato realmente instradato, era disponibile il gruppo
  reale di sei expert da usare come target.
- `online_recall_at_6`: media della Recall@6 sulle predizioni con target. Vale
  1 se tutti e sei gli expert reali sono nella sestina predetta; vale, per
  esempio, `0.5` se ne sono corretti tre su sei.
- `predictor_inference_ms_average`: tempo medio in millisecondi tra l'invio
  della richiesta al servizio CPU e la ricezione delle sei predizioni. Nella
  fase asincrona questo tempo non dovrebbe bloccare il thread GPU, ma descrive
  comunque la latenza del predictor.
- `correct_predicted_experts_total`: somma degli expert corretti tra tutte le
  predizioni con target; ogni expert ripetuto nella sestina viene contato una
  sola volta per quella predizione.
- `predicted_experts_total`: numero totale di expert predetti nelle predizioni
  valutabili, cioe' `predictions_with_target x 6`.
- `correct_predicted_experts_average_per_prediction`: media degli expert
  corretti in una singola sestina predetta. E' il rapporto tra
  `correct_predicted_experts_total` e `predictions_with_target`.
- `fully_correct_six_expert_predictions`: numero di predizioni perfette con
  tutti e sei gli expert del router reale presenti nella sestina predetta.

La sola fase C contiene anche i contatori del worker pinned-RAM:

- `learned_prefetch_enqueues`: quante sestine predicted dal modello sono state
  consegnate al worker come richieste di prefetch, per i layer coperti da
  TEST_7.
- `expert_group_demands_at_predicted_layers`: quante volte DS4 ha poi davvero
  richiesto un gruppo di expert in quei layer. Di norma e' molto vicino agli
  enqueue; puo' differire ai bordi della generazione, per esempio per l'ultima
  previsione del token successivo.
- `full_prefetch_hits`: domande in cui tutti e sei gli expert richiesti erano
  gia' disponibili nella RAM pinned (`6` riusati, `0` da SSD).
- `partial_prefetch_reuse_groups`: domande in cui era disponibile almeno un
  expert corretto, ma non tutti e sei. DS4 riusa quelli disponibili e legge da
  SSD solo gli altri.
- `experts_reused_from_pinned_ram`: numero totale di expert individuali
  effettivamente copiati dalla RAM pinned alla cache GPU al momento della
  domanda.
- `experts_loaded_from_ssd_at_predicted_layers`: numero totale di expert
  individuali che non erano pronti e sono stati letti dal GGUF/SSD nelle stesse
  domande. Per ogni domanda di sei expert, il contributo di questi ultimi due
  campi e' complessivamente sei.

## Primo risultato misurato: `output_8`

La prima esecuzione completa ha terminato A/B/S/C senza errori. Il predictor
ha elaborato 62 coppie token/layer sia nella fase shadow sia nella fase C.

| Misura | Risultato |
|---|---:|
| Recall@6 online | 0,467742 |
| Latenza media del predictor CPU | 13,40 ms in S; 13,47 ms in C |
| Tempo B, replay senza predictor/prefetch | 103,45 s |
| Tempo S, predictor in shadow | 105,57 s |
| Tempo C, learned prefetch | 98,27 s |
| Differenza C rispetto a B | -5,19 s (-5,01%) |

Il dato di C è incoraggiante ma **non è ancora una prova di speedup causato
dal modello**. In questa singola run il worker ha ricevuto 63 richieste learned
e ne ha rese pronte 32, ma non ha registrato alcun `prefetch_hit`: in 31 casi
il gruppo di sei expert predetti non coincideva esattamente con il gruppo
richiesto (`prefetch_lookup_expert_mismatch`). Un Recall@6 parziale non basta
per un hit del worker corrente, che richiede la corrispondenza del gruppo
completo di sei expert.

Il -5,01% può quindi includere variabilità di sistema, cache del sistema
operativo o benefici indiretti delle letture speculative. Per attribuirlo con
sicurezza al learned prefetch occorrono più ripetizioni comparabili e, in
particolare, hit effettivi del worker.

### Risultato dopo il riuso parziale: `output_9`

Con il riuso expert-per-expert, il worker ha completato tutte le 63 richieste
learned e 62 domande al layer previsto hanno registrato
`prefetch_partial_reuse`. Non ci sono ancora `prefetch_hit` completi: in ogni
caso almeno un expert della sestina richiesta era diverso da quello predetto.

| Misura | Risultato |
|---|---:|
| Tempo B, replay senza predictor/prefetch | 100,96 s |
| Tempo S, predictor in shadow | 108,33 s |
| Tempo C, learned prefetch con riuso parziale | 101,50 s |
| Differenza C rispetto a B | +0,54 s (+0,54%) |
| Lettura SSD media, gruppo parzialmente riusato | 11,02 ms |
| Lettura SSD media, gruppo caricato interamente a domanda | 19,48 ms |
| Caricamento totale medio del gruppo parzialmente riusato | 17,70 ms |
| Caricamento totale medio del gruppo completamente demand | 27,81 ms |

Il riuso parziale riduce quindi concretamente la lettura SSD e il tempo di
caricamento del gruppo interessato. In questa run, però, il risparmio non
compensa ancora con margine il costo della pipeline predictor CPU e la normale
variabilità della run: C è sostanzialmente alla pari con B, ma leggermente più
lenta. Il confronto corretto richiede ripetizioni B/C alternate.

### Risultato prima dell'asincronia: `output_10`

`output_10` e' stato eseguito quando il predictor CPU veniva ancora chiamato
in modo **sincrono**: il thread che guida il calcolo DS4 attendeva la risposta
del servizio prima di proseguire. Il riuso parziale funzionava, ma il costo di
questa attesa poteva cancellarne il beneficio.

| Misura | Risultato |
|---|---:|
| Tempo B, replay senza predictor/prefetch | 105,35 s |
| Tempo S, predictor in shadow sincrono | 118,08 s |
| Tempo C, learned prefetch sincrono | 107,17 s |
| Differenza C rispetto a B | +1,83 s (+1,73%) |
| Expert riusati dalla RAM pinned in C | 174 |
| Expert letti da SSD in C | 204 |
| Gruppi con riuso parziale | 62 |
| Hit completi, 6/6 expert | 0 |

Questo risultato mostra che il riuso era reale, ma che aspettare il predictor
CPU nel percorso critico poteva rendere C leggermente piu' lenta del baseline.

### Primo risultato con predictor asincrono: `output_11`

In `output_11` il thread GPU non attende piu' la risposta del predictor. Un
worker CPU persistente riceve le feature, esegue il modello e invia il
prefetch al worker pinned-RAM quando la risposta e' pronta. Di conseguenza,
una previsione puo' anche arrivare troppo tardi per la domanda corrispondente:
e' il compromesso necessario per non bloccare la GPU.

| Misura | Risultato |
|---|---:|
| Tempo B, replay senza predictor/prefetch | 105,53 s |
| Tempo S, predictor asincrono in shadow | 100,61 s |
| Tempo C, learned prefetch asincrono | **95,74 s** |
| Differenza C rispetto a B | **-9,79 s (-9,28%)** |
| Predizioni valutabili | 62 |
| Recall@6 online | 0,467742 |
| Tempo medio del predictor | 14,63 ms in C |
| Expert corretti nelle sestine predette | 174 su 372 |
| Expert riusati dalla RAM pinned in C | 115 |
| Expert letti da SSD in C | 263 |
| Gruppi con riuso parziale | 38 |
| Hit completi, 6/6 expert | 0 |

Il confronto B/C della stessa run e' incoraggiante: C e' piu' veloce del
baseline anche se una parte delle predizioni asincrone non e' arrivata in
tempo. Non e' ancora una dimostrazione conclusiva: le fasi eseguite piu'
tardi possono beneficiare della page cache Linux, cioe' di dati del GGUF gia'
letti in precedenza e rimasti in RAM del sistema operativo. La VRAM e la RAM
pinned, invece, vengono ricreate per ogni processo DS4 e non sono condivise
fra B e C. Il prossimo esperimento deve quindi ripetere B e C piu' volte,
alternando il loro ordine e confrontandone le medie.

Questo esperimento statistico e' implementato separatamente in
[`../test_8/README.md`](../test_8/README.md), per conservare TEST_7 come test
funzionale della pipeline completa A/B/S/C.

### Controllo rapido su quattro layer MoE: `output_12`

`output_12` usa il predictor asincrono sui primi quattro layer MoE fisici,
cioe' gli indici `3`–`6`, e legge il GGUF del prefetch dal percorso esterno
`/mnt/d/ds4/gguf/...`. Rispetto al controllo precedente a un solo layer, ogni
token genera quattro richieste di previsione e quindi quattro opportunita' di
riusare expert gia' nella RAM pinned.

| Misura | Risultato |
|---|---:|
| Tempo B, replay senza predictor/prefetch | 104,95 s |
| Tempo S, predictor asincrono in shadow | 104,49 s |
| Tempo C, learned prefetch asincrono | **99,08 s** |
| Differenza C rispetto a B | **-5,87 s (-5,59%)** |
| Predizioni valutabili | 248 |
| Recall@6 online | 0,454301 |
| Expert corretti nelle sestine predette | 676 su 1.488 |
| Gruppi con riuso parziale | 152 su 252 domande |
| Expert riusati dalla RAM pinned | 421 |
| Expert letti da SSD | 1.091 |
| Hit completi, 6/6 expert | 0 |

Il predictor ha prodotto una sestina perfetta nel confronto offline/online,
ma il worker non ha registrato un hit completo: quella previsione non era
verosimilmente pronta nel preciso istante della domanda oppure non corrisponde
alla stessa richiesta tracciata dal worker. Questo evidenzia la differenza tra
qualita' del modello e puntualita' del prefetch. Il risultato temporale e'
incoraggiante, ma e' una sola run: va confermato con TEST_8 impostato anch'esso
a quattro layer e con le tre coppie B/C alternate.

### Controllo rapido su otto layer MoE: `output_13`

`output_13` estende il predictor asincrono ai layer MoE fisici 3–10. Il
servizio CPU ha elaborato 496 predizioni valutabili e inviato 504 enqueue,
esattamente quanto atteso per otto layer: non si osserva quindi un crollo della
capacita' del predictor al raddoppio delle richieste.

| Misura | Risultato |
|---|---:|
| Tempo B, replay senza predictor/prefetch | 98,57 s |
| Tempo C, learned prefetch asincrono | 100,25 s |
| Differenza C rispetto a B | +1,67 s (+1,70%) |
| Recall@6 online | 0,427419 |
| Gruppi con riuso parziale | 276 su 504 domande |
| Expert riusati dalla RAM pinned | 762 |
| Expert letti da SSD | 2.262 |
| Hit completi, 6/6 expert | 0 |

Il riuso parziale rimane frequente (276 gruppi) e il tempo medio del predictor
e' 13,24 ms, vicino al valore misurato a quattro layer. La singola run C e'
tuttavia piu' lenta del baseline; non e' sufficiente per decidere fra quattro
e otto layer. TEST_8 con tre coppie alternate e' necessario per confrontare in
modo affidabile questa configurazione.

## Riutilizzo parziale degli expert predetti

Inizialmente il worker poteva riutilizzare una lettura pinned-RAM soltanto se
la sestina predetta coincideva esattamente con la sestina richiesta. Il
predictor, però, può prevedere correttamente alcuni expert anche quando non
prevede l'intero gruppo.

TEST_7 ora conserva gli expert predetti nel worker indipendentemente dalla
loro correttezza e, alla domanda del token successivo, costruisce il gruppo
richiesto expert per expert: copia dalla RAM pinned quelli presenti in entrambe
le sestine e legge da SSD soltanto gli expert mancanti. La trace registra
questo caso come `prefetch_partial_reuse`; `prefetch_hit` resta riservato alla
corrispondenza completa dei sei expert.

Ogni record `prefetch_partial_reuse` della trace contiene inoltre
`prefetched_experts_reused` e `ssd_experts_loaded`: i due valori sommano a 6 e
permettono di vedere esattamente quanti expert del gruppo reale erano già
pronti nella RAM pinned e quanti sono stati recuperati a domanda dal GGUF.
Al termine della run, gli stessi conteggi sono anche aggregati in
`summary.json`, nella sezione `C_learned_prefetch`.

Per esempio:

```json
"prefetched_experts_reused": 3,
"ssd_experts_loaded": 3
```

significa che, per quel token e quel layer, 3 dei 6 expert richiesti dal router
erano stati predetti correttamente e si trovavano già nella RAM pinned. DS4 ha
quindi letto dal GGUF/SSD soltanto gli altri 3 expert necessari. In un
`prefetch_hit` completo i valori sono rispettivamente `6` e `0`; in un normale
`prefetch_miss_demand` sono `0` e `6`.

In `summary.json`, per la fase C, sono disponibili anche:

- `correct_predicted_experts_total`: expert corretti tra tutte le predizioni
  con target noto;
- `predicted_experts_total`: expert complessivamente predetti, cioè sei per
  ogni predizione con target;
- `experts_reused_from_pinned_ram`: expert realmente già pronti e riusati al
  momento della domanda;
- `experts_loaded_from_ssd_at_predicted_layers`: expert che sono rimasti da
  leggere dal GGUF/SSD nei layer effettivamente coperti dal predictor;
- `full_prefetch_hits` e `partial_prefetch_reuse_groups`: rispettivamente i
  gruppi con 6/6 expert pronti e quelli con riuso parziale.

## Limite deliberato iniziale

Il lookup token Build 5 contiene solo i 7.157 token distinti osservati nei tre
split. TEST_7, per ora, termina con un errore chiaro se compare un token non
presente: il prompt iniziale usa una frase già contenuta in Build 5. La
gestione di token fuori vocabolario sarà un intervento successivo, distinto
dal test di integrazione iniziale.


## Diagnostica degli expert corretti ma non riusati

Dalle nuove run, `summary.json` collega la qualità della previsione alla disponibilità effettiva nella RAM pinned. Nella fase `C_learned_prefetch` sono disponibili:

- `correct_predicted_experts_reused_from_pinned_ram`: expert sia corretti sia riusati dalla RAM pinned;
- `correct_predicted_experts_not_reused`: expert corretti che non sono stati riusati;
- `correct_predicted_experts_not_reused_prefetch_bank_full`: expert corretti non ammessi perché la banca pinned era piena;
- `correct_predicted_experts_not_reused_worker_not_ready`: expert corretti per cui il worker non aveva ancora completato il caricamento;
- `correct_predicted_experts_not_reused_stale_discarded`: expert corretti rimossi come obsoleti;
- `correct_predicted_experts_not_reused_ready_but_unavailable` e `correct_predicted_experts_not_reused_no_prefetch_state`: casi diagnostici residui.

Le ultime cinque categorie sommano a `correct_predicted_experts_not_reused`. Questa separazione evita di attribuire automaticamente al predictor o al disco un problema causato invece dalla capacità della banca pinned.
