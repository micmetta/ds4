# Dataset e corpus utili per costruire il dataset del predictor di expert

> **Obiettivo:** costruire un dataset supervisionato per addestrare un predictor capace di anticipare gli expert MoE che DeepSeek-V4-Flash userà durante il decoding in DS4, sfruttando progressivamente routing history, router score, hidden state, feature della KV cache e informazioni dell'indexer/attention.
>
> **Ordinamento:** i dataset sono ordinati per **importanza rispetto al nostro obiettivo finale**, non semplicemente per quanto siano già pronti all'uso.
>
> **Verifica link e contenuti:** 27 agosto 2026.

---

## Classifica sintetica

| # | Dataset / corpus | Utilità principale | Contiene routing token×layer? | Contiene vere risposte DeepSeek? | Priorità |
|---|---|---|---|---|---|
| **1** | **DS4 Imatrix Calibration Dataset — antirez / Salvatore Sanfilippo** | Corpus principale da riprodurre nel nostro DS4 instrumentato | No | No, salvo transcript già presenti come parte del prompt | **Massima** |
| **2** | **Doubleword/specdec-calibration** | Routing trace DeepSeek-V4-Flash già pronto per baseline e primi predictor | **Sì** | Non è il suo scopo principale | **Molto alta** |
| **3** | **DS4 quality-testing / DeepSeek V4 Flash official continuations** | Prompt + continuazioni ufficiali DeepSeek da riprodurre in DS4 | No, finché non instrumentiamo DS4 | **Sì** | **Alta** |
| **4** | **0xSero/deepseek-v4-flash-reap-observations-v2** | Copertura di expert e domini; analisi di salienza/attivazione | Non come trace temporale pronto per il nostro target | Non è il suo scopo principale | **Media-alta** |
| **5** | **0xSero/deepseek-v4-flash-reap-observations-v1** | Versione precedente utile per confronto e studio del formato REAP | Non come trace temporale pronto per il nostro target | Non è il suo scopo principale | **Media** |

---

# 1. DS4 Imatrix Calibration Dataset — Salvatore Sanfilippo / antirez

## Perché è il più importante

Questo è il corpus che userei come **base principale per costruire il nostro vero dataset di training**.

Il motivo è che proviene direttamente dal repository ufficiale di DS4 e i prompt sono costruiti per riprodurre il formato reale utilizzato dal server DS4. Include inoltre pattern particolarmente rilevanti per il workload che vogliamo studiare:

- codice C e Metal proveniente da DS4;
- code review e debugging;
- prompt di programmazione;
- prompt general-purpose;
- long-context;
- reasoning;
- GPQA Diamond, SuperGPQA e AIME 2025 tramite `ds4-eval`;
- traduzione e task multilingua;
- summarization, rewriting, extraction e copy-editing;
- conversazioni agentiche;
- tool calling e tool result con sintassi DSML di DS4;
- modalità **think**;
- modalità **nothink**.

Il README attuale del repository indica circa **4682 prompt renderizzati** e circa **2,91 milioni di token** secondo la stima grezza `bytes / 4`. Il numero può cambiare con gli aggiornamenti del repository.

## File più importanti

### `prompts.jsonl`

È il file che preferirei usare per il nostro progetto perché conserva la struttura dei record.

Contiene record con informazioni come:

```text
id
category
mode
source
messages
rendered
```

Questo ci consente, dopo aver eseguito i prompt nel nostro DS4 instrumentato, di mantenere anche metadati utili per effettuare split e analisi per dominio.

### `rendered_prompts.txt`

Contiene gli stessi prompt già trasformati nella forma che DS4 passa al modello.

I prompt sono separati da marker visibili `DS4_IMATRIX_PROMPT`.

Sono disponibili anche:

```text
rendered_prompts_think.txt
rendered_prompts_nothink.txt
manifest.json
```

## Cosa NON contiene

Non contiene direttamente:

- expert IDs token-per-token;
- router score;
- hidden state;
- KV cache;
- indexer output;
- vere continuazioni generate durante quella specifica run.

Quindi **non è ancora il dataset supervisionato del predictor**.

Il suo ruolo sarebbe:

```text
prompts.jsonl
      |
      v
nostro DS4 instrumentato
      |
      +--> token generati
      +--> expert IDs
      +--> router scores
      +--> hidden-state features
      +--> KV-cache features
      +--> indexer/attention features
      |
      v
nostro dataset supervisionato
```

## Come lo userei

Questo sarebbe il corpus principale per i nostri esperimenti finali, perché permette di raccogliere i dati usando:

- la nostra build di DS4;
- il nostro GGUF;
- la nostra quantizzazione;
- CUDA;
- SSD streaming;
- la stessa configurazione di decoding usata nei benchmark finali.

### Link

**Pagina del dataset nel repository DS4:**

https://github.com/antirez/ds4/tree/main/gguf-tools/imatrix/dataset

**`prompts.jsonl`:**

https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/dataset/prompts.jsonl

**`rendered_prompts.txt`:**

https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/dataset/rendered_prompts.txt

**README del dataset:**

https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/dataset/README.md

**README dell'imatrix / procedura di calibrazione:**

https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/README.md

---

# 2. Doubleword/specdec-calibration

## Perché è estremamente utile

È il dataset già pronto più vicino al **primo esperimento sul predictor** perché contiene routing trace di DeepSeek-V4-Flash.

Per gli esperimenti di routing mette a disposizione principalmente:

```text
routing.npy
routing_meta.parquet
```

`routing.npy` contiene gli ID degli expert selezionati per posizione e layer, mentre `routing_meta.parquet` permette di ricostruire a quale request e posizione appartiene ogni osservazione.

La struttura documentata è concettualmente:

```text
routing.npy -> shape (N, L, k)

N = posizioni di routing
L = layer
k = expert routed per posizione
```

Il dataset contiene run di DeepSeek-V4-Flash su:

- HumanEval;
- SPEED-Bench;
- diverse categorie di workload.

## Perché ci serve

Possiamo usarlo **subito**, prima ancora di modificare DS4, per testare predictor basati solo sulla cronologia del routing.

Per esempio:

```text
X(t,l) = [E(t-3,l), E(t-2,l), E(t-1,l), E(t,l)]
Y(t,l) = E(t+1,l)
```

Possiamo confrontare:

- last-routing baseline;
- frequency baseline;
- Markov-1;
- Markov-N;
- piccolo MLP;
- GRU/LSTM;
- altri modelli sequenziali molto piccoli.

## Limiti

Non è stato raccolto con la nostra configurazione DS4 e la card indica sampling con:

```text
temperature = 0.6
```

Inoltre il dataset nasce nell'ambito di speculative decoding / MTP e non rappresenta automaticamente il comportamento della nostra build locale con decoding deterministico.

Per questo lo userei come:

1. **baseline iniziale**;
2. verifica che il routing contenga prevedibilità temporale;
3. eventuale pre-training/analisi preliminare;
4. confronto esterno con i trace raccolti da noi.

Non lo userei come unico dataset finale.

### Link diretto

https://huggingface.co/datasets/Doubleword/specdec-calibration

La pagina Hugging Face contiene anche le istruzioni `hf download` per scaricare l'intero repository o solo le slice necessarie.

---

# 3. DS4 quality-testing — DeepSeek V4 Flash official continuations

## Perché è importante

Nel repository DS4 esiste un dataset separato usato per il quality testing di DeepSeek-V4-Flash.

A differenza del corpus imatrix, qui vengono conservate **vere continuazioni ottenute dalla API ufficiale di DeepSeek**.

La struttura è del tipo:

```text
data/flash/
  prompts/
    case_*.txt
  continuations/
    case_*.txt
  responses/
    case_*.json
  manifest.tsv
```

La procedura ufficiale di DS4 usa **100 casi** per Flash.

I file `responses` possono includere anche le informazioni API richieste durante la raccolta, tra cui i `top_logprobs` quando configurati.

## Perché può essere utile al predictor

Da solo non contiene il routing, ma possiamo riprodurre questi prompt nel nostro DS4 instrumentato.

Questo crea un piccolo benchmark particolarmente interessante:

```text
prompt ufficiale
       |
       v
nostro DeepSeek-V4-Flash in DS4
       |
       +--> routing trace
       +--> internal features
       +--> continuazione DS4
       |
       v
confronto con continuazione ufficiale DeepSeek
```

Ci permette quindi di associare il dataset del predictor a un workload per il quale abbiamo anche una **reference ufficiale del comportamento del modello**.

## Limite

È molto più piccolo del corpus imatrix: è ottimo come:

- validation/test set di alta qualità;
- sanity check;
- studio di equivalenza DS4 ↔ API ufficiale;

ma non lo userei come unico training corpus del predictor.

### Link

**Cartella con i dati Flash:**

https://github.com/antirez/ds4/tree/main/gguf-tools/quality-testing/data/flash

**README della procedura di quality testing:**

https://github.com/antirez/ds4/blob/main/gguf-tools/quality-testing/README.md

**Script che raccoglie le continuazioni ufficiali:**

https://github.com/antirez/ds4/blob/main/gguf-tools/quality-testing/collect_official.py

---

# 4. 0xSero/deepseek-v4-flash-reap-observations-v2

## Cosa contiene

È un dataset di osservazioni usato per REAP e per analizzare l'importanza/copertura degli expert di DeepSeek-V4-Flash.

Una snapshot documentata e riutilizzata nei checkpoint REAP contiene:

```text
21.289 righe
```

con copertura di workload come:

- coding;
- math;
- science;
- tool calling;
- agentic tool trajectories.

La documentazione dei checkpoint che lo utilizzano indica che **tutti gli expert sono stati osservati almeno una volta** nella snapshot considerata.

## Perché può esserci utile

Non lo considererei il nostro dataset principale per la prediction temporale, ma può essere molto utile per:

- studiare la copertura degli expert;
- controllare che il nostro corpus non lasci expert quasi mai osservati;
- costruire statistiche di frequenza e salienza;
- scegliere workload aggiuntivi per riequilibrare il nostro training set;
- confrontare la distribuzione degli expert dei nostri trace con una raccolta indipendente.

In altre parole, può aiutarci a rispondere a:

> il dataset che stiamo raccogliendo in DS4 copre veramente l'intero spazio degli expert o è troppo concentrato su alcuni domini?

## Limiti per il nostro predictor

Non lo tratterei come equivalente a un file del tipo:

```text
token t -> routing
KV(t)
state(t)
token t+1 -> routing
```

Quindi la sua utilità principale è **complementare**, non sostitutiva del dataset temporale che vogliamo costruire.

### Link diretto

https://huggingface.co/datasets/0xSero/deepseek-v4-flash-reap-observations-v2

**Pagina di un checkpoint che documenta la snapshot v2 e la sua provenienza:**

https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP

---

# 5. 0xSero/deepseek-v4-flash-reap-observations-v1

## Perché lo teniamo in lista

È la versione precedente del dataset REAP per DeepSeek-V4-Flash.

La terrei soprattutto per:

- capire l'evoluzione del formato delle osservazioni;
- verificare quali informazioni vengono conservate;
- confrontare statistiche tra revisioni;
- avere una sorgente indipendente aggiuntiva per lo studio degli expert.

È meno importante della v2 perché quest'ultima è stata utilizzata in lavori/checkpoint successivi con una snapshot più ampia e documentata.

## Limiti

Come la v2, non la considero un sostituto del nostro dataset sequenziale DS4 token-per-token.

### Link diretto

https://huggingface.co/datasets/0xSero/deepseek-v4-flash-reap-observations-v1

---

# Strategia consigliata per costruire il nostro dataset finale

## Fase A — Validazione rapida dell'idea

Usare:

```text
Doubleword/specdec-calibration
```

per misurare quanto il routing passato riesca già a predire quello futuro.

Primi modelli:

```text
P0 = last expert-set
P1 = Markov-1
P2 = Markov-N
P3 = small MLP
P4 = GRU/LSTM piccola
```

Metriche iniziali:

- Recall@k degli expert reali;
- Precision@k;
- set overlap;
- hit rate potenziale della cache.

---

## Fase B — Costruzione del nostro vero dataset DS4

Usare principalmente:

```text
antirez/ds4/gguf-tools/imatrix/dataset/prompts.jsonl
```

come corpus di input.

Durante il decoding instrumentiamo DS4 e salviamo almeno:

```text
prompt_id
category
mode
source
sequence_id
token_position
token_id
layer
experts_t
router_scores_t
experts_t_plus_1
```

Poi aggiungiamo progressivamente:

```text
hidden_features_t
kv_features_t
indexer_features_t
attention_features_t
```

---

## Fase C — Dataset di controllo ad alta qualità

Usare:

```text
DS4 quality-testing/data/flash
```

come validation/test slice, perché per quei prompt abbiamo vere continuazioni provenienti dalla API ufficiale DeepSeek.

---

## Fase D — Controllo della copertura degli expert

Usare:

```text
0xSero REAP v2
```

per confrontare:

- frequenza di attivazione degli expert;
- copertura per layer;
- domini nei quali certi expert risultano più importanti;
- eventuali expert quasi assenti dal nostro corpus.

Se necessario, aggiungiamo nuovi prompt al corpus DS4 per ridurre lo sbilanciamento.

---

# Feature da raccogliere nel nostro dataset: stato semantico e computazionale

Questa è la parte centrale del dataset che vogliamo costruire. L'obiettivo non è limitarsi a salvare **quali expert sono stati scelti**, ma anche una fotografia compatta dello **stato interno del modello nel momento in cui quella scelta viene prodotta**.

La domanda che vogliamo poter studiare è:

$$
\boxed{
P(E_{t+1,l}\mid \text{routing passato},\ \text{router state},\ h_{t,l},\ \text{KV state},\ \text{indexer/attention state})
}
$$

Dove:

- $t$ = posizione/token corrente nel decoding;
- $l$ = layer corrente;
- $E_{t,l}$ = insieme degli expert selezionati al token $t$ nel layer $l$;
- $h_{t,l}$ = hidden state disponibile in quel punto del modello;
- `KV state` = feature estratte dalle rappresentazioni Key/Value già disponibili nel decoding;
- `indexer/attention state` = informazione su quali parti del contesto passato l'attention sta considerando rilevanti.

L'idea è confrontare progressivamente predictor sempre più ricchi e verificare se le feature semantico/computazionali aggiungono davvero informazione rispetto al semplice routing history.

---

## 1. `routing_history`

### Cosa rappresenta

È la cronologia degli expert selezionati nei token precedenti, per uno specifico layer.

Esempio:

```text
token t-3 -> [12, 31, 55, 81, 144, 203]
token t-2 -> [12, 31, 55, 90, 144, 211]
token t-1 -> [12, 44, 55, 90, 144, 211]
token t   -> [12, 44, 55, 90, 144, 203]
```

Formalmente, con una finestra di lunghezza $H$:

$$
E_{t-H+1,l},\ldots,E_{t,l}
$$

### Perché è importante

È la baseline più naturale perché il routing MoE può avere **località temporale**: expert usati recentemente possono avere una probabilità maggiore di essere riutilizzati subito dopo.

Questa feature ci permette di rispondere alla prima domanda sperimentale:

> Quanto funziona il trace precedente come predictor?

È inoltre la feature più economica da raccogliere e usare.

### Come salvarla

Possiamo salvare gli ID degli expert per gli ultimi $H$ token:

```json
"routing_history": [
  [12, 31, 55, 81, 144, 203],
  [12, 31, 55, 90, 144, 211],
  [12, 44, 55, 90, 144, 211],
  [12, 44, 55, 90, 144, 203]
]
```

Conviene mantenere sia:

- l'**insieme** degli expert selezionati;
- se DS4 lo conserva, anche l'**ordine/ranking** con cui il router li ha selezionati.

L'ordine potrebbe infatti contenere informazione aggiuntiva rispetto al solo set.

---

## 2. `router_scores` / `router_logits`

### Cosa rappresentano

Il router non produce solamente gli ID degli expert scelti. Prima della selezione Top-k produce degli **score** per gli expert candidati.

Concettualmente:

$$
h_{t,l}
\rightarrow
\text{Router}_l
\rightarrow
[r_1,r_2,\ldots,r_{N_e}]
\rightarrow
\text{Top-k expert}
$$

Dove $N_e$ è il numero di routed expert del layer.

Gli `expert_ids` ci dicono solamente **chi ha vinto**. Gli score ci dicono invece **con quanta convinzione** e quali altri expert erano vicini alla soglia.

Esempio concettuale:

```text
Expert 12 -> 0.91
Expert 44 -> 0.87
Expert 55 -> 0.84
Expert 90 -> 0.81
Expert 144 -> 0.79
Expert 203 -> 0.76
Expert 31 -> 0.75   <- quasi selezionato
Expert 211 -> 0.74  <- quasi selezionato
```

### Perché è importante

Due token possono selezionare lo stesso set di expert ma trovarsi in stati del router molto diversi.

Caso A:

```text
expert scelti nettamente sopra tutti gli altri
```

Caso B:

```text
expert scelti con score quasi identici agli expert esclusi
```

Nel secondo caso il routing del token successivo potrebbe essere molto più instabile.

Gli score possono quindi fornire al predictor informazioni su:

- **confidenza del routing corrente**;
- expert quasi selezionati;
- possibili cambiamenti imminenti del routing;
- continuità o discontinuità dello stato del router.

### Come salvarli

Possibili opzioni, in ordine di ricchezza:

1. tutti gli score dei routed expert;
2. Top-k selezionati + un certo numero di expert appena esclusi;
3. score solo degli expert selezionati;
4. statistiche compresse, per esempio margine tra ultimo selezionato e primo escluso.

Per il primo dataset, se lo spazio lo consente, sarebbe meglio conservare il segnale più ricco possibile e comprimere successivamente offline.

---

## 3. `hidden_features`

### Cosa rappresentano

L'hidden state $h_{t,l}$ è la rappresentazione interna del token corrente nel layer $l$.

È molto più ricco del semplice token ID perché incorpora il **contesto elaborato fino a quel punto**.

Concettualmente:

```text
token t
   |
   v
layer precedenti
   |
   v
hidden state h(t,l)
   |
   +----> attention / altre trasformazioni
   |
   +----> router MoE
```

Il router decide gli expert proprio a partire da una rappresentazione interna di questo tipo. Per questo l'hidden state è probabilmente una delle feature più direttamente correlate alla scelta degli expert.

### Perché è importante

Il routing history dice:

> quali expert sono stati scelti prima?

L'hidden state può invece dire:

> che tipo di informazione semantica/computazionale sta elaborando il modello adesso?

Due token con lo stesso routing passato possono avere hidden state differenti e quindi richiedere expert differenti al passo successivo.

Questa feature è quindi fondamentale per verificare se il predictor può andare oltre la semplice località temporale.

### Perché confrontarlo con la KV cache

$K$ e $V$ sono proiezioni specializzate per l'attention. L'hidden state è invece una rappresentazione più generale.

Per questo conviene fare un'ablation del tipo:

```text
routing only
routing + hidden
routing + KV
routing + hidden + KV
```

Se `routing + hidden + KV` migliora rispetto a `routing + hidden`, avremo evidenza che la KV cache contiene informazione predittiva incrementale e non solo una trasformazione ridondante dello stesso stato.

### Come salvarlo

Salvare l'intero hidden state per ogni token e layer potrebbe produrre un dataset enorme.

Meglio prevedere due modalità:

- **raw** per una piccola subset di dati, utile per analisi esplorative;
- **compressed/projection features** per il corpus principale.

Per esempio:

$$
h_{t,l}\in\mathbb{R}^{d}
\rightarrow
z_h\in\mathbb{R}^{16\text{-}64}
$$

La proiezione deve essere molto economica anche in prospettiva del predictor online.

---

## 4. `kv_features_current`: informazioni del token corrente dalla KV cache

### Cosa rappresentano

Durante il decoding, per ogni nuovo token il blocco di attention costruisce nuove rappresentazioni Key e Value e le aggiunge allo stato di cache utilizzato dai token successivi.

Nel caso classico possiamo pensare a:

$$
K_t = h_tW_K
$$

$$
V_t = h_tW_V
$$

Quindi $K_t$ e $V_t$ sono trasformazioni contestualizzate dello stato corrente, specializzate per l'attention.

Nel codice reale di DeepSeek-V4-Flash/DS4 l'attention può usare rappresentazioni compresse e strutture specifiche invece di una KV cache classica pura; per il dataset dovremo quindi mappare questi concetti sui **tensori effettivamente materializzati da DS4**.

### Significato intuitivo di K

La Key descrive, in modo molto semplificato, **come il token corrente può essere riconosciuto/confrontato dalle query di attention future**.

È quindi una rappresentazione utile per stabilire la rilevanza del token all'interno del contesto.

### Significato intuitivo di V

La Value rappresenta il **contenuto informativo che può essere recuperato** quando l'attention assegna peso a quella posizione.

K e V hanno quindi ruoli differenti:

```text
K -> "quando questa posizione è rilevante?"
V -> "quale informazione porta questa posizione?"
```

### Perché possono aiutare il predictor

Il routing futuro può dipendere dal contenuto semantico che il modello sta elaborando.

Le feature K/V possono catturare una parte di questo stato con una prospettiva diversa dall'hidden state:

- K descrive la collocazione del token nello spazio usato per il matching dell'attention;
- V descrive l'informazione che il token rende disponibile al contesto futuro.

La domanda di ricerca diventa:

> le rappresentazioni interne dell'attention al token $t$ contengono informazione utile a prevedere gli expert del token $t+1$?

### Vincolo temporale fondamentale

Per predire gli expert del token $t+1$ possiamo usare solo informazioni già disponibili al token $t$:

$$
K_{\leq t},V_{\leq t}
$$

Non possiamo usare:

$$
K_{t+1},V_{t+1}
$$

perché a quel punto il token successivo è già in elaborazione e il prefetch arriverebbe troppo tardi.

### Come salvarle

Non conviene salvare necessariamente tutti i tensori K/V grezzi.

Possiamo salvare una proiezione compatta:

$$
z_{KV,t,l}=f(K_{t,l},V_{t,l})
$$

con poche decine di dimensioni.

---

## 5. `kv_history_features`: feature ricavate dalla storia della KV cache

### Cosa rappresentano

La KV cache non contiene solo l'ultimo token: rappresenta il contesto già elaborato.

Usare tutta la cache come input del predictor sarebbe però troppo costoso.

L'idea è quindi estrarre **statistiche compatte** che riassumano come lo stato corrente differisce dalla storia recente.

Tra le feature che abbiamo discusso:

$$
\|K_t\|
$$

$$
\|V_t\|
$$

$$
\cos(K_t,K_{t-1})
$$

$$
\cos(V_t,V_{t-1})
$$

Possiamo inoltre costruire la stessa famiglia di statistiche su una breve finestra recente, evitando di fornire al predictor l'intera cache.

### Cosa ci dicono

- `||K_t||`: intensità/energia della rappresentazione Key corrente;
- `||V_t||`: intensità/energia della Value corrente;
- `cos(K_t, K_{t-1})`: quanto il token corrente è simile al precedente nello spazio delle Key;
- `cos(V_t, V_{t-1})`: quanto il contenuto Value corrente è simile a quello precedente.

Una variazione brusca di queste quantità potrebbe segnalare un cambiamento del tipo di informazione elaborata e quindi anticipare un cambio di expert.

### Perché sono interessanti

Consentono di catturare **dinamica semantica** senza salvare migliaia di valori della cache.

Per esempio:

```text
routing quasi invariato
ma K/V cambiano bruscamente
        |
        v
possibile cambio di expert al prossimo token
```

Questa è esattamente una situazione che un predictor basato solo sul routing potrebbe non vedere.

---

## 6. `indexer_features`

### Cosa rappresentano

DeepSeek-V4-Flash/DS4 usa meccanismi di attention compressa/selettiva in cui una componente di **indexing/selezione** può individuare quali porzioni della storia compressa sono più rilevanti per il token corrente.

Concettualmente:

```text
KV history compressa
        |
        v
     indexer
        |
        +--> posizione/row A
        +--> posizione/row B
        +--> posizione/row C
        +--> ...
```

Queste selezioni indicano **quali parti del contesto passato il modello sta cercando di recuperare in quel momento**.

### Perché sono importanti

Sono potenzialmente una feature molto forte perché non descrivono solo il token corrente, ma anche **quale parte della memoria contestuale è diventata rilevante per esso**.

Due token con hidden state simili potrebbero interrogare regioni molto diverse del contesto precedente.

Quindi l'indexer potrebbe aggiungere informazione del tipo:

> il modello sta tornando a parlare di una parte di contesto associata a codice, matematica, un tool result, una vecchia istruzione, ecc.

Questa variazione del contesto recuperato potrebbe anticipare un cambiamento nel routing MoE.

### Come salvarle

Possibili feature:

- ID/indici delle righe selezionate;
- ranking delle righe selezionate;
- score associati alla selezione, se disponibili;
- overlap fra selezioni al token $t$ e $t-1$;
- rappresentazione compressa dell'insieme selezionato.

Per evitare dipendenza eccessiva da indici assoluti, potrebbe essere utile derivare anche feature relative, come il grado di overlap tra i set selezionati consecutivamente.

---

## 7. `attention_features`

### Cosa rappresentano

Questa categoria raccoglie feature riassuntive dello stato dell'attention che non coincidono necessariamente con K/V o con l'indexer.

L'obiettivo non è salvare matrici di attention complete, ma estrarre segnali economici che dicano **quanto e dove il modello sta concentrando l'attenzione**.

Nel nostro progetto questa categoria va considerata opzionale e dipendente da ciò che DS4 materializza senza overhead eccessivo.

### Perché potrebbero aiutare

Il routing MoE viene eseguito dopo che il token è stato trasformato dai blocchi precedenti del layer. Se l'attention modifica in modo significativo la rappresentazione del token, un riassunto del suo comportamento può anticipare il tipo di expert richiesto dalla successiva FFN/MoE.

È quindi una famiglia di feature da testare solo dopo routing, router scores, hidden e KV/indexer, che sono più direttamente motivate.

---

## 8. `token_id`, `token_position`, `layer` e metadati del prompt

Queste non sono feature semantiche interne in senso stretto, ma sono indispensabili per interpretare correttamente il dataset.

### `token_id`

Identifica il token corrente.

Può essere usato come baseline per capire quanto del routing sia prevedibile semplicemente dal token lessicale rispetto allo stato contestualizzato.

### `token_position`

Indica la posizione nel decoding.

È utile per distinguere l'inizio della generazione da fasi più avanzate e per studiare se la prevedibilità cambia con la lunghezza del contesto.

### `layer`

È indispensabile perché la distribuzione degli expert e il significato delle rappresentazioni interne cambiano tra layer.

### `category`, `mode`, `source`

Provengono soprattutto dal corpus DS4 di Sanfilippo.

Ci permettono di fare split e analisi per:

- coding;
- agentic/tool use;
- reasoning;
- language;
- translation;
- think/nothink;
- long context.

Questi campi sono importanti per verificare che il predictor non impari semplicemente pattern specifici di un dominio.

---

## 9. Target principale: `target_experts_next_token`

### Cosa rappresenta

È la label principale del predictor token-ahead:

$$
Y_{t,l}=E_{t+1,l}
$$

Il record associato allo stato del token $t$ contiene quindi gli expert realmente usati al token successivo nello stesso layer.

### Perché è la label corretta per il prefetch token-ahead

Se al token $t$ riusciamo a predire gli expert di $t+1$, possiamo iniziare il caricamento mentre il modello sta ancora completando il lavoro corrente.

```text
stato al token t
      |
      v
  predictor
      |
      v
expert previsti per t+1
      |
      v
prefetch SSD/RAM -> VRAM
```

---

## 10. Target alternativo: prediction layer-ahead

Oltre alla prediction token-ahead, abbiamo discusso una seconda strategia:

$$
(l,t)\rightarrow(l+k,t)
$$

cioè usare lo stato disponibile in un layer precedente dello **stesso token** per predire gli expert richiesti da un layer successivo.

Esempio:

```text
token t

Layer 10
  |
  +--> hidden/KV/indexer features
             |
             v
         predictor
             |
             v
    expert previsti Layer 15
             |
             v
         prefetch

Layer 11
Layer 12
Layer 13
Layer 14
Layer 15 -> expert già pronto
```

Per supportare anche questo esperimento il dataset dovrebbe poter generare label del tipo:

```text
"target_layer_offset": 5
"target_experts_same_token": [...]
```

### Perché è importante

Il layer-ahead predictor potrebbe avere **più tempo reale per nascondere l'I/O** rispetto alla sola prediction del token successivo.

Per questo vale la pena progettare il dataset in modo da non escludere questa seconda formulazione.

---

## 11. Feature operative del runtime: utili per il prefetch, ma da separare dal predictor semantico

Lo stato del sistema può includere:

```text
expert già in VRAM?
expert in RAM?
expert solo su SSD?
prefetch già in corso?
```

Queste feature sono molto importanti per decidere **quale expert conviene prefetchare**, ma non descrivono quale expert il modello sceglierà.

Per questo conviene separare due problemi:

### Predictor di routing

$$
\text{stato semantico/computazionale}
\rightarrow
P(E_{future})
$$

### Policy di prefetch

$$
P(E_{future})
+
\text{stato VRAM/RAM/SSD}
\rightarrow
\text{azione di prefetch}
$$

Questa separazione è importante per non confondere l'accuratezza del predictor con l'efficacia della policy di caching.

---

# Ablation study consigliata sulle feature

La maniera più pulita di capire quali informazioni aggiungano davvero potere predittivo è addestrare una serie incrementale di modelli.

| Predictor | Feature disponibili | Domanda a cui risponde |
|---|---|---|
| **P0** | routing corrente | Quanto vale la semplice località? |
| **P1** | routing history | Quanto aiuta una finestra temporale? |
| **P2** | P1 + router scores | La confidenza/competizione del router anticipa cambi futuri? |
| **P3** | P2 + hidden features | Lo stato interno generale aggiunge informazione? |
| **P4** | P2 + KV current features | K/V contengono segnale predittivo utile senza hidden? |
| **P5** | P3 + KV current/history | La KV aggiunge informazione oltre l'hidden state? |
| **P6** | P5 + indexer features | Quali parti del contesto recuperate aiutano la prediction? |
| **P7** | P6 + attention features | Ulteriori segnali dell'attention migliorano ancora? |

La comparazione più importante è:

$$
\text{routing + hidden}
\quad\text{vs}\quad
\text{routing + hidden + KV}
$$

Se il secondo modello migliora in modo consistente, allora avremo evidenza che **la KV cache contiene informazione incrementale sul routing futuro**.

---

# Principio fondamentale: feature utili ma economiche

Il predictor deve accelerare DS4, non diventare un nuovo collo di bottiglia.

Per questo non basta chiedersi:

> questa feature aumenta Recall@k?

Bisogna chiedersi:

> aumenta Recall@k abbastanza da compensare il costo per estrarla, copiarla e processarla?

Il percorso ideale è quindi:

```text
segnale interno già disponibile in DS4
              |
              v
feature extraction molto economica
              |
              v
rappresentazione compatta
              |
              v
piccolo predictor
              |
              v
prefetch anticipato
```

Le feature raw possono essere raccolte su una subset offline per capire cosa funziona; il predictor finale dovrebbe invece usare solo feature che possano essere ottenute con overhead molto basso.

---

# Schema consigliato del nostro record di training

Una struttura più completa, capace di supportare gli ablation study descritti sopra, potrebbe essere:

```json
{
  "prompt_id": "...",
  "category": "source",
  "mode": "nothink",
  "source": "ds4.c:14",

  "sequence_id": 27,
  "token_position": 413,
  "token_id": 12345,
  "layer": 17,

  "routing_history": [
    [12, 31, 55, 81, 144, 203],
    [12, 31, 55, 90, 144, 211],
    [12, 44, 55, 90, 144, 211],
    [12, 44, 55, 90, 144, 203]
  ],

  "router_scores": [],
  "hidden_features": [],

  "kv_features_current": {
    "k_projection": [],
    "v_projection": [],
    "k_norm": 0.0,
    "v_norm": 0.0,
    "k_cos_prev": 0.0,
    "v_cos_prev": 0.0
  },

  "kv_history_features": [],

  "indexer_features": {
    "selected_rows": [],
    "selected_scores": [],
    "overlap_with_previous": 0.0
  },

  "attention_features": [],

  "target_experts_next_token": [12, 44, 55, 90, 144, 211],

  "target_layer_offset": 5,
  "target_experts_same_token": []
}
```

Questo è uno **schema concettuale**, non necessariamente il formato fisico finale. Per dataset molto grandi potrebbe essere preferibile usare Parquet/Arrow o array binari separati per le feature dense.

## Versione minima consigliata per il primo tracer DS4

Per evitare di complicare subito l'instrumentazione, partirei da:

```text
prompt_id
category
mode
sequence_id
token_position
token_id
layer
experts_t
routing_history
router_scores_t
target_experts_next_token
```

Poi aggiungerei in quest'ordine:

```text
1. hidden_features
2. kv_features_current
3. kv_history_features
4. indexer_features
5. attention_features
```

In questo modo ogni nuova famiglia di feature può essere valutata con un ablation study separato.

# Ordine pratico con cui li userei

## Per iniziare domani senza modificare DS4

1. **Doubleword/specdec-calibration**
2. creare baseline di prediction del routing;
3. verificare Recall@k e prevedibilità temporale.

## Per costruire il dataset realmente utile al progetto

1. **DS4 Imatrix Calibration Dataset di Sanfilippo** come corpus;
2. instrumentazione di DS4;
3. raccolta dei nostri routing trace;
4. aggiunta router score;
5. aggiunta hidden feature;
6. aggiunta KV/indexer feature;
7. ablation study.

## Per validazione e controllo

1. **DS4 official Flash continuations**;
2. **REAP v2**;
3. **REAP v1** come riferimento secondario.

---

# Conclusione

Il punto fondamentale è che **non esiste un singolo dataset pubblico che contenga già tutto ciò che vogliamo**:

```text
routing history
+ router scores
+ hidden state
+ KV-cache features
+ indexer features
+ future expert labels
```

Per questo la strategia migliore è combinare risorse differenti:

```text
                    RISORSE PUBBLICHE
                           |
          +----------------+----------------+
          |                |                |
      DS4 corpus       Doubleword       REAP / quality
          |                |                |
          v                v                v
  workload realistico   routing prior    coverage / validation
          |                |                |
          +----------------+----------------+
                           |
                           v
                  DS4 instrumentato
                           |
                           v
              NOSTRO DATASET TEMPORALE
                           |
                           v
                    predictor MoE
```

La risorsa più strategica per il **dataset finale** è quindi il corpus di Sanfilippo, perché possiamo trasformarlo direttamente in un dataset ricco eseguendolo nella stessa implementazione DS4 sulla quale misureremo poi il beneficio reale del prefetch.

La risorsa più strategica per il **primo esperimento immediato** è invece `Doubleword/specdec-calibration`, perché contiene già routing trace di DeepSeek-V4-Flash e ci permette di validare il concetto prima di modificare il runtime.
