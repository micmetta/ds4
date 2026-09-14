# Prefill, KV Cache e Predictor degli Expert in DS4

## Obiettivo del documento

Questo documento raccoglie e organizza **tutto il percorso concettuale discusso nella conversazione** a partire dalla domanda:

> **“Cos’è il prefill per un LLM?”**

fino alla domanda:

> **“Ma esistono dataset già pronti per quello che vogliamo fare noi su DeepSeek-V4-Flash?”**

Il focus è quindi su:

- prefill e decode;
- cosa accade realmente nel prefill;
- rapporto tra prefill e KV cache;
- implementazione della KV cache in DS4;
- differenza tra KV cache di inferenza e KV cache persistente su disco;
- possibilità di usare lo stato interno del modello come feature del predictor;
- hidden state, router scores, KV cache e indexer;
- predizione token-ahead e layer-ahead;
- lavori di ricerca già esistenti che usano stati computazionali interni per prevedere gli expert;
- spazio di ricerca ancora interessante per un predictor specifico per DS4 / DeepSeek-V4-Flash.

---

# 1. Cos’è il prefill per un LLM?

Il **prefill** è la fase in cui un LLM elabora **tutti i token già presenti nel prompt e nel contesto** prima di iniziare a generare nuovi token.

Esempio:

```text
Prompt:
"La capitale della Francia è"
```

Prima di generare:

```text
"Parigi"
```

il modello deve processare tutti i token del prompt. Questa fase è il **prefill**.

Dopo il prefill inizia il **decode**, cioè la fase autoregressiva in cui il modello genera un nuovo token alla volta.

## 1.1 Prefill vs Decode

| Fase | Cosa fa | Caratteristica principale |
|---|---|---|
| Prefill | Elabora il prompt già disponibile | molti token elaborati insieme |
| Decode | Genera la continuazione | un token nuovo alla volta |
| Decode con KV cache | Riusa K e V precedenti | evita di ricalcolare l’intero passato |

Con un prompt di 10.000 token, concettualmente:

```text
10.000 token
      |
      v
   PREFILL
      |
      v
costruzione dello stato interno
e della KV cache
      |
      v
primo token generato
```

Poi:

```text
token 10001
token 10002
token 10003
...
```

vengono prodotti in decode.

## 1.2 Perché il prefill è importante per i MoE

In un modello Mixture-of-Experts, ogni token può essere instradato verso expert differenti.

Durante il prefill, se sono presenti migliaia di token, il router può dover prendere moltissime decisioni quasi contemporaneamente.

Durante il decode, invece, la computazione è molto più sequenziale:

```text
token t
   |
   v
routing
   |
   v
expert
   |
   v
token t+1
```

Questa sequenzialità è proprio ciò che rende interessante la previsione degli expert futuri e il loro prefetch.

---

# 2. Cosa accade realmente durante il prefill

Supponiamo di avere:

```text
[t1, t2, t3, t4, t5]
```

I token passano attraverso:

```text
token IDs
   |
   v
embedding
   |
   v
Transformer layer 1
   |
   v
Transformer layer 2
   |
   v
...
   |
   v
Transformer layer N
```

La differenza fondamentale rispetto al decode è che, durante il prefill, **molti token vengono processati insieme sotto forma di matrici**.

## 2.1 Embedding e informazione posizionale

I token vengono trasformati in rappresentazioni numeriche.

Con hidden size `d`, l’input può essere visto come una matrice:

\[
X \in \mathbb{R}^{T \times d}
\]

dove:

- `T` = numero di token;
- `d` = hidden size.

Nei modelli moderni l’informazione posizionale può essere introdotta con meccanismi come RoPE.

## 2.2 Calcolo di Q, K e V

Per ogni layer di attention vengono calcolate le proiezioni:

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

Durante il prefill queste operazioni vengono eseguite per tutti i token disponibili.

## 2.3 Self-attention causale

Nel decoder autoregressivo ogni token può guardare solamente se stesso e i token precedenti.

Con 5 token:

```text
t1 -> t1
t2 -> t1,t2
t3 -> t1,t2,t3
t4 -> t1,t2,t3,t4
t5 -> t1,t2,t3,t4,t5
```

La causal mask impedisce a una posizione di utilizzare informazione proveniente dai token futuri.

La GPU, tuttavia, non esegue cinque inferenze indipendenti: sfrutta operazioni matriciali altamente parallelizzabili.

---

# 3. Il ruolo della KV cache durante il prefill

Uno degli output più importanti del prefill è la costruzione della **KV cache**.

Per ogni layer vengono conservati Key e Value dei token del contesto.

Concettualmente:

```text
Layer 1:
K(t1), K(t2), ..., K(tT)
V(t1), V(t2), ..., V(tT)

Layer 2:
K(t1), K(t2), ..., K(tT)
V(t1), V(t2), ..., V(tT)

...

Layer N
```

Questo evita che, durante il decode, il modello debba ricalcolare da zero K e V di tutto il contesto precedente.

## 3.1 Cosa succede nel decode

Al passo `t`, vengono calcolati i nuovi:

\[
K_t
\]

e:

\[
V_t
\]

che vengono aggiunti alla cache precedente:

\[
K_{\text{cache}}^{(t)}
=
[K_{\text{cache}}^{(t-1)};K_t]
\]

\[
V_{\text{cache}}^{(t)}
=
[V_{\text{cache}}^{(t-1)};V_t]
\]

Il nuovo token può quindi effettuare attention sul passato senza ricalcolare tutte le proiezioni K/V precedenti.

---

# 4. Passaggio attraverso MLP / MoE durante il prefill

Dopo l’attention, ogni token attraversa la parte feed-forward del layer.

In un Transformer dense:

```text
hidden state
    |
    v
   MLP
    |
    v
nuovo hidden state
```

In un MoE:

```text
hidden state
    |
    v
  Router
    |
    +--> Expert A
    +--> Expert B
    +--> ...
```

Il router decide quali expert utilizzare.

Con migliaia di token nel prefill:

```text
token 1     -> experts ...
token 2     -> experts ...
token 3     -> experts ...
...
token 10000 -> experts ...
```

Questo può portare all’attivazione di un insieme molto ampio di expert.

---

# 5. Perché prefill e decode hanno caratteristiche hardware diverse

Il prefill lavora su matrici grandi:

```text
[T x hidden_size]
```

ed è quindi molto adatto a sfruttare il parallelismo della GPU.

Il decode lavora invece su un numero estremamente ridotto di nuovi token per volta.

Per questo, semplificando:

```text
Prefill -> più compute-oriented

Decode  -> molto più sensibile a memory bandwidth,
           cache miss e trasferimenti di pesi
```

Questa distinzione è centrale per il nostro problema di prefetch degli expert.

---

# 6. Perché il predictor è particolarmente interessante nel decode

Nel decode abbiamo una sequenza temporale naturale:

```text
token t
   |
   v
expert usati al tempo t
   |
   v
predictor
   |
   v
expert probabilmente necessari a t+1
   |
   v
prefetch
SSD/RAM --------> VRAM
```

Quindi il predictor può cercare di anticipare un accesso futuro mentre la computazione corrente è ancora in corso.

L’obiettivo non è sostituire la KV cache.

Le due ottimizzazioni risolvono problemi differenti:

```text
KV cache
-> evita di ricalcolare stati di attention passati

Expert predictor
-> prova a evitare/nascondere la latenza di caricamento degli expert
```

Sono quindi complementari.

---

# 7. DS4 implementa già la KV cache?

Sì.

DS4 implementa la KV cache e, per DeepSeek-V4-Flash, non si limita a una classica KV cache completa per tutti i token.

Secondo il model card ufficiale di DS4, ogni layer mantiene una **raw sliding-window KV cache per gli ultimi 128 token**.

Per il contesto più vecchio vengono utilizzate forme di KV compresse.

Schema semplificato:

```text
vecchio contesto                     contesto recente

+--------------------------+       +----------------------+
| compressed KV history    |       | raw KV              |
|                          |       | ultimi 128 token     |
+--------------------------+       +----------------------+
            |                             |
            +--------------+--------------+
                           |
                           v
                        Attention
```

Fonte DS4:
https://github.com/antirez/ds4/blob/main/MODEL_CARD.md

## 7.1 Struttura della KV cache compressa di DeepSeek-V4-Flash in DS4

Nel DS4 attuale:

- layer 0 e 1: raw sliding window;
- layer compressi con rapporto 4:1: compressed KV + indexer;
- layer compressi con rapporto 128:1: compressed KV molto più aggressiva;
- nei layer selettivi ratio-4 l’indexer può selezionare fino a 512 righe compresse rilevanti.

Il model card riporta inoltre:

- 43 layer;
- raw window = 128 token;
- indexer top-k = 512.

Questo consente di supportare contesti molto lunghi senza utilizzare una KV cache standard completa per ogni token e ogni layer.

Fonte:
https://github.com/antirez/ds4/blob/main/MODEL_CARD.md

---

# 8. KV cache di inferenza vs disk KV cache in DS4

È importante distinguere due meccanismi.

## 8.1 KV cache attiva di inferenza

È lo stato K/V usato direttamente durante prefill e decode.

Serve a evitare il ricalcolo del passato.

## 8.2 KV cache persistente su disco

DS4 supporta anche checkpoint della sessione/KV cache su disco, per esempio attraverso:

```bash
--kv-disk-dir /tmp/ds4-kv
```

Questa cache serve a poter riutilizzare un contesto già elaborato anche dopo che una sessione è stata sostituita o il server è stato riavviato.

Schema:

```text
                  KV CACHE
                     |
          +----------+----------+
          |                     |
          v                     v
 KV di inferenza           Disk KV cache
          |                     |
          v                     v
accelerazione decode       persistenza /
corrente                   ripristino
```

DS4 salva nello stato persistente anche informazioni specifiche della sua architettura, tra cui righe raw KV, compressed KV e stato dell’indexer nei layer che lo utilizzano.

Fonti:

https://github.com/antirez/ds4/blob/main/README.md

https://github.com/antirez/ds4/blob/main/ds4_kvstore.c

---

# 9. Possiamo sfruttare la KV cache come feature del predictor?

Sì, ed è una delle idee più interessanti discusse.

La KV cache non contiene solamente identificatori di token.

K e V derivano da rappresentazioni interne contestualizzate:

\[
K_t = h_t W_K
\]

\[
V_t = h_t W_V
\]

dove `h_t` rappresenta uno stato interno del modello.

Quindi K e V incorporano informazione sullo **stato semantico/computazionale corrente**.

---

# 10. Predictor basato solo sul routing

Un baseline può essere:

\[
P(E_{t+1}\mid E_t,E_{t-1},...)
\]

dove:

\[
E_t
\]

è l’insieme degli expert usati al token `t`.

Esempio:

```text
E(t-2), E(t-1), E(t)
          |
          v
      Predictor
          |
          v
      E(t+1)
```

Questo sfrutta temporal locality e regolarità del routing.

---

# 11. Predictor con stato semantico/computazionale

Possiamo aggiungere informazioni interne:

\[
P(E_{t+1}
\mid
E_{\le t},
K_t,
V_t,
...)
\]

Schema:

```text
routing history ----------+
                          |
K/V correnti -------------+--> Predictor --> expert futuri
                          |
altre feature ------------+
```

L’obiettivo è permettere al predictor di sapere non soltanto:

> “dopo questi expert ne vengono spesso altri”

ma anche:

> “dato lo stato interno che il modello sta elaborando in questo momento, quali expert è probabile che serviranno dopo?”

---

# 12. Perché NON usare tutta la KV cache grezza

La KV cache può essere enorme.

Se fossimo al token 5000, avremmo informazione relativa a moltissimi token e layer.

Usare direttamente tutta la cache come input del predictor potrebbe rendere il predictor troppo costoso.

Questo sarebbe controproducente:

```text
risparmio ottenuto dal prefetch
          <
overhead necessario per calcolare la prediction
```

Il predictor deve rimanere estremamente leggero.

---

# 13. Feature compatte derivate dalla KV cache

L’idea proposta è estrarre feature molto più piccole.

Per esempio:

\[
K_t,\;V_t
\]

oppure una breve finestra:

\[
K_t,V_t,K_{t-1},V_{t-1}
\]

oppure statistiche:

\[
\|K_t\|
\]

\[
\|V_t\|
\]

\[
\cos(K_t,K_{t-1})
\]

\[
\cos(V_t,V_{t-1})
\]

Queste feature possono indicare:

- quanto cambia lo stato interno da un token al successivo;
- se il modello sta continuando nello stesso regime semantico;
- se si verifica una transizione;
- quanto la rappresentazione corrente differisce dal passato recente.

Una proiezione compatta potrebbe ridurre centinaia o migliaia di valori a poche decine di feature.

---

# 14. Hidden state: potrebbe essere ancora migliore di K/V

È emersa una distinzione importante.

Il router MoE prende la propria decisione a partire da una rappresentazione interna del token:

```text
hidden state
     |
     v
   Router
     |
     v
expert selezionati
```

Quindi l’hidden state è **più direttamente collegato al routing** rispetto a K e V.

K e V sono proiezioni specializzate per l’attention:

\[
h \rightarrow W_K \rightarrow K
\]

\[
h \rightarrow W_V \rightarrow V
\]

Per questo abbiamo proposto di confrontare:

### A. Routing history

```text
E(t-2), E(t-1), E(t)
          |
          v
      predictor
```

### B. Routing + KV

```text
routing history --+
                  +--> predictor
K_t,V_t ----------+
```

### C. Routing + hidden state

```text
routing history --+
                  +--> predictor
h_t --------------+
```

L’ipotesi da verificare sperimentalmente è che l’hidden state possa essere più informativo della KV cache.

---

# 15. Router scores / router logits

Oltre all’ID degli expert selezionati, è molto interessante salvare anche gli score prodotti dal router.

Per esempio, invece di sapere solamente:

```text
expert selezionati:
[7, 12, 31, 80, ...]
```

potremmo conoscere:

```text
expert 7  -> 0.94
expert 12 -> 0.87
expert 31 -> 0.75
expert 80 -> 0.71
...
```

Questo rivela non soltanto la decisione finale, ma anche **quanto il router fosse sicuro**.

Due token possono scegliere gli stessi expert ma avere distribuzioni di score molto diverse.

Questa informazione potrebbe anticipare una futura transizione di routing.

---

# 16. Problema temporale: evitare feature che arrivano troppo tardi

Se vogliamo predire gli expert del token:

\[
t+1
\]

non possiamo usare:

\[
K_{t+1},V_{t+1}
\]

perché questi valori esistono solamente quando abbiamo già iniziato a elaborare `t+1`.

Per un vero prefetch token-ahead dobbiamo usare solo informazioni disponibili fino a `t`:

\[
K_{\le t},V_{\le t}
\]

Quindi:

\[
(K_t,V_t)
\rightarrow
Predictor
\rightarrow
E_{t+1}
\]

è valido.

---

# 17. Predizione token-ahead

La prima strategia è:

\[
(t,l)\rightarrow(t+1,l)
\]

ossia usare lo stato del token corrente per prevedere gli expert che saranno necessari allo stesso layer al prossimo token.

Schema:

```text
token t
   |
   v
routing / hidden / KV
   |
   v
predictor
   |
   v
expert di token t+1
```

Questo sfrutta la sequenzialità autoregressiva del decode.

---

# 18. Predizione layer-ahead

È stata discussa anche una seconda possibilità.

Durante l’elaborazione di un singolo token, il modello attraversa i layer in ordine:

```text
Layer 0
Layer 1
Layer 2
...
Layer 42
```

Possiamo quindi utilizzare informazioni disponibili a un layer precedente per predire gli expert di un layer successivo dello **stesso token**.

Formalmente:

\[
(l,t)\rightarrow(l+k,t)
\]

Esempio:

```text
token t

Layer 10
  |
  +--> feature disponibili
          |
          v
      predictor
          |
          v
"al layer 15 serviranno probabilmente
 gli expert X,Y,Z"
          |
          v
       prefetch

Layer 11
Layer 12
Layer 13
Layer 14

Layer 15
 -> expert già disponibile
```

Questa strategia potrebbe offrire una finestra temporale più ampia per nascondere il trasferimento SSD/RAM -> VRAM.

---

# 19. Feature dell’indexer di DeepSeek-V4-Flash

DeepSeek-V4-Flash / DS4 utilizza, in alcuni layer compressi, un **indexer** per selezionare le righe compresse del passato più rilevanti.

Questo segnale potrebbe essere estremamente interessante.

L’indexer risponde implicitamente a una domanda del tipo:

> “Quali parti del contesto passato sono più rilevanti per il token corrente?”

Schema:

```text
token t
   |
   v
attention
   |
   +--> raw KV
   |
   +--> compressed KV
   |
   +--> indexer
          |
          +--> riga 182
          +--> riga 517
          +--> riga 911
          ...
               |
               v
           predictor
               |
               v
        expert futuri
```

Possibili feature:

- ID delle righe selezionate;
- score dell’indexer;
- distribuzione degli score;
- statistiche sulle distanze temporali delle righe selezionate;
- overlap delle righe selezionate fra token consecutivi;
- quantità di cambiamento della selezione.

L’ipotesi è che:

> le parti del contesto che diventano rilevanti nell’attention possano essere correlate con gli expert MoE che saranno attivati subito dopo.

---

# 20. Possibile set di feature del predictor

Un vettore di feature potrebbe essere:

\[
X_t =
[
E_{t-w:t},
R_{t-w:t},
h_t,
z(K_t,V_t),
A_t,
I_t
]
\]

dove:

- \(E\) = expert history;
- \(R\) = router scores/logits;
- \(h_t\) = hidden state o sua compressione;
- \(z(K_t,V_t)\) = feature compatte della KV cache;
- \(A_t\) = feature dell’attention;
- \(I_t\) = feature dell’indexer.

Il target potrebbe essere:

\[
Y_t = E_{t+1}
\]

oppure, per layer-ahead:

\[
Y_{t,l}=E_{t,l+k}
\]

---

# 21. Ablation study proposta

Per capire quali segnali aggiungano realmente informazione, abbiamo proposto un confronto incrementale.

| Predictor | Feature |
|---|---|
| P0 | expert precedente |
| P1 | routing history |
| P2 | routing history + router scores |
| P3 | P2 + K |
| P4 | P2 + V |
| P5 | P2 + K + V |
| P6 | P2 + hidden state |
| P7 | P6 + indexer |
| P8 | combinazione completa |

Questo permette di rispondere a domande come:

- K aggiunge qualcosa rispetto al routing?
- V aggiunge qualcosa?
- hidden state è migliore della KV?
- KV aggiunge informazione oltre l’hidden state?
- l’indexer aggiunge ulteriore potere predittivo?
- il guadagno giustifica l’overhead?

---

# 22. Metriche: non basta la sola accuracy

Il predictor esiste per accelerare l’inferenza, non per ottenere una metrica ML astratta.

Bisogna quindi valutare una catena del tipo:

\[
Expert\ Recall
\rightarrow
Cache\ Hit\ Rate
\rightarrow
I/O\ evitato
\rightarrow
latenza
\rightarrow
tokens/s
\]

Un predictor più accurato ma troppo lento può peggiorare le performance complessive.

Metriche interessanti:

- Recall@k;
- Precision@k;
- percentuale degli expert realmente necessari già prefetched;
- cache hit rate;
- byte SSD/RAM -> VRAM evitati;
- stall time;
- overhead del predictor;
- tokens/s finali.

---

# 23. Predictor leggero

Una possibile architettura discussa è:

```text
routing history -----+
                     |
router scores -------+
                     |
KV compact features -+--> tiny predictor
                     |        |
hidden features -----+        v
                        256 expert scores
                              |
                              v
                            Top-k
```

Il predictor potrebbe essere un piccolo MLP o un altro modello estremamente economico.

Il requisito più importante è che il suo costo sia molto inferiore al tempo di trasferimento che cerca di nascondere.

---

# 24. La domanda di ricerca più interessante

La domanda iniziale:

> “Quanto funziona il trace precedente come predictor?”

può essere ampliata in:

> **Quanto è prevedibile il routing MoE futuro a partire dallo stato interno già disponibile durante il decoding, e quali segnali — routing history, router scores, hidden states, KV-cache features e indexer features — offrono il miglior compromesso tra accuratezza di previsione e overhead?**

Ancora più specificamente:

> **La KV cache contiene informazione predittiva sul routing futuro che non è già contenuta nello storico del routing o nell’hidden state?**

Questa è una domanda sperimentale molto interessante perché misura l’**informazione incrementale** offerta dalla KV cache.

---

# 25. Lo stato semantico/computazionale è già stato studiato nella ricerca?

Sì.

L’idea generale di usare **stati intermedi del modello** per predire gli expert è già presente nella letteratura.

Questo significa che la macro-idea:

```text
internal state
    |
    v
expert predictor
```

non è completamente nuova.

Ciò che può essere più interessante è capire quali feature specifiche risultino utili in DeepSeek-V4-Flash / DS4 e, in particolare, se la KV cache o l’indexer offrano informazione aggiuntiva.

---

# 26. ProMoE

**ProMoE: Fast MoE-based LLM Serving using Proactive Caching** propone un sistema di proactive caching che sfrutta risultati intermedi del modello per prevedere l’utilizzo successivo degli expert.

L’obiettivo è caricare gli expert prima che diventino necessari e rimuovere il caricamento dalla critical path.

Schema concettuale:

```text
intermediate state
      |
      v
 predictor
      |
      v
expert futuri
      |
      v
 proactive cache / prefetch
```

Paper:
https://arxiv.org/abs/2410.22134

Questo lavoro è un precedente diretto importante per l’uso degli hidden/intermediate states.

---

# 27. Pre-Attention Expert Prediction and Prefetching

Il lavoro:

**Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models**

utilizza le **attivazioni prima del blocco di attention dello stesso layer** per anticipare il routing.

Usa trasformazioni leggere e punta esplicitamente a ridurre l’overhead del predictor.

Paper:
https://arxiv.org/abs/2511.10676

Il lavoro riporta elevate accuratezze di prediction su diversi MoE.

Questo rafforza l’idea che le rappresentazioni interne contengano informazione sufficiente per anticipare le decisioni del router.

---

# 28. ExpertFlow

**ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference** combina:

- pregating information;
- intermediate computational states;
- runtime statistics;
- scheduling adattivo del prefetch.

L’idea è fondere segnali del modello e segnali hardware/runtime.

Paper:
https://arxiv.org/abs/2510.26730

Questo è rilevante perché suggerisce che il predictor non debba necessariamente usare solo informazioni “semantiche”: può essere utile combinare lo stato del modello con lo stato del sistema.

---

# 29. Fate

Anche sistemi come **Fate** utilizzano uno stato intermedio prima del gate, copiandolo sulla CPU per eseguire prediction in parallelo e generare una prefetch list per layer successivi.

Paper:
https://arxiv.org/abs/2502.12224

Questo è un ulteriore esempio di previsione cross-layer basata sullo stato computazionale.

---

# 30. OD-MoE e il ruolo della KV cache

Un lavoro particolarmente interessante è:

**OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference**

Paper:
https://arxiv.org/abs/2512.03927

OD-MoE utilizza un **shadow / emulative predictor model** che corre davanti al modello principale e anticipa l’attivazione degli expert.

Il punto importante per la nostra discussione è che il lavoro analizza la divergenza dello stato KV dello shadow model rispetto al modello principale.

La sincronizzazione della KV cache migliora fortemente la qualità della previsione.

Concettualmente:

```text
Main model KV
      |
      v
synchronization
      |
      v
Shadow model KV
      |
      v
shadow inference
      |
      v
expert prediction
```

Questo dimostra che lo stato KV è strettamente legato alla capacità di mantenere una traiettoria computazionale coerente.

Tuttavia è differente dalla nostra idea.

OD-MoE usa la KV cache per mantenere allineato uno shadow model.

La nostra ipotesi sarebbe invece:

```text
KV del modello principale
        |
        v
feature extraction
        |
        v
piccolo predictor
        |
        v
expert futuri
```

---

# 31. Cosa sembra già consolidato e cosa è più esplorativo

## Già studiato direttamente

```text
routing history -> expert prediction
```

```text
hidden/intermediate state -> expert prediction
```

```text
pre-gating state -> expert prediction
```

```text
cross-layer prediction
```

```text
runtime-aware prefetch
```

## Più interessante per il nostro caso

La combinazione:

\[
[
routing\ history,
router\ scores,
hidden\ state,
KV\ features,
indexer\ features
]
\rightarrow
future\ experts
\]

e soprattutto l’ablation volta a capire:

> **la KV cache offre informazione predittiva aggiuntiva rispetto all’hidden state e al routing history?**

Questo è il punto che può rendere l’esperimento più specifico e interessante per DeepSeek-V4-Flash.

---

# 32. Perché l’indexer di DS4 può rendere il caso DeepSeek-V4-Flash particolare

DeepSeek-V4-Flash non utilizza una semplice KV cache standard.

DS4 mantiene:

- raw recent KV;
- compressed KV;
- in alcuni layer, un indexer che sceglie quali parti compresse del passato rendere visibili all’attention.

Questo significa che disponiamo potenzialmente di una feature che molti altri MoE non possiedono nello stesso modo:

```text
"quale porzione del passato sta diventando importante adesso?"
```

Se questa informazione risultasse correlata al routing futuro, potrebbe essere particolarmente utile.

---

# 33. Stato semantico vs stato operativo

È utile distinguere due gruppi di feature.

## Stato semantico/computazionale del modello

- hidden state;
- K;
- V;
- similarità K/V nel tempo;
- router logits/scores;
- attention statistics;
- indexer selections;
- routing history.

## Stato operativo del sistema

- expert già in VRAM;
- expert presenti nella cache RAM;
- expert solo su SSD;
- latenza stimata di trasferimento;
- banda disponibile;
- richieste di I/O in corso.

Il primo gruppo aiuta a prevedere **quali expert serviranno**.

Il secondo aiuta a decidere **quali expert conviene veramente prefetcheare**.

Quindi, idealmente:

```text
semantic/computational predictor
             |
             v
probabilità expert futuri
             |
             +------+
                    |
runtime/cache state |
                    v
             prefetch policy
                    |
                    v
              SSD/RAM -> VRAM
```

---

# 34. Possibile pipeline finale del progetto

```text
                 DeepSeek-V4-Flash / DS4
                          |
                          v
                      token t
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
     routing state    hidden state      KV/indexer
          |               |               |
          +---------------+---------------+
                          |
                          v
                    feature vector
                          |
                          v
                    tiny predictor
                          |
                          v
               expert probabilities
                          |
                          v
                cache-aware policy
                          |
                          v
                 prefetch SSD/RAM
                          |
                          v
                        VRAM
                          |
                          v
                    token t+1
```

---

# 35. Punto di arrivo della discussione

Dopo aver stabilito:

1. cosa sono prefill e decode;
2. come viene costruita e usata la KV cache;
3. che DS4 implementa già una KV cache avanzata;
4. che hidden state, router state, KV cache e indexer possono diventare feature;
5. che esiste prior art sull’uso degli stati computazionali interni;
6. che l’uso diretto di feature KV/indexer per il routing futuro è una direzione da verificare sperimentalmente;

la domanda naturale successiva è diventata:

> **“Ma esistono dataset già pronti per quello che vogliamo fare noi su DeepSeek-V4-Flash?”**

Da questo punto in poi inizia quindi la parte successiva del lavoro: individuare corpus e routing trace già disponibili e decidere quali dati dovranno invece essere raccolti direttamente instrumentando DS4.

---

# 36. Riferimenti principali discussi

## DS4 / DeepSeek-V4-Flash

- DS4 repository  
  https://github.com/antirez/ds4

- DS4 Model Card — architettura KV cache compressa  
  https://github.com/antirez/ds4/blob/main/MODEL_CARD.md

- DS4 README — KV cache e disk persistence  
  https://github.com/antirez/ds4/blob/main/README.md

- DS4 KV store implementation  
  https://github.com/antirez/ds4/blob/main/ds4_kvstore.c

## Expert prediction / prefetch

- ProMoE  
  https://arxiv.org/abs/2410.22134

- Pre-Attention Expert Prediction and Prefetching  
  https://arxiv.org/abs/2511.10676

- ExpertFlow  
  https://arxiv.org/abs/2510.26730

- Fate  
  https://arxiv.org/abs/2502.12224

- OD-MoE  
  https://arxiv.org/abs/2512.03927

---

# 37. Riassunto finale

Il percorso della discussione può essere condensato così:

```text
PREFILL
   |
   v
elaborazione parallela del prompt
   |
   v
costruzione dello stato interno
e della KV cache
   |
   v
DECODE
   |
   v
generazione token-per-token
   |
   +--> KV cache evita ricalcoli dell’attention
   |
   +--> MoE router seleziona expert
             |
             v
        expert offloading
             |
             v
     possibile collo di bottiglia
     SSD/RAM -> VRAM
             |
             v
          PREDICTOR
             |
    +--------+---------+---------+
    |        |         |         |
 routing   router    hidden     KV /
 history   scores    state    indexer
    |        |         |         |
    +--------+---------+---------+
             |
             v
       expert futuri
             |
             v
          prefetch
```

La direzione sperimentale centrale diventa quindi:

> **Studiare quanto il routing futuro di DeepSeek-V4-Flash sia prevedibile usando non soltanto il trace degli expert precedenti, ma anche lo stato semantico/computazionale già disponibile durante il decoding, mantenendo però il predictor abbastanza leggero da produrre un guadagno reale sul tempo di inferenza.**
