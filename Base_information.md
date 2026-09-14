# Base Information

## 1. Cos'è un layer in un Transformer

Un **layer** è uno dei blocchi di elaborazione che vengono ripetuti in sequenza all'interno di un Transformer.

Nella tesi, nella sezione dedicata all'architettura Transformer, viene spiegato che i blocchi non vengono utilizzati una sola volta, ma vengono ripetuti formando uno **stack di layer**. L'output prodotto da un layer diventa l'input del layer successivo.

Flusso semplificato:

```text
embedding del token
       │
       ▼
┌─────────────┐
│   Layer 0   │
└──────┬──────┘
       │ rappresentazione modificata
       ▼
┌─────────────┐
│   Layer 1   │
└──────┬──────┘
       │
       ▼
      ...
       │
       ▼
┌─────────────┐
│  Layer N    │
└──────┬──────┘
       │
       ▼
output finale
       │
       ▼
logits → prossimo token
```

Un layer quindi **non è un singolo neurone** e non è nemmeno un singolo expert. È un blocco che prende una rappresentazione vettoriale del token, la elabora e produce una nuova rappresentazione.

---

## 2. Da cosa è composto un layer Transformer

Partendo dalla struttura descritta nella tesi, un layer decoder-only può essere semplificato in due componenti principali:

```text
                 TRANSFORMER LAYER
                        │
              ┌─────────▼─────────┐
              │    ATTENTION      │
              └─────────┬─────────┘
                        │
              Normalization /
              Residual connection
                        │
              ┌─────────▼─────────┐
              │        FFN        │
              │ Feed-Forward Net  │
              └─────────┬─────────┘
                        │
              Normalization /
              Residual connection
                        │
                        ▼
                 layer successivo
```

Le due parti hanno ruoli differenti:

```text
ATTENTION
    ↓
"Da quali altri token devo prendere informazione?"

FFN
    ↓
"Come devo trasformare/elaborare questa informazione?"
```

---

## 3. Attention

L'Attention serve a mettere in relazione il token corrente con gli altri token del contesto.

In modo intuitivo, risponde alla domanda:

> **Quali altri token del contesto sono importanti per capire il token che sto elaborando?**

Nella tesi il meccanismo viene descritto tramite le matrici:

```text
Q = Query
K = Key
V = Value
```

e tramite la Self-Attention.

Esempio:

```text
"The cat is sitting on the ..."
```

Durante l'elaborazione, il modello può assegnare importanza diversa alle informazioni associate a token come:

```text
cat
sitting
on
```

L'Attention permette quindi di **mescolare informazione tra token differenti** e costruire rappresentazioni contestualizzate.

---

## 4. Feed-Forward Network (FFN)

Dopo l'Attention troviamo la **Feed-Forward Network**.

La FFN prende la rappresentazione del singolo token prodotta dai passaggi precedenti e la trasforma ulteriormente.

Nella tesi viene descritta, nella forma classica del Transformer originale, come:

\[
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

Concettualmente:

```text
rappresentazione del token
          │
          ▼
      matrice W1
          │
          ▼
    spazio "espanso"
          │
   funzione di attivazione
          │
          ▼
      matrice W2
          │
          ▼
nuova rappresentazione
del token
```

A differenza dell'Attention, che mette in relazione token differenti, la FFN lavora principalmente sulla rappresentazione del **singolo token**.

---

## 5. Residual Connection e Normalization

Tra i principali sottoblocchi del Transformer sono presenti anche:

- **Residual connection**
- **Normalization**

La residual connection permette di sommare all'output di un sottoblocco anche il suo input originale, preservando l'informazione e facilitando il passaggio del gradiente nelle reti molto profonde.

La normalizzazione serve invece a stabilizzare le rappresentazioni interne e l'addestramento del modello.

Nella tesi questi meccanismi vengono descritti nel contesto dei blocchi **Add & Norm** e delle varianti moderne basate sulla **pre-normalization**.

---

# 6. Dal Transformer classico al Mixture of Experts (MoE)

La tesi descrive principalmente Transformer con una **FFN classica**.

In un Transformer classico, un layer può essere rappresentato in modo molto semplificato così:

```text
Layer L
│
├── Attention
│
└── FFN
```

Nei modelli **Mixture of Experts (MoE)**, invece, la singola FFN viene sostituita concettualmente da molte FFN specializzate, chiamate **expert**, più un componente chiamato **router**.

```text
Layer L
│
├── Attention
│
└── MoE
     │
     ├── Router
     │
     ├── Expert 0
     ├── Expert 1
     ├── Expert 2
     ├── Expert 3
     ├── ...
     └── Expert N
```

Quindi, in modo semplice:

> **Un expert può essere visto come una FFN specializzata all'interno del layer MoE.**

---

## 7. Il ruolo del Router

Il router decide **quali expert utilizzare per il token corrente**.

Non vengono necessariamente eseguiti tutti gli expert disponibili.

Esempio concettuale:

```text
Token t
   │
   ▼
Layer 17
   │
   ├── Attention
   │
   ▼
Router
   │
   ├── Expert 12  ✓
   ├── Expert 41  ✓
   ├── Expert 73  ✓
   ├── Expert 101 ✓
   ├── Expert 180 ✓
   ├── Expert 229 ✓
   │
   └── altri expert ✗
```

Gli expert contrassegnati con `✓` sono quelli realmente selezionati per quel token e per quel layer.

---

# 8. Caso specifico: DeepSeek V4 Flash in DS4

Nel modello DeepSeek V4 Flash che stiamo usando con DwarfStar/DS4, l'ispezione del modello ha mostrato:

```text
layers: 43
experts: count=256
used=6
```

Quindi:

- il modello ha **43 layer**;
- nei layer MoE ci sono **256 expert disponibili**;
- per il routing vengono selezionati **6 expert** per token.

Importante: questi numeri appartengono al modello che stiamo utilizzando nel nostro setup DS4 e **non derivano dalla tesi**.

---

# 9. Significato di E_l(t)

Nel nostro progetto usiamo la notazione:

\[
E_l(t)
\]

per indicare:

> **l'insieme degli expert selezionati dal router nel layer `l` mentre viene elaborato il token `t`.**

Esempio:

```text
token t
   │
   ▼
Layer 17
   │
   ├── Attention
   │
   ▼
 Router
   │
   └── E_17(t) =
       {12, 41, 73, 101, 180, 229}
```

Lo stesso token continua poi ad attraversare i layer successivi:

```text
Layer 17
   ↓
Layer 18
   ↓
Layer 19
   ↓
...
Layer 42
```

e in ogni layer MoE può essere selezionato un insieme differente di expert.

Esempio:

```text
token t

Layer 0  → {7, 18, 35, 90, 113, 201}
Layer 1  → {3, 41, 56, 99, 187, 220}
Layer 2  → {8, 19, 72, 81, 144, 200}
...
Layer 17 → {12, 41, 73, 101, 180, 229}
...
Layer 42 → {5, 31, 67, 104, 171, 240}
```

---

# 10. La nostra idea di predizione

L'obiettivo del progetto è utilizzare ciò che accade durante l'elaborazione del token `t` per prevedere quali expert potrebbero essere necessari al token successivo `t+1`.

L'idea principale è:

\[
E_l(t) \rightarrow \hat{E}_l(t+1)
\]

cioè:

> usare il routing osservato nel layer `l` al token `t` per prevedere gli expert che serviranno nello **stesso layer `l`** al token `t+1`.

---

## 11. Predictor eseguito in parallelo

Il predictor dovrebbe essere eseguito **in parallelo** mentre gli expert reali del token `t` vengono elaborati.

Flusso concettuale:

```text
               Token t
               Layer 17
                  │
       ┌──────────┴───────────┐
       │                      │
       ▼                      ▼
esegui gli expert       PREDICTOR
necessari a t               │
                            ▼
                    predice E_17(t+1)
                            │
                            ▼
                    {12, 41, 98,
                     101, 180, 230}
                            │
                            ▼
                      PREFETCH SSD
```

Il predictor **non sostituisce il router reale** del modello.

Il router continua a decidere normalmente gli expert corretti. Il predictor serve soltanto per **anticipare il caricamento degli expert**.

---

# 12. Collegamento con SSD Streaming

Nel nostro setup il modello è molto più grande della VRAM disponibile.

Per questo motivo DS4 utilizza un flusso del tipo:

```text
SSD
 ↓
expert cache / VRAM
 ↓
GPU computation
```

Quando un expert necessario non è già disponibile nella cache, DS4 deve recuperarlo dallo storage.

La nostra idea è anticipare questo trasferimento.

### Senza predictor

```text
token t+1
   │
   ▼
Layer 17
   │
   ▼
Router
   │
   ▼
scopro che serve Expert 98
   │
   ▼
Expert 98 non è in cache
   │
   ▼
caricamento da SSD
   │
   ▼
attesa
   │
   ▼
esecuzione Expert 98
```

### Con predictor

```text
token t
Layer 17
   │
   ▼
Predictor
   │
   ▼
"Probabilmente al token t+1 servirà Expert 98"
   │
   ▼
prefetch Expert 98 da SSD
   │
   ▼
nel frattempo il modello continua a elaborare token t
   │
   ▼
token t+1
Layer 17
   │
   ▼
Router reale seleziona Expert 98
   │
   ▼
Expert 98 è già disponibile
   │
   ▼
esecuzione immediata
```

L'obiettivo è quindi **nascondere la latenza dello storage dietro alla computazione già in corso**.

---

# 13. Stesso layer, token successivo

È importante distinguere due concetti diversi:

```text
stesso token, layer successivo:
E_l(t) → E_(l+1)(t)
```

e:

```text
stesso layer, token successivo:
E_l(t) → E_l(t+1)
```

La direzione su cui vogliamo concentrarci inizialmente è:

\[
\boxed{E_l(t) \rightarrow E_l(t+1)}
\]

cioè:

```text
stesso layer L
────────────────────────────────────

token t                   token t+1

Layer L                   Layer L
   │                         │
Router                      Router
   │                         │
expert realmente    →    expert che
utilizzati                vorremmo
a t                       prevedere prima
```

Non stiamo quindi cercando di prevedere il layer successivo.

Stiamo cercando di prevedere **quali expert dello stesso layer verranno selezionati quando arriverà il prossimo token**.

---

# 14. Perché questa predizione può essere utile

La predizione `t → t+1` offre una finestra temporale interessante.

Se nel layer `17` del token `t` prevediamo quali expert serviranno al layer `17` del token `t+1`, il sistema ha ancora tempo per:

```text
restanti layer del token t
        ↓
produzione del nuovo token
        ↓
primi layer del token t+1
        ↓
Layer 17 del token t+1
```

Durante questo intervallo è possibile eseguire in background il prefetch degli expert dall'SSD.

Questo è particolarmente utile nel nostro scenario di **SSD streaming**.

---

# 15. Il predictor non deve modificare l'output del modello

Il predictor deve essere una **ottimizzazione di sistema**, non una modifica al comportamento logico del modello.

Il router reale rimane sempre la fonte di verità.

Esempio:

```text
Predictor:
prevede {12, 41, 98, 101, 180, 230}

Router reale:
seleziona {12, 41, 73, 101, 180, 229}
```

Gli expert realmente eseguiti devono essere:

```text
{12, 41, 73, 101, 180, 229}
```

cioè quelli selezionati dal router reale.

Gli expert:

```text
{98, 230}
```

sono semplicemente stati precaricati inutilmente.

Quindi una previsione sbagliata dovrebbe comportare:

```text
prediction miss
      ↓
I/O inutile
      ↓
eventuale cache pollution
```

ma **non deve cambiare i logits o i token generati dal modello**.

---

# 16. Flusso completo ideale

```text
TOKEN t
  │
  ▼
Layer l
  │
  ├───────────────┐
  │               │
  ▼               ▼
Router reale   informazioni
  │           disponibili a t
  ▼               │
E_l(t)             ▼
  │            Predictor
  │               │
  │               ▼
  │          Ê_l(t+1)
  │               │
  ▼               ▼
Esecuzione       Prefetch
expert t           SSD
  │               │
  └───────┬───────┘
          │
          ▼
il modello continua
a processare token t
          │
          ▼
generazione token t+1
          │
          ▼
Layer l del token t+1
          │
          ▼
Router reale
          │
          ▼
E_l(t+1)
          │
          ▼
gli expert corretti
sono idealmente già
presenti nella cache
```

---

# 17. Concetti chiave da ricordare

## Layer
Un **layer** è un blocco completo del Transformer che trasforma la rappresentazione dei token.

## Attention
L'Attention mette in relazione il token corrente con gli altri token del contesto.

## FFN
La FFN trasforma ulteriormente la rappresentazione del singolo token.

## Expert
In un modello MoE, un expert può essere visto, in modo semplificato, come una **FFN specializzata**.

## Router
Il router sceglie quali expert usare per un determinato token in un determinato layer.

## E_l(t)

\[
E_l(t)
\]

è l'insieme degli expert realmente selezionati nel layer `l` per il token `t`.

## Predictor
Il predictor cerca di stimare:

\[
\hat{E}_l(t+1)
\]

mentre il token `t` è ancora in elaborazione.

## Prefetch
Gli expert previsti vengono caricati anticipatamente dall'SSD verso la cache/memoria più veloce.

## Vincolo fondamentale
Il predictor **non deve modificare il routing reale del modello**. Serve solamente ad anticipare i movimenti di memoria.

---

# 18. Collegamento concettuale finale

Partendo dalla struttura descritta nella tesi:

```text
Transformer
    ↓
stack di layer
    ↓
Attention + FFN
```

nel nostro caso MoE possiamo estendere il concetto a:

```text
Transformer MoE
    ↓
stack di layer
    ↓
Attention + Router + Expert
```

e infine arrivare alla nostra ottimizzazione:

```text
Transformer MoE
    ↓
Layer l, token t
    ↓
Router
    ↓
E_l(t)
    │
    ├──────────────→ esecuzione normale
    │
    └──────────────→ predictor
                        ↓
                    Ê_l(t+1)
                        ↓
                    SSD prefetch
                        ↓
                 Layer l, token t+1
```

L'obiettivo è quindi **ridurre il tempo di attesa causato dal caricamento degli expert dallo storage**, sfruttando il tempo di computazione del token corrente per preparare in anticipo gli expert che probabilmente serviranno al token successivo.

---

## Riferimenti concettuali alla tesi

Le parti relative al Transformer classico si basano sulle sezioni della tesi dedicate a:

- **Architettura Transformer**
- **Encoder**
- **Multi-Head Attention**
- **Normalization / Add & Norm**
- **Feed-Forward Network (FFN)**
- **Decoder**
- **LLaMA**

La parte relativa a **Mixture of Experts, router, expert routing e predictive expert prefetch** è invece un'estensione concettuale applicata al lavoro attuale su **DwarfStar / DeepSeek V4 Flash / SSD streaming**.

# 19. DeepSeek V4 Flash è un modello decoder-only

Un punto importante per collegare correttamente la teoria della tesi al modello che stiamo usando è che **DeepSeek V4 Flash non adotta l'architettura encoder-decoder completa del Transformer originale**.

È invece un modello **decoder-only**, cioè utilizza soltanto uno stack autoregressivo di blocchi decoder-like per elaborare il prompt e generare il token successivo.

## 19.1 Transformer standard descritto nella tesi

Nella descrizione classica del Transformer, l'architettura è composta da due parti principali:

```text
INPUT
  │
  ▼
ENCODER
  │
  ▼
rappresentazione contestualizzata dell'input
  │
  ▼
DECODER
  │
  ▼
OUTPUT
```

L'encoder riceve l'intera sequenza di input e costruisce una rappresentazione interna.

Il decoder utilizza poi questa rappresentazione, insieme ai token già generati, per produrre progressivamente l'output.

Questa è la struttura tipica del Transformer originale descritta nella tesi.

---

## 19.2 DeepSeek V4 Flash

DeepSeek V4 Flash segue invece una struttura **decoder-only**.

Il flusso concettuale è:

```text
prompt + token già generati
          │
          ▼
      embedding
          │
          ▼
       Layer 0
          │
          ▼
       Layer 1
          │
          ▼
         ...
          │
          ▼
      Layer 42
          │
          ▼
     output head
          │
          ▼
        logits
          │
          ▼
     prossimo token
```

Non esiste quindi uno stack encoder separato che produce una rappresentazione poi consumata dal decoder.

Il prompt e i token già generati vengono elaborati direttamente dallo stesso stack decoder-only.

---

## 19.3 Generazione autoregressiva

DeepSeek V4 Flash genera il testo in modo autoregressivo.

Il processo può essere rappresentato così:

```text
prompt
  │
  ▼
DECODER-ONLY
  │
  ▼
logits
  │
  ▼
token t
  │
  ├──────────────┐
  │              │
  ▼              │
aggiunto al      │
contesto         │
  │              │
  └──────────────┘
         │
         ▼
   DECODER-ONLY
         │
         ▼
       logits
         │
         ▼
     token t+1
```

Ogni nuovo token viene quindi aggiunto al contesto e attraversa nuovamente lo stesso stack di layer.

---

## 19.4 Differenza concettuale rispetto al Transformer encoder-decoder

La differenza può essere riassunta così:

```text
Transformer standard
────────────────────────────

input
  ↓
ENCODER
  ↓
rappresentazione dell'input
  ↓
DECODER
  ↓
output
```

mentre:

```text
DeepSeek V4 Flash
────────────────────────────

prompt + token precedenti
          ↓
     DECODER-ONLY
          ↓
       logits
          ↓
    prossimo token
```

Quindi:

```text
Transformer classico:
Encoder + Decoder

DeepSeek V4 Flash:
solo stack decoder autoregressivo
```

---

## 19.5 Collegamento con i layer MoE

Dire che DeepSeek V4 Flash è **decoder-only** non significa che i suoi layer siano semplici.

Ogni token attraversa lo stesso stack decoder-only e, all'interno dei layer MoE, vengono eseguite componenti come:

```text
Token t
   │
   ▼
Layer l
   │
   ├── Attention
   │
   ├── Normalization / residual path
   │
   └── MoE
        │
        ├── Router
        ├── Expert selezionati
        └── combinazione dell'output
```

Nel modello che stiamo usando con DS4, l'ispezione ha mostrato:

```text
layers: 43
experts: count=256
used=6
```

Il token attraversa quindi i layer dello stack e, nei layer MoE, il router decide quali expert attivare.

---

## 19.6 Perché questo è importante per il nostro predictor

Il fatto che DeepSeek V4 Flash sia decoder-only e autoregressivo è direttamente collegato alla nostra idea di predizione.

Il token successivo attraverserà di nuovo lo stesso stack di layer.

Quindi, se al token `t` osserviamo:

```text
Layer l
   │
   ▼
E_l(t)
```

possiamo cercare di prevedere:

```text
Layer l
   │
   ▼
E_l(t+1)
```

perché anche il token `t+1` passerà attraverso lo stesso layer `l`.

Il flusso è quindi:

```text
TOKEN t
  │
  ▼
Layer l
  │
  ▼
Router
  │
  ▼
E_l(t)
  │
  ├──────────────→ esecuzione normale degli expert
  │
  └──────────────→ predictor
                       │
                       ▼
                  Ê_l(t+1)
                       │
                       ▼
                  SSD prefetch
                       │
                       ▼
                 TOKEN t+1
                       │
                       ▼
                   Layer l
                       │
                       ▼
                 Router reale
                       │
                       ▼
                  E_l(t+1)
```

Questo rende naturale la formulazione:

\[
E_l(t) \rightarrow \hat{E}_l(t+1)
\]

perché il modello genera i token in sequenza e ciascun nuovo token attraversa nuovamente lo stesso stack decoder-only.

---

## 19.7 Collegamento con la tesi

La tesi fornisce la base teorica del Transformer classico, descrivendo encoder, decoder, Attention, FFN, residual connection e normalizzazione.

Per il nostro lavoro è importante distinguere chiaramente:

```text
Tesi:
Transformer classico
Encoder + Decoder
```

da:

```text
DeepSeek V4 Flash:
Transformer decoder-only
+ Mixture of Experts
+ routing dinamico degli expert
+ generazione autoregressiva
```

Quindi il nostro lavoro parte dai concetti descritti nella tesi, ma li applica a un'architettura moderna decoder-only e MoE, in cui la parte FFN viene sostituita concettualmente da un insieme di expert selezionati dinamicamente dal router.
