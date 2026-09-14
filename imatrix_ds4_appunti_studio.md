# Imatrix in DS4 — Appunti di studio

Questi appunti raccolgono e organizzano tutto ciò che abbiamo discusso sull'**imatrix** nel contesto di **DS4 / DeepSeek-V4-Flash**, con particolare attenzione al collegamento tra attivazioni, errore di quantizzazione e reti `gate`, `up` e `down` degli expert MoE.

---

## 1. Cos'è l'imatrix

**Imatrix** significa *importance matrix*.

Nel contesto di DS4, viene usata durante la costruzione di modelli quantizzati per raccogliere statistiche sulle **attivazioni interne** osservate quando il modello elabora un workload di calibrazione.

L'idea centrale è:

> non tutti gli errori introdotti dalla quantizzazione dei pesi hanno lo stesso effetto sull'output del modello.

Se una certa dimensione dell'input di una matrice è spesso molto attiva, un errore sui pesi associati a quella dimensione può propagarsi maggiormente sull'output.

Quindi l'imatrix non assegna propriamente una “importanza” indipendente a ogni singolo peso. Piuttosto, misura quanto risultano importanti, in media, le **dimensioni degli input** che alimentano determinate matrici dei pesi.

---

## 2. Cosa NON è l'imatrix

L'imatrix non è:

- una copia dei pesi del modello;
- la KV cache;
- un routing trace token-per-token;
- un dataset contenente le vere risposte generate da DeepSeek;
- un dataset supervisionato già pronto per addestrare il nostro predictor degli expert;
- una tabella del tipo “questo singolo peso è importante 0.87”.

In particolare, non conserva una sequenza temporale come:

```text
token t     -> expert [3, 8, 17, ...]
token t+1   -> expert [2, 8, 41, ...]
token t+2   -> expert [...]
```

Le osservazioni vengono **aggregate**.

---

## 3. Relazione con `rendered_prompts.txt`

`rendered_prompts.txt` e imatrix sono due cose diverse.

### `rendered_prompts.txt`

Contiene i prompt già renderizzati nel formato usato da DS4.

Serve come **corpus di input** per stimolare il modello durante la calibrazione.

Non contiene necessariamente le vere continuazioni generate da DeepSeek.

### Imatrix

È invece il risultato delle statistiche raccolte facendo elaborare quei prompt al modello.

Pipeline concettuale:

```text
rendered_prompts.txt
        |
        v
       DS4
        |
        v
DeepSeek-V4-Flash
        |
        v
prefill sui prompt
        |
        v
attivazioni interne degli expert
        |
        v
accumulo delle statistiche
        |
        v
     imatrix.dat
        |
        v
quantizzazione più informata
```

---

## 4. Dove entra in gioco negli expert MoE

Un expert MoE può essere rappresentato, in forma molto semplificata, con tre matrici principali:

- `W_gate`
- `W_up`
- `W_down`

Schema concettuale:

```text
                    hidden state x
                          |
                 +--------+--------+
                 |                 |
              W_gate             W_up
                 |                 |
                 +------ SwiGLU ----+
                          |
                          z
                          |
                       W_down
                          |
                          v
                       output
```

Per `gate` e `up`, l'input è sostanzialmente lo stesso hidden state `x`.

Per `down`, invece, l'input non è più `x`, ma un vettore successivo che possiamo indicare con `z`, ottenuto dopo le trasformazioni precedenti e la non linearità SwiGLU.

---

# 5. Perché si usano gli input `x_i`

Questa è la parte matematica più importante.

Partiamo dal caso più semplice possibile: un solo peso e un solo input.

Supponiamo:

\[
y = wx
\]

dove:

- \(w\) è il peso originale;
- \(x\) è l'input;
- \(y\) è l'output.

La quantizzazione può modificare leggermente il peso:

\[
w' = w + \Delta w
\]

dove:

\[
\Delta w = w' - w
\]

è l'errore introdotto dalla quantizzazione.

Il nuovo output sarà:

\[
y' = w'x
\]

Sostituiamo \(w'\):

\[
y' = (w + \Delta w)x
\]

Applicando la proprietà distributiva:

\[
y' = wx + \Delta w\,x
\]

ma:

\[
y = wx
\]

quindi:

\[
y' = y + \Delta w\,x
\]

Da cui:

\[
\boxed{y'-y = \Delta w\,x}
\]

Questa formula dice una cosa fondamentale:

> l'errore sull'output non dipende soltanto dall'errore sul peso \(\Delta w\), ma anche da quanto è grande l'input \(x\).

---

## 6. Esempio numerico semplice

Supponiamo:

\[
w=2
\]

\[
x=3
\]

L'output originale è:

\[
y=2\cdot3=6
\]

Dopo la quantizzazione, immaginiamo:

\[
w'=2.1
\]

Quindi:

\[
\Delta w=2.1-2=0.1
\]

Il nuovo output è:

\[
y'=2.1\cdot3=6.3
\]

L'errore sull'output è:

\[
y'-y=6.3-6=0.3
\]

La formula precedente dà:

\[
\Delta w\,x=0.1\cdot3=0.3
\]

quindi coincide esattamente.

---

# 7. Perché compare `x²`

Vogliamo ora misurare la **grandezza** dell'errore senza preoccuparci del suo segno.

Una misura naturale è l'errore quadratico:

\[
(y'-y)^2
\]

Abbiamo appena visto che:

\[
y'-y=\Delta w\,x
\]

quindi:

\[
(y'-y)^2=(\Delta w\,x)^2
\]

Usando:

\[
(ab)^2=a^2b^2
\]

otteniamo:

\[
\boxed{(y'-y)^2=(\Delta w)^2x^2}
\]

Ed è proprio qui che compare \(x^2\).

---

## 8. Interpretazione intuitiva di `x²`

Consideriamo lo stesso identico errore sul peso:

\[
\Delta w=0.1
\]

### Caso A

\[
x=1
\]

Errore sull'output:

\[
\Delta y=0.1
\]

Errore quadratico:

\[
(\Delta y)^2=0.01
\]

### Caso B

\[
x=10
\]

Errore sull'output:

\[
\Delta y=1
\]

Errore quadratico:

\[
(\Delta y)^2=1
\]

| Caso | \(x\) | \(\Delta w\) | errore output | errore quadratico |
|---|---:|---:|---:|---:|
| A | 1 | 0.1 | 0.1 | 0.01 |
| B | 10 | 0.1 | 1 | 1 |

Lo stesso errore sul peso produce un errore quadratico 100 volte maggiore quando l'input è 10 volte più grande, perché:

\[
10^2=100
\]

---

## 9. Perché il quadrato elimina il problema del segno

Supponiamo di osservare due attivazioni:

\[
+5
\]

e:

\[
-5
\]

Se facessimo semplicemente la somma:

\[
5+(-5)=0
\]

sembrerebbe che quella dimensione non sia importante.

Ma entrambe sono attivazioni molto forti.

Usando il quadrato:

\[
5^2+(-5)^2=25+25=50
\]

quindi il segno non provoca cancellazione.

---

# 10. Dal caso scalare alla matrice dei pesi

Nelle reti neurali reali non abbiamo un solo peso, ma matrici.

Supponiamo:

\[
y=Wx
\]

con:

\[
W=
\begin{bmatrix}
w_{11} & w_{12} & w_{13}\\
w_{21} & w_{22} & w_{23}
\end{bmatrix}
\]

ed:

\[
x=
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}
\]

Otteniamo:

\[
y_1=w_{11}x_1+w_{12}x_2+w_{13}x_3
\]

\[
y_2=w_{21}x_1+w_{22}x_2+w_{23}x_3
\]

Quindi:

- \(x_1\) moltiplica tutti i pesi della colonna 1;
- \(x_2\) moltiplica tutti i pesi della colonna 2;
- \(x_3\) moltiplica tutti i pesi della colonna 3.

```text
              x1       x2       x3
               |        |        |
               v        v        v

W =          [ *        *        * ]
             [ *        *        * ]
               ^        ^        ^
            col. 1   col. 2   col. 3
```

Per questo una statistica sull'intensità di \(x_j\) è indirettamente collegata a quanto è importante preservare bene la **colonna j-esima** della matrice dei pesi.

---

# 11. La statistica accumulata

Semplificando, per una certa dimensione \(j\), l'imatrix accumula qualcosa del tipo:

\[
\boxed{I_j=\sum_{\text{osservazioni}}x_j^2}
\]

Supponiamo che una certa dimensione venga osservata tre volte:

\[
x_j^{(1)}=2,\quad x_j^{(2)}=3,\quad x_j^{(3)}=4
\]

Allora:

\[
I_j=2^2+3^2+4^2=4+9+16=29
\]

Per un'altra dimensione:

\[
0.1,\quad0.2,\quad0.1
\]

abbiamo:

\[
I_k=0.1^2+0.2^2+0.1^2=0.06
\]

Quindi:

```text
dimensione      importance
x_j             29.00
x_k              0.06
```

La prima dimensione è stata molto più sollecitata dal workload osservato.

---

# 12. Perché NON è l'importanza del singolo peso

L'imatrix non produce concettualmente:

```text
importance(w_11)
importance(w_12)
importance(w_13)
...
```

ma piuttosto qualcosa del tipo:

```text
importance(input_dimension_1)
importance(input_dimension_2)
importance(input_dimension_3)
...
```

Questa statistica viene poi usata per capire quali direzioni/colonne dei pesi risultano più sensibili all'errore di quantizzazione.

La formulazione più precisa è quindi:

> **L'imatrix misura quanto intensamente vengono utilizzate le dimensioni degli input delle matrici dei pesi; da questa informazione si deduce quali errori di quantizzazione sui pesi potrebbero essere più pericolosi.**

---

# 13. `gate`, `up` e `down`

Per `gate` e `up`, l'input è sostanzialmente lo stesso hidden state \(x\), quindi possiamo pensare a statistiche del tipo:

\[
I_j^{gate/up}\approx\sum x_j^2
\]

Schema:

```text
               x
               |
        +------+------+
        |             |
      W_gate        W_up
```

Per `down`, invece, l'input è il vettore \(z\) prodotto dopo le operazioni precedenti:

\[
I_j^{down}\approx\sum z_j^2
\]

Schema:

```text
               hidden state x
                     |
            +--------+--------+
            |                 |
         W_gate             W_up
            |                 |
            +----- SwiGLU -----+
                     |
                     z
                     |
                  W_down
                     |
                   output
```

Nel percorso MoE discusso, il segnale associato a `down` può comprendere anche l'effetto del weighting del routing.

---

# 14. Interazioni tra dimensioni diverse

Nel caso scalare avevamo:

\[
(\Delta w\,x)^2=(\Delta w)^2x^2
\]

Nel caso multidimensionale, però, l'errore totale è una somma.

Con due dimensioni:

\[
y=w_1x_1+w_2x_2
\]

Dopo la quantizzazione:

\[
y'-y=\Delta w_1x_1+\Delta w_2x_2
\]

Elevando al quadrato:

\[
(y'-y)^2=(\Delta w_1x_1+\Delta w_2x_2)^2
\]

Usiamo:

\[
(a+b)^2=a^2+b^2+2ab
\]

Otteniamo:

\[
\boxed{
(y'-y)^2=
(\Delta w_1)^2x_1^2+
(\Delta w_2)^2x_2^2+
2\Delta w_1\Delta w_2x_1x_2
}
\]

Il termine:

\[
\boxed{2\Delta w_1\Delta w_2x_1x_2}
\]

è un **termine incrociato**.

Dipende contemporaneamente da entrambe le dimensioni, quindi rappresenta una loro interazione.

---

## 15. Esempio: rafforzamento degli errori

Supponiamo:

\[
x_1=2,\quad x_2=3
\]

\[
\Delta w_1=0.1,\quad \Delta w_2=0.2
\]

L'errore totale è:

\[
\Delta y=0.1(2)+0.2(3)=0.8
\]

quindi:

\[
(\Delta y)^2=0.64
\]

I contributi individuali sono:

\[
(0.1)^2(2)^2=0.04
\]

\[
(0.2)^2(3)^2=0.36
\]

Somma:

\[
0.40
\]

Il termine incrociato è:

\[
2(0.1)(0.2)(2)(3)=0.24
\]

Quindi:

\[
0.04+0.36+0.24=0.64
\]

Gli errori si **rafforzano**.

---

## 16. Esempio: compensazione degli errori

Se invece:

\[
\Delta w_1=0.1
\]

\[
\Delta w_2=-0.2
\]

allora:

\[
\Delta y=0.1(2)-0.2(3)=-0.4
\]

quindi:

\[
(\Delta y)^2=0.16
\]

Il termine incrociato diventa:

\[
2(0.1)(-0.2)(2)(3)=-0.24
\]

perciò:

\[
0.04+0.36-0.24=0.16
\]

Gli errori si **compensano parzialmente**.

---

# 17. Caso generale con molte dimensioni

Con:

\[
x=[x_1,x_2,\ldots,x_d]
\]

e:

\[
\Delta w=[\Delta w_1,\Delta w_2,\ldots,\Delta w_d]
\]

l'errore è:

\[
\Delta y=\sum_j\Delta w_jx_j
\]

Elevando al quadrato:

\[
\boxed{
(\Delta y)^2=
\sum_j(\Delta w_j)^2x_j^2+
2\sum_{j<k}\Delta w_j\Delta w_kx_jx_k
}
\]

La prima parte contiene i contributi individuali delle dimensioni.

La seconda contiene tutte le interazioni tra coppie di dimensioni.

---

# 18. La matrice `xxᵀ`

Se volessimo conservare anche tutte le interazioni, potremmo considerare:

\[
x=
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}
\]

quindi:

\[
xx^T=
\begin{bmatrix}
x_1^2 & x_1x_2 & x_1x_3\\
x_2x_1 & x_2^2 & x_2x_3\\
x_3x_1 & x_3x_2 & x_3^2
\end{bmatrix}
\]

### Sulla diagonale

Abbiamo:

\[
x_1^2,\quad x_2^2,\quad x_3^2
\]

### Fuori dalla diagonale

Abbiamo:

\[
x_1x_2,\quad x_1x_3,\quad x_2x_3,\ldots
\]

cioè le interazioni tra dimensioni diverse.

---

# 19. L'imatrix come approssimazione diagonale

Un modo utile per interpretare l'imatrix è come una statistica **diagonale** molto più economica.

Conserva principalmente quantità del tipo:

\[
I_j=\sum x_j^2
\]

ma non tutte le correlazioni del tipo:

\[
I_{jk}=\sum x_jx_k
\]

per \(j\neq k\).

Concettualmente:

\[
\begin{bmatrix}
I_1 & \cancel{I_{12}} & \cancel{I_{13}}\\
\cancel{I_{21}} & I_2 & \cancel{I_{23}}\\
\cancel{I_{31}} & \cancel{I_{32}} & I_3
\end{bmatrix}
\]

Questa interpretazione è utile per capire il compromesso: si perde parte dell'informazione sulle correlazioni, ma il costo di memoria diminuisce enormemente.

---

## 20. Perché non memorizzare tutte le interazioni?

Con hidden dimension \(d\):

### Solo diagonale

Servono circa:

\[
O(d)
\]

statistiche.

### Matrice completa

Servirebbero:

\[
O(d^2)
\]

valori.

Per esempio, con:

\[
d=7168
\]

avremmo:

\[
7168^2\approx51\,\text{milioni}
\]

valori per una sola matrice completa di correlazione.

Ripetendo questo concetto per molti expert e molti layer, il costo diventerebbe enorme.

---

# 21. Come l'imatrix aiuta la quantizzazione

La quantizzazione trasforma:

\[
W_{FP}\rightarrow W_Q
\]

introducendo un errore:

\[
\Delta W=W_Q-W_{FP}
\]

L'imatrix permette di sapere quali dimensioni degli input vengono usate più intensamente, e quindi dove un errore di quantizzazione può avere un impatto maggiore.

```text
dimensione poco attiva
        |
        v
errore sui pesi
        |
        v
effetto spesso piccolo


dimensione molto attiva
        |
        v
stesso errore sui pesi
        |
        v
effetto potenzialmente molto maggiore
```

Per questo l'imatrix può guidare il quantizzatore nel preservare meglio le parti più sensibili.

---

# 22. Struttura concettuale per layer ed expert

Nel modello discusso possiamo immaginare logicamente una struttura del tipo:

```text
Layer 0
 +-- Expert 0
 |    +-- gate: [I_1, I_2, ...]
 |    +-- up:   [I_1, I_2, ...]
 |    +-- down: [I_1, I_2, ...]
 |
 +-- Expert 1
 |    +-- gate: [...]
 |    +-- up:   [...]
 |    +-- down: [...]
 |
 ...

Layer 1
 +-- Expert 0
 |    ...
```

Ogni vettore rappresenta statistiche aggregate sulle dimensioni osservate nel workload di calibrazione.

---

# 23. Differenza tra imatrix e dataset del nostro predictor

L'imatrix risponde a una domanda del tipo:

> quali dimensioni sono state importanti **in media** durante il workload?

Il nostro predictor deve invece rispondere a:

> dato lo stato del modello al token \(t\), quali expert serviranno al token o layer successivo?

Per il predictor ci servirebbe qualcosa come:

```text
prompt_id
token_position
layer
expert_ids
router_scores
hidden_features
KV_features
indexer_features
expert_ids_next_token
```

Quindi una buona analogia è:

```text
IMATRIX
= fotografia statistica aggregata

PREDICTOR DATASET
= film temporale token-per-token
```

---

# 24. Perché l'imatrix è comunque utile per il nostro progetto

Anche se non è il dataset del predictor, il collector dell'imatrix è interessante perché dimostra che DS4 ha già un punto in cui vengono osservate le attivazioni interne reali.

Potremmo usare quel meccanismo come riferimento per costruire un tracer più ricco capace di salvare:

- token position;
- layer;
- expert selezionati;
- router scores;
- hidden-state features;
- feature della KV cache;
- eventuali feature dell'indexer;
- expert utilizzati successivamente.

Confronto:

```text
Imatrix collector
      |
      v
"cosa è importante in media?"
      |
      v
quantizzazione

Nostro tracer
      |
      v
"quale stato interno predice gli expert futuri?"
      |
      v
prediction + prefetch
```

---

# 25. Formule da ricordare

## Caso scalare

\[
y=wx
\]

\[
w'=w+\Delta w
\]

\[
\boxed{y'-y=\Delta w\,x}
\]

\[
\boxed{(y'-y)^2=(\Delta w)^2x^2}
\]

## Caso multidimensionale

\[
\Delta y=\sum_j\Delta w_jx_j
\]

\[
\boxed{
(\Delta y)^2=
\sum_j(\Delta w_j)^2x_j^2+
2\sum_{j<k}\Delta w_j\Delta w_kx_jx_k
}
\]

La prima parte rappresenta i contributi individuali; la seconda le interazioni tra dimensioni.

---

# 26. Mappa mentale finale

```text
                workload di calibrazione
                         |
                         v
                    hidden state x
                         |
               +---------+---------+
               |                   |
            W_gate               W_up
               |                   |
               +------ SwiGLU ------+
                         |
                         z
                         |
                      W_down
                         |
                         v
                       output

      molte osservazioni delle attivazioni
                         |
                         v
                      IMATRIX
                         |
                         v
     importanza aggregata delle dimensioni
                         |
                         v
          quantizzazione più informata
```

---

# 27. Domande di ripasso

1. Cos'è una imatrix?
2. Perché non è un routing trace?
3. Perché si osservano gli input \(x_j\)?
4. Da dove deriva \(y'-y=\Delta w\,x\)?
5. Perché compare \(x^2\) nell'errore quadratico?
6. Perché il quadrato evita la cancellazione tra valori positivi e negativi?
7. Perché \(x_j\) è collegato alla colonna \(j\) di una matrice \(W\)?
8. Qual è la differenza tra l'input di `gate/up` e quello di `down`?
9. Cosa sono i termini incrociati \(x_jx_k\)?
10. Perché gli errori di quantizzazione possono rafforzarsi o compensarsi?
11. Cosa rappresenta la matrice \(xx^T\)?
12. Cosa c'è sulla diagonale di \(xx^T\)?
13. Cosa rappresentano gli elementi fuori dalla diagonale?
14. Perché una statistica diagonale è molto più economica di una matrice completa?
15. In che modo l'imatrix aiuta la quantizzazione?
16. Perché “importanza dei singoli pesi” è una semplificazione?
17. Qual è la differenza tra `rendered_prompts.txt` e imatrix?
18. Perché l'imatrix non basta per il nostro predictor?
19. Come potremmo riutilizzare il collector dell'imatrix per raccogliere un dataset temporale?

---

# 28. Riassunto in una frase

> **L'imatrix osserva quanto il modello sollecita le dimensioni degli input delle sue reti interne, accumula statistiche aggregate sulla loro energia e usa questa informazione per quantizzare con maggiore attenzione i pesi i cui errori potrebbero propagarsi maggiormente sull'output.**

---

## Nota terminologica

Durante questi appunti abbiamo usato formule come:

\[
I_j=\sum x_j^2
\]

come modello concettuale per capire il meccanismo. È una rappresentazione utile per studiare la logica dell'importance weighting: il punto essenziale è che l'imatrix deriva l'importanza dalle **attivazioni osservate**, non da un'importanza intrinseca assegnata a priori ai singoli pesi.
