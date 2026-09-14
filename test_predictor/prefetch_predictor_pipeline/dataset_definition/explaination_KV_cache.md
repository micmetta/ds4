# KV cache nei Transformer autoregressivi

## Guida passo per passo collegata alla tesi di Michele Metta

Questo documento raccoglie e organizza, dall'inizio alla fine, i concetti discussi sulla **KV cache**. L'obiettivo è collegare l'ottimizzazione usata durante l'inferenza alle formule e all'architettura Transformer descritte nella tesi, senza dare per scontati i passaggi intermedi.

> **Idea centrale**  
> La KV cache non cambia il modello, i suoi pesi o il risultato matematico dell'attenzione. Memorizza, per ogni layer, le chiavi e i valori già calcolati per i token passati, evitando di ricalcolarli a ogni nuovo passo della generazione autoregressiva.

---

## 1. Da dove partire nella tesi

I punti della tesi più utili per comprendere la KV cache sono:

1. **Sezione 1.4 e descrizione dell'autoregressione**: il decoder genera un token e reinserisce quanto generato per produrre il token successivo.
2. **Sezione 1.4.2.2, Scaled Dot-Product Attention**: vengono introdotte le matrici (Q), (K), (V) e la formula dell'attenzione.
3. **Sezione sulla Multi-Head Attention**: lo stesso principio viene applicato in più head e in più layer.
4. **Sezione 1.4.3.1, Masked Multi-Head Attention del decoder**: la maschera rende l'attenzione causale, impedendo a un token di osservare quelli futuri.
5. **Descrizione di Llama**: Llama è un modello decoder-only e genera autoregressivamente; è quindi il caso d'uso tipico della KV cache.
6. **Configurazione sperimentale con `DEVICE_MAP: cuda:0` e GPU H100**: permette di capire dove, nella tua implementazione, vengono normalmente mantenuti pesi, attivazioni e KV cache durante l'inferenza.

La tesi spiega già il **calcolo matematico** dell'attenzione. La KV cache è il passaggio successivo: spiega **come implementare efficientemente quel calcolo durante il decoding**, quando la sequenza cresce di un token alla volta.

---

## 2. Richiamo della self-attention

Sia:

\[
X\in\mathbb{R}^{T\times d_{model}}
\]

la matrice contenente le rappresentazioni di (T) token. Come descritto nella tesi, il layer calcola:

\[
Q=XW^{(Q)},\qquad K=XW^{(K)},\qquad V=XW^{(V)}
\]

e poi:

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

dove (M) è la maschera causale nel decoder.

### 2.1 Significato intuitivo di Q, K e V

La spiegazione semantica fornita nella tesi può essere riassunta così:

- **Query (Q)**: rappresenta ciò che ogni token sta cercando nel contesto.
- **Key (K)**: descrive con quale tipo di informazione ciascun token può essere confrontato.
- **Value (V)**: contiene l'informazione che verrà effettivamente combinata nell'output.

Per un token (i), il prodotto (q_i k_j^T) misura quanto la query del token (i) è compatibile con la key del token (j). Dopo scaling e softmax, questi punteggi diventano pesi con cui combinare i value.

### 2.2 Attenzione in più layer e più head

L'esempio di questo documento usa una sola head e un solo layer. Nei modelli reali:

- ciascun layer produce le proprie (K) e (V);
- ciascuna KV head produce le proprie proiezioni;
- la cache non è quindi una sola matrice globale, ma una struttura organizzata per layer, batch e head.

Questo si collega alla spiegazione della Multi-Head Attention nella tesi: la KV cache non sostituisce le head, ma conserva i risultati (K) e (V) necessari a ciascuna di esse.

---

## 3. Perché la generazione autoregressiva crea calcoli ripetuti

Nella tesi descrivi il decoding come un processo nel quale il nuovo token viene aggiunto alla sequenza e il decoder continua la generazione. Per esempio:

```text
SELECT name
SELECT name FROM
SELECT name FROM users
```

Se a ogni passo fornissimo nuovamente l'intera sequenza senza riutilizzare alcun risultato, il modello ricalcolerebbe (Q), (K) e (V) anche per token già elaborati.

Al passo con (T) token, verrebbero ricostruite matrici con (T) righe. Al passo successivo, con (T+1) token, le prime (T) righe di (K) e (V) sarebbero uguali a prima, ma verrebbero calcolate di nuovo.

La causalità spiegata nella sezione 1.4.3.1 della tesi garantisce infatti che l'arrivo di un token futuro non possa modificare la rappresentazione già calcolata per un token passato allo stesso layer. È proprio questa proprietà a rendere possibile il riuso.

---

## 4. Prefill e decoding

È utile dividere l'inferenza in due fasi.

### 4.1 Prefill

Durante il **prefill**, il modello elabora in parallelo tutti i token del prompt iniziale. Se il prompt contiene (P) token, per ogni layer calcola le proiezioni (Q), (K) e (V) per tutti quei token.

La maschera causale continua a essere applicata: anche se il prompt viene processato in parallelo, il token in posizione (i) non può usare i token in posizione maggiore di (i).

Alla fine del prefill:

- l'ultima posizione produce le informazioni necessarie a predire il primo token di output;
- le (K) e le (V) del prompt vengono conservate nella KV cache.

### 4.2 Decoding

Durante il **decoding**, la generazione avanza normalmente di un token alla volta. Per ogni nuovo token si calcolano soltanto le sue nuove proiezioni (q_i), (k_i), (v_i), si aggiungono (k_i) e (v_i) alla cache e si esegue l'attenzione della nuova posizione sul contesto accumulato.

Questa distinzione approfondisce ciò che nella tesi descrivi come processo autoregressivo: il prefill prepara lo stato del prompt; il decoding estende quello stato incrementalmente.

---

## 5. Definizione della KV cache

Per ciascun layer (l), al termine del passo (i-1) conserviamo:

\[
K_{\text{cache},l}^{(i-1)}=
\begin{bmatrix}
k_{1,l}\\
k_{2,l}\\
\vdots\\
k_{i-1,l}
\end{bmatrix},
\qquad
V_{\text{cache},l}^{(i-1)}=
\begin{bmatrix}
v_{1,l}\\
v_{2,l}\\
\vdots\\
v_{i-1,l}
\end{bmatrix}
\]

Quando arriva il token (x_i), nel layer (l) si calcolano:

\[
q_{i,l}=x_{i,l}W_l^{(Q)}
\]

\[
k_{i,l}=x_{i,l}W_l^{(K)}
\]

\[
v_{i,l}=x_{i,l}W_l^{(V)}
\]

Si aggiornano quindi le cache:

\[
K_{\text{cache},l}^{(i)}=
\begin{bmatrix}
K_{\text{cache},l}^{(i-1)}\\
k_{i,l}
\end{bmatrix}
\]

\[
V_{\text{cache},l}^{(i)}=
\begin{bmatrix}
V_{\text{cache},l}^{(i-1)}\\
v_{i,l}
\end{bmatrix}
\]

Per evitare ambiguità nella notazione, distinguiamo:

\[
r_{i,l}=q_{i,l}\left(K_{\text{cache},l}^{(i)}\right)^T
\]

\[
s_{i,l}=\frac{r_{i,l}}{\sqrt{d_k}}
\]

\[
p_{i,l}=\operatorname{softmax}(s_{i,l})
\]

\[
z_{i,l}=p_{i,l}V_{\text{cache},l}^{(i)}
\]

Quindi la tua formulazione era corretta: **al passo (i), si aggiungono (k_i) e (v_i) alle cache ottenute fino al passo (i-1), poi la cache aggiornata viene usata per calcolare l'attenzione del token corrente**.

---

## 6. Esempio matriciale completo

Useremo gli stessi token dell'esempio SQL, coerenti con il dominio Text-to-SQL della tesi:

```text
SELECT name FROM users
```

Per rendere visibili tutti i calcoli supponiamo:

- un solo layer;
- una sola attention head;
- embedding di dimensione 2;
- assenza di bias;
- numeri scelti solo a scopo didattico.

### 6.1 Embedding

\[
x_1=\begin{bmatrix}1&0\end{bmatrix}
\quad\text{per `SELECT`}
\]

\[
x_2=\begin{bmatrix}0&1\end{bmatrix}
\quad\text{per `name`}
\]

\[
x_3=\begin{bmatrix}1&1\end{bmatrix}
\quad\text{per `FROM`}
\]

\[
x_4=\begin{bmatrix}2&1\end{bmatrix}
\quad\text{per `users`}
\]

### 6.2 Matrici dei pesi

\[
W^{(Q)}=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

\[
W^{(K)}=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
\]

\[
W^{(V)}=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
\]

Poniamo (d_k=2), quindi il fattore di scaling è (sqrt{2}).

---

## 7. Generazione senza KV cache

### 7.1 Primo passaggio: `SELECT name`

\[
X_{1:2}=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

Calcoliamo:

\[
Q_{1:2}=X_{1:2}W^{(Q)}=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

\[
K_{1:2}=X_{1:2}W^{(K)}=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
\]

\[
V_{1:2}=X_{1:2}W^{(V)}=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
\]

Gli score non scalati sono:

\[
Q_{1:2}K_{1:2}^T=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
\]

La maschera causale, descritta nella tesi per il decoder, impedisce a `SELECT` di vedere `name`:

\[
\operatorname{mask}(QK^T)=
\begin{bmatrix}
1&-\infty\\
1&1
\end{bmatrix}
\]

Dopo scaling e softmax:

\[
P_{1:2}=
\begin{bmatrix}
1&0\\
0.5&0.5
\end{bmatrix}
\]

Infine:

\[
Z_{1:2}=P_{1:2}V_{1:2}=
\begin{bmatrix}
1&0\\
1&0.5
\end{bmatrix}
\]

L'ultima riga, (z_2=[1,0.5]), contribuisce attraverso i componenti successivi del Transformer alla predizione del token `FROM`.

### 7.2 Secondo passaggio: `SELECT name FROM`

Senza cache viene fornita di nuovo l'intera sequenza:

\[
X_{1:3}=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
\]

Si ricalcolano tutte le proiezioni:

\[
Q_{1:3}=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
\]

\[
K_{1:3}=
\begin{bmatrix}
1&1\\
0&1\\
1&2
\end{bmatrix}
\]

\[
V_{1:3}=
\begin{bmatrix}
1&0\\
1&1\\
2&1
\end{bmatrix}
\]

Le prime due righe di (K) e (V) sono identiche a quelle già ottenute, ma vengono ricalcolate.

La matrice degli score è:

\[
Q_{1:3}K_{1:3}^T=
\begin{bmatrix}
1&0&1\\
1&1&2\\
2&1&3
\end{bmatrix}
\]

Dopo la maschera:

\[
S_{1:3}=
\begin{bmatrix}
1&-\infty&-\infty\\
1&1&-\infty\\
2&1&3
\end{bmatrix}
\]

Le prime due righe ripetono calcoli del passo precedente. Per la nuova predizione serve l'ultima riga:

\[
r_3=\begin{bmatrix}2&1&3\end{bmatrix}
\]

Dopo scaling:

\[
s_3=\frac{r_3}{\sqrt{2}}
\approx\begin{bmatrix}1.414&0.707&2.121\end{bmatrix}
\]

e softmax:

\[
p_3\approx\begin{bmatrix}0.284&0.140&0.576\end{bmatrix}
\]

L'output è:

\[
z_3=p_3V_{1:3}
\approx\begin{bmatrix}1.576&0.716\end{bmatrix}
\]

Questa rappresentazione contribuisce alla predizione di `users`.

### 7.3 Terzo passaggio: `SELECT name FROM users`

Senza cache vengono nuovamente calcolate tutte le righe:

\[
Q_{1:4}=
\begin{bmatrix}
1&0\\0&1\\1&1\\2&1
\end{bmatrix}
\]

\[
K_{1:4}=
\begin{bmatrix}
1&1\\0&1\\1&2\\2&3
\end{bmatrix}
\]

\[
V_{1:4}=
\begin{bmatrix}
1&0\\1&1\\2&1\\3&1
\end{bmatrix}
\]

e l'intera matrice degli score:

\[
Q_{1:4}K_{1:4}^T=
\begin{bmatrix}
1&0&1&2\\
1&1&2&3\\
2&1&3&5\\
3&1&4&7
\end{bmatrix}
\]

Con la maschera causale:

\[
S_{1:4}=
\begin{bmatrix}
1&-\infty&-\infty&-\infty\\
1&1&-\infty&-\infty\\
2&1&3&-\infty\\
3&1&4&7
\end{bmatrix}
\]

Per avanzare nella generazione è realmente nuova soltanto l'ultima riga:

\[
r_4=\begin{bmatrix}3&1&4&7\end{bmatrix}
\]

---

## 8. Lo stesso esempio con KV cache

### 8.1 Prefill di `SELECT name`

Il primo calcolo è uguale perché la cache è inizialmente vuota. Dopo aver ottenuto (K_{1:2}) e (V_{1:2}), li conserviamo:

\[
K_{\text{cache}}^{(2)}=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
\]

\[
V_{\text{cache}}^{(2)}=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
\]

### 8.2 Arriva soltanto `FROM`

Per (x_3=[1,1]) calcoliamo:

\[
q_3=x_3W^{(Q)}=\begin{bmatrix}1&1\end{bmatrix}
\]

\[
k_3=x_3W^{(K)}=\begin{bmatrix}1&2\end{bmatrix}
\]

\[
v_3=x_3W^{(V)}=\begin{bmatrix}2&1\end{bmatrix}
\]

Aggiorniamo le cache:

\[
K_{\text{cache}}^{(3)}=
\begin{bmatrix}
1&1\\
0&1\\
1&2
\end{bmatrix},
\qquad
V_{\text{cache}}^{(3)}=
\begin{bmatrix}
1&0\\
1&1\\
2&1
\end{bmatrix}
\]

Le prime due righe non vengono ricalcolate. Calcoliamo direttamente:

\[
r_3=q_3\left(K_{\text{cache}}^{(3)}\right)^T
=\begin{bmatrix}2&1&3\end{bmatrix}
\]

Poi:

\[
p_3=\operatorname{softmax}\left(\frac{r_3}{\sqrt{2}}\right)
\approx\begin{bmatrix}0.284&0.140&0.576\end{bmatrix}
\]

\[
z_3=p_3V_{\text{cache}}^{(3)}
\approx\begin{bmatrix}1.576&0.716\end{bmatrix}
\]

Il risultato è uguale a quello ottenuto senza cache.

### 8.3 Arriva soltanto `users`

Per (x_4=[2,1]):

\[
q_4=\begin{bmatrix}2&1\end{bmatrix},\qquad
k_4=\begin{bmatrix}2&3\end{bmatrix},\qquad
v_4=\begin{bmatrix}3&1\end{bmatrix}
\]

Le cache diventano:

\[
K_{\text{cache}}^{(4)}=
\begin{bmatrix}
1&1\\0&1\\1&2\\2&3
\end{bmatrix},
\qquad
V_{\text{cache}}^{(4)}=
\begin{bmatrix}
1&0\\1&1\\2&1\\3&1
\end{bmatrix}
\]

Calcoliamo soltanto la nuova riga:

\[
r_4=q_4\left(K_{\text{cache}}^{(4)}\right)^T
=\begin{bmatrix}3&1&4&7\end{bmatrix}
\]

È esattamente l'ultima riga ottenuta ricostruendo l'intera matrice (4\times4), ma non abbiamo ricalcolato le righe precedenti.

---

## 9. Confronto diretto

| Passaggio | Senza KV cache | Con KV cache |
|---|---|---|
| Prompt `SELECT name` | Calcola (Q,K,V) per 2 token | Calcola (Q,K,V) per 2 token e salva (K,V) |
| Arriva `FROM` | Ricalcola (Q,K,V) per 3 token | Calcola solo (q_3,k_3,v_3) |
| Arriva `users` | Ricalcola (Q,K,V) per 4 token | Calcola solo (q_4,k_4,v_4) |
| Score al passo 4 | Costruisce una matrice (4\times4) | Calcola il vettore (1\times4) della nuova posizione |
| Risultato matematico | Risultato di riferimento | Uguale, salvo normali dettagli numerici dell'implementazione |
| Memoria | Non conserva (K,V) | Conserva (K,V) per token, layer e KV head |

Nel piccolo esempio, per ciascuna proiezione:

- senza cache vengono elaborate (2+3+4=9) righe;
- con cache vengono elaborate (2+1+1=4) righe.

La KV cache, però, **non elimina ogni costo dipendente dalla lunghezza del contesto**. Al passo (i), (q_i) deve ancora essere confrontata con tutte le (i) chiavi e i pesi devono combinare tutti gli (i) valori. Elimina il ricalcolo delle proiezioni dei token passati e delle loro vecchie righe di attenzione.

---

## 10. Perché Q non viene messa in cache

Durante il decoding causale, (q_i) viene usata solo per calcolare l'output della posizione corrente:

\[
z_i=\operatorname{softmax}\left(
\frac{q_i(K_{\text{cache}}^{(i)})^T}{\sqrt{d_k}}
\right)V_{\text{cache}}^{(i)}
\]

Al passo successivo si usa una nuova query:

\[
z_{i+1}=\operatorname{softmax}\left(
\frac{q_{i+1}(K_{\text{cache}}^{(i+1)})^T}{\sqrt{d_k}}
\right)V_{\text{cache}}^{(i+1)}
\]

Quindi:

- (k_i) sarà confrontata anche con query future;
- (v_i) potrà contribuire anche agli output futuri;
- (q_i) non sarà usata per calcolare l'attenzione dei token futuri.

Nell'esempio SQL, quando il token corrente è `FROM`, (q_{\text{FROM}}) interroga le key di `SELECT`, `name` e `FROM`. Quando il token corrente diventa `users`, serve (q_{\text{users}}); non dobbiamo ricalcolare l'attenzione di `FROM`.

Un modo intuitivo per ricordarlo è:

> **Q è la domanda del token corrente. K e V sono l'indice e il contenuto del passato che potranno essere consultati anche dalle domande future.**

Il vero motivo che rende valida questa strategia è la maschera causale illustrata nella tesi: il nuovo token non può cambiare retroattivamente l'output delle posizioni precedenti.

---

## 11. Cosa accadrebbe senza maschera causale

### 11.1 Con maschera

Con tre token:

\[
S=
\begin{bmatrix}
q_1k_1^T&-\infty&-\infty\\
q_2k_1^T&q_2k_2^T&-\infty\\
q_3k_1^T&q_3k_2^T&q_3k_3^T
\end{bmatrix}
\]

Quando arriva il quarto token, si aggiunge soltanto una nuova riga effettivamente utilizzabile:

\[
\begin{bmatrix}
q_4k_1^T&q_4k_2^T&q_4k_3^T&q_4k_4^T
\end{bmatrix}
\]

I prodotti (q_1k_4^T,q_2k_4^T,q_3k_4^T) sarebbero mascherati. Le vecchie query non servono.

### 11.2 Senza maschera

In un singolo layer bidirezionale, l'aggiunta di (k_4) richiederebbe anche una nuova colonna:

\[
\begin{bmatrix}
q_1k_4^T\\q_2k_4^T\\q_3k_4^T
\end{bmatrix}
=Q_{\text{precedente}}k_4^T
\]

Per un aggiornamento incrementale di quel singolo layer, conservare le vecchie query potrebbe quindi essere utile.

### 11.3 Perché una Q cache non sarebbe comunque sufficiente

Senza causalità, ogni vecchio token potrebbe prestare attenzione al nuovo token. La softmax di ciascuna vecchia riga avrebbe un termine aggiuntivo nel denominatore:

\[
p_{1,j}=\frac{e^{s_{1,j}}}{\sum_{m=1}^{3}e^{s_{1,m}}}
\qquad\longrightarrow\qquad
p'_{1,j}=\frac{e^{s_{1,j}}}{\sum_{m=1}^{4}e^{s_{1,m}}}
\]

Di conseguenza cambierebbero gli output precedenti:

\[
z_1\neq z'_1,\qquad z_2\neq z'_2,\qquad z_3\neq z'_3
\]

Questi output sono gli input del layer successivo. Cambiandoli, cambierebbero anche le (Q), (K) e (V) dei layer superiori e le vecchie cache diventerebbero obsolete.

| Tipo di attenzione | I vecchi output cambiano con un nuovo token? | Vecchie Q utili? | Cache incrementale semplice |
|---|---:|---:|---:|
| Causale | No | No | Sì, KV cache |
| Bidirezionale, singolo layer con input fissati | Sì | Sì, per i nuovi score | Richiede aggiornamenti aggiuntivi |
| Bidirezionale, Transformer multilayer | Sì, con propagazione ai layer successivi | Sì, ma non bastano | In generale serve ricalcolare |

Questo collega direttamente la KV cache alla sezione della tesi sulla Masked Multi-Head Attention: **la cache funziona così bene perché la causalità rende stabile il passato quando la sequenza viene estesa**.

---

## 12. Dove viene memorizzata la KV cache

La cache viene normalmente collocata nella memoria del dispositivo che esegue l'attenzione:

- inferenza su GPU: generalmente **VRAM**;
- inferenza su CPU: **RAM**;
- esecuzione distribuita o offload: può essere ripartita o trasferita tra più dispositivi e livelli di memoria.

### 12.1 Collegamento alla tua tesi

Nella configurazione sperimentale riporti:

```text
DEVICE_MAP: cuda:0
```

e specifichi che il modello viene caricato interamente su una NVIDIA H100. Durante l'inferenza standard, se anche i layer di attenzione vengono eseguiti su quella GPU, la KV cache viene normalmente allocata nella VRAM della stessa GPU.

Questo non deriva dal significato matematico di (K) e (V), ma dall'esigenza prestazionale: a ogni decoding step la GPU deve leggere ripetutamente la cache per eseguire:

\[
q_iK_{\text{cache}}^T
\qquad\text{e}\qquad
p_iV_{\text{cache}}
\]

Se la cache fosse in RAM mentre l'attenzione viene calcolata sulla GPU, sarebbero necessari trasferimenti CPU-GPU, con latenza e consumo di banda aggiuntivi.

### 12.2 Il caso del tuo ROG

Sulla RTX 4080 Laptop da 12 GB, la VRAM deve ospitare una combinazione di:

\[
\text{pesi del modello}
+\text{KV cache}
+\text{attivazioni e buffer temporanei}
+\text{overhead del runtime}
\]

Perciò non basta verificare che i soli pesi entrino in 12 GB: contesto, batch e cache possono utilizzare la memoria rimanente. L'offload in RAM può aumentare la capacità disponibile, ma introduce trasferimenti più lenti.

---

## 13. Quanto occupa la KV cache

Una stima semplificata, per una singola sequenza, è:

\[
\text{Memoria KV}\approx
2\cdot L\cdot T\cdot H_{kv}\cdot d_h\cdot b
\]

dove:

- (2) rappresenta la presenza sia di (K) sia di (V);
- (L) è il numero di layer;
- (T) è il numero complessivo di token memorizzati, prompt più output già generato;
- (H_{kv}) è il numero di key-value head;
- (d_h) è la dimensione di ciascuna head;
- (b) è il numero di byte per elemento, per esempio 2 in FP16/BF16.

Per batch size (B), la stima diventa:

\[
\text{Memoria KV}\approx
2\cdot B\cdot L\cdot T\cdot H_{kv}\cdot d_h\cdot b
\]

Da questa formula segue che:

- un prompt più lungo crea una cache iniziale più grande;
- ogni token generato la fa crescere;
- un batch più grande moltiplica l'occupazione;
- più layer e più KV head aumentano la memoria richiesta;
- una precisione più bassa riduce i byte per elemento.

### 13.1 Multi-Head Attention, MQA e GQA

La tesi introduce la Multi-Head Attention. Nella forma classica ogni query head ha key e value head corrispondenti. Architetture moderne possono ridurre la memoria mediante:

- **Multi-Query Attention (MQA)**: molte query head condividono una sola KV head;
- **Grouped-Query Attention (GQA)**: gruppi di query head condividono un numero ridotto di KV head.

Poiché nella formula compare (H_{kv}), ridurre le KV head riduce direttamente la dimensione della cache. Nei modelli Llama che adottano GQA, questo è uno dei motivi per cui la cache è più contenuta rispetto a una Multi-Head Attention con lo stesso numero di query head.

---

## 14. Cosa ottimizziamo e cosa non cambia

### 14.1 Cosa viene evitato

La KV cache evita di:

- ricalcolare (k_j) e (v_j) dei token passati a ogni step;
- ricalcolare le vecchie righe degli score di attenzione;
- riprocessare l'intero prefisso attraverso ogni layer per ottenere il solo token successivo.

### 14.2 Cosa resta necessario

Per ogni nuovo token resta necessario:

- calcolare (q_i,k_i,v_i) in ogni layer;
- confrontare (q_i) con tutte le key memorizzate;
- applicare scaling e softmax;
- combinare tutti i value;
- eseguire proiezioni di output, residual connection, normalizzazioni e feed-forward network;
- selezionare il token successivo.

Con cache, il costo dell'attenzione di un singolo nuovo token cresce ancora con la lunghezza del contesto, ma si evita di ricostruire l'intera elaborazione del prefisso a ogni passo.

### 14.3 Accuratezza e output

La KV cache è un'ottimizzazione dell'esecuzione:

- non aggiunge conoscenza al modello;
- non migliora direttamente l'accuratezza Text-to-SQL;
- non modifica la query SQL desiderata;
- non cambia i pesi appresi durante fine-tuning o QLoRA.

Nel caso ideale produce lo stesso output del ricalcolo completo. Piccole differenze possono apparire solo per dettagli numerici o implementativi, non perché il principio della cache definisca un'attenzione diversa.

Nel contesto della tesi, quindi, la KV cache non modifica il metodo di generazione della query SQL: **riduce il lavoro necessario per eseguire la fase autoregressiva descritta nel capitolo sull'architettura e usata nell'applicazione**.

---

## 15. KV cache ed expert cache nel progetto DS4

Nel tuo progetto attuale è fondamentale non confondere due strutture:

| Struttura | Contenuto | Scopo | Dipendenza principale |
|---|---|---|---|
| KV cache | (K) e (V) calcolati per i token, per layer e KV head | Evitare il ricalcolo del passato nell'attenzione | Lunghezza del contesto, batch, layer, KV head |
| Expert cache | Pesi degli expert MoE mantenuti in un livello di memoria veloce | Evitare o anticipare caricamenti degli expert | Pattern di routing, capacità della cache, politica di sostituzione |

Il parametro DS4:

```bash
--ssd-streaming-cache-experts 8GB
```

riguarda la cache dei **pesi degli expert**, non la KV cache.

Il tuo predictor di prefetch mira ad anticipare quali expert MoE serviranno nei passi successivi, affinché possano essere trasferiti prima dal livello di memoria più lento. La KV cache, invece, accumula lo stato dell'attenzione relativo ai token già processati.

Le due strutture possono però competere indirettamente per RAM o VRAM e la KV cache può contenere segnali sullo stato semantico/computazionale della sequenza utili da studiare come feature del predictor. Questo non significa che la KV cache dica direttamente quale expert verrà scelto: l'eventuale valore predittivo deve essere misurato sperimentalmente.

---

## 16. Pseudocodice concettuale

### 16.1 Prefill

```python
hidden = embeddings(prompt_tokens)
cache = []

for layer in layers:
    Q = hidden @ layer.W_q
    K = hidden @ layer.W_k
    V = hidden @ layer.W_v

    attention_output = causal_attention(Q, K, V)
    hidden = layer.complete_block(hidden, attention_output)

    cache.append((K, V))

next_token = sample_or_argmax(output_head(hidden[-1]))
```

### 16.2 Singolo decoding step

```python
hidden = embedding(current_token)

for layer_index, layer in enumerate(layers):
    q_i = hidden @ layer.W_q
    k_i = hidden @ layer.W_k
    v_i = hidden @ layer.W_v

    K_cache, V_cache = cache[layer_index]
    K_cache = append(K_cache, k_i)
    V_cache = append(V_cache, v_i)

    scores = (q_i @ transpose(K_cache)) / sqrt(d_k)
    probabilities = softmax(scores)
    attention_output = probabilities @ V_cache

    hidden = layer.complete_block(hidden, attention_output)
    cache[layer_index] = (K_cache, V_cache)

next_token = sample_or_argmax(output_head(hidden))
```

Il codice reale deve anche gestire batch, head, posizioni, rotary positional embedding, layout dei tensori, precisione numerica e kernel ottimizzati. Lo pseudocodice serve a mostrare il flusso logico.

---

## 17. Precisazioni importanti

### 17.1 La cache contiene K e V per ogni layer

Non si calcolano (K) e (V) una sola volta all'ingresso del modello. Ogni layer riceve una rappresentazione diversa e genera le proprie proiezioni. Di conseguenza, ogni layer ha la propria porzione di KV cache.

### 17.2 Il token corrente viene incluso nella cache prima della sua attenzione

Nel decoder standard, la posizione (i) può prestare attenzione a sé stessa e alle posizioni precedenti. Per questo (k_i) e (v_i) vengono inclusi nelle cache usate per calcolare (z_i).

### 17.3 La maschera resta importante nel prefill

Durante il decoding token-per-token non esistono ancora posizioni future nella cache. Nel prefill, invece, più posizioni sono elaborate insieme: serve quindi la maschera triangolare descritta nella tesi.

### 17.4 Posizioni e RoPE

Nei decoder moderni, la posizione del token deve avanzare correttamente anche quando al modello viene passato un solo nuovo token. Con Rotary Positional Embedding, le trasformazioni posizionali vengono applicate alle query e alle key; le key conservate in cache devono essere coerenti con le posizioni già assegnate.

### 17.5 Fine della cache

La cache è legata alla specifica sequenza o richiesta. Quando la generazione termina, può essere liberata. In sistemi server con batching, prefix caching o riuso esplicito dei prefissi, il ciclo di vita può essere più complesso.

---

## 18. Schema mentale finale

### Durante il prefill

\[
\text{prompt completo}
\longrightarrow Q,K,V\text{ per tutti i token}
\longrightarrow \text{attenzione causale}
\longrightarrow \text{salvataggio di }K,V
\]

### Al passo di decoding (i)

\[
x_i\longrightarrow(q_i,k_i,v_i)
\]

\[
k_i,v_i\longrightarrow\text{aggiornamento delle cache}
\]

\[
q_i,K_{\text{cache}}^{(i)}
\longrightarrow r_i
\longrightarrow s_i
\longrightarrow p_i
\]

\[
p_i,V_{\text{cache}}^{(i)}\longrightarrow z_i
\]

### Frase da ricordare

> La KV cache conserva, in ciascun layer, le key e i value dei token già processati. Durante il decoding causale, il nuovo token calcola una nuova query, una nuova key e un nuovo value; aggiunge key e value alla cache e usa la propria query per consultare l'intero passato, senza ricalcolarlo.

---

## 19. Collegamento conclusivo con la tesi

La tesi costruisce tutti i prerequisiti necessari:

- introduce (Q), (K), (V) e la Scaled Dot-Product Attention;
- mostra che la Multi-Head Attention replica il calcolo su più spazi di rappresentazione;
- descrive lo stack di decoder;
- spiega la maschera causale;
- descrive l'autoregressione, cioè il reinserimento dei token generati;
- applica un modello Llama decoder-only alla generazione Text-to-SQL;
- esegue il modello su GPU mediante `cuda:0`.

La KV cache è la conseguenza implementativa di questi elementi: poiché il decoder è causale, le (K) e le (V) passate restano valide; poiché la generazione è autoregressiva, vengono riutilizzate a ogni nuovo token; poiché l'attenzione viene eseguita in ogni layer, la cache viene mantenuta per ogni layer; poiché il modello della tesi gira sulla H100, la cache viene normalmente tenuta nella VRAM per essere letta rapidamente.

In altre parole, la tesi descrive **che cosa calcola** il decoder, mentre la KV cache chiarisce **come evitare di ripetere inutilmente una parte di quel calcolo durante l'inferenza**.

