# Build 7 - dataset, architettura ed esperimenti

## Stato del progetto

Build 7 verifica se il contesto tabellare di un terzo layer causale migliora il
prefetch degli expert del token successivo. Il modello selezionato al momento e'
la FFN compatta con residuale gated delle probabilita' di transizione e con
regolarizzazione L2 $= 3 \cdot 10^{-6}$.

La selezione dell'architettura, dell'epoca e degli iperparametri e' stata fatta
soltanto su train e validation. Dopo questa scelta definitiva, il checkpoint
selezionato e' stato valutato una sola volta anche sul test set, senza alcun
ulteriore addestramento o modifica dei pesi.

Il miglior modello Build 7 e il miglior risultato **assoluto** corrente e'
`top4_patience10_l2_3e6_bce_v1`, con validation Recall@6 $=0.539002$.
Ogni nuovo esperimento deve riportare sia il proprio
valore sia la differenza rispetto a questo riferimento.

Il precedente miglior risultato assoluto ritrovato nei report era Build 6:
`candidate_09` del tuner `successive_halving_current_previous_hidden_kv_fusion_v1`,
con validation Recall@6 $=0.534194$. Il nuovo Top-4 lo supera di $0.004808$.
Il precedente migliore Build 7 completo
`state2048_bs2048_l2_3e6_bce_v1` rimane il riferimento architetturale per
misurare l'effetto della selezione dei rami: raggiungeva $0.530249$.

## Obiettivo e causalita'

Per predire i sei expert da prefetcharre per il token $t+1$ al layer MoE $l$,
Build 7 usa esclusivamente segnali gia' disponibili al token $t$:

```text
token t, layer l-2 ---+
token t, layer l-1 ---+--> expert target del token t+1, layer l
token t, layer l   ---+
```

Il target resta invariato rispetto a Build 6: un vettore multilabel di 256
expert del layer target. Build 7 e' limitata a $l=5,\ldots,42$, perche' il
layer $l-2$ deve essere anch'esso un layer MoE comparabile.

## Dati e feature

### State fixed-NNZ dei tre layer

L'unica espansione strutturale rispetto alla Build 6 e' il ramo state:
contiene il blocco corrente $l$ e due blocchi precedenti $l-1$ e $l-2$.

| Blocco | Feature incluse |
|---|---|
| Layer $l$ | routing history dei 6 expert, one-hot del layer, ID e valori dei 16 router logits, 5 statistiche KV |
| Layer $l-1$ | routing history, ID e valori dei router logits, 5 statistiche KV |
| Layer $l-2$ | le stesse feature tabellari di $l-1$ |

Il layer $l-2$ non aggiunge un nuovo one-hot del numero di layer: e'
deterministico dato $l$. Non sono stati aggiunti un terzo raw hidden 4096D o
un terzo KV 512D; questo isola il contributo incrementale dello state
tabellare.

Lo spazio ha 13.157 feature possibili. Ogni riga attiva 130 feature reali
(44 correnti, 43 precedenti e 43 pre-precedenti), piu' una sentinella/bias:
il formato fixed-NNZ usa quindi 131 slot contigui, non un vettore denso.

### Altri input

| Ramo | Input |
|---|---|
| Raw hidden | vettori quantizzati int8 4096D dei layer $l$ e $l-1$ |
| KV | vettori normalizzati float16 512D dei layer $l$ e $l-1$ |
| Posizione | `token_position` standardizzata, 1D |
| Transizioni | $\mathbf q_{t,l}\in\mathbb{R}^{256}$, probabilita' di transizione tra expert |

La matrice di transizione e' stimata esclusivamente dal train. Le righe train
ricevono probabilita' out-of-fold per prompt; validation e test ricevono la
matrice globale stimata dal train. Quindi la costruzione di $\mathbf q_{t,l}$
non usa mai il target della riga stessa.

Il dataset finale contiene 1.959.546 esempi: 1.570.008 train, 190.608
validation e 198.930 test.

## Architettura selezionata

```text
state fixed-NNZ l/l-1/l-2 13.157D -> Dense(2048) -> ReLU -> Dropout(0.3) --+
raw hidden l 4096D                -> Dense(64)   -> ReLU ------------------+
raw hidden l-1 4096D              -> Dense(64)   -> ReLU ------------------+--> Concatenate
KV l 512D                         -> Dense(256)  -> ReLU ------------------+        |
KV l-1 512D                       -> Dense(256)  -> ReLU ------------------+        v
token_position 1D -----------------------------------------------------------+   Dense(2048) -> ReLU -> Dropout(0.3)
                                                                                           |
                                                                                           v
                                                                                    base logits $\mathbf z_{\mathrm{FFN}}$ (256D)

$\mathbf q$ 256D -> $\log(\mathbf q+\epsilon)$ -> LayerNormalization -> $\alpha\,g(\mathbf x)$ --+ 
                                                                                                      |
                                                                       $\mathbf z_{\mathrm{final}}=\mathbf z_{\mathrm{FFN}}+\alpha g(\mathbf x)\operatorname{LN}(\log(\mathbf q+\epsilon))$
                                                                                                      |
                                                                                                   sigmoid
```

Il ramo $\mathbf q$ non passa nella concatenazione, perche' nella precedente
architettura concatenata era stato ignorato quasi completamente dalla FFN. Qui
influenza direttamente i 256 logits finali:

$$
\mathbf z_{\mathrm{final}}=
\mathbf z_{\mathrm{FFN}}+
\alpha\,g(\mathbf x)\,
\operatorname{LN}\!\left(\log(\mathbf q+\epsilon)\right).
$$

$\epsilon=10^{-6}$ evita il logaritmo di zero. I valori del log sono in
genere negativi perche' $0<q_i\leq1$; LayerNormalization centra e scala le
256 componenti per esempio, quindi gli expert relativamente piu' probabili
ricevono valori normalizzati piu' alti e possono essere positivi. $\alpha$ e'
uno scalare apprendibile, inizializzato a $0.1$; $g(\mathbf x)$ e' un gate
sigmoidale scalare appreso dal contesto.

La loss e' binary cross-entropy e il criterio di scelta e' il massimo
`val_recall_at_6`. Adam usa learning rate $1.5\cdot10^{-4}$.

## Tuning generico disponibile

Sono disponibili anche gli script generici
`tune_build7_two_previous_layers_bce.py` e
`train_build7_two_previous_layers_ffn.py`. Il tuner valuta 30 candidate con
selezione progressiva 3/7/20 epoche e usa BCE senza ranking loss. Lo spazio
include stato da 512, 1024, 2048 o 4096 unita', proiezioni raw hidden/KV e
layer di fusione differenti.

Questi script non sono stati usati per scegliere i risultati riportati sotto:
gli esperimenti effettivi hanno fissato l'architettura Build 6 `candidate_10`
con residuale gated, per misurare separatamente il nuovo state Build 7 e la
regolarizzazione.

## Esperimenti eseguiti

Tutti gli esperimenti sono addestrati da zero, con early stopping su
`val_recall_at_6` e pazienza 5. Salvo indicazioni diverse: state 2048,
batch 2048, dropout state/fusione 0.3, raw hidden 64/64, KV 256/256 e
fusione 2048.

| Esperimento | Modifica controllata | Migliore validation Recall@6 | Interpretazione |
|---|---|---:|---|
| `gated_residual_bce_v1` | Prima conferma Build 7 su L4, batch 4096 | 0.525510 | Il solo state $l-2$ non migliora automaticamente il risultato Build 6; confronto non perfettamente omogeneo per batch diverso. |
| `state2048_bs2048_bce_v1` | Baseline controllato A100 | 0.525279 | Riferimento per tutti i test successivi. |
| `state4096_bs2048_bce_v1` | State aumentato da 2048 a 4096 | 0.521591 | Piu' capacita' nel ramo state peggiora la generalizzazione; non e' un limite di memoria, ma non aggiunge informazione utile. |
| `state2048_bs2048_l2_1e6_bce_v1` | L2 $=10^{-6}$ | 0.528947 | Migliora la generalizzazione e ritarda l'overfitting. |
| `state2048_bs2048_l2_3e6_bce_v1` | L2 $=3\cdot10^{-6}$ | **0.530249** | Miglior configurazione corrente. |
| `state2048_bs2048_l2_1e5_bce_v1` | L2 $=10^{-5}$ | 0.520689 | Regolarizzazione eccessiva: sotto-adattamento evidente. |
| `state2048_bs2048_l2_3e6_branch_dropout01_bce_v1` | Dropout 0.1 dopo le quattro proiezioni raw hidden/KV | 0.526346 | Peggiora: i rami continui perdono informazione utile. Non usare branch dropout in questa configurazione. |

### Confronto della L2

| L2 | Validation Recall@6 |
|---:|---:|
| 0 | 0.525279 |
| $10^{-6}$ | 0.528947 |
| $3\cdot10^{-6}$ | **0.530249** |
| $10^{-5}$ | 0.520689 |

L2 $=3\cdot10^{-6}$ recupera circa $0.0050$ Recall@6 rispetto alla
baseline non regolarizzata. Con $10^{-5}$ anche il train resta molto piu'
basso, segnale di sotto-adattamento e non di semplice early stopping.

### Nota sul batch e sul test state 4096

Il test con state 4096 e batch 4096 ha causato un errore TensorFlow GPU
`Unexpected Event status: 1` sia su L4 sia su A100. Il tensore temporaneo
dell'embedding avrebbe forma $(4096,131,4096)$, cioe' oltre il limite signed
int32 di circa $2.147\times10^9$ elementi. Con batch 2048 il test e' stato
eseguito correttamente e ha comunque ottenuto un risultato inferiore.

## Configurazione da mantenere

```text
state_hidden_units = 2048
l2_regularization  = 3e-6
branch_dropout     = 0.0
batch_size         = 2048
dropout state/fusione = 0.3
```

Il modello migliore salvato su Drive e' nella cartella:

```text
DS4_Colab/build7_results/gated_residual_state2048_bs2048_l2_3e6_bce_v1
```

## Ablation del modello selezionato

L'ablation e' stata eseguita sul checkpoint selezionato con L2
$=3\cdot10^{-6}$, esclusivamente su train e validation. Non modifica i pesi:
per ogni ramo sostituisce il solo input con zero oppure ne permuta
deterministicamente le righe. La permutazione e' la misura principale, perche'
mantiene distribuzione e scala del ramo ma ne rompe l'allineamento con
contesto e target.

Per Build 7 lo state fixed-NNZ e' stato inoltre diviso in tre blocchi:
state corrente $l$, state precedente $l-1$ e nuovo state pre-precedente
$l-2$. Le perdite non sono additive, perche' i rami possono rappresentare
informazione parzialmente ridondante.

### Significato di `zero` e `permutation`

Le due ablation sono complementari e non misurano esattamente la stessa cosa.

| Metodo | Operazione | Domanda a cui risponde |
|---|---|---|
| `zero` | Sostituisce il ramo scelto con valori nulli. Per un sotto-blocco state azzera solo i suoi valori fixed-NNZ, lasciando invariati gli altri blocchi. | Quanto peggiora il modello se questo segnale non e' disponibile? |
| `permutation` | Assegna a ogni esempio il ramo dello stesso split preso da un altro esempio, mediante una permutazione deterministica. Distribuzione, scala e shape restano realistiche, ma si rompe l'allineamento con contesto e target. | Quanto conta che questo segnale sia quello corretto per quello specifico contesto? |

Ad esempio, l'ablation di state $l-2$ con `zero` annulla soltanto le feature
tabellari $l-2$ della riga corrente. Con `permutation`, invece, la riga
mantiene feature $l-2$ valide ma appartenenti a un altro token/prompt. Per
questo `permutation` e' la misura principale per ordinare l'importanza:
evita di introdurre un input artificialmente nullo e mantiene il ramo nella
sua distribuzione naturale.

Le letture utili sono:

- perdita alta con entrambi i metodi: il ramo e' utile;
- `permutation` molto piu' dannosa di `zero`: il modello usa soprattutto
  l'allineamento semantico del ramo con il resto dell'esempio;
- perdita quasi nulla con entrambi: il ramo e' poco usato oppure ridondante;
- perdita grande sul train ma molto minore su validation: il contributo e'
  meno trasferibile e puo' contenere adattamento specifico del train.

Le ablation sono importanza **condizionata al modello gia' addestrato**, non
una prova di causalita' intrinseca della feature: rami correlati possono
dividersi o sostituirsi parte della stessa informazione.

### Importanza su validation

Le tabelle di importanza sono ordinate in modo decrescente per **perdita su
`Validation permutation`**. Questa e' la colonna scelta perche' validation
misura la generalizzazione su prompt non usati per aggiornare i pesi e la
permutazione conserva distribuzione, scala e shape del ramo, rompendo soltanto
il suo allineamento con l'esempio. `zero` resta una misura complementare, ma
puo' creare un input artificialmente fuori distribuzione.

`State fixed-NNZ completo` e gli state $l$, $l-1$, $l-2$ si sovrappongono:
il primo contiene gia' gli altri tre blocchi. Il loro ordine e' utile per
descrivere il contributo condizionato di ciascun blocco, ma non devono essere
trattati come gruppi indipendenti ne' sommati per scegliere una Top-N.

| Posizione | Ramo ablated | Recall@6 dopo permutazione | Perdita dalla baseline |
|---:|---|---:|---:|
| Baseline | Nessuna ablation | 0.530250 | 0.000000 |
| 1 | `expert_transition_probabilities` | 0.230614 | **0.299636** |
| 2 | State fixed-NNZ completo | 0.358670 | 0.171580 |
| 3 | State corrente $l$ | 0.416940 | 0.113309 |
| 4 | Raw hidden $l$ | 0.475427 | 0.054823 |
| 5 | Raw hidden $l-1$ | 0.498421 | 0.031829 |
| 6 | KV $l$ | 0.512515 | 0.017734 |
| 7 | KV $l-1$ | 0.513553 | 0.016697 |
| 8 | State $l-1$ | 0.520209 | 0.010041 |
| 9 | Nuovo state $l-2$ | 0.520426 | 0.009824 |
| 10 | Token position | 0.530157 | 0.000093 |

### Confronto zero e permutazione

Questa tabella usa lo stesso ordinamento: perdita validation dopo
permutazione decrescente.

| Ramo | Train zero | Train permutazione | Validation zero | Validation permutazione |
|---|---:|---:|---:|---:|
| `expert_transition_probabilities` | 0.067960 | **0.317073** | 0.103513 | **0.299636** |
| State fixed-NNZ completo | 0.137872 | 0.213983 | 0.109040 | 0.171580 |
| State corrente $l$ | 0.073605 | 0.126702 | 0.073819 | 0.113309 |
| Raw hidden $l$ | 0.056499 | 0.080051 | 0.035897 | 0.054823 |
| Raw hidden $l-1$ | 0.031272 | 0.048021 | 0.018668 | 0.031829 |
| KV $l$ | 0.019805 | 0.027497 | 0.012107 | 0.017734 |
| KV $l-1$ | 0.020281 | 0.026800 | 0.011747 | 0.016697 |
| State $l-1$ | 0.018277 | 0.026313 | 0.004716 | 0.010041 |
| Nuovo state $l-2$ | 0.018048 | 0.026012 | 0.004490 | 0.009824 |
| Token position | 0.000046 | 0.000196 | 0.000017 | 0.000093 |

I valori in tabella sono perdite di Recall@6 rispetto alla baseline dello
split corrispondente, non Recall@6 assolute.

### Interpretazione

Il residuale gated rende le transition probabilities il ramo piu'
determinante del modello: su validation, permutare $\mathbf q$ fa scendere
Recall@6 da $0.530250$ a $0.230614$. Il risultato e' coerente su train
($0.317073$ di perdita) e validation ($0.299636$), quindi il ramo non e'
solo memorizzato sul train.

Il nuovo state $l-2$ viene utilizzato ma ha un contributo limitato e quasi
uguale a quello di $l-1$: rispettivamente $0.009824$ e $0.010041$ con
permutazione su validation. Il suo effetto e' maggiore sul train
($0.026012$) che su validation, segnale che una parte dell'informazione
aggiuntiva e' ridondante con $l$/$l-1$ oppure meno trasferibile tra prompt.
Questo spiega perche' l'aggiunta di $l-2$ non ha da sola prodotto un forte
incremento di Recall@6.

Il blocco corrente $l$ resta invece nettamente piu' informativo dei due
precedenti, mentre il ramo `token_position` e' di fatto irrilevante. Rispetto
all'ablation Build 6 gated-residual precedente, dove la permutazione di
$\mathbf q$ su validation causava circa $0.0919$ di perdita, il valore Build 7
di $0.299636$ conferma che nel modello selezionato il segnale di transizione
entra realmente nel ranking finale.

## Esperimento minimale state corrente + q

Per verificare se i due segnali apparentemente piu' forti fossero sufficienti
da soli, e' stata provata una rete con soltanto state corrente $l$ e
transition probabilities $\mathbf q$ con residuale gated. Sono stati rimossi
raw hidden, KV, `token_position` e gli state $l-1$/$l-2$.

```text
state corrente l fixed-NNZ 4.411D
    -> somma pesata dell'Embedding(2048) -> ReLU -> Dropout(0.3)
    -> Dense di fusione(2048) -> ReLU -> Dropout(0.3)
    -> Dense lineare di output(256) -> base logits z_state 256D --------+
                                                                    |
q 256D -> log -> LayerNormalization -> alpha x gate(fusione) -> residuale 256D
                                                                    |
                                                                    +--> logits finali -> sigmoid
```

Per questa variante e' stato materializzato `11_current_fixed_nnz_state`: 44
feature non nulle piu' sentinella, cioe' 45 slot invece dei 131 dello state
Build 7 completo. L'embedding state passa da circa 27 milioni a circa 9
milioni di parametri; il modello completo e' stimato intorno a 14 milioni di
parametri.

I due passaggi da 2048 unita' appartengono entrambi al **solo ramo state**:
il primo e' la somma pesata fixed-NNZ dell'embedding, equivalente a un Dense
sparse senza materializzare il vettore 4.411D; il secondo e' il Dense di
fusione che trasforma quella rappresentazione prima dei logits. Non esiste un
secondo gruppo concatenato in questo test. Il secondo input, $\mathbf q$, non
entra in nessuno dei due Dense: viene trasformato e sommato direttamente ai
logits finali tramite il residuale gated. Il gate usa la rappresentazione in
uscita dal Dense di fusione per decidere quanto fidarsi di $\mathbf q$.

I **base logits** non compaiono spontaneamente dal Dense di fusione: sono
l'output di un ulteriore classificatore lineare da 2048 a 256 unita', una per
expert. Indicando con $\mathbf h_{\mathrm{fusion}}\in\mathbb{R}^{2048}$
l'uscita del secondo dropout:

$$
\mathbf z_{\mathrm{state}}
=
W_{\mathrm{out}}\mathbf h_{\mathrm{fusion}}+\mathbf b_{\mathrm{out}}
\in\mathbb{R}^{256}.
$$

La componente $z_{\mathrm{state},i}$ e' il punteggio prodotto dal solo state
per l'expert $i$, prima della sigmoid. Il residuale delle transition
probabilities lo corregge componente per componente:

$$
\mathbf z_{\mathrm{final}}
=
\mathbf z_{\mathrm{state}}
+
\alpha\,g(\mathbf h_{\mathrm{fusion}})
\operatorname{LN}\!\left(\log(\mathbf q+\epsilon)\right).
$$

La sigmoid viene applicata soltanto dopo questa somma; poiche' e' monotona,
non cambia l'ordinamento dei logits usato per selezionare gli expert Top-6.

| Run | Configurazione | Migliore validation Recall@6 | Delta vs miglior Build 7 | Esito |
|---|---|---:|---:|---|
| `current_state_q_gated_l2_3e6_bce_v1` | state corrente + $\mathbf q$ gated; state/fusione 2048; L2 $=3\cdot10^{-6}$; batch 2048 | 0.470912 | $-0.068090$ | Esperimento completato. Forte sotto-adattamento: i due rami non sono sufficienti. Non e' un candidato selezionato. |

### Confronto con il miglior modello Build 7

Il miglior modello corrente Build 7, il Top-4 con patience 10, raggiunge
validation Recall@6 $=0.539002$. La variante minimale raggiunge $0.470912$:
perde quindi $0.068090$ Recall@6, cioe' circa il 12.6% del valore del miglior
modello.

Anche il valore train finale della variante minimale (circa $0.4927$) rimane
molto inferiore al train del modello completo ($0.602351$, misurato nella sua
valutazione di ablation). Il gap train/validation della variante minimale e'
invece contenuto, circa $0.0218$ ($0.4927 - 0.4709$). L'evidenza indica dunque
**sotto-adattamento dovuto alla rimozione di informazione utile**, non un
problema da risolvere aggiungendo ulteriore regolarizzazione.

In particolare, l'ablation del modello completo mostrava che gli hidden raw
restano utili anche dopo aver condizionato su state e $\mathbf q$; questo e'
il motivo per cui il Top-4 reintroduce raw hidden $l$ e raw hidden $l-1$.

## Esperimento Top-4 non sovrapposto

Il test in corso mantiene i quattro gruppi indipendenti con maggiore
importanza per **permutazione su validation**, evitando di contare due volte
lo state completo e il suo sotto-blocco corrente:

1. `expert_transition_probabilities` ($0.299636$);
2. state corrente $l$ ($0.113309$);
3. raw hidden corrente $l$ ($0.054823$);
4. raw hidden precedente $l-1$ ($0.031829$).

La rete resta intenzionalmente il piu' simile possibile al modello Build 7
migliore: stessi 2048 per state e fusione, stessi hidden raw proiettati a 64,
dropout 0.3 dopo state e fusione, L2 $=3\cdot10^{-6}$ e identico residuale
gated di $\mathbf q$. Vengono rimossi solo KV, token position e state
$l-1$/$l-2$.

```text
state corrente l fixed-NNZ (45 slot) -> Embedding pesato(2048) -> ReLU -> Dropout(0.3) --+
raw hidden l 4096D                 -> dequantizzazione -> Dense(64) -> ReLU ----------------+-->
raw hidden l-1 4096D               -> dequantizzazione -> Dense(64) -> ReLU ----------------+    concat 2176D
                                                                                                     |
                                                                                                     v
                                                                                    Dense fusione(2048) -> ReLU -> Dropout(0.3)
                                                                                                     |
                                                                                                     v
                                                                                     Dense lineare(256) -> base logits

q 256D -> log -> LayerNormalization -> alpha x gate(fusione) -> residuale 256D
base logits + residuale -> logits finali -> sigmoid
```

Il modello ha circa 14.55 milioni di parametri. Il primo run registrato e'
`top4_current_state_two_raw_hidden_q_gated_l2_3e6_bce_v1`, con state 2048,
fusione 2048, L2 $=3\cdot10^{-6}$, batch 2048 ed early stopping di pazienza
5 su `val_recall_at_6`. L'addestramento e' terminato con il miglior checkpoint
di validation ripristinato.

| Run | Gruppi inclusi | Gruppi rimossi | Patience | Train Recall@6 | Validation Recall@6 | Delta vs Build 7 completo | Delta vs migliore Build 6 |
|---|---|---|---:|---:|---:|---:|---:|
| `top4_current_state_two_raw_hidden_q_gated_l2_3e6_bce_v1` | state $l$, raw hidden $l$, raw hidden $l-1$, $\mathbf q$ gated | KV $l/l-1$, token position, state $l-1/l-2$ | 5 | 0.571731 | 0.537095 | +0.006846 vs 0.530249 | +0.002901 vs 0.534194 |
| `top4_patience10_l2_3e6_bce_v1` | state $l$, raw hidden $l$, raw hidden $l-1$, $\mathbf q$ gated | KV $l/l-1$, token position, state $l-1/l-2$ | 10 | **0.576967** | **0.539002** | **+0.008753** vs 0.530249 | **+0.004808** vs 0.534194 |

I valori train sono le metriche registrate da Keras alla stessa epoca del
miglior checkpoint validation (epoca 147 per patience 5, epoca 187 per
patience 10). Durante il training il dropout e' attivo, quindi non sono una
seconda valutazione inference-mode del checkpoint; per il confronto tra i due
run, eseguiti con la stessa procedura, sono comunque coerenti.

Con patience 10 il modello supera la stessa configurazione con patience 5 di
$0.001907$ Recall@6. Il miglioramento e' modesto ma non solo teorico: il run
ha continuato oltre il primo plateau e ha trovato un checkpoint migliore prima
dello stop. Questa e' quindi la configurazione selezionata per la valutazione
finale.

La semplificazione selettiva ha quindi funzionato: rispetto al modello Build 7
completo, rimuovere KV, token position e state $l-1$/$l-2$ riduce la capacita'
da circa 33.77 a 14.55 milioni di parametri e migliora la validation. Questo
e' coerente con le ablation: i rami eliminati erano deboli, ridondanti o meno
trasferibili, mentre i due raw hidden reinseriti colmano il sotto-adattamento
osservato nel test soltanto state+$\mathbf q$.

### Valutazione finale Top-6, Top-10, Top-20, Top-50 e posizioni degli expert target

Il checkpoint `top4_patience10_l2_3e6_bce_v1` e' stato valutato localmente in
sola inferenza sui tre split, senza addestramento e senza modificare i pesi.
Il report completo e riproducibile e'
`build_dataset_7/ml/neural_network/top4_patience10_l2_3e6_bce_v1/reports/recall_at_6_10_20_50_rank_distribution_train_validation_test.json`.
La valutazione streaming processa un batch alla volta, quindi include anche il
train completo senza materializzare in RAM tutti gli score $N\times256$.

| Split | Recall@6 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|
| Train | 0.605569 | 0.725152 | 0.835630 | 0.929502 |
| Validation | 0.539006 | 0.655730 | 0.775345 | 0.891281 |
| Test | 0.542485 | 0.659771 | 0.779376 | 0.894476 |

Validation e test sono molto vicini: il test e' persino superiore di $0.003479$
Recall@6. Questo conferma che il checkpoint scelto su validation trasferisce
al test senza un degrado osservabile. Il divario train--validation a Recall@6
e' $0.066563$ (train $0.605569$, validation $0.539006$): il modello e'
moderatamente overfit, ma non in modo tale da produrre una discrepanza fra
validation e test. Le metriche train registrate durante `fit` non vanno
confuse con questa tabella: qui dropout e' disattivato e viene sempre valutato
lo stesso checkpoint in inference-mode.

Il confronto omogeneo con il checkpoint Build 6 precedentemente selezionato
mostra un miglioramento a ogni soglia di recupero:

| Split | Modello | Recall@6 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Validation | Build 6 `candidate_09` | 0.536895 | 0.652267 | 0.770273 | 0.884137 |
| Validation | Build 7 Top-4 | **0.539006** | **0.655730** | **0.775345** | **0.891281** |
| Test | Build 6 `candidate_09` | 0.540792 | 0.657125 | 0.775077 | 0.887555 |
| Test | Build 7 Top-4 | **0.542485** | **0.659771** | **0.779376** | **0.894476** |

| Split | Delta Build 7 − Build 6 Recall@6 | Delta Recall@10 | Delta Recall@20 | Delta Recall@50 |
|---|---:|---:|---:|---:|
| Validation | **+0.002111** | **+0.003463** | **+0.005072** | **+0.007144** |
| Test | **+0.001693** | **+0.002646** | **+0.004299** | **+0.006921** |

I guadagni Build 7 rispetto a Build 6 sono $+0.002111$, $+0.003463$,
$+0.005072$ e $+0.007144$ su validation per Recall@6/10/20/50; sul test sono
$+0.001693$, $+0.002646$, $+0.004299$ e $+0.006921$. Il vantaggio cresce
all'aumentare di $k$: il Top-4 non migliora solo i sei expert immediatamente
prefetchabili, ma ordina meglio anche gli expert target appena oltre la soglia.

### Confronto della capacita' del modello

| Modello selezionato | Parametri trainable | Differenza rispetto a Build 6 |
|---|---:|---:|
| Build 6 `candidate_09` | 27.632.576 | riferimento |
| Build 7 Top-4 `top4_patience10_l2_3e6_bce_v1` | **14.545.794** | **-13.086.782 (-47,36%)** |

Il Top-4 Build 7 usa quindi solo il 52,64% dei parametri di Build 6, pur
migliorando tutte le metriche Recall@k riportate sopra. La riduzione non dipende
da un singolo strato: Build 7 elimina i rami KV e gli state $l-1/l-2$, mantiene
proiezioni raw hidden piu' compatte e conserva il ramo $\mathbf q$ come
residuale gated. Il risultato mostra che, per questo task, la selezione dei
quattro gruppi informativi e' piu' efficace dell'aumento indiscriminato della
capacita' del modello.

La distribuzione esatta delle posizioni dei sei expert target e':

| Posizione | Train | Validation | Test |
|---:|---:|---:|---:|
| 1 | 15.12% | 13.93% | 13.94% |
| 2 | 13.17% | 11.65% | 11.72% |
| 3 | 11.02% | 9.55% | 9.62% |
| 4 | 8.91% | 7.71% | 7.82% |
| 5 | 6.99% | 6.19% | 6.26% |
| 6 | 5.36% | 4.88% | 4.90% |
| 7 | 4.05% | 3.80% | 3.86% |
| 8 | 3.19% | 3.07% | 3.11% |
| 9 | 2.58% | 2.59% | 2.58% |
| 10 | 2.14% | 2.22% | 2.19% |
| Oltre Top-10 | 27.48% | 34.43% | 34.02% |

Le fasce cumulative rendono piu' immediata la lettura operativa:

| Fascia di posizione del target | Train | Validation | Test |
|---|---:|---:|---:|
| 1--6 | 60.56% | 53.90% | 54.25% |
| 7--10 | 11.96% | 11.67% | 11.73% |
| 11--20 | 11.05% | 11.96% | 11.96% |
| 21--50 | 9.39% | 11.59% | 11.51% |
| Oltre Top-50 | 7.05% | 10.87% | 10.55% |

Su validation e test circa 11.7% degli expert target cade subito fuori dal
Top-6, nelle posizioni 7--10: e' la quota piu' promettente per un eventuale
reranking o per un budget di prefetch piu' ampio. Recall@20 raggiunge circa
77.5--77.9%, mentre Recall@50 circa 89.1--89.4%; rimane quindi una quota di
circa 10--11% oltre la posizione 50 che non verrebbe risolta ampliando
indiscriminatamente il Top-k.

## Tuning Top-4 eseguito: successive halving A1-A3

Dato il risultato del Top-4 con patience 10, viene avviato un tuning
architetturale che conserva gli stessi quattro input e lo stesso residuale
gated di $\mathbf q$. Il tuning usa **solo train e validation**: il test non
viene caricato ne' consultato.

Gli iperparametri che hanno prodotto il miglior modello corrente sono rimasti fissi:

| Iperparametro fisso | Valore |
|---|---:|
| L2 | $3\cdot10^{-6}$ |
| Learning rate Adam | $1.5\cdot10^{-4}$ |
| Dropout state e fusione | $0.3$ |
| Dropout raw hidden | $0$ |
| Batch size | 2048 |
| Loss | BCE |

Lo spazio A non prova il dropout sui raw hidden. In precedenza il branch
dropout aveva peggiorato il modello completo; perciò qui si confrontano prima
le sole capacita' rappresentazionali, senza confondere l'effetto della
regolarizzazione aggiuntiva.

| Ramo | Scelte |
|---|---|
| State corrente fixed-NNZ | 1024, 2048, 4096 |
| Raw hidden $l$ | 64, 256, 1024 |
| Raw hidden $l-1$ | 64, 256 |
| Fusione | 1024, 2048 |

Sono quindi esattamente $3\times3\times2\times2=36$ candidate, inclusa la
baseline Top-4 attuale $2048/64/64/2048$. Il raw hidden corrente ha anche
l'opzione 1024 perche' e' il ramo hidden piu' importante; il precedente si
ferma a 256, coerentemente con la sua minore importanza e con il miglior
precedente Build 6, che usava un ramo precedente stretto.

| Fase | Candidate | Budget totale per candidata | Early stopping | Promozione |
|---|---:|---:|---|---:|
| A1 | 36 | 5 epoche | nessuno | migliori 12 |
| A2 | 12 | fino a 20 epoche | nessuno | migliori 6 |
| A3 | 6 | fino a 60 epoche | patience 5 | nessuna automatica |

Le candidate promosse hanno ripreso dai loro checkpoint completi (pesi e
stato Adam), non da zero. Il tuning e' terminato senza eseguire alcun
addestramento finale automatico; i report sono in
`build7_top4_successive_halving_a1_a3_v1/reports/tuning_report.json`.

### Risultati A1-A3

In A1 sono state promosse 12 candidate su 36; in A2 ne sono rimaste 6. La
tabella seguente riporta il massimo `val_recall_at_6` osservato durante A3,
non una valutazione finale sul test. Il confronto e' utile per scegliere quali
configurazioni meritano un riaddestramento completo, ma non sostituisce il
benchmark selezionato fino a 1000 epoche.

| Posizione A3 | Candidata | State | Raw hidden $l$ | Raw hidden $l-1$ | Fusione | Migliore validation Recall@6 | Delta vs miglior Build 7 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `candidate_32` | 4096 | 256 | 256 | 2048 | **0.5306** | $-0.0084$ |
| 2 | `candidate_20` | 2048 | 256 | 256 | 2048 | 0.5299 | $-0.0091$ |
| 3 | `candidate_34` | 4096 | 1024 | 64 | 2048 | 0.5195 | $-0.0195$ |
| 4 | `candidate_22` | 2048 | 1024 | 64 | 2048 | 0.5175 | $-0.0215$ |
| 5 | `candidate_36` | 4096 | 1024 | 256 | 2048 | 0.5128 | $-0.0262$ |
| 6 | `candidate_24` | 2048 | 1024 | 256 | 2048 | 0.5119 | $-0.0271$ |

Il riferimento nella colonna delta e' sempre
`top4_patience10_l2_3e6_bce_v1` con validation Recall@6 $=0.539002$.
L'A3 dura al massimo 60 epoche e applica patience 5, mentre il riferimento ha
raggiunto il suo massimo all'epoca 187 con patience 10: una perdita in questo
screening breve non dimostra da sola che una candidata non possa recuperare
con un training completo.

Emergono comunque tre segnali solidi. Tutte le finaliste usano fusione 2048;
il ramo state 4096 offre un vantaggio iniziale molto piccolo rispetto a 2048
quando entrambi gli hidden hanno 256 unita' ($0.5306$ contro $0.5299$), a
costo di epoche sensibilmente piu' lente; aumentare soltanto raw hidden $l$
a 1024 produce invece train Recall@6 molto alto ma validation decisamente
inferiore. Quest'ultimo e' un segnale di sovra-adattamento o di capacita'
sbilanciata, non una prova che due raw hidden larghi e simmetrici siano
inutili.

La scelta manuale successiva raccomandata e' riaddestrare da zero e in nuove
directory `candidate_20` e `candidate_32` fino a 1000 epoche con patience 10,
senza sovrascrivere `top4_patience10_l2_3e6_bce_v1`. Il modello migliore
attuale resta invariato finche' una di queste run complete non supera
0.539002 su validation.

## Tuning B eseguito: state e raw hidden simmetrici piu' larghi

Prima di portare candidate A3 a training completo, e' stata eseguita una seconda
ricerca mirata. A1-A3 ha provato raw hidden $l$ fino a 1024, ma raw hidden
$l-1$ solo fino a 256; non ha quindi verificato se capacita' elevata e
distribuita simmetricamente sui due rami possa generalizzare meglio.

| Ramo | Scelte B |
|---|---|
| State corrente fixed-NNZ | 4096, 8192 |
| Raw hidden $l$ e $l-1$ | 1024/1024, 2048/2048, 4096/4096 |
| Fusione | 2048, 4096 |

Le combinazioni sono $2\times3\times2=12$:

| ID | State | Raw hidden $l$ | Raw hidden $l-1$ | Fusione |
|---:|---:|---:|---:|---:|
| B01 | 4096 | 1024 | 1024 | 2048 |
| B02 | 4096 | 1024 | 1024 | 4096 |
| B03 | 4096 | 2048 | 2048 | 2048 |
| B04 | 4096 | 2048 | 2048 | 4096 |
| B05 | 4096 | 4096 | 4096 | 2048 |
| B06 | 4096 | 4096 | 4096 | 4096 |
| B07 | 8192 | 1024 | 1024 | 2048 |
| B08 | 8192 | 1024 | 1024 | 4096 |
| B09 | 8192 | 2048 | 2048 | 2048 |
| B10 | 8192 | 2048 | 2048 | 4096 |
| B11 | 8192 | 4096 | 4096 | 2048 |
| B12 | 8192 | 4096 | 4096 | 4096 |

Restano fissi L2 $=3\cdot10^{-6}$, learning rate Adam
$=1.5\cdot10^{-4}$, batch 2048, dropout 0.3 dopo state e fusione, nessun
dropout dopo i raw hidden e BCE. Anche B usa soltanto train e validation:
il test set non viene caricato ne' consultato.

| Fase | Candidate | Budget totale per candidata | Early stopping | Promozione |
|---|---:|---:|---|---:|
| B1 | 12 | 5 epoche | nessuno | migliori 6 |
| B2 | 6 | fino a 20 epoche | nessuno | migliori 3 |
| B3 | 3 | fino a 60 epoche | patience 5 | nessuna automatica |

### Risultati B1-B3

Le fasi hanno promosso 12 candidate a 6 e poi a 3. Le tre finaliste B3 sono
terminate con patience 5 molto prima del limite di 60 epoche: il massimo di
validation era gia' stato raggiunto nelle prime 8--11 epoche e le epoche
successive hanno aumentato il divario train/validation. I valori sotto sono
il massimo `val_recall_at_6` osservato in tutte le fasi B della candidata.

| Posizione B | Candidata | State | Raw hidden $l/l-1$ | Fusione | Migliore validation Recall@6 | Delta vs miglior Build 7 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `candidate_B11` | 8192 | 4096 / 4096 | 2048 | **0.5177** | $-0.0213$ |
| 2 | `candidate_B05` | 4096 | 4096 / 4096 | 2048 | 0.5170 | $-0.0220$ |
| 3 | `candidate_B12` | 8192 | 4096 / 4096 | 4096 | 0.5159 | $-0.0231$ |

Il riferimento e' `top4_patience10_l2_3e6_bce_v1`, con validation
Recall@6 $=0.539002$. Il migliore B rimane inoltre sotto alle migliori A3:
`candidate_32` raggiungeva $0.5306$ e `candidate_20` $0.5299$ nello
screening A3. Non esiste quindi alcun motivo empirico per portare una
candidata B a 1000 epoche.

La conclusione di B e' negativa ma informativa: aumentare insieme state,
raw hidden e fusione produce train Recall@6 molto alto (fino a circa 0.69
gia' a 20 epoche), ma peggiora la validation dopo il primo picco. In questo
regime L2 $=3\cdot10^{-6}$ e dropout soltanto su state/fusione non bastano a
regolarizzare la capacita' aggiuntiva. La fusione 4096 non mostra alcun
vantaggio stabile; state 8192 non compensa il costo computazionale e il
sovra-adattamento.

Non verra' avviata alcuna run B a 1000 epoche. La selezione manuale per i
training completi resta limitata alle due migliori candidate A3,
`candidate_20` e `candidate_32`, entrambe da riaddestrare da zero con
patience 10 in directory nuove. Il checkpoint selezionato
`top4_patience10_l2_3e6_bce_v1` resta immutato finche' un nuovo training
completo non supera $0.539002$ su validation.

## Tuning C eseguito parzialmente: limite di capacita' nelle prime cinque epoche

Il Tuning C non e' un tentativo di selezionare subito un nuovo modello finale:
misura esclusivamente se la validation Recall@6 continua a crescere nelle
primissime epoche aumentando drasticamente la capacita'. Tutte le candidate
usano gli stessi quattro input Top-4 e lo stesso residuale gated di
$\mathbf q$. Ciascuna viene addestrata da zero per **esattamente cinque
epoche**, senza early stopping e senza alcun test set.

| ID | State corrente fixed-NNZ | Raw hidden $l$ | Raw hidden $l-1$ | Fusione |
|---|---:|---:|---:|---:|
| C01 | 16384 | 8192 | 8192 | 8192 |
| C02 | 16384 | 8192 | 8192 | 16384 |
| C03 | 16384 | 16384 | 16384 | 8192 |
| C04 | 16384 | 16384 | 16384 | 16384 |
| C05 | 32768 | 8192 | 8192 | 8192 |
| C06 | 32768 | 8192 | 8192 | 16384 |
| C07 | 32768 | 16384 | 16384 | 8192 |
| C08 | 32768 | 16384 | 16384 | 16384 |

Sono fissi L2 $=3\cdot10^{-6}$, learning rate Adam
$=1.5\cdot10^{-4}$, batch 2048, dropout 0.3 dopo state e fusione, nessun
dropout sui raw hidden e BCE. Il confronto e' intenzionalmente circoscritto
alla curva validation delle epoche 1--5: una buona metrica iniziale non basta
per dichiarare una configurazione migliore del checkpoint selezionato a
$0.539002$.

Il tuner C salva atomicamente dopo ogni epoca completata il checkpoint
`latest` (pesi e stato Adam), il miglior checkpoint validation, la storia
delle metriche e il contatore delle epoche per candidata. Se la runtime viene
interrotta durante un'epoca, il riavvio con `--resume` riparte dall'ultima
epoca completata della stessa candidata: al massimo si ripete l'epoca
interrotta, senza perdere le candidate o le epoche gia' salvate.

### Risultati C01--C04

Le prime quattro candidate, tutte con state 16384, hanno completato le cinque
epoche previste. Nessuna mostra un segnale iniziale competitivo rispetto alle
candidate A3 o al checkpoint selezionato.

| Candidata | State | Raw hidden $l/l-1$ | Fusione | Validation Recall@6 epoca 5 | Delta vs miglior Build 7 |
|---|---:|---:|---:|---:|---:|
| `candidate_C01` | 16384 | 8192 / 8192 | 8192 | **0.5104** | $-0.0286$ |
| `candidate_C02` | 16384 | 8192 / 8192 | 16384 | 0.5093 | $-0.0297$ |
| `candidate_C03` | 16384 | 16384 / 16384 | 8192 | 0.5092 | $-0.0298$ |
| `candidate_C04` | 16384 | 16384 / 16384 | 16384 | 0.5073 | $-0.0317$ |

Il riferimento e' `top4_patience10_l2_3e6_bce_v1`, con validation
Recall@6 $=0.539002$. Anche il migliore C01 resta sotto al migliore B
($0.5177$) e molto sotto alle migliori candidate A3 (`candidate_32` a
$0.5306$, `candidate_20` a $0.5299$). La fusione 16384 non migliora il
risultato rispetto a 8192; allargare entrambi gli hidden a 16384 lo peggiora
ulteriormente.

Le epoche hanno richiesto circa 103--197 secondi ciascuna, contro circa
16--23 secondi per le candidate A3 moderate. Quindi la capacita' aggiuntiva
non offre neppure un vantaggio iniziale proporzionato al costo.

### Interruzione tecnica su C05 e decisione

`candidate_C05` (state 32768, raw hidden 8192/8192, fusione 8192) non ha
completato la prima epoca. TensorFlow ha terminato il processo nel kernel GPU
del gradiente sparse dell'Embedding con un conteggio di work element negativo:

```text
Check failed: work_element_count >= 0
```

Non e' un esaurimento della VRAM A100, ma un overflow signed-int32 del numero
di elementi temporanei: con batch 2048, 45 slot e state 32768 si ottengono
$2048\times45\times32768=3,019,898,880$ elementi, oltre $2^{31}-1$.
Un batch 1024 eviterebbe questo limite tecnico, ma renderebbe il confronto
non omogeneo e non e' giustificato dai risultati C01--C04. C05--C08 non sono
quindi eseguite; non usare `--resume` per questa directory, poiche'
ripeterebbe C05 con lo stesso batch e fallirebbe di nuovo.

La conclusione del tuning C e' che non conviene aumentare ulteriormente la
capacita'. Le sole candidate da considerare per un training lungo restano
`candidate_20` e `candidate_32` di A3; il checkpoint
`top4_patience10_l2_3e6_bce_v1` resta il modello selezionato finche' una run
completa non supera validation Recall@6 $=0.539002$.

### Archiviazione e spazio su Drive

Il Tuning C ha prodotto checkpoint eccezionalmente grandi: i soli checkpoint
`latest` e `best_validation` delle candidate C01--C04 occupavano circa 65 GB,
ma non hanno alcun valore operativo perche' l'esperimento e' stato scartato e
non verra' ripreso. Per evitare di occupare spazio inutilmente, la cartella
pesante dei checkpoint C e' stata eliminata da Google Drive. Su Drive resta
soltanto `top4_extreme_capacity_first5_c1_partial/reports/`, contenente
configurazione e stato/metriche del tuning (circa 8 KB). I checkpoint locali
della runtime possono essere eliminati dopo questa archiviazione leggera.

Solo dopo aver fissato definitivamente l'architettura, eseguire una sola
valutazione sul test set con il checkpoint selezionato.
