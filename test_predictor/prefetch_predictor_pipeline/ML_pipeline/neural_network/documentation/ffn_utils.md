DOCUMENTAZIONE DI build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn
=============================================================================

Scopo del documento
-------------------

Questo documento descrive esclusivamente la funzione Python:

    build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn

contenuta in ffn_utils.py.

La funzione costruisce la rete neurale feed-forward multi-input attualmente
usata come riferimento per il training ottimizzato con state fixed-NNZ. Questa
variante evita di ricostruire, a ogni batch, il vettore state sparso completo
di 4.413 valori: conserva esattamente la stessa trasformazione lineare del
ramo state, ma lavora soltanto sulle feature effettivamente presenti.

La rete usa contemporaneamente:

1. lo state del router;
2. lo stato nascosto grezzo del modello DeepSeek;
3. l'embedding congelato del token corrente;
4. il vettore Key-Value compatto corrente;
5. la posizione standardizzata del token.


Configurazione descritta in questo documento
---------------------------------------------

La configurazione di riferimento corrente e' la seguente:

    --state-hidden-units 512
    # --fusion-hidden-layers non passato: nessun layer di fusione aggiuntivo
    --activation relu
    --dropout 0.3
    --hidden-projection-dimension 64
    --token-projection-dimension 64
    --raw-hidden-projection-activation linear
    --token-projection-activation linear
    --batch-size 4096
    --l2-regularization 0.0     (valore predefinito, se non passato)

Con questa configurazione non vengono creati ulteriori strati nascosti dopo
la concatenazione: il vettore concatenato entra direttamente nel classificatore
finale Dense(256).

I due parametri sono volutamente separati:

| Parametro | A quale parte della rete si riferisce | Esempio |
|---|---|---|
| `--state-hidden-units` | Il solo ramo state fixed-NNZ: dopo la somma pesata produce `Dense(512)`. | `--state-hidden-units 512` |
| `--fusion-hidden-layers` | Eventuali Dense posti **dopo** la concatenazione dei cinque rami e **prima** di `Dense(256)`. | `--fusion-hidden-layers 512,256` |

Se `--fusion-hidden-layers` non viene passato, la lista e' vuota e non viene
creato alcun layer di fusione. Per compatibilita' con le run precedenti,
`--hidden-layers 512,256` continua a funzionare e viene interpretato come
`--state-hidden-units 512` e `--fusion-hidden-layers 256`; nei nuovi comandi
va usata invece la coppia di parametri separati. I due formati non possono
essere passati insieme.

Le due proiezioni 4.096D -> 64D hanno anch'esse parametri separati. Il valore
predefinito `linear` conserva esattamente la rete di riferimento; per la nuova
ablazione con ReLU bisogna passare entrambi i parametri con valore `relu`.
Questo permette al futuro tuning di provare separatamente le quattro
combinazioni `linear/linear`, `relu/linear`, `linear/relu` e `relu/relu`.


Architettura della rete
-----------------------

    state 4.413D
      (routing history, layer, router ID/score, 5 statistiche KV)
            |
            v
    Dense(512) -> ReLU -> Dropout(0.3) ------------------+
                                                           |
    hidden raw 4.096D -> Dense(64) -----------------------+
                                                           |
    token embedding congelato 4.096D -> Dense(64) --------+
                                                           +--> Concatenate(1.153D) --> Dense(256) --> sigmoid
    KV completo normalizzato 512D -> identita' -----------+       (expert_scores)
                                                           |
    token_position standardizzata 1D ---------------------+

La dimensione 1.153 della concatenazione deriva da:

    512  valori dal ramo state
     64  valori dal ramo hidden raw
     64  valori dal ramo token embedding
    512  valori dal vettore Key-Value diretto
      1  valore dalla posizione del token
    ----
   1153  valori totali

L'ultimo Dense produce 256 probabilita' indipendenti, una per ogni esperto del
modello Mixture-of-Experts. La sigmoid trasforma ciascun logit in probabilita'.
Recall@6 valuta se gli esperti corretti sono fra i sei punteggi piu' alti.


1. Ramo state: 4.413 feature possibili, 44 feature usate per esempio
------------------------------------------------------------------------

Lo spazio completo dello state ha 4.413 colonne:

    256   possibili esperti della routing history
     40   possibili layer
   4096   possibili ID del router: 16 posizioni di rank x 256 esperti
     16   valori numerici dei router logits
      5   statistiche Key-Value correnti
   ----
   4413   feature totali

Un singolo esempio ha sempre 44 feature state memorizzate:

      6  esperti selezionati al token precedente (routing history H=1)
      1  layer corrente
     16  ID degli esperti nelle 16 posizioni della classifica del router
     16  valori standardizzati dei 16 router logits
      5  statistiche standardizzate del vettore Key-Value corrente
   ----
     44  feature

La rappresentazione fixed-NNZ aggiunge un quarantacinquesimo slot sentinella,
con indice 4413 e valore 1.0. Non e' una feature del dataset: rappresenta il
termine additivo costante, cioe' il bias del Dense(512) originale.

Pertanto gli input del ramo state hanno shape:

    state_feature_indices : (numero_esempi, 45), uint16 su disco
    state_feature_values  : (numero_esempi, 45), float32 su disco

Durante il training, per un batch da 4.096 esempi, hanno shape:

    state_feature_indices : (4096, 45), int32 in TensorFlow
    state_feature_values  : (4096, 45), float32 in TensorFlow


Implementazione del Dense(512) state tramite Embedding
------------------------------------------------------

Il ramo state e' logicamente:

    state_4413D @ kernel_(4413, 512) + bias_(512)

Non viene costruito esplicitamente state_4413D. Al suo posto il codice usa:

    tf.keras.layers.Embedding(
        input_dim=4414,
        output_dim=512,
        embeddings_regularizer=None,
        name="state_fixed_nnz_linear_weights",
    )(state_indices)

Embedding viene usato qui come tabella di pesi, non come embedding linguistico.
La tabella apprendibile ha shape:

    (4414, 512)

Le righe da 0 a 4412 sono i pesi delle 4.413 feature state. La riga 4413 e'
il bias. Per ogni esempio, state_indices seleziona 44 righe delle feature
attive e la riga sentinella. L'output e':

    state_weight_rows : (batch_size, 45, 512)

Con batch_size=4096:

    state_weight_rows : (4096, 45, 512)

I metodi Keras successivi fanno quanto segue:

    Reshape((45, 1))
        Trasforma i 45 valori in una colonna, shape (batch_size, 45, 1).

    Multiply()
        Moltiplica ogni vettore di pesi da 512 valori per il valore della sua
        feature. Le feature categoriche hanno valore 1; router logits e
        statistiche Key-Value hanno i rispettivi valori standardizzati.

    Lambda(tf.reduce_sum(..., axis=1))
        Somma i 45 vettori pesati e restituisce (batch_size, 512). Questo e'
        esattamente il risultato del Dense state, incluso il bias sentinella.

    Activation("relu")
        Applica ReLU alle 512 attivazioni del ramo state.

    Dropout(0.3)
        Solo durante l'addestramento azzera casualmente il 30 per cento delle
        attivazioni state. Durante validation e test non modifica i valori.


Dal primo esempio train al vettore state da 512 valori
-------------------------------------------------------

Questa sezione segue soltanto il ramo state del primo esempio train appena
mostrato. Non descrive ancora cosa accade agli hidden raw, al token embedding o
al vettore Key-Value a 512 valori.

### Passo 0 — Cosa esiste prima di entrare nella rete

Lo spazio state completo ha **4.413 feature possibili**, con indici da `0` a
`4412`. Non significa però che ogni esempio debba contenere 4.413 valori:
questo primo esempio ne attiva soltanto **44**. Il dataset fixed-NNZ salva
quindi esclusivamente gli indici delle 44 feature attive e i loro valori,
invece di salvare un vettore quasi interamente composto da zeri.

Ai 44 slot reali viene aggiunto un quarantacinquesimo slot tecnico:
`indice = 4413`, `valore = 1.0`. Questo non rappresenta una feature del
dataset; è la sentinella usata dal ramo state per aggiungere il termine di
bias alla somma pesata. Perciò ogni riga dei file fixed-NNZ ha esattamente
**45 slot = 44 feature state attive + 1 sentinella del bias**.

Nel file `state_feature_indices_train.npy`, la prima riga contiene dunque i
45 indici delle feature attive di questo esempio, seguiti dalla sentinella:

```text
15, 90, 96, 112, 119, 202, 256,
392, 671, 1010, 1154, 1432, 1591, 1950, 2189, 2499, 2671,
3096, 3336, 3615, 3630, 3948, 4389,
4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399,
4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407,
4408, 4409, 4410, 4411, 4412,
4413
```

Nella riga corrispondente di `state_feature_values_train.npy` ci sono i valori
associati agli stessi indici. Le coppie principali sono:

| Slot | Indice | Feature selezionata | Valore associato |
|---:|---:|---|---:|
| 1 | 15 | Esperto 15 nella routing history | `1` |
| 2 | 90 | Esperto 90 nella routing history | `1` |
| 3 | 96 | Esperto 96 nella routing history | `1` |
| 4 | 112 | Esperto 112 nella routing history | `1` |
| 5 | 119 | Esperto 119 nella routing history | `1` |
| 6 | 202 | Esperto 202 nella routing history | `1` |
| 7 | 256 | Layer 3 | `1` |
| 8–23 | vari | I 16 ID degli esperti classificati dal router | `1` ciascuno |
| 24–39 | 4392–4407 | I 16 punteggi del router standardizzati | il rispettivo punteggio |
| 40–44 | 4408–4412 | Le 5 statistiche Key-Value standardizzate | il rispettivo valore |
| 45 | 4413 | Sentinella del bias | `1` |

I 16 ID del router, quindi le feature degli slot da 8 a 23, sono:

| Rank router | Indice state | Feature attiva |
|---:|---:|---|
| 0 | 392 | `router_logit_expert_id_0_96` |
| 1 | 671 | `router_logit_expert_id_1_119` |
| 2 | 1010 | `router_logit_expert_id_2_202` |
| 3 | 1154 | `router_logit_expert_id_3_90` |
| 4 | 1432 | `router_logit_expert_id_4_112` |
| 5 | 1591 | `router_logit_expert_id_5_15` |
| 6 | 1950 | `router_logit_expert_id_6_118` |
| 7 | 2189 | `router_logit_expert_id_7_101` |
| 8 | 2499 | `router_logit_expert_id_8_155` |
| 9 | 2671 | `router_logit_expert_id_9_71` |
| 10 | 3096 | `router_logit_expert_id_10_240` |
| 11 | 3336 | `router_logit_expert_id_11_224` |
| 12 | 3615 | `router_logit_expert_id_12_247` |
| 13 | 3630 | `router_logit_expert_id_13_6` |
| 14 | 3948 | `router_logit_expert_id_14_68` |
| 15 | 4389 | `router_logit_expert_id_15_253` |

I 16 valori standardizzati del router, negli slot da 24 a 39, sono:

```text
router_logit_value_0  = -1.61500335
router_logit_value_1  = -1.66757727
router_logit_value_2  = -2.11209822
router_logit_value_3  = -2.00715661
router_logit_value_4  = -1.93500447
router_logit_value_5  = -1.81196034
router_logit_value_6  = -1.93749273
router_logit_value_7  = -1.83856726
router_logit_value_8  = -1.71391559
router_logit_value_9  = -1.85004640
router_logit_value_10 = -1.84914732
router_logit_value_11 = -1.78769433
router_logit_value_12 = -1.74574232
router_logit_value_13 = -1.69826663
router_logit_value_14 = -1.67387772
router_logit_value_15 = -1.67472768
```

Le cinque statistiche Key-Value, negli slot da 40 a 44, sono:

```text
kv_current_l2_norm = -0.802221656
kv_current_rms     = -0.802221656
kv_current_mean    = -0.060272869
kv_current_std     = -0.797440290
kv_current_max_abs =  1.572362300
```

Il significato della coppia `(indice, valore)` e' semplice: l'indice dice
*quale feature e' presente*, il valore dice *con quale intensita' contribuisce*.
Per una feature categorica, come `layer_3`, l'intensita' e' `1`. Per una
feature numerica, come `router_logit_value_0`, l'intensita' e' il suo valore
standardizzato, qui `-1.61500335`.

### Prima del Passo 1 — Dalla riga di comando ai valori reali del batch

La funzione `build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn` non viene
eseguita direttamente da sola. Viene chiamata dallo script:

```text
ML_pipeline/neural_network/
train_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py
```

Il comando di training lancia proprio questo script. La parte rilevante del
comando corrente e' la seguente:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/\
train_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/\
build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --batch-size 4096 \
  --state-hidden-units 512 \
  --hidden-projection-dimension 64 \
  --token-projection-dimension 64
```

Di seguito e' riportato l'intero percorso dei valori, dall'avvio del comando
fino al primo aggiornamento dei pesi della rete.

#### 1. Il terminale avvia lo script di training

Il comando fornisce allo script i percorsi dei dati e gli iperparametri. In
particolare:

```text
--fixed-nnz-state-data-dir
    indica la cartella con gli array fixed-NNZ dello state;

--batch-size 4096
    indica che ogni step di training deve contenere 4.096 esempi;

--state-hidden-units 512
    indica che il ramo state termina in 512 valori;

--fusion-hidden-layers 512,256
    e' opzionale e inserirebbe due Dense, prima da 512 e poi da 256 neuroni,
    tra la concatenazione e il classificatore finale. Nella configurazione
    documentata non viene passato, quindi non viene inserito alcun Dense;

--hidden-projection-dimension 64
--token-projection-dimension 64
    indicano le due proiezioni lineari da 4.096 a 64 valori.
```

#### 2. Lo script legge gli array reali dal disco

Lo script chiama:

```python
train_state_indices, train_state_values, state_metadata = load_fixed_nnz_state_inputs(
    args.fixed_nnz_state_data_dir, "train", y_train.shape[0]
)
```

Per lo split train, questa funzione apre realmente:

```text
state_feature_indices_train.npy  -> shape (1652640, 45)
state_feature_values_train.npy   -> shape (1652640, 45)
fixed_nnz_state_metadata.json    -> feature_count=4413, input_slots=45
```

Quindi, prima ancora che Keras inizi il training, `train_state_indices` e
`train_state_values` contengono gia' tutte le 1.652.640 righe reali del train
set. La prima riga e' l'esempio dettagliato in questa sezione; le righe
successive rappresentano gli altri esempi.

Lo script legge anche le label, gli indici token, le posizioni, gli hidden int8
e i vettori Key-Value. Tutti gli array dello stesso split hanno lo stesso
numero di righe e la riga `i` di ogni array descrive lo stesso esempio.

#### 3. Lo script costruisce la rete, ma non le passa ancora un batch

Poi il trainer invoca la funzione documentata qui:

```python
model = build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(
    state_metadata["feature_count"],
    state_metadata["input_slots"],
    frozen_embeddings,
    state_hidden_units,
    fusion_hidden_layers,
    args.activation,
    args.dropout,
    args.learning_rate,
    args.l2_regularization,
    tf,
    args.hidden_projection_dimension,
    args.token_projection_dimension,
    args.raw_hidden_projection_activation,
    args.token_projection_activation,
    None,          # KV diretto nella configurazione di riferimento.
    "relu",        # Usato soltanto se viene scelta una proiezione KV.
    train_scale,
)
```

In questa chiamata, `state_metadata["input_slots"]` vale `45`. Per questo la
funzione crea gli input Keras con `shape=(45,)`. In questo preciso momento gli
oggetti creati da `tf.keras.Input` sono soltanto le porte simboliche della
rete: non contengono ancora la prima riga train, ne' un batch reale.

E' utile distinguere con precisione i tre tipi di oggetto coinvolti:

```text
tf.keras.Input(...)
    definisce una porta di ingresso e la forma dei dati che puo' ricevere;
    all'inizio e' simbolico, quindi non contiene esempi reali.

train_dataset
    conserva il collegamento agli array e fornisce i dati reali del batch.

tf.keras.layers.Embedding e tf.keras.layers.Dense
    contengono parametri apprendibili della rete, cioe' pesi e bias; non
    conservano gli esempi del dataset.
```

In particolare, `Embedding(4414, 512)` conserva la matrice dei pesi dello
state, mentre `state_indices` e `state_feature_values` ricevono i dati reali
del batch. I risultati intermedi prodotti dai layer durante il forward pass
esistono soltanto temporaneamente; dopo il batch rimangono soltanto i pesi
aggiornati dall'ottimizzatore Adam.

#### 4. Lo script crea il dataset che consegnera' i batch alla rete

Subito dopo costruisce il dataset TensorFlow:

```python
train_dataset = make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
    train_state_indices,
    train_state_values,
    y_train,
    train_token_indices,
    train_positions,
    train_hidden,
    train_kv,
    args.batch_size,
    True,
    tf,
    args.shuffle_block_size,
).repeat()
```

La funzione `make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset`
crea un generatore di batch. A ogni iterazione seleziona una porzione di righe
degli array. Per il primo batch senza considerare lo shuffle, la porzione e':

```python
selected = slice(0, 4096)
```

e restituisce un dizionario con i valori reali:

```python
{
    "state_feature_indices": state_feature_indices[selected].astype(np.int32, copy=False),
    "state_feature_values": state_feature_values[selected],
    "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
    "token_position": token_positions[selected].astype(np.float32, copy=False),
    "raw_hidden_state": quantized_hidden[selected],
    "kv_current_state": normalized_kv[selected].astype(np.float16, copy=False),
}, labels[selected].astype(np.float32)
```

In altre parole, il generatore prende 4.096 righe reali da ogni array e le
unisce in un unico batch coerente. Per il ramo state produce concretamente:

```text
"state_feature_indices" -> matrice reale (4096, 45)
"state_feature_values"  -> matrice reale (4096, 45)
```

Lo shuffle non separa mai una riga di indici dalla corrispondente riga di
valori: cambia l'ordine dei blocchi di esempi, ma tutte le sorgenti usano la
stessa selezione `selected`.

#### 5. `model.fit` riceve un batch reale dal dataset

L'addestramento inizia qui nel trainer:

```python
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    validation_data=validation_dataset.repeat(),
    validation_steps=validation_steps,
    epochs=args.epochs,
    callbacks=callbacks,
    shuffle=False,
    verbose=2,
)
```

`model.fit` chiede il primo elemento a `train_dataset`. Il dataset restituisce
il dizionario del passo precedente e le label reali. Keras collega ogni valore
al rispettivo input usando il nome della chiave:

```text
chiave del dizionario                 input Keras corrispondente
----------------------------------------------------------------
"state_feature_indices"   ->          state_indices
"state_feature_values"    ->          state_feature_values
"raw_hidden_state"        ->          raw_hidden
"kv_current_state"        ->          kv_current
"token_embedding_index"   ->          token_indices
"token_position"          ->          token_position
```

Quindi e' soltanto in questo momento che la porta simbolica:

```python
state_indices = tf.keras.Input(shape=(45,), ...)
```

riceve la matrice reale `(4096, 45)` proveniente da
`state_feature_indices_train.npy`. Nello stesso forward pass,
`state_feature_values` riceve la matrice reale associata proveniente da
`state_feature_values_train.npy`.

La stessa idea puo' essere riassunta cosi':

```text
Prima di model.fit(...)
    Input Keras = porta vuota con forma dichiarata.

Durante un batch di model.fit(...)
    Input Keras = porta attraverso cui transitano i valori reali delle 4.096
    righe del batch corrente.

Dopo il batch
    i valori e le attivazioni intermedie non vengono conservati come dataset;
    restano aggiornati soltanto i pesi apprendibili di Embedding e Dense.
```

#### 6. La rete calcola output, loss e aggiornamento dei pesi

Con i valori reali ora inseriti negli input, Keras esegue tutti i layer della
funzione di costruzione: lookup Embedding state, moltiplicazione per i valori,
somma, ReLU, dropout, altri rami, concatenazione e Dense(256) finale.

Confronta quindi `expert_scores` con `labels[selected]` usando la
cross-entropia binaria. Adam calcola come modificare i pesi apprendibili e li
aggiorna. Terminato il primo step, `model.fit` chiede un nuovo batch al dataset
e ripete lo stesso ciclo fino a `train_steps` batch per ogni epoca.

### Passo 1 — Gli array diventano input TensorFlow

La funzione crea i due input Keras seguenti:

```python
state_indices = tf.keras.Input(
    shape=(state_input_slots,), dtype="int32", name="state_feature_indices"
)
state_feature_values = tf.keras.Input(
    shape=(state_input_slots,), dtype="float32", name="state_feature_values"
)
```

Nella configurazione corrente, `state_input_slots` vale `45`. Nel batch da
4.096 esempi, i due tensori hanno quindi forma:

```text
state_indices        : (4096, 45)
state_feature_values : (4096, 45)
```

Queste righe non vengono scritte manualmente dentro `tf.keras.Input`. Gli input
Keras definiscono soltanto il **contratto di forma** che la rete accetta. I
valori reali arrivano quando si avvia il trainer da terminale. I due parametri
del comando che determinano il riempimento dei tensori state sono:

```bash
--fixed-nnz-state-data-dir \
  test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state

--batch-size 4096
```

`--fixed-nnz-state-data-dir` indica al trainer dove leggere i file:

```text
arrays/state_feature_indices_train.npy
arrays/state_feature_values_train.npy
artifacts/fixed_nnz_state_metadata.json
```

I primi due file contengono tutte le righe reali del train set. Il file dei
metadati contiene, fra le altre informazioni, `input_slots = 45`: per questo
la funzione riceve `state_input_slots=45`. Non esiste un argomento CLI
`--state-input-slots`; quel valore viene letto automaticamente dai metadati,
cosi' il trainer usa sempre la forma con cui sono stati materializzati gli
array fixed-NNZ.

`--batch-size 4096` decide invece quante righe consecutive vengono estratte
dagli array e consegnate insieme alla rete in un singolo step di training. In
ogni step, quindi, il dataset TensorFlow fornisce:

```text
prime 4096 righe del batch da state_feature_indices_train.npy
prime 4096 righe corrispondenti da state_feature_values_train.npy
                         |
                         v
state_indices        : (4096, 45)
state_feature_values : (4096, 45)
                         |
                         v
build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(...)
```

Al batch successivo vengono fornite altre 4.096 righe; lo shuffle del trainer
decide l'ordine dei blocchi di esempi, ma ogni riga di indici resta sempre
allineata alla riga con gli stessi valori.

Il primo esempio del batch conserva esattamente le 45 coppie mostrate al passo
0. TensorFlow usa `int32` per gli indici, anche se gli indici sono salvati su
disco come `uint16`, perche' il layer Keras `Embedding` richiede indici interi
compatibili con TensorFlow.

Ogni riga di `state_indices` e ogni riga di `state_feature_values` devono essere
lette insieme, colonna per colonna. `state_indices` dice **quale** feature e'
attiva; `state_feature_values` dice **quanto** pesa quella feature. Per la prima
riga train, alcuni esempi di corrispondenza sono:

| Colonna dello slot | `state_indices` | `state_feature_values` | Significato |
|---:|---:|---:|---|
| 1 | `15` | `1` | L'esperto 15 e' nella routing history. |
| 7 | `256` | `1` | Il layer corrente e' `layer_3`. |
| 8 | `392` | `1` | L'esperto 96 e' al rank 0 del router. |
| 24 | `4392` | `-1.61500335` | Primo router logit standardizzato. |
| 40 | `4408` | `-0.802221656` | Norma L2 Key-Value standardizzata. |
| 44 | `4412` | `1.572362300` | Massimo valore assoluto Key-Value standardizzato. |
| 45 | `4413` | `1` | Sentinella che aggiunge il bias. |

Le feature categoriche hanno valore `1`, perche' segnalano soltanto presenza o
assenza. I punteggi del router e le statistiche Key-Value sono invece feature
numeriche: il loro valore standardizzato e' conservato in
`state_feature_values` e verra' usato per pesare il rispettivo vettore di pesi.

### Passo 2 — Ogni indice recupera il proprio vettore di 512 pesi

Il seguente snippet crea la tabella dei pesi e la interroga con i 45 indici:

```python
state_weight_rows = tf.keras.layers.Embedding(
    input_dim=state_feature_count + 1,  # 4413 + 1 = 4414 righe di pesi.
    output_dim=state_hidden_units,      # 512 pesi per riga.
    embeddings_regularizer=regularizer, # None: L2 = 0.0 nella run corrente.
    name="state_fixed_nnz_linear_weights",
)(state_indices)
```

`tf.keras.layers.Embedding` viene spesso usato per convertire l'ID di una
parola nel suo embedding. Qui viene usato per un motivo diverso ma con la
stessa logica: come tabella che, dato l'indice di una feature state, recupera
velocemente il suo vettore di pesi apprendibili.

Questa scelta e' normale ed e' equivalente a un Dense classico. In un Dense
con 4.413 feature di input e 512 neuroni di output, ogni feature ha un arco
verso ciascuno dei 512 neuroni successivi. Esiste quindi un peso per ogni arco:

```text
4.413 feature x 512 neuroni = 2.259.456 pesi
```

Tutti questi pesi formano un'unica matrice:

```text
W_state : (4413, 512)
```

Non esiste un unico vettore di pesi condiviso identicamente da tutte le
feature. Ogni feature possiede invece una riga diversa della matrice:

```text
feature state i
    -> W_state[i, 0]   peso verso il neurone 0
    -> W_state[i, 1]   peso verso il neurone 1
    -> ...
    -> W_state[i, 511] peso verso il neurone 511
```

La riga `W_state[i]` e' quindi un vettore di 512 pesi specifico della feature
`i`. Il layer `Embedding(4414, 512)` memorizza questa matrice come tabella di
4.414 righe: le prime 4.413 sono `W_state`; l'ultima riga, indice 4.413, e' il
vettore bias sentinella da 512 valori.

I quattro elementi della chiamata hanno ruoli distinti:

| Elemento | Valore nella run corrente | Cosa comunica a Keras |
|---|---:|---|
| `input_dim` | `4414` | La tabella ha 4.414 righe indicizzabili, da `0` a `4413`. |
| `output_dim` | `512` | Ogni riga della tabella e' un vettore di 512 pesi. |
| `embeddings_regularizer` | `None` | Non aggiunge penalizzazione L2, perche' `l2_regularization=0.0`. |
| `name` | `state_fixed_nnz_linear_weights` | Nome con cui il layer appare nel riepilogo Keras. |

`state_indices` non e' uno di questi quattro parametri di costruzione del
layer. Il layer viene prima definito tra parentesi tonde:

```python
tf.keras.layers.Embedding(...)
```

Poi viene immediatamente *chiamato* con gli indici del batch:

```python
(state_indices)
```

L'intera espressione:

```python
tf.keras.layers.Embedding(...)(state_indices)
```

significa quindi: “crea questa tabella di pesi e usala per cercare una riga per
ogni indice contenuto in `state_indices`”. `state_indices` e' pertanto
l'input del layer, cioe' il primo elemento che il layer osserva quando la rete
esegue il forward pass.

`tf.keras.layers.Embedding` e' qui una tabella apprendibile di forma:

```text
(4414, 512)
```

Ogni feature state possiede dunque il proprio vettore di 512 pesi. Per il
primo esempio:

```text
indice 15   -> recupera la riga dei pesi dell'esperto history 15
indice 90   -> recupera la riga dei pesi dell'esperto history 90
...
indice 4392 -> recupera la riga dei pesi di router_logit_value_0
...
indice 4412 -> recupera la riga dei pesi di kv_current_max_abs
indice 4413 -> recupera la riga dei pesi del bias sentinella
```

Ogni recupero produce 512 numeri. Per un solo esempio, il risultato ha forma
`(45, 512)`; per il batch completo, `state_weight_rows` ha forma:

```text
(4096, 45, 512)
```

La riga di pesi dell'indice `15`, per esempio, non e' un valore fisso del
dataset: e' un vettore di 512 parametri che Adam inizializza e poi impara
durante l'addestramento. Lo stesso vale per ogni altro indice.

Il vantaggio rispetto al Dense standard riguarda soltanto il modo in cui si
calcola lo stesso risultato. Un Dense riceverebbe tutte le 4.413 feature, anche
le 4.369 che valgono zero in questo esempio. Il lookup con Embedding recupera
soltanto le 44 righe effettivamente attive, piu' la riga sentinella del bias.
Gli step successivi moltiplicano queste righe per i valori delle feature e le
sommano, ottenendo lo stesso risultato matematico del Dense state.

### Passo 3 — Ogni vettore viene pesato dal valore della sua feature

Nel passo precedente, `Embedding` ha trasformato gli indici state in una
tabella di vettori di pesi. Per un batch da 4.096 esempi il risultato ha forma
`(4096, 45, 512)`: per ogni esempio ci sono i vettori di pesi delle 44
feature state attive più il vettore della sentinella del bias. Queste 45 entità
occupano 45 posizioni tecniche nella matrice. Invece i valori reali delle
feature sono ancora nella matrice `state_feature_values`, con forma
`(4096, 45)`: ogni posizione contiene un solo numero.

Il problema è quindi questo: il codice deve associare ogni valore della
feature al vettore di 512 pesi della **stessa feature**; poi deve moltiplicare
quel singolo valore per tutti i 512 pesi del vettore. Le due forme iniziali
non esprimono in modo esplicito questa corrispondenza:

```text
pesi recuperati da Embedding: (4096, 45, 512)
valori delle feature state:   (4096, 45)
```

Per esempio, nella prima riga train la feature memorizzata nella posizione 1
ha indice `15` e valore `1`; la feature nella posizione 24 ha indice `4392` e
valore `-1.61500335`; la feature nella posizione 44 ha indice `4412` e valore
`1.572362300`. Per preparare ciascuno di questi valori a essere applicato al
rispettivo vettore da 512 pesi, il codice trasforma i 45 valori scalari in una
colonna:

```python
state_value_column = tf.keras.layers.Reshape(
    (state_input_slots, 1), name="state_fixed_nnz_value_column"
)(state_feature_values)
```

La forma passa da `(4096, 45)` a:

```text
state_value_column : (4096, 45, 1)
```

`Reshape` non modifica alcun valore e non crea nuove feature. Aggiunge soltanto
l'ultima dimensione di grandezza 1: ogni valore diventa una piccola colonna
`(1,)`. Questa forma rende esplicito a Keras che il valore della feature nella
posizione 1 deve essere applicato ai suoi 512 pesi, quello della feature nella
posizione 24 ai suoi 512 pesi, e così via. Senza questa dimensione aggiuntiva, le forme
`(4096, 45)` e `(4096, 45, 512)` non sarebbero allineate per eseguire questa
moltiplicazione feature per feature.

#### Esempio reale: trasformazione della prima riga train

Consideriamo solamente il primo esempio del batch, cioè
`state_feature_values[0, :]`. Prima di `Reshape` è un vettore con 45 valori:

```text
shape prima: (45,)

[1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 -1.61500335, -1.66757727, -2.11209822, -2.00715661,
 -1.93500447, -1.81196034, -1.93749273, -1.83856726,
 -1.71391559, -1.85004640, -1.84914732, -1.78769433,
 -1.74574232, -1.69826663, -1.67387772, -1.67472768,
 -0.802221656, -0.802221656, -0.060272869, -0.797440290,
  1.572362300, 1]
```

I primi 23 valori `1` appartengono alle 6 feature della routing history, alla
feature del layer e ai 16 identificatori categorici del router. Seguono i 16
punteggi del router, le 5 statistiche Key-Value e l'ultimo `1` della
sentinella del bias.

`Reshape((45, 1))` conserva esattamente questi stessi 45 numeri e aggiunge
soltanto una dimensione finale. Per il primo esempio la trasformazione reale è:

```text
shape dopo: (45, 1)

[[ 1.000000000],
 [ 1.000000000],
 [ 1.000000000],
 ...,
 [-1.615003350],
 [-1.667577270],
 ...,
 [-0.802221656],
 [-0.797440290],
 [ 1.572362300],
 [ 1.000000000]]
```

Quindi il valore della prima feature memorizzata passa semplicemente da `1` a
`[1]`, quello della feature nella posizione 24 passa da `-1.61500335` a
`[-1.61500335]`, e così via fino alla sentinella del bias nella posizione 45.
Applicato a tutti i 4.096 esempi del batch, questo stesso cambio da
vettore a colonna produce la forma `(4096, 45, 1)`.

Ora le due entità da combinare sono allineate feature per feature, nelle stesse
45 posizioni tecniche:

```text
state_weight_rows  : (4096, 45, 512)
state_value_column : (4096, 45,   1)
```

`state_weight_rows` contiene i pesi apprendibili: per ciascuna delle 44 feature
state attive di ogni esempio, più la sentinella del bias, contiene una riga di
512 pesi. `state_value_column` contiene i valori reali delle medesime feature:
per ciascuna delle stesse 45 posizioni, contiene un solo valore scalare,
scritto come colonna di lunghezza 1.

È necessario moltiplicarli perché un neurone dense non usa soltanto il peso
associato a una feature: usa il **peso scalato dal valore della feature**. In
forma compatta, il contributo di una feature `i` ai 512 neuroni state è
`x_i × W_i`, dove `x_i` è il valore osservato nell'esempio e `W_i` è la sua
riga di 512 pesi. Le feature categoriche attive hanno `x_i = 1`, mentre i
punteggi router e le statistiche Key-Value usano il loro valore reale già
standardizzato.

Il codice esegue quindi questa moltiplicazione feature per feature:

```python
state_weighted_rows = tf.keras.layers.Multiply(
    name="state_fixed_nnz_weighted_rows"
)([state_weight_rows, state_value_column])
```

Keras applica automaticamente il valore scalare a tutti i 512 elementi della
riga associata. Per il primo esempio:

```text
feature history esperto 15:
    1 x pesi_history_15 (riga con indice 15)
    = pesi_history_15

feature router_logit_value_0:
    -1.61500335 x pesi_router_logit_value_0 (riga con indice 4392)
    = contributo pesato del primo logit router

feature kv_current_max_abs:
    1.572362300 x pesi_kv_current_max_abs (riga con indice 4412)
    = contributo pesato della statistica Key-Value

sentinella:
    1 x pesi_sentinella_bias (riga con indice 4413)
    = vettore bias
```

Il risultato mantiene forma `(4096, 45, 512)`: ogni esempio ha ancora 45
contributi separati, ciascuno di 512 valori.

#### Collegamento tra Passo 3 e Passo 4: l'equivalente di `xW + b`

I due passi insieme ricostruiscono esattamente la parte lineare di un normale
layer dense. La matrice concettuale dei pesi state ha forma `(4413, 512)`:
ogni sua riga contiene i 512 pesi che partono da una feature state; ogni sua
colonna contiene tutti i pesi entranti in uno specifico neurone del layer da
512 neuroni.

Nel Passo 3, per ogni feature attiva `i` e per ogni neurone `j`, viene
calcolato il singolo prodotto `x_i × W_i,j`. Per il primo esempio, il valore
`-1.61500335` della feature `router_logit_value_0`, per esempio, scala i 512
pesi della sua riga nella matrice `W`.

Il Passo 4 non esegue una nuova moltiplicazione: i termini `x_i W_{i,j}` sono
già stati tutti calcolati nel Passo 3. Il suo unico compito è completare il
prodotto scalare sommando quei termini sulle feature. Per ciascun neurone
`j = 1, ..., 512`, il risultato concettuale di un normale `Dense(512)` è:


<div align="center">
  <strong>z<sub>j</sub> = ∑<sub>i = 1</sub><sup>4413</sup> (x<sub>i</sub> · W<sub>i,j</sub>) + b<sub>j</sub>, per j = 1, ..., 512</strong>
</div>

Nel nostro esempio quasi tutti gli `x_i` valgono zero. Il formato fixed-NNZ
non li materializza e la somma effettiva contiene soltanto l'insieme `A` delle
44 feature attive:


<div align="center">
  <strong>z<sub>j</sub> = ∑<sub>i ∈ A</sub> (x<sub>i</sub> · W<sub>i,j</sub>) + b<sub>j</sub>, per j = 1, ..., 512</strong>
</div>

Questa è esattamente la componente `z = xW + b` di `Dense(512)`. Nel formato
fixed-NNZ il bias `b_j` non viene aggiunto in un'operazione separata: la
sentinella con valore `1` seleziona la sua riga di pesi e il Passo 4 la include
nella stessa somma delle 44 feature state attive.

### Passo 4 — I 45 contributi diventano il Dense(512) dello state

Alla fine del Passo 3 abbiamo `state_weighted_rows` con forma
`(4096, 45, 512)`. Per ogni esempio del batch, questo tensore contiene ancora
45 contributi distinti: i 44 contributi delle feature state attive e il
contributo della sentinella del bias. Ciascun contributo è un vettore di 512
valori, uno per ogni neurone del ramo state.

Per ottenere l'input finale dei 512 neuroni state, è necessario sommare tutti
i contributi che arrivano alla stessa dimensione del neurone. Non dobbiamo
sommarli lungo l'asse del batch (`axis=0`), perché un esempio deve restare
separato dagli altri; non dobbiamo nemmeno sommare i 512 valori di un vettore
(`axis=2`), perché ogni valore deve restare associato al proprio neurone.
Bisogna invece sommare le 45 posizioni delle feature (`axis=1`).

Il layer `tf.keras.layers.Lambda` permette di inserire questa operazione
TensorFlow personalizzata nel grafo della rete Keras. Riceve
`state_weighted_rows` come input, esegue `tf.reduce_sum` e restituisce un nuovo
tensore Keras che i layer successivi possono usare normalmente. In questo caso
la funzione Lambda non possiede pesi apprendibili: serve soltanto a eseguire
la somma necessaria per completare l'equivalente del primo `Dense(512)`.

In altre parole, questo Passo 4 produce la stessa **parte lineare** che un
normale layer `Dense(512)` calcolerebbe per ciascuno dei suoi 512 neuroni:

```text
z = xW + b
```

Il tensore `state_linear` contiene questi 512 valori `z`, uno per neurone, ma
non è ancora l'output finale del ramo state. Nel Passo 5 il codice applica
ReLU a ciascun valore; durante l'addestramento, il layer Dropout successivo
azzera inoltre casualmente il 30% delle attivazioni. Quindi la sequenza reale
del ramo state è: somma lineare `xW + b` → ReLU → Dropout(0.3).

Il codice somma quindi i 45 contributi lungo l'asse delle posizioni delle
feature:

```python
state_linear = tf.keras.layers.Lambda(
    lambda values: tf.reduce_sum(values, axis=1),
    output_shape=(state_hidden_units,),
    name="state_fixed_nnz_linear_sum",
)(state_weighted_rows)
```

`tf.reduce_sum(values, axis=1)` somma le 44 feature attive più il contributo
della sentinella del bias. Rimuove quindi soltanto la dimensione delle 45
posizioni e mantiene batch e neuroni, producendo:

```text
state_linear : (4096, 512)
```

Per il primo esempio, in forma concettuale:

```text
state_linear =
    pesi_history_15 + pesi_history_90 + ... + pesi_history_202
    + pesi_layer_3
    + pesi_ID_router_rank_0_96 + ... + pesi_ID_router_rank_15_253
    + (-1.61500335 x pesi_router_logit_value_0) + ...
    + (1.572362300 x pesi_kv_current_max_abs)
    + pesi_sentinella_bias
```

Questa e' esattamente la logica di un Dense(512) applicato allo state completo
di 4.413 feature, ma viene calcolata senza moltiplicare inutilmente i 4.369
valori che in questo esempio non contribuiscono.

### Passo 5 — ReLU e dropout completano il ramo state

Il codice applica ReLU:

```python
state_values = tf.keras.layers.Activation(
    activation, name="state_hidden_1"
)(state_linear)
```

Nella run corrente `activation="relu"`. Per ciascuno dei 512 valori:

```text
se il valore e' negativo -> diventa 0
se il valore e' positivo -> resta invariato
```

#### Collegamento tra Passo 3, Passo 4 e Passo 5: output del neurone `j`

I tre passi completano il calcolo di ciascuno dei 512 neuroni del ramo state.
Per un neurone `j`, il Passo 3 produce separatamente i contributi pesati
`x_i · W_{i,j}` delle 44 feature attive; il Passo 4 li somma, includendo il
bias, e ottiene il valore lineare `z_j`; infine il Passo 5 applica ReLU e
ottiene l'attivazione `a_j` che prosegue nella rete:

<div align="center">
  <strong>a<sub>j</sub> = ReLU(z<sub>j</sub>) = max(0, z<sub>j</sub>)</strong>
</div>

dove:

<div align="center">
  <strong>z<sub>j</sub> = ∑<sub>i ∈ A</sub> (x<sub>i</sub> · W<sub>i,j</sub>) + b<sub>j</sub></strong>
</div>

Quindi, se la somma ottenuta nel Passo 4 per il neurone `j` è negativa, ReLU
restituisce `0`; se è positiva o zero, ReLU la mantiene invariata. Il dropout
che segue non cambia questa definizione dell'attivazione: solo durante il
training può azzerare temporaneamente alcune delle attivazioni `a_j` per
regolarizzare il modello.

Poi, soltanto durante l'addestramento, il codice applica dropout:

```python
state_values = tf.keras.layers.Dropout(
    dropout, name="state_dropout_1"
)(state_values)
```

Con `dropout=0.3`, circa il 30 per cento delle attivazioni ReLU viene
temporaneamente azzerato a ogni batch di training. Durante validation, test e
inferenza non viene azzerata nessuna attivazione.

Il ramo state termina quindi con:

```text
state_values : (4096, 512)
```

### Passo 6 — Dequantizzazione del ramo hidden raw

Dopo il ramo state, il codice procede con il vettore hidden raw del modello
DeepSeek. L'input `raw_hidden` contiene 4.096 valori salvati come interi a 8
bit, quindi prima della proiezione apprendibile deve essere riconvertito in un
valore in virgola mobile approssimato:

```python
hidden_values = tf.keras.layers.Rescaling(
    scale=raw_hidden_dequantization_scale,
    offset=0.0,
    dtype="mixed_float16",
    name="raw_hidden_dequantization",
)(raw_hidden)
```

Nella run corrente `raw_hidden_dequantization_scale = 0.0306656`. Il layer
`Rescaling` moltiplica ogni valore int8 per questa scala; non modifica la
dimensione del vettore:

```text
raw_hidden int8       : (4096, 4096)
hidden_values float16 : (4096, 4096)
```

La scala `0.0306656` non è scelta manualmente: è stata salvata negli artefatti
degli hidden quantizzati dopo una calibrazione eseguita soltanto sul train set.
Con `--calibration-percentile 99.99`, la calibrazione Build 5 ha trovato il
limite `±3.89453`. Poiché i valori int8 usati dalla quantizzazione variano da
circa `-127` a `+127`, la scala è:

```text
0.0306656 = 3.89453 / 127
```

Durante la preparazione offline degli input, ogni valore hidden floating-point
è stato quantizzato approssimativamente così:

```text
hidden_int8 = arrotonda(hidden_float / 0.0306656)
```

Il layer `Rescaling` esegue l'operazione inversa approssimata durante il
forward pass:

```text
hidden_float_approssimato = hidden_int8 × 0.0306656
```

Usare il percentile 99.99 evita che pochissimi valori estremi rendano la scala
troppo grande, riducendo la precisione int8 della grande maggioranza dei
valori. Il trainer legge questa scala automaticamente dai metadati nella
cartella passata con `--quantized-hidden-data-dir` e la inoltra alla funzione
che costruisce la rete come `raw_hidden_dequantization_scale`.

#### Verifica empirica della scala su tutti gli esempi

Le 4.096 righe train sono usate soltanto per scegliere la scala. Dopo questa
calibrazione, lo script attraversa invece **tutti** i vettori train,
validation e test in batch da 4.096 righe. Per ciascun batch esegue questi
controlli sui valori float originali:

```text
valori_saturi += conta(abs(valori_originali) >= limite_di_clipping)
valori_int8 = clip(arrotonda(valori_originali / scala), -127, 127)
valori_ricostruiti = valori_int8 × scala
errore_quadratico += somma((valori_originali - valori_ricostruiti)²)
```

Significato di ogni istruzione:

1. `valori_saturi += conta(abs(valori_originali) >= limite_di_clipping)`
   conta, nel batch corrente, quanti valori originali sono esterni o uguali
   all'intervallo rappresentabile `[-3.89453125, +3.89453125]`. Questi valori
   finiranno sul limite `-127` oppure `+127` dopo la quantizzazione. Il
   contatore viene accumulato su tutti i batch dello split.

2. `valori_int8 = clip(arrotonda(valori_originali / scala), -127, 127)`
   esegue la quantizzazione effettiva. Prima divide ogni valore floating-point
   per la scala `0.0306656`, trasformandolo nel numero intero int8 più vicino;
   poi `clip` limita eventuali valori fuori intervallo a `-127` o `+127`.
   `valori_int8` è il batch che viene salvato negli array quantizzati.

3. `valori_ricostruiti = valori_int8 × scala`
   dequantizza immediatamente il batch appena creato per ottenere
   l'approssimazione floating-point che la rete riceverà successivamente dal
   layer `Rescaling`.

4. `errore_quadratico += somma((valori_originali - valori_ricostruiti)²)`
   confronta ogni valore originale con la sua ricostruzione dequantizzata,
   eleva al quadrato la differenza e accumula il risultato su tutti i batch.
   Questo totale viene infine usato per calcolare l'RMSE dello split.

Alla fine di ogni split, lo script calcola:

```text
frazione_di_saturazione = valori_saturi / (numero_esempi × 4096)
RMSE = radice(errore_quadratico_totale / (numero_esempi × 4096))
```

`errore_quadratico_totale` parte da `0.0` all'inizio di ogni split e accumula,
batch dopo batch, la somma di tutti gli errori quadratici:

```text
errore_quadratico_totale = Σ (valore_originale - valore_ricostruito)²
```

La sommatoria copre ogni esempio dello split e tutte le sue 4.096 dimensioni.
Per il train, per esempio, comprende `1.652.640 × 4.096` differenze al
quadrato. L'RMSE prima calcola la media di questi errori quadratici, dividendo
per il numero totale di valori, e poi applica la radice quadrata per riportare
l'errore nella stessa unità dei valori hidden originali.

Questa verifica considera tutte le 4.096 dimensioni di ogni vettore: nella
Build 5 ha quindi controllato 1.652.640 vettori train, 200.640 validation e
209.400 test. I risultati sono:

| Split | Saturazione | Errore RMSE di ricostruzione |
|---|---:|---:|
| Train | `0.01003%` | `0.00947644` |
| Validation | `0.00981%` | `0.00946018` |
| Test | `0.01036%` | `0.00950761` |

Le percentuali quasi identiche nei tre split mostrano che il campione uniforme
usato per la calibrazione ha rappresentato bene la distribuzione generale dei
hidden. Non è una garanzia matematica assoluta, ma è un'evidenza empirica
solida che la scala scelta è adeguata per la grande maggioranza dei valori.

La riconversione è necessaria per il modello attuale perché il successivo
`Dense(64)` ha pesi apprendibili in virgola mobile e, durante l'addestramento,
TensorFlow deve calcolare anche i gradienti di quei pesi. Un prodotto diretto
fra input int8 e pesi floating-point non è il percorso standard di un layer
`Dense` Keras addestrabile.

L'int8 mantiene comunque il suo vantaggio principale prima della rete: gli
hidden occupano circa un quarto dello spazio rispetto a float32 e richiedono
meno letture da disco e trasferimenti dalla memoria. La dequantizzazione
int8 → float16 è economica rispetto alla proiezione apprendibile 4.096D →
64D che la segue. In teoria si potrebbe evitare questa riconversione solo con
un'implementazione quantizzata personalizzata, con pesi, scale e kernel GPU
dedicati; non sarebbe più il Dense floating-point semplice usato in questa
architettura.

### Passo 7 — Proiezione lineare del ramo hidden raw

Il vettore hidden dequantizzato viene poi proiettato da 4.096 a 64 valori:

```python
hidden_values = tf.keras.layers.Dense(
    hidden_projection_dimension,
    activation=raw_hidden_projection_activation,
    dtype="mixed_float16",
    name="raw_hidden_projection",
)(hidden_values)
```

Con `hidden_projection_dimension = 64`, l'output è
`hidden_values : (4096, 64)`. Con
`raw_hidden_projection_activation="linear"` il layer esegue soltanto una
trasformazione affine apprendibile. Con valore `"relu"`, applica ReLU ai 64
valori ottenuti, azzerando quelli negativi prima della concatenazione.

Questo `Dense(64)` crea una matrice di pesi propria, distinta da quella del
ramo state:

```text
W_hidden    : (4096, 64)
bias_hidden : (64,)
```

Per ogni esempio calcola:

```text
hidden_proiettato = hidden_dequantizzato × W_hidden + bias_hidden
```

Questi pesi e bias sono trainabili e vengono aggiornati dall'ottimizzatore
Adam durante `model.fit()`. Nella baseline, con `activation="linear"`,
l'output della formula passa direttamente alla concatenazione; nella nuova
ablazione, con `activation="relu"`, la ReLU viene applicata dopo questa
formula e prima della concatenazione.

L'opzione `dtype="mixed_float16"` imposta una precisione mista soltanto per
questo layer: i calcoli principali della proiezione e il suo output usano
`float16`, riducendo memoria e, in teoria, sfruttando più efficacemente la
GPU; i pesi trainabili vengono invece normalmente mantenuti in `float32`, per
rendere più stabile il loro aggiornamento da parte di Adam. Questa opzione non
rende i pesi int8 e non modifica la struttura della rete. Le prove con la
mixed precision globale dell'intera rete non hanno migliorato i tempi nel
nostro ambiente; qui descrive la precisione esplicitamente richiesta dal solo
ramo hidden raw.

Più precisamente, `W_hidden` e `bias_hidden` sono conservati come parametri
master in `float32`. Durante il forward pass Keras crea però una conversione
temporanea in `float16`, perché il `compute_dtype` del layer è `float16`. Il
calcolo logico è quindi:

```text
hidden_values in input            : float16
W_hidden e bias_hidden memorizzati: float32
W_hidden e bias_hidden nel forward: float16 temporaneo
hidden_values in output           : float16
```

Non viene quindi eseguita, a livello della policy Keras, una moltiplicazione
diretta fra hidden float16 e pesi float32. Dopo il backward pass, Adam usa i
parametri master float32 per applicare aggiornamenti piccoli senza perderli a
causa dell'arrotondamento float16. Il kernel GPU può usare accumuli interni a
precisione maggiore a seconda dell'implementazione TensorFlow/CUDA, ma la
policy del layer rimane `compute_dtype=float16` e `variable_dtype=float32`.

Il flusso può essere riassunto così:

```text
W_hidden e bias_hidden salvati in float32
              │
              │ cast per il calcolo
              v
hidden float16 × W_hidden float16 → output hidden_values float16
              │
              v
gradienti e aggiornamento Adam sui parametri master float32
```

### Passo 8 — Lookup dell'embedding congelato del token corrente

Il codice usa l'indice compatto del token corrente per recuperare la sua riga
da una tabella GGUF congelata di embedding 4.096D:

```python
token_values = tf.keras.layers.Embedding(
    input_dim=frozen_token_embeddings.shape[0],
    output_dim=frozen_token_embeddings.shape[1],
    embeddings_initializer=tf.keras.initializers.Constant(
        frozen_token_embeddings.astype(np.float32)
    ),
    trainable=False,
    name="frozen_gguf_token_embedding",
)(token_indices)
```

Nel Build 5 la tabella contiene 7.157 righe da 4.096 valori. `trainable=False`
impedisce che questi embedding, estratti dal GGUF, vengano modificati durante
l'addestramento. Il lookup produce `token_values : (4096, 4096)`.

Significato di ogni parametro del layer:

| Codice | Valore reale Build 5 | Funzione |
|---|---:|---|
| `input_dim=frozen_token_embeddings.shape[0]` | `7157` | Numero di righe della tabella: una per ogni token distinto osservato nel Build 5 e trasformato in indice compatto. Gli indici validi vanno da `0` a `7156`. |
| `output_dim=frozen_token_embeddings.shape[1]` | `4096` | Lunghezza di ogni riga recuperata: è la dimensione originale dell'embedding token del modello DeepSeek. |
| `embeddings_initializer=...` | matrice `(7157, 4096)` | Indica con quali valori inizializzare l'intera tabella dei pesi del layer. Qui non usa valori casuali: usa gli embedding estratti dal tensore GGUF `token_embd.weight`. |
| `tf.keras.initializers.Constant(...)` | — | Crea un inizializzatore che copia esattamente la matrice fornita nella tabella del layer al momento della sua creazione. |
| `frozen_token_embeddings.astype(np.float32)` | float16 → float32 | Converte la copia caricata dal GGUF in float32 prima di inizializzare il layer Keras, che nella configurazione corrente conserva normalmente la propria tabella di pesi in float32. Evita conversioni implicite e mantiene stabile l'integrazione con gli altri layer e con TensorFlow. Non recupera precisione assente nel GGUF: ogni valore float16 viene soltanto rappresentato esattamente in float32. Non modifica il file float16 salvato su disco. |
| `trainable=False` | `False` | Congela la tabella: durante il backward pass non riceve aggiornamenti da Adam. Gli embedding restano quelli del modello DeepSeek. |
| `name="frozen_gguf_token_embedding"` | stringa fissa | Assegna un nome leggibile al layer nel riepilogo Keras, nel modello salvato e nei grafici. |

La parte finale `)(token_indices)` non è un ulteriore parametro di
configurazione. Prima `tf.keras.layers.Embedding(...)` crea il layer con la
tabella descritta sopra; poi `(token_indices)` applica il layer al tensore degli
indici del batch. Con batch da 4.096 esempi, `token_indices` ha forma `(4096,)`:
per ogni indice recupera la corrispondente riga 4096D, producendo
`token_values` con forma `(4096, 4096)`.

#### Limitazione attuale da risolvere per l'inferenza reale

Il layer non riceve il token ID originale del vocabolario DeepSeek: riceve un
**indice compatto** compreso fra `0` e `7156`. I due spazi di indici non devono
essere confusi:

```text
token ID originale DeepSeek : da 0 a 129279
token ID originali osservati nella Build 5 (train + validation + test) : da 3 a 127939
indice compatto del lookup Build 5 : da 0 a 7156
```

I 7.157 token originali distinti della Build 5, anche se hanno ID originali
potenzialmente molto grandi, ricevono ciascuno un indice compatto diverso. Per
esempio, un token originale come `127939` viene prima cercato nel file
`token_embedding_artifacts/token_ids.npy` e poi convertito nel proprio indice
compatto, per esempio `5421`; `5421` è soltanto un esempio illustrativo e non
un valore da assumere fisso.

Se venisse passato direttamente un token ID originale, per esempio `127939`,
oppure un indice compatto maggiore di `7156`, il layer
`Embedding(input_dim=7157, ...)` genererebbe un errore di indice fuori
intervallo. Non esiste attualmente un embedding di fallback né un comportamento
automatico per un token sconosciuto.

Durante train, validation e test Build 5 questo problema non può verificarsi:
la tabella è stata estratta dagli stessi esempi dei tre split e contiene tutti
i loro token ID originali distinti, ciascuno convertibile nel proprio indice
compatto `0...7156`. Durante l'uso reale del modello, invece, può comparire un
token originale non osservato nella Build 5. Prima dell'integrazione in
produzione dovremo quindi scegliere e implementare una strategia esplicita,
per esempio: ampliare il lookup fino a includere il token nuovo, preparare una
tabella completa del vocabolario, oppure introdurre un indice OOV (*out of
vocabulary*) con un embedding dedicato. La pipeline corrente non implementa
ancora nessuna di queste soluzioni.

### Passo 9 — Proiezione lineare dell'embedding token

Subito dopo il lookup, il codice riduce l'embedding congelato a 64 valori
apprendibili:

```python
token_values = tf.keras.layers.Dense(
    token_projection_dimension,
    activation=token_projection_activation,
    name="token_embedding_projection",
)(token_values)
```

Con `token_projection_dimension = 64`, l'output è
`token_values : (4096, 64)`.

Anche qui `token_projection_activation="linear"` riproduce la baseline,
mentre `"relu"` azzera le componenti negative dei 64 valori prima della
concatenazione. I due parametri sono separati per poter valutare con il tuning
se ReLU è utile nel solo ramo hidden, nel solo ramo token, in entrambi oppure
in nessuno dei due.

### Passo 10 — Vettore Key-Value e posizione del token diretti

`kv_current` e `token_position` sono già input Keras creati in precedenza e
non subiscono trasformazioni ulteriori nel codice prima della concatenazione:

```text
kv_current     : (4096, 512), float16, già normalizzato offline
token_position : (4096, 1),   float32, già standardizzata offline
```

In questo punto la parola “normalizzato” riferita al KV è un nome descrittivo
degli artefatti, non un metodo matematico diverso dalla standardizzazione. In
realtà entrambi gli input usano una **standardizzazione z-score** calcolata
soltanto sul train set:

```text
valore_standardizzato = (valore_originale - media_train) / deviazione_standard_train
```

In generale, “normalizzazione” è un termine più ampio: può indicare, per
esempio, il ridimensionamento min-max in `[0, 1]`, la divisione per una norma
L2 o una standardizzazione. “Standardizzazione” indica invece precisamente la
formula z-score precedente, che porta ciascuna variabile ad avere media circa
zero e deviazione standard circa uno sul train set.

La differenza fra KV e posizione non è quindi la formula, ma le statistiche
usate:

| Input | Statistiche z-score calcolate sul train | Perché |
|---|---|---|
| `kv_current` 512D | Una media e una deviazione standard per ciascuna delle 512 componenti **e per ciascun layer**. | Le componenti KV e i layer possono avere distribuzioni e scale molto diverse; trattarle separatamente evita che alcune dimensioni o alcuni layer dominino il classificatore. |
| `token_position` 1D | Un'unica media e un'unica deviazione standard su tutte le posizioni train. | È una sola variabile scalare, con lo stesso significato in ogni esempio; non ci sono 512 componenti da trattare separatamente. |

Questa scelta rende comparabili le scale dei valori prima della concatenazione.
Non introduce parametri apprendibili e non usa validation o test per calcolare
le statistiche, evitando data leakage.

Il vettore Key-Value compatto corrente entra quindi interamente con tutte le
sue 512 dimensioni: non usa PCA né una proiezione lineare. La posizione del
token conserva invece il suo unico valore standardizzato.

### Passo 11 — Concatenazione dei cinque rami

Solo dopo che tutti i rami hanno raggiunto la loro forma finale, il codice li
unisce nell'ordine seguente:

```python
values = tf.keras.layers.Concatenate(name="state_hidden_token_kv_fusion")(
    [state_values, hidden_values, token_values, kv_current, token_position]
)
```

Per ogni esempio, i valori occupano queste posizioni del vettore finale da
1.153 dimensioni:

```text
posizioni    0-511   : state_values, 512 valori
posizioni  512-575   : hidden_values, 64 valori
posizioni  576-639   : token_values, 64 valori
posizioni  640-1151  : kv_current, 512 valori
posizione       1152 : token_position, 1 valore
```

Quindi `values` ha forma `(4096, 1153)`.

### Passo 12 — Classificatore finale degli esperti

Con `fusion_hidden_layers=[]` il ciclo degli eventuali layer di fusione
aggiuntivi è vuoto. Il vettore concatenato passa direttamente al classificatore finale
`Dense(256, activation="sigmoid")`:

```python
outputs = tf.keras.layers.Dense(
    NUM_EXPERTS, activation="sigmoid", dtype="float32", name="expert_scores"
)(values)
```

Il layer riceve `values` con forma `(4096, 1153)`. Per ogni esempio, i 1.153
valori sono quelli costruiti nel Passo 11:

```text
state 512D + hidden raw proiettato 64D + token proiettato 64D
+ KV corrente diretto 512D + posizione token 1D = 1153D
```

Il classificatore crea una nuova matrice di pesi e un bias:

```text
W_classificatore    : (1153, 256)
bias_classificatore : (256,)
```

Il flusso per un esempio è:

```text
vettore concatenato values (1153D)
    │
    │  Dense: logits = values × W_classificatore + bias_classificatore
    v
256 logits reali, uno per esperto
    │
    │  sigmoid applicata separatamente a ogni logit
    v
expert_scores (256D), ogni valore compreso fra 0 e 1
```

Per l'esperto `e`, il calcolo è:

<div align="center">
  <strong>logit<sub>e</sub> = ∑<sub>k=1</sub><sup>1153</sup> values<sub>k</sub> · W<sub>k,e</sub> + bias<sub>e</sub></strong><br>
  <strong>expert_score<sub>e</sub> = sigmoid(logit<sub>e</sub>) = 1 / (1 + exp(-logit<sub>e</sub>))</strong>
</div>

La sigmoid trasforma ciascun logit in un valore fra `0` e `1`, interpretabile
come il punteggio/probabilità prevista per quello specifico esperto. Le 256
sigmoid sono indipendenti: non vengono normalizzate per sommare a 1, come
accadrebbe con softmax. Questo è necessario perché il target può contenere più
esperti corretti contemporaneamente. Durante il training,
`BinaryCrossentropy` confronta le 256 probabilità con le 256 label multi-label;
durante la valutazione `Recall@6` verifica se gli esperti corretti compaiono
tra i sei `expert_scores` più alti.

`dtype="float32"` nel classificatore finale mantiene sigmoid, loss e Recall@6
numericamente stabili anche quando altri layer eseguono calcoli in float16.
