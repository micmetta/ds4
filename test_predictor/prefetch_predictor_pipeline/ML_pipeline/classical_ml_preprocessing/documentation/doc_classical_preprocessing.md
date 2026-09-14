# Preprocessing per i modelli classici di scikit-learn

## Obiettivo

Questa fase prepara una rappresentazione dei dati specifica per i primi modelli
classici di Machine Learning. Non modifica e non sostituisce il dataset base
della build scelta: lo legge come sorgente verificata e produce una seconda
rappresentazione, adatta a scikit-learn.

I primi modelli che confronteremo saranno:

1. `OneVsRestClassifier(LogisticRegression(...))`;
2. `RandomForestClassifier`;
3. `DecisionTreeClassifier`;
4. `KNeighborsClassifier`.

L'obiettivo supervisionato resta invariato: dopo aver osservato le feature del
token corrente $t$ nel layer routed $l$, predire gli expert che il router
selezionera' nello stesso layer al token successivo:

$$
E_l(t) \longrightarrow E_l(t+1)
$$

## Posizione nella pipeline completa

```text
raw/routing/ traces DS4
        |
        v
build_dataset.py
        |
        v
derived/prefetch_dataset_build_1.jsonl
        |
        v
preprocessing.py
        |
        v
training/                              dataset base: Parquet + array .npy
        |
        v
classical_ml_preprocessing/            nuova fase descritta in questo file
        |
        v
matrici sparse + artefatti fit sul train
        |
        v
script futuro di training scikit-learn
```

`training/` continuera' quindi a esistere come sorgente generale, leggibile e
indipendente dal tipo di modello. La nuova cartella contiene soltanto ciò che
serve ai modelli classici.

## Struttura prevista

```text
ML_pipeline/
├── training/                                  input base gia' costruito
│   ├── prefetch_training_build_1.parquet
│   └── arrays/
│       ├── X_train.npy
│       ├── X_validation.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_validation.npy
│       └── y_test.npy
│
└── classical_ml_preprocessing/
    ├── documentation/
    │   └── doc_classical_preprocessing.md     questo documento
    ├── preprocessing_classical.py             JSON/Parquet -> matrici sparse scikit-learn
    ├── artifacts/                             creati dallo script
    │   ├── categorical_encoder.joblib
    │   ├── numeric_scaler.joblib
    │   ├── feature_names.json
    │   └── preprocessing_summary.json
    └── arrays/                                creati dallo script
        ├── X_classic_train.npz
        ├── X_classic_validation.npz
        ├── X_classic_test.npz
        ├── y_train.npy
        ├── y_validation.npy
        └── y_test.npy
```

I file `X_classic_*.npz` useranno una matrice sparsa CSR. Il formato e'
importante: il one-hot encoding crea molte colonne, ma ogni riga attiva solo
pochissime di esse; una matrice densa sprecherebbe memoria.

## Input

Lo script futuro leggerà il Parquet:

```text
build_dataset_1/ml/training/prefetch_training_build_1.parquet
```

Il Parquet contiene nomi di colonne espliciti, quindi è la sorgente più chiara
per selezionare gruppi di feature. Gli split `train`, `validation` e `test`
sono già presenti nella colonna `split` e non verranno ricalcolati.

Le label saranno lette dagli array già creati in `training/arrays/`, oppure
ricostruite dalle colonne `target_expert_0` ... `target_expert_5`; lo script
controllerà che la versione utilizzata coincida con il Parquet. Nel risultato
salverà una copia degli array `y_*.npy` nella nuova cartella, così gli script
di training classico avranno un input autosufficiente.

## Flusso di esecuzione di `preprocessing_classical.py`

Lo script viene orchestrato da `main()`. Lo schema seguente usa i nomi esatti
delle funzioni chiamate dal codice e sintetizza il loro compito:

```text
main()
  |
  +-- leggi_argomenti()
  |     legge --training-dir e --output-dir, oppure usa i percorsi di default
  |
  +-- carica_tabella_base(training_dir)
  |     trova il solo Parquet prefetch_training*.parquet, controlla colonne,
  |     split e eventuali hidden_pca_*
  |     risultato: tabella Pandas completa + colonne numeriche della build
  |
  +-- crea split_tables
  |     separa la tabella esistente in train, validation e test senza
  |     riassegnare righe o prompt a split diversi
  |
  +-- carica_e_verifica_label(training_dir, split, split_tables[split])
  |     chiamata una volta per train, validation e test
  |     carica y_<split>.npy e controlla shape, dtype, Top-6 e target Parquet
  |     risultato: mapping split -> label multi-hot [N_split, 256]
  |
  +-- crea_encoder_categorico()
  |     crea OneHotEncoder con min_frequency=41 e gestione degli ID sconosciuti
  |
  +-- StandardScaler()
  |     crea lo scaler per token_position, 16 raw logit e, se presenti,
  |     le coordinate hidden_pca_*
  |
  +-- encoder.fit(train[categorical_columns])
  |     apprende categorie, colonne one-hot e bucket infrequente dal solo train
  |
  +-- scaler.fit(train[numeric_columns])
  |     apprende media e deviazione standard dal solo train
  |
  +-- trasforma_feature(split_tables[split], encoder, scaler)
  |     chiamata per train, validation e test
  |     applica transform, concatena one-hot e numerici e restituisce una CSR
  |     risultato: mapping split -> X_classic sparse
  |
  `-- salva_artefatti(output_dir, encoder, scaler, matrices, labels)
        salva X_classic_<split>.npz, y_<split>.npy, encoder, scaler,
        nomi delle feature e riepilogo JSON
```

Il punto più importante è la separazione tra `fit` e `transform`:

```text
train       -> fit encoder + fit scaler, poi transform
validation  -> solo transform con encoder/scaler già appresi
test        -> solo transform con encoder/scaler già appresi
runtime     -> solo transform con gli stessi artefatti salvati
```

Questa sequenza mantiene lo spazio delle feature identico fra tutti gli split
ed evita data leakage: validation e test non contribuiscono mai a decidere
categorie, bucket infrequente, media o deviazione standard.

## Feature categoriche: one-hot encoding

Queste feature sono ID discreti. Non hanno un ordine numerico significativo e
non devono essere date al modello come interi ordinari:

```text
token_id
layer
routing_history_expert_0 ... routing_history_expert_5
router_logit_expert_id_0 ... router_logit_expert_id_15
```

Saranno trasformate tramite un unico `OneHotEncoder`:

```python
OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=41,
    sparse_output=True,
)
```

### Esempio: i sei slot di `routing_history`

Consideriamo un routing realmente osservato nel corpus:

```text
E_l(t) = [96, 119, 202, 112, 90, 15]
```

Con `history_length = 1`, prima dell'encoding la riga contiene:

```text
routing_history_expert_0 = 96
routing_history_expert_1 = 119
routing_history_expert_2 = 202
routing_history_expert_3 = 112
routing_history_expert_4 = 90
routing_history_expert_5 = 15
```

L'encoder crea un blocco di colonne distinto per ogni slot. Le colonne attive
in questa riga sono quindi concettualmente:

```text
routing_history_expert_0_96  = 1
routing_history_expert_1_119 = 1
routing_history_expert_2_202 = 1
routing_history_expert_3_112 = 1
routing_history_expert_4_90  = 1
routing_history_expert_5_15  = 1
```

Tutte le altre colonne dei sei blocchi valgono `0`. Di conseguenza l'ordine
Top-6 è mantenuto: ad esempio `expert 96` nello slot `0` e `expert 96` nello
slot `4` sono due feature differenti. Il massimo teorico è `6 × 256 = 1536`
colonne per questi sei campi, anche se l'encoder crea soltanto le categorie
presenti nel train e può raggruppare quelle rare nel bucket `infrequente`.

#### Cosa riceve effettivamente il modello

Concettualmente il modello riceve l'intera riga binaria: per lo slot `0`, se
il valore è `96`, la feature `routing_history_expert_0_96` vale `1` e tutte le
altre possibili feature dello stesso slot valgono `0`. Non riceve quindi il
numero intero `96` come valore ordinabile o confrontabile.

La matrice CSR conserva fisicamente solo le sei colonne attive e il loro valore
`1`; gli zeri non vengono scritti né mantenuti in memoria. I loro indici CSR
sono soltanto coordinate interne della matrice sparse: non sono gli ID degli
expert e non vengono interpretati dal modello come numeri.

Per esempio, una Logistic Regression calcola concettualmente lo score di un
expert da predire sommando i pesi associati alle feature attive:

```text
score = peso(slot_0_96)
      + peso(slot_1_119)
      + peso(slot_2_202)
      + peso(slot_3_112)
      + peso(slot_4_90)
      + peso(slot_5_15)
      + intercetta
```

Le colonne non attive hanno valore `0` e non contribuiscono alla somma.

### Perché questa configurazione

- `sparse_output=True`: mantiene la matrice in formato sparso CSR.
- `min_frequency=41`: nel dataset BUILD_1 ogni token di decoding viene
  ripetuto nei 40 layer routed. Un ID osservato in una sola posizione di
  decoding compare quindi 40 volte nel train e viene raggruppato nella
  categoria "infrequente", anziché creare una colonna dedicata.
- `handle_unknown="infrequent_if_exist"`: un ID presente in validation, test
  o runtime ma mai osservato nel train viene associato allo stesso bucket
  "infrequente", se esso esiste. Il preprocessing non fallisce e il modello
  riceve comunque il segnale "categoria rara/sconosciuta".

L'encoder sarà sottoposto a `fit` **solo sui prompt train**, quindi solo sulle
righe con `split == "train"`. Validation e test riceveranno esclusivamente
`transform`. Questo evita data leakage.

## Feature numeriche: standardizzazione

Queste feature hanno un significato numerico e saranno trasformate con
`StandardScaler`:

```text
token_position
router_logit_value_0 ... router_logit_value_15
```

In BUILD_DATASET_2 lo script rileva automaticamente anche:

```text
hidden_pca_0 ... hidden_pca_63
```

Questi 64 valori sono già la feature 4 compressa tramite PCA layer-wise
train-only. In questa fase **non** viene calcolata una nuova PCA: le coordinate
già prodotte vengono soltanto standardizzate insieme alle altre feature
numeriche, usando media e deviazione del solo train.

Per ciascuna colonna numerica il train determina media $\mu$ e deviazione
standard $\sigma$:

$$
z = \frac{x - \mu}{\sigma}
$$

Validation, test e futuro runtime useranno poi gli stessi $\mu$ e $\sigma$,
senza ricalcolarli. Questo e' particolarmente importante per Logistic
Regression e KNN; per alberi e Random Forest la scala numerica influisce molto
meno, ma usare un'unica rappresentazione mantiene il confronto ordinato.

## Matrice finale `X_classic`

Lo script applica separatamente:

```text
feature categoriche  -> OneHotEncoder sparse
feature numeriche    -> StandardScaler
```

Le due parti vengono concatenate tramite `scipy.sparse.hstack` in una sola
matrice sparse CSR:

```text
X_classic_train.npz
X_classic_validation.npz
X_classic_test.npz
```

La dimensione finale non viene fissata nella documentazione perché dipende
esclusivamente dalle categorie apprese sul train e dal raggruppamento delle
categorie rare. Lo script salverà il numero di righe, colonne e valori non
nulli in `preprocessing_summary.json`.

## Target multi-label

Il target resta l'array già verificato:

```text
y_<split>.npy, shape [N_split, 256], dtype uint8
```

Ogni riga ha esattamente sei valori uguali a 1, corrispondenti agli expert di
$E_l(t+1)$; le altre 250 posizioni valgono 0. Questa rappresentazione permette
di confrontare i modelli con la stessa label, indipendentemente dal loro modo
di supportare il multi-label.

`LogisticRegression` dovrà essere inserita in `OneVsRestClassifier`, che
allena un classificatore binario per ciascuno dei 256 expert. Decision Tree,
Random Forest e KNN saranno verificati nella loro configurazione multi-output
diretta.

## Artefatti riutilizzabili

Il preprocessing salverà gli oggetti appresi esclusivamente sul train:

| Artefatto | Contenuto | Utilizzo futuro |
|---|---|---|
| `categorical_encoder.joblib` | Categorie one-hot e bucket infrequente | Trasformare validation, test e input runtime nello stesso spazio. |
| `numeric_scaler.joblib` | Medie e deviazioni standard del train | Standardizzare le feature numeriche senza leakage. |
| `feature_names.json` | Ordine e nomi delle colonne finali | Interpretare modelli e matrici sparse. |
| `preprocessing_summary.json` | Shape, densità e conteggi per split | Controllo di correttezza dell'output. |

In una fase runtime, il predictor dovrà usare gli stessi encoder/scaler salvati
qui. Rifare un `fit` sui dati runtime produrrebbe colonne o scale diverse e
renderebbe incompatibile il modello addestrato.

## Valutazione futura

La metrica iniziale sarà Recall@6:

$$
\mathrm{Recall@6} =
\frac{|\mathrm{Top6}_{pred} \cap E_l(t+1)|}{6}
$$

Il denominatore è sempre 6 perché il routing reale seleziona Top-6 expert.
Anche Precision@6 avrebbe lo stesso valore, poiché il predictor emetterà
esattamente sei expert; per ora non verrà quindi calcolata separatamente.

In seguito potranno essere aggiunte Recall@8/16 e Precision@8/16 per studiare
il trade-off tra copertura degli expert reali e costo del prefetch.

## Cosa non fa ancora questa fase

Questo preprocessing classico non includerà ancora:

1. il vettore token embedding 4096D estratto dal GGUF;
2. il calcolo della PCA: BUILD_DATASET_2 usa invece le 64 coordinate PCA già
   presenti nel suo Parquet;
3. KV cache, attention o indexer features;
4. il training o la valutazione dei quattro modelli;
5. la rete neurale Keras, che avrà una pipeline distinta e più adatta a
   embedding appresi.

Queste scelte mantengono il primo confronto tra modelli classici semplice,
riproducibile e direttamente interpretabile.

## Come eseguire `preprocessing_classical.py`

Dalla cartella `ML_pipeline`, assicurandosi che l'ambiente uv sia sincronizzato:

```bash
cd ~/ds4/test_predictor/prefetch_predictor_pipeline/ML_pipeline
uv sync
uv run classical_ml_preprocessing/preprocessing_classical.py
```

Il comando legge automaticamente `training/` e crea gli output rigenerabili
nelle sottocartelle `classical_ml_preprocessing/arrays/` e
`classical_ml_preprocessing/artifacts/`.

Se necessario, è possibile indicare esplicitamente i percorsi:

```bash
uv run classical_ml_preprocessing/preprocessing_classical.py \
  --training-dir training \
  --output-dir classical_ml_preprocessing
```

### Esecuzione per BUILD_DATASET_2

Per creare la rappresentazione sparse che include la feature 4 PCA-64, usa il
Parquet e le label della seconda build ma conserva i risultati in una cartella
ML separata della stessa build:

```bash
cd ~/ds4/test_predictor/prefetch_predictor_pipeline/ML_pipeline

uv run classical_ml_preprocessing/preprocessing_classical.py \
  --training-dir ../build_dataset_2/ml/training \
  --output-dir ../build_dataset_2/ml/classical_ml
```

Lo script rileverà `hidden_pca_0` ... `hidden_pca_63` senza altri parametri.
Le feature numeriche standardizzate saranno quindi 81:

```text
1 token_position + 16 router_logit_value_* + 64 hidden_pca_* = 81
```

Le feature categoriche, il One-Hot Encoder, `min_frequency=41`, lo split per
prompt e le label multi-label restano identici a BUILD_DATASET_1. Il numero
finale di colonne sparse potrà aumentare o diminuire leggermente in base alle
categorie osservate nel train della nuova raccolta.

Al termine, il primo file da controllare è:

```text
classical_ml_preprocessing/artifacts/preprocessing_summary.json
```

Riporta shape, numero di valori non nulli e densità delle matrici sparse per
train, validation e test.

## Risultato della prima esecuzione con `min_frequency=41`

La prima esecuzione reale di BUILD_1 ha prodotto questo output:

```text
michele@AsusMik:~/ds4/test_predictor/prefetch_predictor_pipeline/ML_pipeline$ \
uv run classical_ml_preprocessing/preprocessing_classical.py
1) Carico il Parquet base...
   Esempi totali: 199,000
2) Carico e verifico le label multi-label...
3) Fit di OneHotEncoder e StandardScaler sul solo train...
4) Trasformo train, validation e test in matrici sparse CSR...
5) Salvo matrici e artefatti riutilizzabili...

Preprocessing classico completato.
train: X=(157080, 6168), nnz=6,440,280, y=(157080, 256)
validation: X=(20600, 6168), nnz=844,600, y=(20600, 256)
test: X=(21320, 6168), nnz=874,120, y=(21320, 256)
Output: /home/michele/ds4/test_predictor/prefetch_predictor_pipeline/build_dataset_1/ml/classical_ml
```

I messaggi delle cinque fasi indicano:

| Riga stampata | Significato |
|---|---|
| `Carico il Parquet base` | Legge il dataset tabellare verificato in `training/prefetch_training_build_1.parquet`. |
| `Esempi totali: 199,000` | Conferma che il dataset base contiene tutti i 199.000 record supervisionati derivati dai trace DS4. |
| `Carico e verifico le label multi-label` | Carica `y_train.npy`, `y_validation.npy` e `y_test.npy`; controlla shape, tipo e che ogni record abbia esattamente sei expert target. |
| `Fit ... sul solo train` | One-hot encoder e scaler apprendono categorie, bucket infrequente, medie e deviazioni esclusivamente dalle 157.080 righe train. Validation e test non partecipano al `fit`. |
| `Trasformo ... in matrici sparse CSR` | Applica gli stessi trasformatori già appresi a tutti e tre gli split e concatena one-hot e feature numeriche in una matrice sparse. |
| `Salvo ... artefatti riutilizzabili` | Scrive le matrici `.npz`, le label copiate e gli oggetti `joblib` necessari a ripetere identicamente la trasformazione. |

### Come leggere le righe di train, validation e test

Ogni riga finale ha la forma:

```text
<split>: X=(numero_esempi, numero_feature), nnz=valori_non_nulli, y=(numero_esempi, 256)
```

| Campo | Significato nel risultato osservato |
|---|---|
| `train`, `validation`, `test` | Partizione per prompt già definita: rispettivamente 80, 10 e 10 prompt. Il numero di righe non è 80/10/10 esatto perché prompt diversi generano numeri diversi di token. |
| Prima dimensione di `X` | Numero di esempi dello split: 157.080 train, 20.600 validation, 21.320 test. La somma è 199.000. |
| `6168` | Numero totale di feature dopo le trasformazioni: 6.151 colonne one-hot e 17 feature numeriche standardizzate (`token_position` e 16 raw logits). La soglia `min_frequency=41` ha creato il bucket infrequente e ridotto le colonne rispetto alla codifica senza raggruppamento. |
| `nnz` | Numero di valori non nulli nella matrice sparse. In questo run ogni riga ha esattamente 41 valori non nulli: 24 one-hot (uno per ogni feature categoriale) e 17 numerici standardizzati. Per train, infatti, $157080 \times 41 = 6\,440\,280$. |
| `y=(N, 256)` | Target multi-label: una colonna per ciascuno dei 256 expert routed. Ogni riga ha esattamente sei valori uguali a 1, cioè gli expert di $E_l(t+1)$. |
| `Output: ...` | Cartella che contiene `arrays/` e `artifacts/`, cioè i file da caricare nello script di training scikit-learn. |

Il fatto che anche validation e test abbiano esattamente 41 valori non nulli
per riga conferma un dettaglio importante: gli ID token sconosciuti rispetto al
train vengono assegnati al bucket one-hot "infrequente", invece di produrre un
blocco `token_id` completamente nullo.
