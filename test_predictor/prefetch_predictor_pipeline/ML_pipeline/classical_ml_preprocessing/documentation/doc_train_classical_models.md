# Training dei modelli classici di scikit-learn

## Obiettivo

Questo script addestra e valuta i primi modelli classici per il prefetch
predictor usando le matrici sparse prodotte da `preprocessing_classical.py`.
Non ricrea feature, split o label: legge gli artefatti già verificati in
`classical_ml_preprocessing/arrays/`, addestra i modelli richiesti e salva sia
i classificatori sia le metriche.

I modelli disponibili sono:

1. `OneVsRestClassifier(LogisticRegression(...))`;
2. `DecisionTreeClassifier`;
3. `RandomForestClassifier`;
4. `KNeighborsClassifier`.

L'obiettivo supervisionato resta invariato: osservate le feature del token
corrente $t$ nel layer routed $l$, predire i sei expert selezionati dal router
nello stesso layer al token successivo:

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
preprocessing_classical.py
        |
        v
classical_ml_preprocessing/arrays/     matrici sparse + label verificate
        |
        v
train_classical_models.py              fase descritta in questo file
        |
        v
models/ + reports/                     classificatori e risultati
```

Il training classico dipende quindi dal preprocessing, ma non lo modifica.
Gli stessi input possono essere riusati per addestrare più configurazioni e
confrontarle in modo riproducibile.

## Struttura prevista

```text
classical_ml_preprocessing/
├── documentation/
│   ├── doc_classical_preprocessing.md
│   └── doc_train_classical_models.md       questo documento
├── preprocessing_classical.py
├── train_classical_models.py               matrici sparse -> modelli + report
├── arrays/                                 input creato dal preprocessing
│   ├── X_classic_train.npz
│   ├── X_classic_validation.npz
│   ├── X_classic_test.npz
│   ├── y_train.npy
│   ├── y_validation.npy
│   └── y_test.npy
├── models/                                 creati dallo script
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   └── knn.joblib
└── reports/                                creati dallo script
    ├── logistic_regression.json
    ├── decision_tree.json
    ├── random_forest.json
    ├── knn.json
    └── training_summary.json
```

Le cartelle `models/` e `reports/` vengono create automaticamente. I file
esistenti con lo stesso nome vengono sostituiti dal nuovo run.

## Input

Per ogni split lo script legge una coppia di file dalla cartella `arrays/`:

```text
X_classic_<split>.npz    matrice CSR delle feature
y_<split>.npy            target multi-hot
```

`carica_split()` controlla prima di addestrare che:

- entrambi i file esistano;
- il numero di righe di `X` e `y` coincida;
- `y` abbia shape `[N, 256]`;
- ogni riga del target contenga esattamente `TOP_K = 6` expert attivi.

Questi controlli evitano di valutare accidentalmente una matrice e label
generate da esecuzioni incompatibili del preprocessing.

## Flusso di esecuzione di `train_classical_models.py`

Lo script è orchestrato da `main()`. Lo schema usa i nomi esatti delle
funzioni chiamate dal codice:

```text
main()
  |
  +-- leggi_argomenti()
  |     legge modelli, cartelle di input/output, flag test e limite KNN
  |
  +-- leggi_nomi_modello(args.models)
  |     valida i nomi, rimuove duplicati e conserva l'ordine indicato
  |
  +-- carica_split(data_dir, "train")
  +-- carica_split(data_dir, "validation")
  |     carica sempre train e validation e ne verifica la coerenza
  |
  +-- carica_split(data_dir, "test")             solo con --evaluate-test
  |
  +-- per ogni modello selezionato
  |     |
  |     +-- crea_modello(nome_modello)
  |     |     crea il classificatore con parametri riproducibili
  |     |
  |     +-- addestra_e_valuta(...)
  |     |     per KNN può ridurre il train; esegue fit e Recall@6 su train,
  |     |     validation ed eventualmente test
  |     |
  |     +-- joblib.dump(...)
  |     |     salva il modello addestrato in models/<nome>.joblib
  |     |
  |     `-- salva_report(...)
  |           scrive reports/<nome>.json
  |
  `-- scrive reports/training_summary.json
        riunisce configurazione globale e report di tutti i modelli del run
```

Validation è sempre valutata; il test è deliberatamente opzionale, così può
restare non osservato durante la scelta del modello e dei suoi parametri.

## Modelli e configurazioni iniziali

| Nome CLI | Classificatore | Configurazione iniziale |
|---|---|---|
| `logistic_regression` | `ParallelProgressOneVsRestClassifier(LogisticRegression)` oppure `SequentialOneVsRestClassifier` | solver `saga`, `class_weight="balanced"`, `max_iter` configurabile da CLI (default 1000), parallelismo configurabile da CLI (default tutti i core). |
| `decision_tree` | `DecisionTreeClassifier` | `random_state=42`. |
| `random_forest` | `RandomForestClassifier` | 30 alberi, `max_depth=16`, `min_samples_leaf=20`, `max_features="sqrt"`, `n_jobs=1`, `random_state=42`. |
| `knn` | `KNeighborsClassifier` | 5 vicini, pesi `distance`, `n_jobs=-1`. |

`random_state=42` rende riproducibili le parti casuali dei modelli e del
sotto-campionamento KNN. Logistic Regression usa One-vs-Rest perché il target
è multi-label: viene allenato un classificatore binario per ognuno dei 256
expert.

Per il nuovo esperimento Logistic Regression, `max_iter` è stato portato da
200 a 1000. La modifica è mirata ai warning del primo run: alcuni dei 256
classificatori binari non erano convergenti al limite precedente. Nessun'altra
feature o parametro viene modificato in questo confronto.

### `max_iter` dinamico per Logistic Regression

`--logistic-max-iter` controlla dinamicamente il valore di `max_iter` usato da
ciascuno dei 256 classificatori binari della Logistic Regression. Il default
è `1000`; il valore scelto viene salvato sia nel report del modello sia nel
training summary. Questo parametro non modifica gli altri algoritmi.

### Parallelismo e progresso della Logistic Regression

`--logistic-n-jobs` decide come addestrare i 256 classificatori binari:

- `--logistic-n-jobs -1` (default) usa tutti i core disponibili tramite
  `ParallelProgressOneVsRestClassifier`. Il processo principale stampa un
  messaggio non ordinato appena riceve ciascun modello completato, per esempio:

```text
Logistic Regression: expert 143/256 completato (37/256 totali)
```

- `--logistic-n-jobs 1` usa `SequentialOneVsRestClassifier`, addestra un
  expert alla volta e stampa un avanzamento ordinato, per esempio:

```text
Logistic Regression: expert 37/256 completato
```

In modalità parallela il numero tra parentesi è il contatore affidabile dei
completamenti, mentre l'ID dell'expert dipende dall'ordine con cui i worker
terminano. La modalità sequenziale è tipicamente più lenta, ma ordina gli ID.
I valori positivi maggiori di uno limitano il numero di worker; `0` e valori
minori di `-1` non sono ammessi.

### Limiti del Random Forest

La prima configurazione, con 100 alberi senza limiti di profondità e
`n_jobs=-1`, ha esaurito la RAM durante il training BUILD_1. Il Random Forest
ora è volutamente un baseline più compatto:

- `n_estimators=30` limita il numero totale di alberi conservati;
- `max_depth=16` impedisce che ogni albero continui a ramificarsi fino a
  memorizzare dettagli troppo specifici del train;
- `min_samples_leaf=20` vieta foglie sostenute da pochissimi esempi;
- `max_features="sqrt"` considera a ogni split circa la radice quadrata delle
  6.168 feature, riducendo costo e correlazione tra alberi;
- `n_jobs=1` costruisce un albero alla volta, evitando che più alberi
  concorrenti sommino il loro picco di memoria.

Questi parametri non sono ancora una ricerca di iperparametri: servono a far
completare in modo affidabile il primo confronto. La scelta del modello resta
basata sulla Recall@6 di validation.

## Gestione speciale di KNN

KNN può essere molto costoso perché, in fase di predizione, confronta ogni
esempio di validation o test con gli esempi usati per il training. Per questo
lo script limita per default il train KNN a 20.000 righe:

```text
--knn-max-train-samples 20000
```

`seleziona_train_knn()` estrae senza reinserimento un sottoinsieme casuale con
generatore inizializzato da `RANDOM_STATE`, ordina gli indici e applica gli
stessi indici a `X_train` e `y_train`. Il modello resta quindi valutato su
validation completa, ma è addestrato su un train ridotto e ripetibile.

Per usare tutto il train:

```text
--knn-max-train-samples 0
```

Un valore negativo non è ammesso. Se il limite è maggiore o uguale al numero
di righe train, non viene eseguito alcun campionamento.

## Score e Recall@6

Dopo il `fit`, `estrai_score_positivi()` invoca `predict_proba()` e ricava uno
score positivo per ciascuno dei 256 expert. Gestisce sia la matrice restituita
da alcuni classificatori sia la lista di 256 matrici restituita dai modelli
multi-output.

Per ogni esempio, `calcola_recall_at_6()` seleziona i sei score più alti. La
metrica media è:

$$
\mathrm{Recall@6} =
\frac{|\mathrm{Top6}_{pred} \cap E_l(t+1)|}{6}
$$

Il denominatore è sempre sei poiché ogni label contiene esattamente i Top-6
expert del router. Se il predictor emette esattamente sei expert, Precision@6
e Recall@6 coincidono; perciò il report iniziale salva soltanto Recall@6.

## Report e modelli salvati

Per ogni modello viene creato `reports/<nome_modello>.json` con:

| Campo | Significato |
|---|---|
| `train_examples_used` | Numero di righe train impiegate nel `fit`; può essere ridotto solo per KNN. |
| `training_seconds` | Durata del `fit` in secondi. |
| `train_evaluation_seconds` | Durata del calcolo degli score sul train effettivamente usato nel fit. |
| `train_recall_at_6` | Recall@6 sul train effettivamente usato nel fit; serve a diagnosticare overfitting. |
| `validation_seconds` | Durata del calcolo degli score su validation. |
| `validation_recall_at_6` | Recall@6 media su validation. |
| `logistic_max_iter` | Limite di iterazioni scelto per Logistic Regression; presente solo nel suo report. |
| `logistic_n_jobs` | Numero di worker scelto per Logistic Regression; presente solo nel suo report. |
| `test_seconds` | Durata del calcolo degli score sul test, presente solo con `--evaluate-test`. |
| `test_recall_at_6` | Recall@6 media sul test, presente solo con `--evaluate-test`. |

`training_summary.json` contiene inoltre `random_state`, `top_k`, il flag
`test_evaluated` e un mapping `models` che raccoglie i report del run.

Il corrispondente file `models/<nome_modello>.joblib` contiene il classificatore
già addestrato; per fare inferenza deve ricevere feature nello stesso spazio
CSR creato dagli encoder e dallo scaler salvati dal preprocessing.

## Come eseguire il training standard

Dalla cartella `ML_pipeline`, con l'ambiente uv sincronizzato:

```bash
cd ~/ds4/test_predictor/prefetch_predictor_pipeline/ML_pipeline
uv sync
uv run classical_ml_preprocessing/train_classical_models.py
```

Senza argomenti vengono addestrati Logistic Regression, Decision Tree e Random
Forest. Per ogni modello il report contiene Recall@6 su train e validation.
La metrica train serve soltanto a individuare overfitting; la scelta fra i
modelli deve basarsi sulla validation. KNN resta escluso dal run standard a
causa del suo costo; può essere avviato separatamente:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models knn
```

Per addestrare una selezione esplicita, mantenendo l'ordine desiderato:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models logistic_regression,random_forest,knn \
  --knn-max-train-samples 20000
```

Per rilanciare soltanto il Random Forest dopo il precedente arresto per RAM:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models random_forest
```

## Conservare risultati di parametri diversi

Quando viene addestrata soltanto Logistic Regression, lo script genera
automaticamente il suffisso `max_iter_<valore>` a partire da
`--logistic-max-iter`. Non è quindi necessario scrivere manualmente
`--output-suffix`. Per un run con `max_iter=1000`:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models logistic_regression \
  --logistic-max-iter 1000 \
  --logistic-n-jobs -1
```

Per impostare realmente, per esempio, `max_iter=1000000`, usa:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models logistic_regression \
  --logistic-max-iter 1000000
```

Un limite così alto può richiedere molto tempo. Per run con più modelli,
oppure per aggiungere un'altra etichetta al nome, `--output-suffix` rimane
disponibile e ha priorità sul suffisso automatico.

Vengono creati:

```text
models/logistic_regression_max_iter_1000.joblib
reports/logistic_regression_max_iter_1000.json
reports/training_summary_logistic_regression_max_iter_1000.json
```

Quando `--models` contiene un solo algoritmo, il summary include sempre anche
il suo nome. Se invece il run addestra più algoritmi, resta
`training_summary_<suffisso>.json`, perché riunisce più report.

## Valutazione finale sul test

Durante la scelta del modello si usa normalmente soltanto validation. Dopo
aver fissato la configurazione, il test può essere valutato con:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --models random_forest \
  --evaluate-test
```

Il flag carica anche `X_classic_test.npz` e `y_test.npy`, calcola Recall@6 e
aggiunge le rispettive chiavi al report del modello. Rieseguire lo script con
lo stesso nome di modello aggiorna il relativo file `.joblib` e `.json`.

## Percorsi personalizzati

I percorsi predefiniti sono relativi alla directory che contiene lo script:

```text
--data-dir     classical_ml_preprocessing/arrays
--models-dir   classical_ml_preprocessing/models
--reports-dir  classical_ml_preprocessing/reports
```

Possono essere sostituiti, per esempio per separare esperimenti diversi:

```bash
uv run classical_ml_preprocessing/train_classical_models.py \
  --data-dir classical_ml_preprocessing/arrays \
  --models-dir experiments/run_01/models \
  --reports-dir experiments/run_01/reports
```

## Cosa non fa ancora questa fase

Questo script non include ancora:

1. ricerca di iperparametri o cross-validation;
2. calibrazione delle probabilità o scelta di soglie per expert;
3. metriche diverse da Recall@6;
4. confronto automatico o selezione del miglior modello;
5. inferenza runtime integrata nel predictor DS4;
6. modelli neurali o feature embedding/PCA.

Il suo scopo è fornire un primo benchmark semplice, riproducibile e comparabile
fra modelli classici sullo stesso dataset preprocessato.
