# `preprocessing.py` — dal JSONL derivato ai dati per il training

## Scopo dello script

`preprocessing.py` non addestra ancora un modello di Machine Learning. Il suo
unico compito e' trasformare il dataset supervisionato JSONL creato da
`build_dataset.py` in due rappresentazioni complementari:

1. una tabella Parquet, facile da aprire e ispezionare in CLion;
2. tensori NumPy `.npy`, pronti per uno script di training.

Il record di partenza rappresenta una coppia `(token t, layer l)` e richiede
di predire gli expert del medesimo layer al token successivo:

$$
E_l(t) \quad \longrightarrow \quad E_l(t+1)
$$

BUILD_1 usa soltanto i 40 layer routed (dal layer 3 al layer 42), ciascuno con
256 expert possibili e una scelta reale Top-6.

## Input e output

Se eseguito senza argomenti:

```bash
uv run preprocessing.py
```

lo script legge:

```text
build_dataset_1/derived/prefetch_dataset_build_1.jsonl
```

e crea:

```text
build_dataset_1/ml/training/
├── prefetch_training_build_1.parquet
├── feature_columns.json
├── split_by_prompt.json
├── preprocessing_summary.json
└── arrays/
    ├── X_train.npy
    ├── X_validation.npy
    ├── X_test.npy
    ├── y_train.npy
    ├── y_validation.npy
    ├── y_test.npy
    ├── target_expert_ids_train.npy
    ├── target_expert_ids_validation.npy
    └── target_expert_ids_test.npy
```

I due percorsi e il seed possono essere cambiati senza modificare il codice:

```bash
uv run preprocessing.py \
  --input derived/prefetch_dataset_build_1.jsonl \
  --output training \
  --seed 42
```

## Flusso completo

```text
main()
  |
  +-- leggi_argomenti()
  |     legge --input, --output e --seed (oppure usa i valori di default)
  |
  +-- carica_record_jsonl(percorso_input)
  |     legge prefetch_dataset_build_1.jsonl riga per riga
  |       |
  |       `-- controlla_record(esempio, numero_riga)
  |             verifica H=1, Top-6 e 16 raw logits prima di accettare il record
  |     risultato: lista di record JSON validati in memoria
  |
  +-- crea_split_per_prompt(record, seed)
  |     assegna ogni prompt a train/validation/test per categoria, senza leakage
  |     risultato: mapping prompt_id -> split
  |
  +-- crea_colonne_tabellari_e_tensori(record, split_per_prompt)
  |     appiattisce i record JSON e costruisce DataFrame, X, y e target Top-6
  |     risultato: tabella Pandas + array NumPy completi
  |
  `-- salva_output(tabella, X, y, target_ids, ...)
        scrive Parquet, metadati JSON e array .npy separati per split
        risultato: build_dataset_1/ml/training/
```

`main()` e' soltanto l'orchestratore: chiama queste funzioni nell'ordine
mostrato, stampa le quattro fasi a terminale e riporta infine shape e conteggi
degli output.

## 1. Dipendenze e costanti

Lo script importa:

- `numpy`: costruisce e salva le matrici numeriche;
- `pandas`: costruisce la tabella a colonne;
- `pyarrow`: e' il motore usato da Pandas per scrivere Parquet;
- moduli standard per JSON, percorsi, seed e split.

Le dipendenze sono dichiarate in `pyproject.toml` e installate nella `.venv`
locale da uv. Non servono per ora scikit-learn, PyTorch o altre librerie ML:
questo non e' ancora il training.

Le costanti esplicitano le ipotesi della BUILD_1:

| Costante | Valore | Significato |
|---|---:|---|
| `NUM_EXPERTS` | 256 | Dimensione dell'universo degli expert routed. |
| `TOP_K` | 6 | Expert selezionati dal router per token/layer. |
| `ROUTER_SCORES` | 16 | Numero di raw logits salvati in BUILD_1. |
| `SEED` | 42 | Seed riproducibile per lo split dei prompt. |

Se in futuro la raccolta usa, ad esempio, `DS4_DATASET_ROUTER_MAX_SCORES=32`,
anche `ROUTER_SCORES` dovra' essere aggiornato da 16 a 32, altrimenti la
validazione blocchera' volutamente il preprocessing.

## 2. Lettura e validazione del JSONL

La funzione `carica_record_jsonl()` legge una riga JSON alla volta dal dataset
derivato. Per ogni riga chiama `controlla_record()` prima di conservarla in
memoria.

La validazione richiede questi campi:

```text
prompt_id, prompt_category,
token_position, token_id, layer,
routing_history, router_logits,
target_experts_next_token
```

Controlla inoltre che:

1. `routing_history.history_length` sia esattamente `1`;
2. la history e il target contengano entrambi 6 ID expert;
3. `router_logits.stored_count` sia esattamente 16.

Questi controlli non trasformano i dati: impediscono di creare silenziosamente
un dataset incoerente. Per esempio, una nuova raccolta con history `H=3` o con
32 logits deve prima avere un preprocessing aggiornato in modo consapevole.

## 3. Split corretto: per prompt, non per riga

`crea_split_per_prompt()` raggruppa prima tutti i `prompt_id` per
`prompt_category`. Per ciascuna categoria:

1. ordina gli ID per avere una base stabile;
2. li mescola con `random.Random(seed)`;
3. assegna l'80% a `train`, il 10% a `validation` e il restante 10% a `test`.

Nel dataset V0 corrente ogni categoria ha 10 prompt. Il risultato e' quindi,
per ogni categoria, 8 prompt train, 1 validation e 1 test.

Questo dettaglio e' fondamentale: un singolo prompt produce migliaia di righe
molto correlate (token successivi e 40 layer). Uno split casuale per righe
metterebbe token dello stesso prompt sia nel train sia nel test; il modello
vedrebbe in test esempi quasi identici a quelli di training e la metrica
sarebbe troppo ottimistica. Con lo split per prompt questo leakage non avviene.

Il mapping completo e riproducibile e' salvato in:

```text
training/split_by_prompt.json
```

## 4. Costruzione della tabella Parquet

`crea_colonne_tabellari_e_tensori()` percorre ogni record JSONL e costruisce
prima colonne Python, poi un `pandas.DataFrame`. Le liste JSON vengono
**appiattite** in colonne con nomi espliciti: cio' e' preferibile a inserire
oggetti annidati nel Parquet, perche' CLion puo' visualizzarle come una
normale tabella.

Le colonne sono:

| Gruppo | Colonne | Perche' vengono salvate |
|---|---|---|
| Split e prompt | `split`, `prompt_id`, `prompt_category`, `sequence_id` | Tracciabilita' dell'esempio e ispezione del corpus. |
| Token/layer | `token_position`, `token_id`, `layer` | Stato della sequenza e layer per cui si fa la predizione. |
| History | `history_length`, `history_token_position_0`, `routing_history_expert_0` ... `_5` | Feature 1, cioe' $E_l(t)$ con `H=1`. |
| Routing corrente | `experts_t_0` ... `_5` | Copia leggibile del routing corrente, utile a verificare i dati. |
| Raw logits | `router_logit_expert_id_0` ... `_15` e `router_logit_value_0` ... `_15` | Feature 2: Top-16 ordinata per raw logit. |
| Target | `target_expert_0` ... `_5` | I sei ID realmente scelti al token successivo, $E_l(t+1)$. |

Con `H=1`, `experts_t` e `routing_history.expert_ids[0]` sono per definizione
gli stessi sei ID, entrambi pari a $E_l(t)$. Nel Parquet sono mantenuti entrambi
per leggibilita' e controllo; nella matrice di feature `X` vengono invece
inclusi una sola volta, per non duplicare informazione identica.

Il file risultante e':

```text
training/prefetch_training_build_1.parquet
```

## 5. Costruzione della matrice delle feature `X`

Per ogni esempio viene creato un vettore `feature` e poi tutti i vettori sono
impilati in una matrice NumPy `X` con tipo `float32`.

L'ordine esatto delle 41 colonne e':

```text
0      token_position
1      token_id
2      layer
3..8   routing_history_expert_0 .. routing_history_expert_5
9..24  router_logit_expert_id_0 .. router_logit_expert_id_15
25..40 router_logit_value_0 .. router_logit_value_15
```

Quindi:

$$
3 + 6 + 16 + 16 = 41
$$

feature per esempio.

L'ordine e' scritto anche in `training/feature_columns.json`, cosi' un futuro
script di training non deve ricostruirlo a mano.

### Cosa `X` include e cosa non include

`X` include esclusivamente dati disponibili prima di conoscere il routing del
token futuro: posizione, ID del token corrente, layer, routing osservato a
$t$ e raw logits osservati a $t$.

Non include il target $E_l(t+1)$: inserirlo in `X` sarebbe leakage.

Non include ancora il vettore di token embedding da 4096 dimensioni. BUILD_1
salva nel JSONL soltanto `token_id` e il riferimento alla tabella
`token_embd.weight` del GGUF. Per materializzare i 4096 valori servira' uno
step dedicato che estragga correttamente la tabella dal GGUF e faccia il lookup
per ogni `token_id`.

Infine, `token_id` ed expert ID compaiono in `X` come numeri `float32` solo per
avere un formato tabellare unico. Semanticamente sono **categorie discrete**:
il futuro modello dovrebbe preferibilmente usare embedding/encoding categorici,
non assumere che l'expert 200 sia “il doppio” dell'expert 100. Per questa
ragione lo script non applica qui `StandardScaler` o PCA.

## 6. Costruzione della label multi-label `y`

Per ciascun record il target originale e':

```json
"target_experts_next_token": [e_1, e_2, e_3, e_4, e_5, e_6]
```

Lo script costruisce un vettore di 256 elementi inizialmente tutti zero:

$$
y \in \{0,1\}^{256}
$$

e imposta a 1 le sei posizioni corrispondenti agli ID del target:

$$
y[e] =
\begin{cases}
1 & \text{se } e \in E_l(t+1) \\
0 & \text{altrimenti}
\end{cases}
$$

Esempio semplificato: se il target e' `[12, 44, 55, 90, 144, 203]`, `y[12]`,
`y[44]`, `y[55]`, `y[90]`, `y[144]` e `y[203]` valgono 1; tutte le altre 250
posizioni valgono 0.

Questa rappresentazione e' adatta a un classificatore multi-label che produca
uno score/probabilita' per ciascuno dei 256 expert. A runtime potremo scegliere
poi i Top-6 score predetti da inviare al prefetch.

Gli array `target_expert_ids_<split>.npy` salvano comunque la lista originale
ordinata dei sei ID. Sono utili per metriche Top-k o per verificare i risultati,
senza dover riconvertire il vettore multi-hot.

## 7. Salvataggio per split

`salva_output()` filtra le righe tramite il vettore `split_per_riga` e produce:

| File | Tipo e shape | Contenuto |
|---|---|---|
| `X_<split>.npy` | `float32`, `[N_split, 41]` | Input numerico nell'ordine documentato. |
| `y_<split>.npy` | `uint8`, `[N_split, 256]` | Target multi-label. Ogni riga ha sei `1`. |
| `target_expert_ids_<split>.npy` | `int16`, `[N_split, 6]` | Target Top-6 originale e ordinato. |

Lo script salva anche `preprocessing_summary.json`, con numero di esempi,
numero di prompt e shape effettive per train, validation e test. E' il primo
file da controllare dopo un run.

## Cosa aspettarsi dal primo run

Con il dataset BUILD_1 attuale, formato da 199.000 record derivati, il terminale
mostrera' quattro fasi e un riepilogo simile a:

```text
1) Leggo il dataset JSONL derivato...
   Record letti: 199,000
2) Creo lo split train/validation/test per prompt e categoria...
3) Creo la tabella Parquet e i tensori X/y...
4) Salvo i file di training...

Preprocessing BUILD_1 completato.
Parquet: .../build_dataset_1/ml/training/prefetch_training_build_1.parquet
train:      ... esempi, 80 prompt, X=[..., 41], y=[..., 256]
validation: ... esempi, 10 prompt, X=[..., 41], y=[..., 256]
test:       ... esempi, 10 prompt, X=[..., 41], y=[..., 256]
```

I conteggi di esempi non sono necessariamente esattamente 80/10/10 in termini
di righe: prompt diversi generano numeri diversi di token. La separazione
80/10/10 si applica ai **100 prompt**, non ai 199.000 record.

## Limiti intenzionali di questa prima versione

Questi non sono errori; sono scelte per mantenere BUILD_1 semplice e
verificabile:

1. supporta solo `H=1` e 16 raw logits;
2. non materializza `token_embedding` 4096D;
3. non include ancora hidden features, KV cache o attention/indexer;
4. non standardizza e non applica PCA: queste trasformazioni dovranno essere
   apprese solo dal train e poi riusate identiche su validation/test/runtime;
5. non addestra ne' valuta un modello.

Il prossimo script potra' caricare gli array di `training/`, costruire un
modello multi-label e misurare, ad esempio, il recall degli expert reali nei
Top-6 predetti.

## Come eseguire `preprocessing.py`

Assicurati di trovarti nella cartella della pipeline e che l'ambiente uv sia
stato gia' inizializzato con le dipendenze `numpy`, `pandas` e `pyarrow`:

```bash
cd ~/ds4/test_predictor/prefetch_predictor_pipeline/ML_pipeline
uv sync
```

Per eseguire il preprocessing con i percorsi e il seed di default:

```bash
uv run preprocessing.py
```

Il comando usa automaticamente l'interprete isolato in `.venv/`, legge
`derived/prefetch_dataset_build_1.jsonl` e crea/aggiorna la cartella
`training/`.

Per scegliere esplicitamente input, output oppure seed:

```bash
uv run preprocessing.py \
  --input derived/prefetch_dataset_build_1.jsonl \
  --output training \
  --seed 42
```

Alla fine verifica anzitutto il riepilogo stampato nel terminale e poi apri:

```text
training/preprocessing_summary.json
training/prefetch_training_build_1.parquet
```

Il primo conferma numero di esempi e shape di train/validation/test; il
secondo consente di ispezionare le colonne del dataset in formato tabellare in
CLion.
