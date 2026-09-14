# Baseline deterministica: persistenza del routing

## Obiettivo

`baseline.py` misura quanto sia predittivo il routing già osservato al token
corrente. Non addestra alcun modello e non usa token ID, layer o raw logits:
applica soltanto la regola deterministica

$$
\hat{E}_l(t+1) = E_l(t).
$$

Per ogni riga del dataset, gli stessi sei expert che DS4 ha selezionato nel
layer routed $l$ durante il decoding del token $t$ vengono proposti come
prefetch per il token successivo. Il confronto avviene contro il target reale
$E_l(t+1)$ già salvato nel Parquet.

## Perché è necessaria

Questa baseline rappresenta il riferimento minimo più naturale per il nostro
predictor: al termine del routing di $E_l(t)$ il sistema reale conosce già quei
sei expert. Prima di attribuire un miglioramento a un modello ML dobbiamo quindi
sapere quanta continuità temporale abbia il routing da solo.

La Recall@6 della baseline permette di interpretare la Logistic Regression:

- se è vicina alla Recall@6 validation della Logistic Regression (`0.3139` nel
  primo run), il modello ML aggiunge poco alla semplice persistenza;
- se è sensibilmente più bassa, token ID, layer e router logits stanno fornendo
  informazione predittiva aggiuntiva;
- ogni modello futuro dovrà essere confrontato almeno con questa baseline.

## Input e output

Lo script legge esclusivamente il Parquet già prodotto da `preprocessing.py`:

```text
build_dataset_1/ml/training/prefetch_training_build_1.parquet
```

Per ogni riga usa i campi seguenti:

| Colonne | Significato |
|---|---|
| `split` | Split assegnato per prompt: train, validation oppure test. |
| `experts_t_0` ... `experts_t_5` | I sei expert reali di $E_l(t)$; sono la predizione della baseline. |
| `routing_history_expert_0` ... `routing_history_expert_5` | La routing history con $H=1$. Lo script verifica che coincida esattamente con `experts_t_*`, quindi la baseline usa solo informazione disponibile anche al predictor. |
| `target_expert_0` ... `target_expert_5` | I sei expert reali di $E_l(t+1)$; sono il target. |

Il risultato predefinito viene salvato in:

```text
classical_ml_preprocessing/reports/baseline.json
```

Il JSON contiene la regola applicata, `top_k=6`, il flag `test_evaluated` e,
per ogni split valutato, `examples` e `recall_at_6`.

## Flusso di esecuzione

```text
main()
  |
  +-- leggi_argomenti()
  |     legge input, output e il flag opzionale --evaluate-test
  |
  +-- carica_tabella(input)
  |     carica solo split, experts_t_* e target_expert_* dal Parquet
  |     e verifica la presenza di train, validation e test
  |
  +-- valuta_split(tabella, "train")
  |     |
  |     `-- calcola_recall_persistenza(...)
  |           usa experts_t_* come sei predizioni e confronta con target_expert_*
  |
  +-- valuta_split(tabella, "validation")
  |
  +-- valuta_split(tabella, "test")       solo con --evaluate-test
  |
  `-- salva_report(...)
        salva il JSON riutilizzabile in reports/
```

`calcola_recall_persistenza()` verifica inoltre che routing e target contengano
sempre sei ID distinti nell'intervallo `[0, 255]`. Per ciascun esempio calcola:

$$
\mathrm{Recall@6} =
\frac{|E_l(t) \cap E_l(t+1)|}{6}.
$$

La metrica finale è la media su tutte le righe dello split. Poiché la baseline
restituisce sempre esattamente sei expert, Precision@6 e Recall@6 coincidono.

## Come eseguire

Dalla cartella `ML_pipeline`:

```bash
cd ~/ds4/test_predictor/prefetch_predictor_pipeline/ML_pipeline
uv run classical_ml_preprocessing/baseline.py
```

Il comando valuta train e validation, mantenendo il test set non osservato.
Solo dopo aver scelto la configurazione del predictor puoi includere il test:

```bash
uv run classical_ml_preprocessing/baseline.py --evaluate-test
```
