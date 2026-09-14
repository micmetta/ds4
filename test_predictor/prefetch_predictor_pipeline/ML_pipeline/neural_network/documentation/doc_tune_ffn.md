# Ricerca iperparametri FFN con KerasTuner

## Obiettivo

`tune_ffn.py` esegue una `RandomSearch` di KerasTuner su una FFN con input sparse. Cerca automaticamente il numero di layer nascosti, il numero di nodi per layer, l'attivazione, il dropout e il learning rate.

Ogni trial viene valutato esclusivamente su **validation Recall@6**. Il test set non viene caricato: rimane riservato alla valutazione finale del modello e della configurazione scelti.

Lo script disabilita XLA perche' il primo Dense riceve input TensorFlow sparse:
questa combinazione non e' ancora interamente supportata dal compilatore XLA
su GPU. TensorFlow continua a eseguire i layer della rete sulla GPU CUDA.

Ogni trial usa dataset ripetibili e step espliciti per epoca, evitando
l'avviso Keras `input ran out of data`.

## Spazio di ricerca

Per default lo script prova:

- da 1 a 3 layer nascosti;
- 128, 256, 384 o 512 nodi per ogni layer presente;
- attivazione `relu` oppure `gelu`;
- dropout tra `0.0` e `0.4`, con passo `0.1`;
- learning rate logaritmico tra `0.0001` e `0.003`.

Tutti questi limiti sono modificabili da CLI. Ogni trial usa Adam, binary cross-entropy, batch size 512 ed early stopping su `val_recall_at_6`.

## Output

```text
<output-dir>/
├── tuner/
│   └── random_search/
│       └── ... stato e risultati KerasTuner ...
├── models/
│   └── best_ffn.keras
└── reports/
    └── tuning_report.json
```

`tuner/` permette di ispezionare i singoli trial. `best_ffn.keras` e' il modello del trial migliore, mentre il JSON salva il migliore set di iperparametri e il suo Recall@6 validation.

KerasTuner usa TensorBoard internamente per registrare gli iperparametri dei
trial. `tensorboard` e' quindi una dipendenza del progetto e viene installato
da `uv sync` insieme a TensorFlow e KerasTuner.

Lo script rifiuta una `--output-dir` gia' esistente: questo evita di cancellare
o sovrascrivere lo stato di un tuning precedente.

## Esecuzione

Dalla cartella `ML_pipeline/`, dopo `uv sync`:

```bash
uv run neural_network/tune_ffn.py \
  --data-dir ../build_dataset_2/ml/classical_ml/arrays \
  --output-dir ../build_dataset_2/ml/neural_network/ffn_random_search_v1 \
  --max-trials 20 \
  --epochs 20
```

Per una ricerca iniziale economica, usare pochi trial e poche epoche; dopo aver scelto l'architettura, eseguire `train_ffn.py` in una nuova cartella di output con gli iperparametri selezionati.
