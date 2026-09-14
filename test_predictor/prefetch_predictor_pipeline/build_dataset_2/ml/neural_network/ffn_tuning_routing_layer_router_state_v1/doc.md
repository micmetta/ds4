# FFN tuning — routing + layer + router state (v1)

## Scopo

Ricerca di iperparametri per una FFN che predice i 6 expert del layer routed al token successivo, usando esclusivamente informazioni osservate sullo stesso layer al token corrente.

Il dataset di input è `../../ablations/03_routing_layer_router_state/arrays` e contiene 4.408 feature per esempio:

- 256 feature multi-hot degli expert scelti dal router al token corrente;
- 40 feature one-hot del layer routed corrente;
- 4.096 feature one-hot degli ID dei 16 router logits (16 posizioni × 256 ID expert possibili);
- 16 valori dei router logits, standardizzati colonna per colonna con statistiche apprese solo sul train set.

## Configurazione della ricerca

- strategia: KerasTuner `RandomSearch`;
- trial: 15;
- layer nascosti esplorati: 1 oppure 2;
- unità per layer: 256, 384 oppure 512;
- attivazione: ReLU;
- dropout: da 0.0 a 0.4;
- learning rate: da `0.00005` a `0.0005`;
- batch size: 512;
- massimo: 500 epoche per trial;
- early stopping: pazienza 5 su `val_recall_at_6`, con ripristino dei pesi migliori.

## Risultato migliore

La migliore configurazione selezionata è:

- 1 layer nascosto da 512 unità;
- ReLU;
- dropout `0.3`;
- learning rate `0.00006997208338540213`;
- validation Recall@6: **0.4167**;
- durata complessiva del tuning: circa **25 minuti e 7 secondi**.

Benché la ricerca includesse anche due layer nascosti, il trial migliore ne usa uno solo. Il campo `units_2: 384` presente nel report è quindi inattivo per il modello vincente: KerasTuner registra comunque l'iperparametro, ma non viene usato quando `hidden_layers=1`.

## Contenuti

- `models/best_ffn.keras`: pesi del modello migliore del tuning;
- `reports/tuning_report.json`: configurazione e metrica migliori;
- `tuner/`: stato e risultati dei trial KerasTuner.

## Nota sperimentale

Il test set non è stato usato in questa selezione: il confronto qui è esclusivamente sul validation set. Prima di una valutazione finale su un holdout mai consultato, conviene riaddestrare una volta la configurazione selezionata in una nuova cartella di esperimento.
