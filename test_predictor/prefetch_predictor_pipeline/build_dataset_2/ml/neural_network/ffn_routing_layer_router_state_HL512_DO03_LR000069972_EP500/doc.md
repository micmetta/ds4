# FFN — routing + layer + router state, HL512

## Scopo

Replica indipendente della configurazione selezionata dal tuning `ffn_tuning_routing_layer_router_state_v1`, per confermarne il risultato prima di qualunque valutazione finale su un holdout.

L'input proviene da `../../ablations/03_routing_layer_router_state/arrays` e ha 4.408 feature: routing multi-hot, layer corrente, ID dei 16 router logits codificati one-hot e i corrispondenti 16 valori standardizzati sul train set.

## Configurazione

- hidden layers: `512`;
- attivazione: ReLU;
- dropout: `0.3`;
- learning rate: `0.00006997208`;
- batch size: 512;
- massimo epoche: 500;
- early stopping: pazienza 5 su `val_recall_at_6`, con ripristino dei pesi migliori.

## Risultato

- feature di input: 4.408;
- parametri addestrabili: 2.388.736;
- epoche eseguite: 41;
- migliore validation Recall@6: **0.4169** (epoca 36);
- durata: circa 101 secondi.

Il risultato conferma il tuning (`0.4167`) e migliora di circa `0.0008` il precedente addestramento diretto con 384 unità (`0.4161`) sullo stesso validation set. Il test set non è stato usato da questo esperimento.

## Contenuti

- `models/ffn.keras`: modello con i pesi dell'epoca migliore;
- `reports/training_report.json`: iperparametri, cronologia e risultato della validazione.
