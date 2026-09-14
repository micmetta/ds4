# FFN: routing multi-hot + layer + router state

Questo esperimento addestra una FFN sulla rappresentazione
`03_routing_layer_router_state` per predire `E_l(t+1)` dalle informazioni di
`E_l(t)` dello stesso layer.

## Input

Il dataset `../../ablations/03_routing_layer_router_state/arrays` contiene
4.408 feature sparse:

- 256 expert del routing history come multi-hot;
- 40 feature one-hot del layer;
- 4.096 feature one-hot fisse per gli ID dei 16 raw router logits;
- 16 valori dei router logits standardizzati colonna per colonna sul train.

## Configurazione

- hidden layer: 384 unita';
- attivazione: ReLU;
- dropout: 0.3;
- learning rate Adam: 0.00016428;
- batch size: 512;
- massimo 500 epoche;
- early stopping patience: 5.

## Risultato

L'addestramento si e' fermato dopo 28 epoche. La migliore Recall@6 sul
validation set e' 0.4161.

Il test set non e' stato valutato: questa run serve a confrontare su validation
l'effetto del router state rispetto alla rappresentazione `02`.

## Contenuto

- `models/ffn.keras`: modello con i pesi dell'epoca validation migliore;
- `reports/training_report.json`: iperparametri, history e metrica validation.
