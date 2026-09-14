# FFN tuning: routing multi-hot + layer (v1)

Questo esperimento seleziona gli iperparametri di una feed-forward network
(FFN) con KerasTuner RandomSearch. Il tuning usa il validation set per
confrontare le configurazioni; il test set non viene usato.

## Feature di input

Il dataset di input e' `../../ablations/02_routing_multihot_layer/arrays`.
L'obiettivo e' predire gli esperti del layer `l` sul token successivo,
`E_l(t+1)`, usando le informazioni osservate per lo stesso layer sul token
precedente, `E_l(t)`. Ogni esempio contiene 296 feature sparse float32:

- 256 feature `routing_multihot`: i sei esperti scelti dal router in
  `E_l(t)`, codificati come vettore multi-hot;
- 40 feature `layer`: l'identificativo del layer `l`, codificato one-hot.

Le label sono vettori multi-label di 256 elementi e rappresentano i sei
esperti target in `E_l(t+1)`. Il layer resta quindi invariato tra feature e
label; cambia soltanto il token. La metrica di selezione e' Recall@6 sul
validation set.

## Configurazione del tuning

- 15 trial RandomSearch;
- fino a 2 hidden layer;
- unita' per layer: da 128 a 512, con passo 128;
- attivazioni: ReLU o GELU;
- dropout: da 0.0 a 0.4, con passo 0.1;
- learning rate: campionato logaritmicamente tra 0.0001 e 0.003;
- batch size: 512;
- massimo 100 epoche per trial, con early stopping patience 5.

## Risultato migliore

La migliore configurazione trovata ha un solo hidden layer da 384 unita',
attivazione ReLU, dropout 0.3 e learning rate 0.0001642831. Ha ottenuto
Recall@6 validation pari a 0.4120.

## Contenuto

- `models/best_ffn.keras`: modello migliore selezionato dal tuning;
- `reports/tuning_report.json`: parametri, metrica e percorsi dell'esperimento;
- `tuner/`: stato e risultati dei singoli trial KerasTuner.
