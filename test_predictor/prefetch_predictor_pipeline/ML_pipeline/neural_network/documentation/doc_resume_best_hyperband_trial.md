# Proseguire il miglior trial Hyperband della Build 5

## Obiettivo

Il trial `0024` del tuning Build 5 ha raggiunto `validation_recall_at_6 =
0.49085259437561035` all'epoca 20. La sua configurazione e':

## Architettura della rete migliore

Lo state ha semanticamente 4.413 feature, ma entra nella rete nel formato
fixed-NNZ compatto di 45 coppie indice/valore per esempio: 44 feature attive
piu' la sentinella del bias. La prima riga del diagramma rappresenta quindi il
`Dense(512)` equivalente sullo state completo, calcolato senza materializzare
i valori zero.

```text
state 4.413D
  (routing history, layer, router ID/score, 5 statistiche KV)
        |
        v
Dense(512) -> ReLU -> Dropout(0.3) ------------------+
                                                       |
hidden raw 4.096D -> Dense(512) -> ReLU -------------+
                                                       |
token embedding congelato 4.096D -> Dense(64) -> ReLU +
                                                       +--> Concatenate(1.601D)
KV completo normalizzato 512D -> identita' ----------+          |
                                                       |          v
token_position standardizzata 1D ---------------------+   Dense(512) -> ReLU
                                                                  |
                                                                  v
                                                           Dropout(0.3)
                                                                  |
                                                                  v
                                                           Dense(256) -> sigmoid
                                                              (expert_scores)
```

Le 1.601 componenti concatenate sono `512 + 512 + 64 + 512 + 1`. Il
`Dense(512)` dopo la concatenazione e' il layer di fusione selezionato dal
trial migliore; non era presente nel diagramma piÃ¹ piccolo da 1.153D.

| Componente | Valore |
|---|---:|
| Ramo state fixed-NNZ | 512 neuroni, ReLU, dropout 0,3 |
| Proiezione raw hidden | 4.096 -> 512, ReLU |
| Proiezione token embedding congelato | 4.096 -> 64, ReLU |
| Ramo Key-Value | 512 valori diretti, senza proiezione |
| Fusione | Dense(512), ReLU, dropout 0,3 |
| Classificatore | Dense(256), sigmoid |
| Learning rate Adam | 0,00015 |

Il checkpoint e':

```text
.../hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1/
trial_checkpoints/0024/best.weights.h5
```

## Cosa significa riprendere il training

Il file `best.weights.h5` salva i pesi della rete, non lo stato interno di
Adam (momenti e varianze accumulati). La ripresa e' quindi un **warm-start**:

1. la nuova rete viene ricostruita con la stessa architettura;
2. vengono caricati esattamente i pesi che avevano ottenuto Recall@6 0,49085;
3. Adam viene creato di nuovo con learning rate 0,00015;
4. l'addestramento continua sulle epoche numerate 21..1000, al massimo 980
   epoche aggiuntive;
5. l'early stopping con pazienza 5 ferma il run se la nuova traiettoria non
   migliora per cinque epoche consecutive.

Questo non e' un proseguimento bit-per-bit dello stesso processo Adam, ma e'
il modo corretto e riproducibile per continuare da un checkpoint di soli pesi.

Lo script misura anche il Recall@6 iniziale sul validation set. Salva e
ripristina il migliore fra il checkpoint iniziale e tutte le epoche nuove:
quindi il risultato finale non puo' essere peggiore di 0,49085 solo perche'
una successiva epoca e' sfavorevole.

## Opzioni introdotte nel trainer

| Opzione | Uso in questo esperimento |
|---|---|
| `--initial-weights` | Percorso del checkpoint Hyperband `0024/best.weights.h5`. |
| `--initial-epoch 20` | Segnala che il checkpoint deriva dalla fine dell'epoca 20. |
| `--epochs 1000` | Limite totale: vengono eseguite le epoche 21..1000. |
| `--early-stopping-patience 5` | Arresta dopo cinque epoche nuove senza miglioramento. |

Nella nuova directory di output vengono salvati `models/best.weights.h5`,
`models/ffn.keras` e `reports/training_report.json`. Il report registra sia
il checkpoint di origine sia il Recall@6 misurato prima della ripresa.

## Risultato della ripresa eseguita

La ripresa dal checkpoint `0024` e' stata eseguita il 14 settembre 2026 con
la configurazione riportata sopra. L'early stopping ha concluso l'esperimento
dopo 36 epoche aggiuntive, cioe' dopo l'epoca numerata 56. Il massimo e' stato
ottenuto all'epoca 51:

| Misura | Valore |
|---|---:|
| Recall@6 validation del checkpoint iniziale | 0,490852605 |
| Migliore Recall@6 validation dopo la ripresa | **0,502358287** |
| Miglioramento assoluto | **+0,011505682** |
| Miglioramento relativo | **+2,34%** |
| Tempo complessivo della ripresa | 2.397,59 s, circa 40 minuti |
| Recall@6 sul test set, valutato una sola volta | **0,506056192** |

I pesi migliori e il modello Keras finale sono rispettivamente:

```text
.../retrain_best_hyperband_trial_0024_EP1000_PAT5/models/best.weights.h5
.../retrain_best_hyperband_trial_0024_EP1000_PAT5/models/ffn.keras
```

Il checkpoint originale Hyperband del trial `0024` e' rimasto invariato.

Il precedente checkpoint `0024` aveva ottenuto Recall@6 test `0,494631`.
La ripresa lunga migliora quindi anche sul test di circa `+0,011425` punti di
Recall@6. Il modello della ripresa e' il **migliore risultato Build 5 ottenuto
finora**, sia su validation sia sul test set.
