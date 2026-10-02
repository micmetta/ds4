# Importanza rapida delle feature Build 5 con KNN

`run_build5_knn_feature_importance.py` fornisce una classifica rapida del
**segnale predittivo** dei gruppi di feature della Build 5. Non interpreta i
pesi del modello neurale migliore e non sostituisce una futura analisi per
permutazione del checkpoint neurale.

## Metodo

1. Estrae un sottoinsieme riproducibile di train e validation.
2. Costruisce la stessa famiglia di input disponibile nella Build 5.
3. Addestra un KNN completo.
4. Riaddestra lo stesso KNN rimuovendo un solo gruppo per volta.
5. Ordina i gruppi dalla maggiore alla minore perdita di Recall@6 validation:
   `Recall@6 completo - Recall@6 senza gruppo`.

Una perdita positiva elevata indica che il gruppo contiene segnale utile per
questo KNN. Un valore vicino a zero indica ridondanza con gli altri gruppi o
assenza di segnale sfruttabile dal KNN; non prova che il gruppo sia inutile per
la FFN.

## Gruppi confrontati

Lo state classico 4.413D e' separato in: routing history multi-hot (256D),
layer, ID dei 16 router logits, valori dei 16 router logits e cinque
statistiche KV. Sono poi confrontati raw hidden 4.096D, token embedding
congelato 4.096D, KV corrente 512D e token position.

Per tenere il KNN veloce, raw hidden, embedding e KV completo vengono ridotti
con una proiezione gaussiana fissa a 32 dimensioni, identica e riproducibile
per tutte le ablazioni. Gli array originali non vengono modificati.

Per eseguire una variante senza riduzione, usare `--projection-dimensions 0`.
In quel caso il KNN riceve tutte le 4.096 dimensioni di raw hidden, tutte le
4.096 dell'embedding e tutte le 512 del KV. Le dimensioni non vengono
proiettate né eliminate; ricevono soltanto una standardizzazione calcolata sul
train, indispensabile per evitare che la distanza KNN sia dominata dalla scala
numerica di un singolo ramo. Questa variante richiede molta piu' RAM e tempo.

## Comando consigliato

Il default usa 5.000 esempi train e 5.000 validation: e' deliberatamente un
primo screening rapido. Se il tempo e' accettabile, aumentare entrambi i
limiti mantenendo gli stessi valori tra esperimenti confrontabili.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/run_build5_knn_feature_importance.py
```

Il report ordinato viene scritto in:

```text
build_dataset_5/ml/ablations/build5_knn_feature_importance/reports/
build5_knn_feature_importance.json
```

Il test set non viene usato: la graduatoria resta selezionata esclusivamente
sul validation set.

## Risultato della prima esecuzione Build 5

La prima esecuzione ha usato i valori predefiniti: 5.000 esempi train, 5.000
esempi validation, KNN con cinque vicini pesati per distanza e proiezione fissa
a 32 dimensioni per raw hidden, token embedding e KV completo.

Il KNN completo ha ottenuto Recall@6 validation `0.095533`. La tabella mostra
la perdita di Recall@6 ottenuta riaddestrando il KNN senza un gruppo alla
volta. Un valore positivo alto indica che, per questo KNN, il gruppo contiene
segnale predittivo utile.

| Posizione | Gruppo rimosso | Recall@6 senza gruppo | Perdita Recall@6 |
|---:|---|---:|---:|
| 1 | KV completo normalizzato 512D | 0.083400 | +0.012133 |
| 2 | Routing history multi-hot | 0.083967 | +0.011567 |
| 3 | Cinque statistiche KV nello state | 0.084000 | +0.011533 |
| 4 | Layer | 0.089600 | +0.005933 |
| 5 | ID dei router logits | 0.092700 | +0.002833 |
| 6 | Valori dei router logits | 0.093500 | +0.002033 |
| 7 | Token position | 0.094067 | +0.001467 |
| 8 | Raw hidden 4.096D proiettato | 0.095267 | +0.000267 |
| 9 | Token embedding congelato 4.096D proiettato | 0.111533 | -0.016000 |

La prima indicazione e' che il KNN ricava il segnale piu' immediato dal KV
completo, dalla storia di routing e dalle statistiche KV gia' incluse nello
state. Layer e stato del router aggiungono un contributo minore ma positivo.

### Limiti importanti dell'interpretazione

Questa non e' una classifica dell'importanza dei pesi della FFN migliore. Il
KNN completo raggiunge solo `0.095533`, molto meno del modello neurale migliore
Build 5 (Recall@6 validation `0.502358287`). La FFN puo' quindi sfruttare
interazioni non lineari fra feature che il KNN basato sulla distanza non sa
rappresentare.

Il valore negativo del token embedding non dimostra che l'embedding sia nocivo
per la FFN: significa soltanto che, in questo KNN e con questa proiezione
casuale 32D, rimuoverlo migliora la geometria dei vicini. Analogamente, il
contributo quasi nullo del raw hidden per il KNN non prova che sia inutile alla
rete neurale. I risultati sono uno screening rapido dei gruppi con segnale
direttamente sfruttabile dal KNN, non una graduatoria definitiva della rete.

## Variante senza riduzione delle feature dense

Una seconda esecuzione ha usato gli stessi 5.000 esempi train e 5.000
validation, ma con `--projection-dimensions 0`: raw hidden ed embedding sono
rimasti a 4.096 dimensioni, mentre il KV completo e' rimasto a 512 dimensioni.
Il report corrispondente e' in
`ablations/build5_knn_feature_importance_full_dim/reports/`.

Il KNN completo ha ottenuto Recall@6 validation `0.076800`, inferiore al
risultato `0.095533` della variante con proiezione 32D. La tabella seguente
riporta la graduatoria leave-one-group-out della variante senza riduzione.

| Posizione | Gruppo rimosso | Recall@6 senza gruppo | Perdita Recall@6 |
|---:|---|---:|---:|
| 1 | Raw hidden 4.096D completo | 0.063633 | +0.013167 |
| 2 | KV completo normalizzato 512D | 0.064867 | +0.011933 |
| 3 | Cinque statistiche KV nello state | 0.076067 | +0.000733 |
| 4 | Valori dei router logits | 0.076200 | +0.000600 |
| 5 | Routing history multi-hot | 0.076567 | +0.000233 |
| 6 | Token position | 0.076567 | +0.000233 |
| 7 | ID dei router logits | 0.076700 | +0.000100 |
| 8 | Layer | 0.076733 | +0.000067 |
| 9 | Token embedding congelato 4.096D completo | 0.121600 | -0.044800 |

### Cosa indica il confronto

- Il KV completo e' il segnale piu' robusto: la sua ablazione causa una perdita
  vicina a `0.012` sia con proiezione 32D sia senza riduzione.
- Il raw hidden contiene segnale utile per il KNN quando conserva tutte le
  dimensioni (`+0.013167`), mentre la proiezione casuale 32D ne aveva perso
  quasi tutto (`+0.000267`).
- L'embedding congelato peggiora fortemente la geometria del KNN senza
  riduzione: rimuoverlo fa salire Recall@6 da `0.076800` a `0.121600`.

Il peggioramento del KNN completo nella variante 13.000D circa non significa
che le feature aggiuntive siano inutili. Con soli 5.000 esempi train, le
distanze in uno spazio cosi' grande diventano meno discriminanti: e' la
cosiddetta *maledizione della dimensionalita'*. L'embedding 4.096D puo' anche
pesare molto nella distanza pur non definendo vicini semanticamente utili.

La FFN migliore non ha questa limitazione: apprende una proiezione
`Dense(64) -> ReLU` per l'embedding e una `Dense(512) -> ReLU` per raw hidden.
Per questo il risultato KNN non autorizza a rimuovere l'embedding dalla rete
neurale. Un possibile prossimo screening KNN e' normalizzare ciascun gruppo
come blocco, cosi' che il suo peso nella distanza non dipenda automaticamente
dal numero delle sue dimensioni.
