# Valutazione FFN sul test set

## Obiettivo

`evaluate_ffn.py` carica un modello FFN Keras gia' addestrato e calcola
Recall@6 sul solo split `test`. Non esegue addestramento, non modifica il
modello e non usa train o validation.

Va eseguito una sola volta dopo aver bloccato architettura e iperparametri,
per mantenere il test set come valutazione finale indipendente.

## Input

| Opzione | Significato |
|---|---|
| `--data-dir` | Cartella `arrays/` contenente `X_test.npz` oppure `X_classic_test.npz` e `y_test.npy` |
| `--model-path` | File `.keras` del modello da valutare |
| `--report-path` | Nuovo file JSON in cui salvare il risultato della valutazione |
| `--batch-size` | Esempi per mini-batch durante l'inferenza; default `512` |

Il numero di feature del modello deve coincidere con quello del test set.

## Output

Il report JSON conserva dataset, modello, dimensioni del test set, batch size
e Recall@6 test. Lo script rifiuta un `--report-path` gia' esistente per non
sovrascrivere una valutazione finale.

## Migliore candidato Build 5 attuale

Il migliore risultato di **validation** disponibile al momento e'
`Recall@6 = 0,502358287`. Proviene dal warm-start del trial Hyperband `0024`
ed e' salvato in:

```text
build_dataset_5/ml/neural_network/
retrain_best_hyperband_trial_0024_EP1000_PAT5/models/ffn.keras
```

Il modello e' stato valutato una sola volta sul test set Build 5 da 209.400
esempi e ha ottenuto **Recall@6 test `0,506056192`**. Il report e'
`retrain_best_hyperband_trial_0024_EP1000_PAT5/reports/test_evaluation.json`.
Il precedente checkpoint Hyperband `0024`, che aveva validation Recall@6
0,490852605, aveva invece ottenuto Recall@6 test 0,494631. Il nuovo modello e'
quindi il migliore risultato Build 5 sia su validation sia su test.

## Esecuzione

Dalla cartella `ML_pipeline/`:

```bash
uv run neural_network/evaluate_ffn.py \
  --data-dir ../build_dataset_2/ml/ablations/02_routing_multihot_layer/arrays \
  --model-path ../build_dataset_2/ml/neural_network/ffn_routing_multihot_layer_HL384_DO03_LR00016428_EP500/models/ffn.keras \
  --report-path ../build_dataset_2/ml/neural_network/ffn_routing_multihot_layer_HL384_DO03_LR00016428_EP500/reports/test_report.json \
  --batch-size 512
```
