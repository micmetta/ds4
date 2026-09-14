# State fixed-NNZ: input contiguo equivalente allo state sparse

## Perche' esiste questo percorso

La FFN originale riceve lo state come matrice CSR sparse. A ogni batch Python estrae una slice CSR e la converte in `tf.SparseTensor`. Il profiler TensorBoard del build 5 ha mostrato che questa fase e' un collo di bottiglia dell'input pipeline.

Questo percorso e' **aggiuntivo**: non modifica ne' cancella `X_train.npz`, `X_validation.npz`, `X_test.npz`, ne' lo script di training sparse originale. Si puo' sempre continuare a usare:

```text
train_quantized_hidden_token_embedding_kv_ffn.py
```

## Formato materializzato

Nel build 5 ogni riga CSR state ha esattamente 44 feature non nulle. Per ciascun esempio vengono quindi salvati due array contigui:

- `state_feature_indices_<split>.npy`: shape `(N, 45)`, `uint16`;
- `state_feature_values_<split>.npy`: shape `(N, 45)`, `float32`.

Le prime 44 posizioni sono gli indici e i valori originali della riga CSR, nello stesso ordine. La posizione 45 e' una sentinella: indice `4413` e valore `1.0`.

Questa sentinella rappresenta il bias dell'originario `Dense(512)`. La rete usa una tabella trainabile di pesi `(4414, 512)` e calcola:

```text
state_linear = sum_j(state_feature_values[j] * W[state_feature_indices[j]])
state_hidden = ReLU(state_linear)
```

L'ultima riga di `W` e' quindi il bias. Senza regolarizzazione L2 (la configurazione corrente usa `0.0`) questa e' matematicamente equivalente a:

```text
Dense(512)(state_CSR) = state_CSR @ kernel + bias
```

Il numero di parametri resta identico: `4413 × 512 + 512 = 4414 × 512 = 2.259.968`.

## Vantaggio e limiti

Il formato aggiunge circa 0,5 GiB per tutti gli split, non circa 17 GiB richiesti da una densificazione float16 di 4.413 dimensioni. Conserva le 44 feature non nulle e quindi non trasforma la prima branca in un calcolo denso molto piu' costoso.

Non rende automaticamente identiche al bit le run precedenti: ordine delle somme floating-point e inizializzazione della tabella embedding possono causare piccole differenze numeriche. La semantica delle feature e la capacita' del layer lineare sono invece le stesse. Per questo le metriche devono essere confrontate con una run separata, con nuovo output directory.

## Materializzazione

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/pipeline/state_features/materialize_fixed_nnz_state_inputs.py \
  --state-arrays-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --chunk-rows 65536
```

Il comando legge soltanto le matrici CSR gia' esistenti e scrive un nuovo output. Verifica che tutti gli split abbiano lo stesso numero di feature e lo stesso numero di feature non nulle per esempio.

## Training dedicato

Usare `train_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py`, non il trainer sparse originale. Gli altri input e l'architettura restano invariati:

- hidden raw 4096D `int8`, dequantizzato e proiettato a 64D;
- embedding token 4096D congelato, proiettato a 64D;
- `token_position` standardizzata 1D;
- 5 statistiche KV nello state;
- KV corrente completo 512D diretto, normalizzato per layer train-only.

Prima della run completa conviene usare `--profile-only-batches` per verificare che `FlatMap::Generator` non sia piu' dominato dalla conversione CSR -> `SparseTensor`.

## Risultato del profiling build 5

Con batch 4096 e 8 batch profilati, il trainer sparse originale impiegava circa 52,2 ms per chiamata del `Generator` (261 ms in 5 chiamate); il trainer fixed-NNZ impiega circa 3,6 ms (29 ms in 8 chiamate). La quota del tempo campionato attribuita a I/O/Python e' scesa dal 35,5% al 9,4%.

Il device step medio e' salito da 9,0 a 19,4 ms: e' atteso, perche' il calcolo della somma pesata dei 45 slot viene ora eseguito dalla GPU. Il risultato importante e' che il preprocessing Python per la conversione CSR -> `SparseTensor` non domina piu' la pipeline. Occorre comunque misurare alcune epoche complete per valutare il tempo reale e la Recall@6.

Il benchmark build 5 di 30 epoche a batch 4096 ha confermato il vantaggio sul tempo reale: 1.676 s con fixed-NNZ contro 5.717 s con il trainer CSR sparse (3,41 volte piu' veloce; -70,7%). Alla trentesima epoca il `val_recall_at_6` era 0,4594 contro 0,4583 della corrispondente run sparse. Le inizializzazioni non sono identiche al bit, quindi il confronto della metrica va interpretato come coerenza della curva, non come una dimostrazione causale di miglioramento della qualita'.
