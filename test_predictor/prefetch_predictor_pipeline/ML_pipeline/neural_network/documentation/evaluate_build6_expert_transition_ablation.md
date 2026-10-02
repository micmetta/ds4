# Ablazione post-addestramento dei rami della FFN Build 6

Lo script `evaluate_build6_expert_transition_ablation.py` misura quanto ogni ramo sia effettivamente usato da un modello gia' addestrato. Non esegue ottimizzazione, non modifica pesi e non sostituisce i test di feature importance KNN: risponde alla domanda diversa "quale informazione sfrutta questa specifica FFN?".

## Metodo

Per ogni split richiesto viene prima calcolata la Recall@6 del modello completo. Viene poi alterato un solo ramo per volta, mantenendo tutti gli altri input e le label originali.

- `zero`: sostituisce il ramo con zero. Nel caso `state_fixed_nnz`, conserva gli indici ma azzera i valori: la somma pesata dell'embedding e' quindi esattamente nulla.
- `permutation`: applica una permutazione riproducibile delle righe del solo ramo. La feature mantiene valori, forma e distribuzione empirica, ma viene associata a un altro esempio; viene quindi rimossa l'informazione predittiva condizionata all'esempio corrente.
- `drop_from_baseline`: `Recall@6 completo - Recall@6 ablato`. Un valore positivo segnala che la rimozione del ramo peggiora il modello; un valore vicino a zero indica che il modello quasi non lo usa o che l'informazione e' ridondante.

I rami valutabili sono state fixed-NNZ, raw hidden corrente e precedente, KV corrente e precedente, token position e `expert_transition_probabilities` 256D.

## Interpretazione della feature di transizione

Per il ramo `expert_transition_probabilities` la permutazione e' il test principale. Un forte calo in questa condizione indica che la rete usa davvero il vettore causale corretto $Q_{t,l}$, e non semplicemente la sua scala o la sua distribuzione media. Se invece la Recall rimane invariata, la rete lo ignora o l'informazione e' gia' ricostruita dagli altri rami.

L'ablazione post-addestramento non misura il potenziale teorico della feature: un ramo puo' essere informativo nel KNN e risultare poco usato dalla FFN se la sua fusione, lo spazio di ricerca o l'ottimizzazione non ne hanno favorito l'uso.

## Risultato su `expert_transition_bce_v1_best_long`

L'ablazione della candidata `candidate_10` continuata fino a early stopping ha
prodotto il report:

```text
build_dataset_6/ml/neural_network/expert_transition_bce_v1_best_long/
reports/neural_branch_ablation_validation_test.json
```

La Recall@6 baseline e' `0.531629` su validation e `0.535918` su test. Per il
ramo `expert_transition_probabilities`, il calo e' risultato praticamente
nullo:

| Split | Azzeramento `q` | Permutazione `q` |
| --- | ---: | ---: |
| Validation | `0.000099` | `0.000075` |
| Test | `0.000087` | `0.000087` |

La conclusione e' che questa specifica FFN non usa materialmente `q` nella
predizione, pur essendo il vettore informativo come baseline diretto e nel KNN.
La causa piu' plausibile e' la fusione per semplice concatenazione diretta, non
un'assenza di segnale nella feature.

### Rami ordinati per importanza effettiva nella candidata 10

La tabella ordina i rami per perdita su **test con permutazione**, che e' la
misura principale: conserva la distribuzione di ogni ramo ma ne rompe la
relazione con l'esempio corrente. Le colonne zero sono un controllo aggiuntivo.
Una perdita maggiore significa che il modello usa maggiormente quel ramo.

| Posizione | Ramo | Perdita test, permutazione | Perdita test, zero | Perdita validation, permutazione | Perdita validation, zero |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | State fixed-NNZ | `0.482305` | `0.380582` | `0.478298` | `0.380835` |
| 2 | Raw hidden layer `l` | `0.059645` | `0.040525` | `0.059372` | `0.039180` |
| 3 | Raw hidden layer `l-1` | `0.036432` | `0.023088` | `0.035706` | `0.022068` |
| 4 | KV layer `l` | `0.019626` | `0.013603` | `0.019120` | `0.013099` |
| 5 | KV layer `l-1` | `0.018592` | `0.012665` | `0.018426` | `0.012335` |
| 6 | Token position | `0.000117` | `0.000082` | `0.000233` | `0.000071` |
| 7 | `expert_transition_probabilities` 256D | `0.000087` | `0.000087` | `0.000075` | `0.000099` |

Lo state fixed-NNZ domina la predizione. I raw hidden aggiungono la seconda e
la terza quota informativa; i KV forniscono un contributo minore ma coerente.
Il vettore `q` e token position sono invece indistinguibili da un ramo ignorato
in questa specifica architettura.

## Prossimo esperimento: fusione residua gated delle transition probabilities

La concatenazione di `q_{t,l}` al vettore di fusione consente alla Dense
successiva di ignorarlo, come confermato dall'ablazione. Il prossimo esperimento
non cambiera' la definizione causale della feature, ma il punto in cui essa
interviene nella rete. L'idea e' sommare il suo contributo direttamente ai 256
logit finali, prima della sigmoid:

$$
z_{\mathrm{finale}} = z_{\mathrm{FFN}} + g(x)\,\alpha\,
\widetilde{\log\bigl(q_{t,l}+\varepsilon\bigr)}.
$$

- $z_{\mathrm{FFN}}$: i 256 logit prodotti dai rami storici della FFN;
- $q_{t,l}$: vettore causale di 256 probabilita' di transizione;
- $\varepsilon>0$: piccola costante numerica che rende il logaritmo sempre
  definito;
- $\log(q+\varepsilon)$: espande le differenze tra le probabilita' piccole,
  che nel vettore originale hanno media circa $1/256$;
- $\widetilde{\cdot}$: normalizzazione applicata **solo al ramo q**, non a
  tutti i rami concatenati;
- $\alpha$: intensita' scalare appresa del contributo della feature;
- $g(x)\in[0,1]$: gate appreso dal contesto $x$, che puo' attenuare q negli
  esempi in cui la transizione e' poco affidabile.

### Architettura precedente: concatenazione diretta che non ha sfruttato `q`

La migliore candidata trovata dal tuner con la nuova feature (`candidate_10`)
introduceva `q` soltanto come altre 256 componenti del vettore concatenato:

```text
state fixed-NNZ -> 2048D -> ReLU -> Dropout(0.3) -------------------+
raw hidden l -> Dense(64) -> ReLU ----------------------------------+
raw hidden l-1 -> Dense(64) -> ReLU --------------------------------+
KV l -> Dense(256) -> ReLU -----------------------------------------+--> Concatenate (2945D)
KV l-1 -> Dense(256) -> ReLU ---------------------------------------+          |
token position 1D ---------------------------------------------------+          v
q_{t,l} 256D diretto ------------------------------------------------+   Dense(2048) -> ReLU -> Dropout(0.3)
                                                                                 |
                                                                                 v
                                                                      Dense(256) -> sigmoid
```

L'ablazione ha mostrato che azzerare o permutare le 256 componenti di `q` non
cambia praticamente Recall@6. La Dense di fusione puo' quindi aver assegnato
pesi trascurabili a quelle componenti, pur mantenendo le rappresentazioni degli
altri rami.

Lo schema previsto e':

```text
rami storici --> FFN di fusione --> Dense(256) ------------------> z_FFN --+
                                                                         |
q 256D --> log(q + epsilon) --> normalizzazione solo q --> alpha --> x gate --+--> z_finale --> sigmoid
```

### Significato passo-passo del ramo `q`

- $q_{t,l}\in\mathbb{R}^{256}$: e' il vettore
  `expert_transition_probabilities` dell'esempio, cioe'
  $q_{t,l}=[q_{t,l}(0),\ldots,q_{t,l}(255)]$. Ogni componente rappresenta la
  plausibilita' dell'expert corrispondente al token successivo, ricavata dagli
  expert osservati al token corrente nello stesso layer.
- $\log(q_{t,l}+\varepsilon)$: il logaritmo mette in evidenza differenze
  relative tra probabilita' piccole. $\varepsilon>0$ e' una piccola
  costante positiva che evita un logaritmo non definito nel caso limite di
  probabilita' nulla. Poiche' ogni componente di $q_{t,l}$ e' una probabilita'
  compresa tra zero e uno, prima della normalizzazione il suo logaritmo e'
  normalmente non positivo: per esempio $\log(0.5)\approx-0.693$,
  $\log(0.01)\approx-4.605$ e $\log(1/256)\approx-5.545$. Un valore meno
  negativo corrisponde quindi a un expert piu' probabile.
- $\operatorname{LN}(\log(q_{t,l}+\varepsilon))$: una `LayerNormalization`
  viene applicata soltanto alle 256 componenti trasformate di $q_{t,l}$,
  separatamente per ogni esempio. Dopo questa normalizzazione i valori hanno
  media circa zero e possono quindi essere positivi o negativi: il residuale
  puo' aumentare alcuni logit e diminuire altri in base alla probabilita'
  relativa degli expert. In particolare, poiche' il logaritmo e' crescente,
  gli expert con probabilita' maggiore tendono ad avere un valore logaritmico
  sopra la media delle 256 componenti e quindi un valore normalizzato positivo:

  $$
  \operatorname{LN}(v_i)=
  \frac{v_i-\mu_v}{\sqrt{\sigma_v^2+\delta}},
  \qquad v_i=\log(q_{t,l}(i)+\varepsilon).
  $$

  Non e' tuttavia una garanzia rigida che i primi sei expert abbiano sempre
  segno positivo: il segno dipende dalla distribuzione completa di $q_{t,l}$
  per quello specifico esempio. Inoltre `LayerNormalization` ha scala e bias
  allenabili, che consentono alla rete di adattare ulteriormente il ramo. Non
  normalizza e non modifica i rami state, hidden, KV o token position.
- $\alpha$: e' uno scalare allenabile, inizializzato a $0.1$, che regola
  globalmente l'intensita' del contributo residuale di $q_{t,l}$ sui 256
  logit.
- $g(x)$: e' il gate, uno scalare tra zero e uno appreso
  dal contesto $x$ dei rami storici. Per ciascun esempio decide quanto la
  rete debba fidarsi della transizione: vicino a zero la attenua, vicino a uno
  la conserva.
- $z_{\mathrm{finale}}$: il residuale 256D
  $\alpha\,g(x)\,\operatorname{LN}(\log(q_{t,l}+\varepsilon))$ viene
  sommato componente per componente ai 256 logit $z_{\mathrm{FFN}}$. La
  componente $i$ di $q_{t,l}$ influenza direttamente il logit dell'expert $i$,
  senza dover attraversare la Dense
  di fusione.
- $\sigma(z_{\mathrm{finale}})$: la sigmoid converte i logit finali in 256
  score tra zero e uno. La Top-6 e Recall@6 sono calcolate su questi score;
  essendo monotona, la sigmoid non cambia l'ordinamento dei logit.

La forma completa e':

$$
z_{\mathrm{finale}} = z_{\mathrm{FFN}} +
\alpha\,g(x)\,\operatorname{LN}(\log(q+\varepsilon)).
$$

La somma residuale rende il contributo di `q` visibile direttamente nella
classifica degli expert: la rete non deve prima conservarlo dentro una grande
Dense di fusione. Il gate mantiene comunque la flessibilita' di non usarlo
quando non e' utile. Il test va addestrato separatamente, mantenendo BCE
classica, per confrontarlo in modo pulito con `candidate_10`.

## Esecuzione locale

Il test e' sola inferenza. Si esegue quindi localmente sulla GPU disponibile, senza usare Compute Units Colab. Per valutare la candidata 10 BCE con transition probabilities:

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/evaluate_build6_expert_transition_ablation.py \
  --tuning-report test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/expert_transition_bce_v1/reports/tuning_report.json \
  --candidate-id candidate_10 \
  --checkpoint test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/expert_transition_bce_v1_best_long/models/ffn.keras \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/training/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/02_current_previous_fixed_nnz_state \
  --current-quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/05_current_hidden_int8 \
  --previous-quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/08_previous_hidden_int8 \
  --current-kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/09_current_kv_f16 \
  --previous-kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/10_previous_kv_f16 \
  --token-position-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/token_position_inputs \
  --expert-transition-probabilities-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/transition_features/expert_transition_probabilities_v1 \
  --output-report test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/expert_transition_bce_v1_best_long/reports/neural_branch_ablation_validation_test.json \
  --splits validation,test \
  --modes zero,permutation \
  --batch-size 4096
```

Il checkpoint deve corrispondere esattamente all'architettura indicata nel report. Per una variante con Batch Normalization dopo la concatenazione, aggiungere `--fusion-batch-normalization` e usare il suo checkpoint.
Per il checkpoint della fusione residuale gated, aggiungere
`--expert-transition-gated-residual`: il flag ricostruisce il percorso di
$q_{t,l}$ separato dalla concatenazione prima di caricare i pesi.

## Risultato della fusione residuale gated

L'esperimento gated residual, addestrato da zero con BCE classica, ha prodotto
il checkpoint e il report:

```text
build_dataset_6/ml/neural_network/expert_transition_gated_residual_bce_v1/
models/ffn.keras
reports/neural_branch_ablation_validation_test.json
```

La Recall@6 baseline e' `0.531674` su validation e `0.535893` su test. Questi
valori sono sostanzialmente equivalenti alla variante con concatenazione diretta
di $q_{t,l}$ (`0.531629` validation, `0.535918` test): il nuovo meccanismo non
ha ancora migliorato la metrica finale, ma ha cambiato in modo netto quali
informazioni il modello usa.

| Posizione | Ramo | Perdita test, permutazione | Perdita test, zero | Perdita validation, permutazione | Perdita validation, zero |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | State fixed-NNZ | `0.334739` | `0.202996` | `0.333080` | `0.204488` |
| 2 | `expert_transition_probabilities` 256D | `0.091052` | `0.022844` | `0.091884` | `0.023484` |
| 3 | Raw hidden layer $l$ | `0.056807` | `0.043122` | `0.056158` | `0.042606` |
| 4 | Raw hidden layer $l-1$ | `0.031829` | `0.019946` | `0.030509` | `0.019159` |
| 5 | KV layer $l$ | `0.018670` | `0.013027` | `0.018102` | `0.012910` |
| 6 | KV layer $l-1$ | `0.017948` | `0.013427` | `0.017201` | `0.012510` |
| 7 | Token position | `0.000066` | `0.000017` | `0.000228` | `0.000118` |

Il risultato centrale e' la perdita per permutazione di $q_{t,l}$: circa
`0.091` su test e validation. Nella concatenazione diretta era circa
`0.000087`; ora e' il secondo ramo per importanza effettiva, dopo lo state
fixed-NNZ e prima degli hidden layer. La permutazione e' molto piu' dannosa
dell'azzeramento ($0.091052$ contro $0.022844$ su test): non basta quindi la
scala media del residuale, ma conta l'allineamento del vettore $q_{t,l}$ con il
singolo esempio. La fusione residuale gated forza pertanto la rete a usare il
segnale causale corretto.

Il fatto che la Recall@6 finale non aumenti non rende inutile il risultato:
separa due questioni diverse. La prima, ora risolta, era far entrare $q_{t,l}$
nella decisione; la seconda e' calibrare quanto e in quali casi il suo segnale
debba correggere la FFN storica. Un semplice aumento della larghezza della
Dense di concatenazione non e' il passo prioritario, poiche' il problema della
concatenazione era l'ignorare il ramo, non la mancanza di capacita' del modello.

## Diagnostica di $\alpha$, gate e contributo residuale

L'ablazione mostra che il residuale usa $q_{t,l}$, ma non spiega perche' questo
non incrementi ancora Recall@6. Per scegliere eventuali modifiche
architetturali, questa diagnostica va eseguita solo su train e validation: il
test resta sigillato e verra' usato una sola volta per la valutazione finale
della configurazione scelta. Lo script locale
`inspect_build6_expert_transition_gated_residual.py` misura, senza modificare
il checkpoint, quattro aspetti complementari:

- $\alpha$: lo scalare allenabile finale del residuale; se e' vicino a zero,
  il modello ha attenuato globalmente $q_{t,l}$; se e' grande, il residuale
  puo' invece dominare i logit storici.
- $g(x)$: media, deviazione standard, quantili e quota di esempi con gate
  minore di $0.1$ o maggiore di $0.9$. Indica se l'uso di $q_{t,l}$ e'
  selettivo oppure quasi costante.
- ampiezza del residuale: media di $|r|$, media di $|z_{\mathrm{FFN}}|$ e loro
  rapporto, dove $r=\alpha g(x)\operatorname{LN}(\log(q+\varepsilon))$.
  Serve a capire se il ramo e' numericamente troppo debole o troppo invasivo.
- effetto sulla graduatoria: Recall@6 dei soli logit base e dei logit finali,
  quota di esempi la cui Top-6 cambia e numero medio di expert in comune tra le
  due Top-6. Mostra se $q_{t,l}$ sta correggendo pochi casi mirati oppure
  riscrivendo la graduatoria in modo eccessivo.

Il comando da eseguire localmente e':

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/inspect_build6_expert_transition_gated_residual.py \
  --tuning-report test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/expert_transition_bce_v1/reports/tuning_report.json \
  --candidate-id candidate_10 \
  --checkpoint test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/expert_transition_gated_residual_bce_v1/models/ffn.keras \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/training/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/02_current_previous_fixed_nnz_state \
  --current-quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/05_current_hidden_int8 \
  --previous-quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/08_previous_hidden_int8 \
  --current-kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/09_current_kv_f16 \
  --previous-kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/ablations/10_previous_kv_f16 \
  --token-position-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/token_position_inputs \
  --expert-transition-probabilities-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/transition_features/expert_transition_probabilities_v1 \
  --output-report test_predictor/prefetch_predictor_pipeline/build_dataset_6/ml/neural_network/expert_transition_gated_residual_bce_v1/reports/gated_residual_diagnostics_validation_test.json \
  --splits train,validation \
  --batch-size 4096
```

### Risultato della diagnostica train/validation

Il report prodotto e' il seguente:

```text
build_dataset_6/ml/neural_network/expert_transition_gated_residual_bce_v1/
reports/gated_residual_diagnostics_train_validation.json
```

| Misura | Train | Validation |
| --- | ---: | ---: |
| Recall@6 dei logit finali gated | `0.634723` | `0.531674` |
| Recall@6 dei soli logit base $z_{\mathrm{FFN}}$ | `0.640308` | `0.508262` |
| Recall@6 di $q_{t,l}$ da solo | `0.363000` | `0.365505` |
| Gate medio $g(x)$ | `0.841228` | `0.841652` |
| Mediana del gate | `0.863279` | `0.861883` |
| Quota di gate $>0.9$ | `34.78\%` | `33.80\%` |
| Quota di gate $<0.1$ | `0.00\%` | `0.00\%` |
| Media $|r|/|z_{\mathrm{FFN}}|$ | `0.145355` | `0.145995` |
| Esempi con Top-6 modificata dal residuale | `68.58\%` | `71.66\%` |
| Expert comuni tra Top-6 base e finale | `5.104/6` | `5.040/6` |

Lo scalare appreso e' $\alpha=0.904925$, molto maggiore del valore iniziale
$0.1$. Il residuale non e' quindi stato attenuato: in valore assoluto vale in
media circa il $14.6\%$ del logit base e modifica almeno un expert della Top-6
in circa sette esempi su dieci. Il gate non e' selettivo: e' mediamente alto,
non scende mai sotto $0.1$ e supera $0.9$ in circa un terzo degli esempi sia su
train sia su validation.

All'interno di **questo stesso checkpoint**, il residuale peggiora la Recall@6
train di `0.005585` ma migliora quella validation di `0.023411`. Questo mostra
che $q_{t,l}$ fornisce una correzione sostanziale e con miglior comportamento
fuori dal train; non e' corretto dedurne direttamente che peggiori la
generalizzazione. I soli logit base non sono infatti un modello allenato
indipendentemente: sono stati co-adattati durante il training insieme al
residuale finale.

La Recall@6 finale resta tuttavia sotto al miglior modello storico senza la
feature di transizione (`0.536904` validation). Il confronto non e' ancora
perfettamente controllato, perche' la candidata 10 del tuner transition usa
rami diversi dalla miglior architettura storica: hidden corrente `64D`, hidden
precedente `64D`, KV corrente `256D`, KV precedente diretto, mentre il migliore
storico aveva hidden corrente `512D`, hidden precedente `64D`, KV corrente
diretto e KV precedente `128D`. Il prossimo confronto informativo deve quindi
inserire il residuale gated nella **stessa** architettura e negli stessi
iperparametri del migliore storico, mantenendo BCE e l'early stopping su
validation.

## Piano del prossimo confronto controllato (non ancora implementato)

Il prossimo esperimento non deve aumentare la larghezza della concatenazione:
la diagnostica ha gia' mostrato che il residuale gated permette a $q_{t,l}$ di
entrare nella decisione. Occorre invece isolare il suo effetto rispetto al
miglior backbone disponibile.

- **Baseline di confronto:** architettura storica migliore senza transition
  probabilities, con state fixed-NNZ `2048D`, hidden corrente `512D`, hidden
  precedente `64D`, KV corrente diretto, KV precedente `128D`, fusione `2048D`
  e dropout `0.3`.
- **Nuova variante controllata:** identica baseline, con l'unica aggiunta del
  residuale gated $q_{t,l}$ sui 256 logit finali. Il ramo $q$ non deve entrare
  nella concatenazione.
- **Training:** nuova inizializzazione per entrambe le reti, BCE classica,
  stesso seed e stessa procedura di macro-block shuffle; early stopping
  monitorato su `val_loss` con patience `5`.
- **Selezione:** tutte le decisioni architetturali e di iperparametri devono
  usare soltanto il validation set. Il test rimane sigillato fino a quando una
  configurazione non viene fissata.
- **Valutazione dopo il training:** confrontare Recall@6 validation e, per la
  variante gated, ripetere l'ablazione/permutazione di $q_{t,l}$ per verificare
  che il ramo sia effettivamente usato anche sul backbone migliore.

Questo piano e' soltanto documentato: non sono stati ancora creati script,
wrapper o comandi di training per la nuova variante.
