# Hyperband Build 5 — tuning della FFN fixed-NNZ

## Scopo

Lo script `neural_network/tune_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py`
cerca configurazioni della rete Build 5 senza modificare dataset, artefatti o
run esistenti. Usa train e validation; il test set resta riservato alla
valutazione finale.

Usa **Hyperband** con `val_recall_at_6` come obiettivo da massimizzare. Le
activation dei rami raw hidden e token sono fisse a ReLU.

## Spazio di ricerca

| Parte della rete | Scelte |
|---|---|
| Raw hidden | `4096D -> Dense(64, 512 o 1024) -> ReLU` |
| Token embedding congelato | `4096D -> Dense(64, 512 o 1024) -> ReLU` |
| KV corrente | diretto `512D`, oppure `512D -> Dense(64, 128 o 512) -> ReLU` |
| Dopo la concatenazione | nessun layer, oppure `Dense(512) -> ReLU` |
| Learning rate Adam | `0.00003`, `0.00005`, `0.00006997208`, `0.0001`, `0.00015` |

Il valore `0.00006997208` e' incluso esattamente perché è la baseline.
Lo spazio teorico contiene `3 × 3 × 4 × 2 × 5 = 360` combinazioni. Hyperband
non le addestra tutte fino in fondo: elimina presto quelle con Recall@6
validation meno promettente.

Parametri fissi nello screening:

```text
state fixed-NNZ        : Dense(512) -> ReLU -> Dropout(0.3)
batch size             : 4096
shuffle block size     : 65536
L2                     : 0.0
policy TensorFlow      : float32
```

La mixed precision globale resta esclusa: nelle prove Build 5 è risultata più
lenta sul sistema corrente.

## Comando per lo screening

Il primo screening usa al massimo 20 epoche nel budget maggiore e pazienza 3.
Con `max_epochs=20`, `factor=3` e una iterazione Hyperband, KerasTuner avvia
22 configurazioni iniziali distinte. Le configurazioni migliori vengono
promosse in round successivi, per un totale di 30 esecuzioni di trial: alcune
sono quindi la continuazione della stessa configurazione, non nuovi candidati.
Non vengono esplorate tutte le 360 combinazioni.

```bash
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline \
  python test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/tune_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn.py \
  --state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/09_routing_layer_router_state_kv_current/arrays \
  --fixed-nnz-state-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/15_routing_layer_router_state_kv_current_fixed_nnz_state \
  --quantized-hidden-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8 \
  --kv-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/14_routing_layer_router_state_kv_current_materialized_kv_f16 \
  --token-data-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/ablations/10_routing_layer_router_state_kv_current_token_embedding/arrays \
  --token-embedding-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/token_embedding_artifacts \
  --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1 \
  --state-hidden-units 512 \
  --dropout 0.3 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --max-epochs 20 \
  --early-stopping-patience 3 \
  --factor 3 \
  --hyperband-iterations 1
```

Un trial 1024D può richiedere molta più VRAM. Se il backend non lo esegue, il
tuner registra il trial fallito e continua con gli altri, fino al limite di
dieci fallimenti consecutivi.

## Salvataggio, Ctrl+C e ripresa

```text
<output-dir>/
├── tuner/build5_fixed_nnz_hyperband/   stato interno KerasTuner e trial
├── trial_progress/<trial-id>.jsonl     metriche e durata di ogni epoca terminata
├── trial_progress/<trial-id>_timing.json durata totale del trial
├── trial_checkpoints/<trial-id>/best.weights.h5
└── reports/
    ├── search_configuration.json
    ├── trials_snapshot.json
    ├── tuning_report.json
    └── interruption_report.json        creato dopo Ctrl+C
```

Alla fine di ogni trial viene scritto `trials_snapshot.json`; dopo ogni epoca
terminata viene aggiunta una riga JSONL. `Ctrl+C` viene intercettato: lo script
scrive un ultimo snapshot e il report di interruzione. I trial completati
restano quindi analizzabili integralmente con iperparametri, score, tempi,
metriche per epoca e checkpoint migliore. Del trial attivo restano le epoche concluse,
ma non viene considerato un confronto finale.

Per riprendere usare lo stesso comando e aggiungere `--resume`:

```bash
... --output-dir test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1 --resume
```

La configurazione viene confrontata con `reports/search_configuration.json`;
se dati o spazio di ricerca differiscono, la ripresa viene rifiutata per non
mescolare trial incompatibili.

## Fase 2: conferma lunga

`reports/tuning_report.json` elenca fino a cinque configurazioni migliori.
Ognuna sarà rieseguita con il trainer ordinario, in una nuova output directory,
usando:

```text
--epochs 1000
--early-stopping-patience 5
```

La fase lunga non è automatica: separarla conserva confronti chiari e non
spende risorse sulle configurazioni scartate.

La conferma lunga del migliore trial `0024` e' stata eseguita separatamente,
partendo dai suoi pesi dell'epoca 20. Con `--epochs 1000` e pazienza 5 si e'
fermata all'epoca 56, con il massimo all'epoca 51:

```text
validation Recall@6 iniziale (trial 0024) : 0.490852605
validation Recall@6 migliore dopo ripresa : 0.502358287
```

Questo e' il **migliore risultato di validation Build 5 ottenuto finora**.
La sua valutazione finale, eseguita una sola volta sul test set da 209.400
esempi, ha ottenuto Recall@6 **`0.506056192`**: e' il migliore risultato Build
5 ottenuto finora anche sul test. Il modello risultante e' in
`retrain_best_hyperband_trial_0024_EP1000_PAT5/models/ffn.keras`; il
checkpoint dei soli pesi migliori e' `models/best.weights.h5`. Il checkpoint
Hyperband originale `trial_checkpoints/0024/best.weights.h5` e' stato
conservato invariato.

La tabella completa della configurazione vincente, il significato del
warm-start da trial `0024` e i percorsi esatti dei checkpoint sono documentati
nel file [`doc_resume_best_hyperband_trial.md`](doc_resume_best_hyperband_trial.md),
intitolato *Proseguire il miglior trial Hyperband della Build 5*.

## Valutazione una sola volta del checkpoint migliore sul test set

Il test set non viene usato durante Hyperband. Dopo avere scelto il miglior
trial tramite validation, si può valutare il suo checkpoint una sola volta sul
test set con `evaluate_fixed_nnz_state_quantized_hidden_token_embedding_kv_ffn_checkpoint.py`.
Lo script ricostruisce esattamente la rete, carica soltanto i suoi pesi salvati
e produce predizioni: non esegue `model.fit`, non modifica pesi e non usa train
o validation.

Nel tuning `hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1`, il migliore era
il trial KerasTuner `0024` (la numerazione interna parte da zero), con
validation Recall@6 `0.49085259437561035`. Il checkpoint
`trial_checkpoints/0024/best.weights.h5`, non il numero progressivo `#25`
mostrato nel log, e' stato valutato una sola volta sul test e ha ottenuto
Recall@6 `0.494631`.

Successivamente quel checkpoint e' stato riaddestrato nella fase lunga e ha
raggiunto il nuovo record di validation `0.502358287`. La sua valutazione
finale una sola volta sul test ha ottenuto Recall@6 `0.506056192`, superiore
al precedente `0.494631`. Questo e' il checkpoint da considerare migliore per
le future integrazioni runtime.

Il report JSON della valutazione conserva il checkpoint, la configurazione e
il Recall@6 sul test. Una volta letto quel valore, non va scelto un altro trial
in base al test: il test resta una verifica finale imparziale.

## Aggiornamento: Build 6 con informazioni del layer precedente

La Build 6 e' un esperimento separato dalla Build 5: elimina il token embedding
congelato e mantiene distinti i rami delle feature del layer corrente `l` e del
layer precedente causale `l-1`. Il suo tuning non usa Hyperband, ma successive
halving: 30 candidate fino a 3 epoche, 15 promosse fino a 7 epoche e 5 finaliste
fino a 20 epoche. Il resoconto completo e riproducibile e' in
[`tuning_report.json`](../../../build_dataset_6/ml/neural_network/successive_halving_current_previous_hidden_kv_fusion_v1/reports/tuning_report.json).

La candidata migliore e' `candidate_09`, con:

```text
state esteso fixed-NNZ       : 2048 -> ReLU -> Dropout(0.3)
hidden raw layer l           : 4096D -> Dense(512) -> ReLU
hidden raw layer l-1         : 4096D -> Dense(64) -> ReLU
KV layer l                   : diretto 512D
KV layer l-1                 : 512D -> Dense(128) -> ReLU
fusione                      : Dense(2048) -> ReLU -> Dropout(0.3)
output                       : Dense(256) -> sigmoid
Adam learning rate           : 0.0001
```

Al termine delle 20 epoche di tuning ha raggiunto validation Recall@6
`0.534193993`. Il checkpoint e' stato poi proseguito su una NVIDIA A100 fino
all'epoca 23; il checkpoint migliore risultante e' conservato localmente in
`build_dataset_6/ml/neural_network/a100_resume_epoch21_patience5/models/ffn.keras`.

### Valutazione Top-6, Top-10, Top-20, Top-50 e posizioni degli expert target

Il checkpoint A100 e' stato valutato senza addestramento sui set validation e
test. La valutazione estesa, eseguita localmente sulla CPU senza modificare i
pesi, e' in [`recall_at_6_10_20_50_rank_distribution.json`](../../../build_dataset_6/ml/neural_network/a100_resume_epoch21_patience5/reports/recall_at_6_10_20_50_rank_distribution.json).
Il report precedente, limitato alle sole metriche Top-6 e Top-10, e' in
[`recall_at_6_and_10.json`](../../../build_dataset_6/ml/neural_network/a100_resume_epoch21_patience5/reports/recall_at_6_and_10.json).

| Split | Recall@6 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|
| Validation | 0.536895 | 0.652267 | 0.770273 | 0.884137 |
| Test | 0.540792 | 0.657125 | 0.775077 | 0.887555 |

La distribuzione delle posizioni dei sei expert target e' calcolata su
entrambi gli split dal medesimo report JSON.

| Posizione | Validation | Test |
|---|---:|---:|
| 1 | 13.83% | 13.88% |
| 2 | 11.58% | 11.66% |
| 3 | 9.54% | 9.64% |
| 4 | 7.72% | 7.79% |
| 5 | 6.17% | 6.21% |
| 6 | 4.84% | 4.90% |
| 7 | 3.77% | 3.83% |
| 8 | 3.07% | 3.08% |
| 9 | 2.53% | 2.57% |
| 10 | 2.17% | 2.16% |
| Oltre Top-10 | 34.77% | 34.29% |

I due profili sono molto simili: il comportamento osservato sul test non e'
un'anomalia isolata rispetto alla validation.

Le differenze fra le Recall cumulative permettono di distribuire ogni expert
target nelle fasce della classifica. Per esempio, la quota nelle posizioni
11--20 e' `Recall@20 - Recall@10`; la quota oltre la posizione 50 e' `1 -
Recall@50`.

| Fascia di posizione del target | Validation | Test |
|---|---:|---:|
| 1--6 | 53.69% | 54.08% |
| 7--10 | 11.54% | 11.63% |
| 11--20 | 11.80% | 11.80% |
| 21--50 | 11.39% | 11.25% |
| Oltre Top-50 | 11.59% | 11.24% |

La differenza di `0.116333` tra Recall@10 e Recall@6 non e' quindi rumore:
una parte importante degli expert reali e' gia' riconosciuta dalla rete, ma
riceve uno score che la colloca subito fuori dai sei expert che DS4 puo'
precaricare. La maggiore concentrazione e' nelle posizioni 7 e 8.

Recall@20 raggiunge `0.770273` in validation e `0.775077` sul test: circa il
77% degli expert target e' quindi gia' incluso nei venti score maggiori. Il
divario tra Top-10 e Top-20 mostra che resta una quota rilevante di target
riconosciuti ma ancora troppo in basso per essere utile al prefetch attuale.

Recall@50 sale a `0.884137` in validation e `0.887555` sul test. Rimane pero'
una quota di circa l'11% degli expert target oltre i primi 50 score: non e'
quindi sufficiente ampliare indiscriminatamente il Top-k, mentre resta utile
spingere verso l'alto gli expert target gia' vicini alla soglia Top-6.

### Loss combinata BCE + ranking pairwise

La loss storica resta la **binary cross-entropy (BCE)** sulle 256 sigmoid e
non viene sostituita: quando `--ranking-loss-weight=0` (valore predefinito),
la compilazione usa ancora esattamente `BinaryCrossentropy()` come nelle run
precedenti e nei checkpoint gia' prodotti.

Per un esperimento separato si puo' attivare la loss combinata:

```text
loss totale = binary cross-entropy + lambda * ranking loss
```

La ranking loss e' pairwise. Per ogni esempio confronta ciascuno dei sei
expert target con ciascun expert non target: applica una penalita' hinge solo
quando lo score del target non e' almeno `margin` sopra quello del non target.
In forma compatta:

```text
ranking loss = media[ max(0, margin - score(target) + score(non_target)) ]
```

#### Derivabilita' e significato della loss

La loss combinata e' una vera funzione di loss utilizzabile con la
backpropagation. Indicando con $P$ i sei expert target, con $N$ i 250 expert
non target e con $s$ gli score sigmoid prodotti dalla rete, per ogni esempio
viene calcolata come:

$$
\mathcal{L}_{\mathrm{tot}} =
\mathcal{L}_{\mathrm{BCE}} +
\lambda \cdot \mathcal{L}_{\mathrm{rank}}
$$

$$
\mathcal{L}_{\mathrm{rank}} =
\frac{1}{|P|\,|N|}
\sum_{p \in P}\sum_{n \in N}
\max\!\left(0,\; m - s(p) + s(n)\right)
$$

dove $m$ e' il parametro `margin` e $\lambda$ e' `ranking-loss-weight`.

Significato di ogni elemento:

- $\mathcal{L}_{\mathrm{tot}}$: loss totale minimizzata dall'ottimizzatore Adam.

- $\mathcal{L}_{\mathrm{BCE}}$: binary cross-entropy storica, calcolata sui 256 score sigmoid e sulle rispettive label binarie. Rimane il termine che insegna alla rete quali expert siano target.

- $\lambda$: peso della ranking loss (`--ranking-loss-weight`). Se vale $0$, la loss totale coincide esattamente con la BCE; se aumenta, l'ottimizzazione attribuisce maggiore importanza all'ordine relativo tra expert target e non target.

- $\mathcal{L}_{\mathrm{rank}}$: penalita' pairwise che misura quanto gli expert target siano insufficientemente davanti agli expert non target.

- $P$: insieme degli expert target dell'esempio; nel dataset corrente contiene i sei expert scelti dal router, quindi $|P| = 6$.

- $N$: insieme degli expert non target dello stesso esempio; dato che l'output contiene 256 expert, contiene $256 - 6 = 250$ elementi.

- $p \in P$: uno specifico expert corretto da promuovere nella classifica.

- $n \in N$: uno specifico expert non corretto da mantenere dietro agli expert target.

- $s(p)$: score sigmoid predetto dalla rete per l'expert target $p$.

- $s(n)$: score sigmoid predetto dalla rete per l'expert non target $n$.

- $m$: margine desiderato (`--ranking-loss-margin`). Una coppia $(p,n)$ non riceve penalita' soltanto se $s(p) \geq s(n) + m$.

- $\max(0,\; m - s(p) + s(n))$: hinge loss della singola coppia. Vale $0$ quando il target e' gia' davanti al non-target di almeno il margine; altrimenti cresce in proporzione a quanto manca per rispettare quel margine.

- $\sum_{p \in P}\sum_{n \in N}$: considera ogni coppia target/non-target, cioe' $6 \cdot 250 = 1.500$ confronti per esempio.

- $\frac{1}{|P|\,|N|}$: media i confronti validi, evitando che il valore della ranking loss cresca soltanto per il numero di coppie considerate.

#### Esempio numerico: BCE sola e BCE + ranking loss

Per rendere leggibile l'esempio, si considerano soltanto quattro expert invece
dei 256 reali. Gli expert target sono $A$ e $B$; $C$ e $D$ sono non target.
La rete produce questi score sigmoid:

| Expert | Label reale $y$ | Score predetto $s$ |
|---|---:|---:|
| $A$ | $1$ | $0{,}80$ |
| $B$ | $1$ | $0{,}45$ |
| $C$ | $0$ | $0{,}70$ |
| $D$ | $0$ | $0{,}10$ |

La classifica e' quindi $A$, $C$, $B$, $D$: il target $B$ si trova dietro al
non-target $C$. Nel caso reale lo stesso fenomeno corrisponde a un expert
target che rischia di restare fuori dal Top-6.

**Caso 1 -- solo BCE.** Per $E$ expert, con label binaria $y_e \in \{0,1\}$
e score sigmoid $s_e$, la binary cross-entropy media e':

$$
\mathcal{L}_{\mathrm{BCE}} =
-\frac{1}{E}\sum_{e=1}^{E}
\left[y_e\log(s_e) + (1-y_e)\log(1-s_e)\right].
$$

Nel piccolo esempio $E=4$, quindi:

$$
\mathcal{L}_{\mathrm{BCE}} =
-\frac{1}{4}\left[
\log(0{,}80) + \log(0{,}45) +
\log(1-0{,}70) + \log(1-0{,}10)
\right]
\approx 0{,}583.
$$

La BCE penalizza correttamente sia $B$ troppo basso sia $C$ troppo alto. Li
tratta pero' come due errori sugli score individuali: non esprime in modo
esplicito che, per il prefetch, e' importante avere $B$ *davanti* a $C$.

**Caso 2 -- BCE + ranking loss.** Si scelgano $m = 0{,}20$ e $\lambda = 1$.
Le quattro coppie target/non-target sono $(A,C)$, $(A,D)$, $(B,C)$ e $(B,D)$:

$$
\begin{aligned}
\ell(A,C) &= \max(0, 0{,}20 - 0{,}80 + 0{,}70) = 0{,}10 \\
\ell(A,D) &= \max(0, 0{,}20 - 0{,}80 + 0{,}10) = 0 \\
\ell(B,C) &= \max(0, 0{,}20 - 0{,}45 + 0{,}70) = 0{,}45 \\
\ell(B,D) &= \max(0, 0{,}20 - 0{,}45 + 0{,}10) = 0.
\end{aligned}
$$

Quindi:

$$
\mathcal{L}_{\mathrm{rank}} = \frac{0{,}10 + 0 + 0{,}45 + 0}{4} = 0{,}1375
$$

e la loss totale vale:

$$
\mathcal{L}_{\mathrm{tot}} = 0{,}583 + 1 \cdot 0{,}1375 \approx 0{,}7205.
$$

Il contributo aggiuntivo piu' grande e' quello della coppia $(B,C)$: la loss
combinata segnala esplicitamente che il target $B$ deve superare il non-target
$C$ di almeno $0{,}20$. Non cambia le label ne' l'output a 256 score; aggiunge
soltanto un gradiente che rende piu' utile la loro classifica per Recall@K.
Nell'implementazione reale la stessa media viene calcolata su $6 \cdot 250$
coppie per esempio.

La BCE e' derivabile rispetto agli score. Il termine pairwise usa
`max(0, ...)`, implementato come `tf.nn.relu`: e' derivabile a tratti e nel
punto angoloso usa un sottogradiente, come avviene normalmente per hinge loss
e ReLU. Le maschere target/non-target dipendono soltanto dalle label reali,
non dagli score predetti; quindi non interrompono il gradiente.

Nella loss non vengono usati `top_k`, ordinamenti o ranghi discreti. Queste
operazioni non sono derivabili e renderebbero impossibile ottimizzare
direttamente Recall@K. La pairwise ranking loss e' invece un *surrogate*
derivabile: penalizza un target quando il suo score non supera quello di un
non-target di almeno `margin`, spingendolo verso le prime posizioni della
classifica.

La media sui $|P|\,|N| = 6 \cdot 250 = 1.500$ confronti validi mantiene la scala
del termine ranking stabile: aumentare il numero di coppie non ne amplifica
automaticamente il peso rispetto alla BCE. `lambda` controlla esplicitamente
questo compromesso; con `lambda = 0` resta esattamente la BCE storica.

Questo non cambia l'output della rete, che resta uno score per tutti i 256
expert, ma spinge direttamente gli expert corretti davanti ai negativi. E'
quindi coerente con i risultati del report
[`recall_at_6_10_20_50_rank_distribution.json`](../../../build_dataset_6/ml/neural_network/a100_resume_epoch21_patience5/reports/recall_at_6_10_20_50_rank_distribution.json): il 23.43% dei target test e' gia' tra le posizioni 7--20 e il 34.68% tra le posizioni 7--50, ma oggi non puo' contribuire al Top-6 usato da DS4.

Gli script Build 6 accettano ora due parametri espliciti:

- `--ranking-loss-weight <lambda>`: peso della componente ranking; `0` conserva la BCE storica.
- `--ranking-loss-margin <margin>`: separazione minima desiderata fra target e non target; il valore predefinito e' `0.05`.

La selezione continua a massimizzare **Recall@6 validation**, non Recall@10:
il runtime prefetch corrente utilizza infatti solo sei expert predetti per
layer. La nuova loss va quindi valutata in una nuova directory di training o
tuning, senza modificare le run BCE gia' esistenti.

### Primo esperimento ranking pairwise: lambda 1.0, margin 0.05

La prima verifica della loss combinata e' stata eseguita come addestramento
nuovo da pesi casuali sulla GPU NVIDIA **L4** di Google Colab (non sulla A100,
non disponibile in quel momento). Ha mantenuto esattamente l'architettura
Build 6 migliore sopra descritta e gli stessi parametri di ottimizzazione
della conferma BCE: Adam con learning rate `0.0001`, dropout `0.3`, batch
`4096`, macro-block shuffle `65536`, massimo `1000` epoche ed early stopping
con pazienza `5` su `val_recall_at_6`.

I nuovi parametri erano:

```text
ranking_loss_weight = 1.0
ranking_loss_margin = 0.05
loss = BCE + 1.0 * ranking_pairwise
```

L'addestramento si e' fermato all'epoca 43; il miglior valore validation
osservato e' circa **Recall@6 = 0.5300** (il report finale, dopo il ripristino
dei pesi migliori, riporta `0.529966`). La loss validation combinata e' scesa
fino a circa `0.0726`, ma dopo l'epoca 35 la Recall@6 non ha piu' mostrato un
miglioramento stabile. La durata sulla L4 e' stata di `1h 21m 44s`.

Il risultato e' inferiore al checkpoint BCE di riferimento addestrato su A100,
che ottiene validation Recall@6 `0.536895`: la differenza e' circa `-0.0069`
Recall@6 (circa `-1.3%` relativo). Per questo la configurazione `lambda=1.0`,
`margin=0.05` non sostituisce il modello BCE migliore e non va ancora portata
all'integrazione DS4.

Gli artefatti della run sono stati copiati con successo su Google Drive in
`DS4_Colab/build6_results/l4_ranking_lambda1_margin005/`. La directory locale
temporanea Colab si chiamava `build6_a100_ranking_lambda1_margin005` per
eredita' della cella precedente, ma il dispositivo effettivamente usato era la
L4 e la directory persistente usa quindi il nome corretto `l4`. Un confronto
successivo ragionevole e' ridurre il peso a `lambda=0.25`, mantenendo inizialmente
lo stesso margin `0.05`, per conservare BCE come termine dominante.

### Esperimento separato: rami profondi con BCE classica

Per verificare se una trasformazione non lineare aggiuntiva cattura pattern piu'
fini, e' disponibile l'architettura alternativa `deeper_branches`. Il percorso
compatto migliore resta **immutato**: continua a essere il valore predefinito
`--architecture compact`, usa il builder storico e mantiene validi sia il
checkpoint A100 sia tutti i report gia' raccolti. La nuova variante deve sempre
essere addestrata in una nuova directory di output.

La variante profonda usa soltanto la BCE classica (`--ranking-loss-weight=0`) e
mantiene separati i segnali di layer `l` e `l-1`:

```text
state esteso 8.785D -> fixed-NNZ(2048) -> ReLU -> Dense(1024) -> ReLU -> Dropout(0.3)

hidden l:   4.096D -> Dense(512) -> ReLU -> Dense(128 oppure 256) -> ReLU -> Dropout(0.1)
hidden l-1: 4.096D -> Dense(64)  -> ReLU -> Dense(32)             -> ReLU -> Dropout(0.1)
KV l:       512D diretto
KV l-1:     512D -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dropout(0.1)
token_position: 1D diretto

Concatenate -> Dense(2048) -> ReLU -> Dense(1024) -> ReLU -> Dropout(0.3)
            -> Dense(256) -> sigmoid
```

Il dropout non viene applicato dopo ogni ReLU: viene posto al termine di ogni
ramo profondo e della fusione. In questo modo regolarizza le rappresentazioni
composite senza rendere eccessivamente rumorosi i layer intermedi. Il layer
state conserva il dropout `0.3` gia' usato dal modello migliore; i nuovi rami
usano inizialmente `0.1`.

In questo esperimento il secondo layer di fusione e' fissato a `1024` neuroni;
non viene quindi scelta la variante da `512`. La dimensione della concatenazione
non e' piu' quella compatta `3265D`: dipende
dalle scelte dei secondi layer ed e' costruita automaticamente dal modello. Con
le opzioni previste varia da `1761D` a `1921D`.

Configurazione iniziale consigliata per il primo confronto controllato:

```text
--architecture deeper_branches
--state-hidden-units 2048 --state-second-units 1024
--current-hidden-projection-dimension 512 --current-hidden-second-units 128
--previous-hidden-projection-dimension 64 --previous-hidden-second-units 32
--current-kv-projection-dimension direct
--previous-kv-projection-dimension direct
--previous-kv-first-units 128 --previous-kv-second-units 64
--fusion-hidden-layers 2048 --fusion-second-units 1024
--dropout 0.3 --branch-dropout 0.1
--learning-rate 0.0001 --ranking-loss-weight 0
```

Solo dopo questa run BCE comparabile avrebbe senso provare l'alternativa
`128/256` del secondo layer hidden corrente, oppure riattivare la ranking loss
come un esperimento distinto.

### Risultato: rami profondi BCE su A100

La configurazione profonda documentata sopra e' stata addestrata da pesi
casuali su una NVIDIA A100 da 40 GB, con BCE classica, Adam learning rate
`0.0001`, batch `4096`, macro-block shuffle `65536`, massimo `1000` epoche ed
early stopping con pazienza `5` su `val_recall_at_6`.

Il miglior risultato validation e' stato **Recall@6 = 0.532726**, raggiunto
all'epoca 43; l'addestramento si e' arrestato all'epoca 48 per assenza di
miglioramenti successivi. La durata totale e' stata circa 21 minuti.

Questo valore non batte il checkpoint BCE compatto migliore, che mantiene
validation Recall@6 **0.536895**: la variante profonda e' inferiore di
`0.004169` Recall@6 (circa `-0.8%` relativo). Per ora quindi il modello
compatto resta il riferimento da preservare e da usare per le valutazioni o
l'integrazione; l'architettura profonda rimane soltanto un esperimento
registrato, non un sostituto.

### Esperimento separato: tuning ampliato con BCE + ranking loss

Il file
`tune_build6_expanded_ranking_previous_layer_ffn.py` e' un wrapper separato
del tuner Build 6 storico: non modifica, non sostituisce e non cancella
`tune_build6_fixed_nnz_previous_layer_ffn.py`, i suoi checkpoint o i suoi
report. Riusa invece il medesimo successive halving riproducibile
`30 -> 15 -> 5`, con budget a 3, 7 e 20 epoche e checkpoint/resume dopo ogni
candidata completata.

I passaggi operativi per trasferire il codice, preparare Colab, avviare il
tuning e salvarne gli artefatti sono raccolti in
[`colab_build6_expanded_ranking_tuning.md`](colab_build6_expanded_ranking_tuning.md).

L'esperimento fissa la loss combinata a:

$$
\mathcal{L}_{\mathrm{tot}} =
\mathcal{L}_{\mathrm{BCE}} + 1{,}0 \cdot \mathcal{L}_{\mathrm{rank}},
\qquad m = 0{,}05.
$$

Questi valori non sono esposti come parametri del nuovo comando: vengono
imposti dal wrapper per evitare run non confrontabili. La selezione continua a
massimizzare `val_recall_at_6`, non la loss, per scegliere le candidate piu'
utili al prefetch.

Lo spazio di ricerca e' quello della FFN Build 6 compatta a rami separati,
senza token embedding, con le sole estensioni richieste sui due punti di
aggregazione:

| Parte della rete | Scelte |
|---|---|
| State esteso fixed-NNZ | `8785D -> Dense(512, 1024, 2048 o 4096) -> ReLU -> Dropout(0.3)` |
| Raw hidden layer $l$ | `4096D -> Dense(64, 512 o 1024) -> ReLU` |
| Raw hidden layer $l-1$ | `4096D -> Dense(64, 512 o 1024) -> ReLU` |
| KV layer $l$ | diretto `512D`, oppure `512D -> Dense(64, 128 o 256) -> ReLU` |
| KV layer $l-1$ | diretto `512D`, oppure `512D -> Dense(64, 128 o 256) -> ReLU` |
| Dopo la concatenazione | nessun layer, oppure `Dense(512, 1024, 2048 o 4096) -> ReLU -> Dropout(0.3)` |
| Learning rate Adam | `0.00003`, `0.00005`, `0.0001`, `0.00015` |

Le dimensioni `4096` sono intenzionalmente incluse: il tuning e' destinato a
una A100 da 40 GB e non a una L4 da 20 GB. Il campionamento estrae comunque
solo 30 configurazioni distinte dal grande spazio teorico; non addestra tutte
le combinazioni possibili.
