# DS4 — handoff completo del progetto di prefetch MoE

> Documento di contesto per una nuova sessione Codex/LLM. È pensato per essere
> letto insieme alla documentazione puntuale già presente nella repository.
> I percorsi sono relativi alla root `/home/michele/ds4`.

## 1. Obiettivo del progetto

`ds4` è un motore di inferenza C/CUDA specializzato per DeepSeek V4 Flash e
non un lettore GGUF generico. Il lavoro svolto qui ha due obiettivi distinti,
che non devono essere confusi:

1. **Inferenza runtime**: nascondere la latenza di lettura degli expert MoE
   dal GGUF su SSD, usando il prefetch `SSD -> RAM pinned -> VRAM` mentre il
   decode prosegue.
2. **Modello di prefetch ML**: predire con alta Recall@6 i sei expert che il
   router reale richiederà a un layer futuro; più hit pronti in RAM pinned
   significano meno fallback on-demand e, potenzialmente, più speedup.

Il record runtime e il miglior candidato ML non coincidono necessariamente:
un modello può avere Recall@6 migliore offline ma non essere ancora stato
integrato e misurato nel benchmark end-to-end.

## 2. Regole e punti di ingresso della repository

Leggere prima [`AGENT.md`](AGENT.md). I principi fondamentali sono:

- preservare correttezza, inferenza CUDA, SSD streaming, Metal e distributed;
- non introdurre varianti permanenti o codice fragile solo per un benchmark;
- usare `make` per validare le modifiche C/CUDA;
- non avviare contemporaneamente processi del modello molto grandi;
- il backend CPU è un riferimento/debug, non il percorso prestazionale.

Mappa sintetica:

| Area | Percorso |
|---|---|
| Motore di inferenza | `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h` |
| Pipeline dati e ML | `test_predictor/prefetch_predictor_pipeline/` |
| Definizione canonica feature | `test_predictor/prefetch_predictor_pipeline/dataset_definition/features.md` |
| Esperimenti NN Build 7 | `test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/` |
| Feature importance KNN/classical ML | `test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/` |
| Test runtime record precedente | `test_predictor/test_13/`, `test_predictor/test_14/` |
| Cache RAM pinned condivisa | `test_predictor/test_15/` |
| Meccanica dettagliata RAM pinned | `test_predictor/test_9/gestione_pinned_ram_expert_ds4.md` |

## 3. Dataset: cosa rappresenta un esempio

Un record associa informazioni causali disponibili durante il forward di un
token/layer MoE al **set dei sei expert realmente scelti dal router al layer
target**. Il target è dunque multilabel su 256 expert e la metrica critica è
la Recall@6: quanti dei sei expert reali sono presenti nei primi sei predetti.

Le tracce vengono raccolte su prompt separati deterministicamente. Le split
sono per prompt, non per riga, così non c'è leakage tra train e validation/test.
La split canonica è:

```text
test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/training/split_by_prompt.json
```

### Evoluzione fino a Build 7

Build 5 contiene la base estesa con state/router, hidden e KV materializzati.
Build 7 deriva da quella base per portare esplicitamente nel record le due
storie di layer precedenti e concentrarsi sui layer che necessitano davvero di
una previsione learned.

Script di costruzione:

```text
test_predictor/prefetch_predictor_pipeline/pipeline/create_build7_two_previous_layers_dataset.py
test_predictor/prefetch_predictor_pipeline/pipeline/preprocess_build7_dataset.py
```

Input Build 5:

```text
build_dataset_5/derived/prefetch_dataset_build_5_v0_v1_v2_raw_hidden_kv_current.jsonl
```

Output Build 7:

```text
build_dataset_7/derived/prefetch_dataset_build_7_two_previous_layers.jsonl
build_dataset_7/ml/training/arrays/
```

La conversione ha prodotto 1.959.546 record a partire da 2.062.680 record.
I layer target learned sono **5–42**; i layer fisici 3 e 4 fanno bootstrap
deterministico e non vengono usati come target learned. La split risultante è:

| Split | Esempi |
|---|---:|
| train | 1.570.008 |
| validation | 190.608 |
| test | 198.930 |

Il file Build 5 sorgente e le feature Build 7 preesistenti non devono essere
sovrascritti: ogni esperimento e materializzazione usa una nuova directory di
output.

## 4. Feature Build 7 già disponibili

La fonte autorevole, con semantica e causalità di ogni campo, è
[`features.md`](test_predictor/prefetch_predictor_pipeline/dataset_definition/features.md).
Le feature principali già usate dal migliore modello Build 7 originale sono:

- **state corrente fixed-NNZ**: routing/state sparso in forma fissa;
- **hidden raw quantizzato corrente `l`** e precedente `l-1` (4096D);
- probabilità di transizione tra expert;
- informazioni storiche causali di routing e layer.

Artefatti principali:

```text
build_dataset_7/ml/training/arrays/
build_dataset_7/ml/ablations/11_current_fixed_nnz_state/
build_dataset_7/ml/ablations/05_current_hidden_int8/
build_dataset_7/ml/ablations/08_previous_hidden_int8/
build_dataset_7/ml/transition_features/expert_transition_probabilities_v1/
```

Gli script che materializzano gli input principali sono sotto
`pipeline/state_features/` e `pipeline/transition_features/`; in particolare:

```text
pipeline/state_features/materialize_build7_current_only_fixed_nnz_state.py
pipeline/transition_features/materialize_build7_expert_transition_probabilities.py
```

Usare array memory-mapped/streaming: molti artefatti train sono dell'ordine di
diversi GiB e non vanno caricati tutti in RAM in una volta sola.

## 5. Nuove feature candidate (priority_1_2_v1)

Sono state create senza modificare Build 7. Script:

```text
test_predictor/prefetch_predictor_pipeline/pipeline/feature_candidates/
  materialize_build7_priority_features.py
```

Output e manifest:

```text
build_dataset_7/ml/feature_candidates/priority_1_2_v1/
build_dataset_7/ml/feature_candidates/priority_1_2_v1/artifacts/collection_manifest.json
```

Tutte le feature sono calcolabili causalmente al punto di previsione: non
leggono il router del token target futuro.

| Famiglia | Directory | Contenuto | Esito sperimentale |
|---|---|---|---|
| `hidden_delta` | `01_hidden_delta/` | Delta int8 dequantizzato tra hidden corrente e precedente (4096D) + 4 statistiche: norme e coseno | Utile e confermata dall'ablation |
| `routing_dynamics` | `02_routing_dynamics/` | 256 età/recency per expert + 5 statistiche globali di turnover/Jaccard/run length | Utile e confermata dall'ablation |
| `layer_expert_ema` | `03_layer_expert_ema/` | EMA causale per `(sequenza, layer, expert)`, decay 0,9 | Segnale non robusto, da escludere per ora |
| `router_confidence` | `04_router_confidence/` | 33 statistiche compatte su logits router corrente e storia | Debole, da escludere per ora |

### Risultati offline delle candidate

Tutti i valori sono Recall@6 **validation**; i run ripresi hanno proseguito
fino a massimo 1000 epoche con early stopping patience 10.

| Input aggiunto al baseline Build 7 | Migliore R@6 validation | Evidenza ablation |
|---|---:|---|
| nessuno, Top-4 originario | 0,539006 | baseline originale |
| `hidden_delta` | 0,540081 | zero 0,525594; permutation 0,529714 |
| `routing_dynamics` | 0,544162 | zero 0,535619; permutation 0,533451 |
| `layer_expert_ema` | 0,539546 | zero 0,539491; permutation 0,539511: non informativa |
| `router_confidence` (60 epoche) | 0,530200 | screening debole |
| `hidden_delta + routing_dynamics` | 0,544672 | hidden zero 0,531719; routing zero 0,529729; entrambi zero 0,521415 |
| AutoML su `hidden_delta + routing_dynamics` (`trial_006`) | **0,548003** | migliore modello selezionato e poi verificato su test |

Il migliore modello ML offline attuale è il risultato AutoML su
**`hidden_delta + routing_dynamics`**. Il test è stato consultato una sola
volta dopo la selezione su validation e conferma il miglioramento: Recall@6
test **0,551882**. Non è ancora integrato nel servizio runtime, quindi il
record di speedup resta associato al checkpoint Top-4 originario.

## 6. Architetture e script della rete neurale

### Modello Build 7 originale runtime

Checkpoint di riferimento integrato nei test record:

```text
build_dataset_7/ml/neural_network/top4_patience10_l2_3e6_bce_v1/models/ffn.keras
```

Script di training/evaluation:

```text
ML_pipeline/neural_network/train_build7_top4_gated_residual_bce.py
ML_pipeline/neural_network/evaluate_build7_top4_gated_residual_topk.py
```

Struttura sintetica: encoder state fixed-NNZ 2048, proiezioni hidden
corrente/precedente 64, transizioni 256D, fusione 2048, testata gated residual
su 256 expert; dropout 0,3, L2 `3e-6`, Adam `1.5e-4`, BCE multilabel.

Valutazione multi-Recall salvata in:

```text
build_dataset_7/ml/neural_network/top4_patience10_l2_3e6_bce_v1/reports/
  recall_at_6_10_20_50_rank_distribution_train_validation_test.json
```

| Split | R@6 | R@10 | R@20 | R@50 |
|---|---:|---:|---:|---:|
| train | 0,605569 | 0,725152 | 0,835630 | 0,929502 |
| validation | 0,539006 | 0,655730 | 0,775345 | 0,891281 |
| test | 0,542485 | 0,659771 | 0,779376 | 0,894476 |

### Screening e ablation NN delle nuove feature

```text
ML_pipeline/neural_network/train_build7_priority_feature_candidate.py
ML_pipeline/neural_network/evaluate_build7_priority_feature_ablation.py
ML_pipeline/neural_network/train_build7_routing_dynamics_hidden_delta.py
ML_pipeline/neural_network/evaluate_build7_routing_dynamics_hidden_delta_ablation.py
```

I report persistono nelle rispettive directory sotto:

```text
build_dataset_7/ml/neural_network/priority_feature_screening_v1/
```

### Ricerca automatica di architettura/iperparametri completata

```text
ML_pipeline/neural_network/tune_build7_hidden_delta_routing_dynamics.py
```

Questo è un AutoML leggero e riproducibile, non una libreria esterna: genera
una baseline e configurazioni random deterministiche non duplicate, poi usa
**successive halving**. Con default `--trials 24 --stage-epochs 10,40,150
--survivors 8,3`, tutte le 24 configurazioni ricevono 10 epoche, le migliori
8 proseguono fino a 40, e le migliori 3 fino a 150. La selezione usa soltanto
Recall@6 validation, quindi il test resta intatto per la verifica finale.

Spazio di ricerca: capacità state/fusion, proiezioni hidden, dimensioni delle
due nuove branche, dropout, learning rate e L2. La base delle feature è sempre
il modello Build 7 vincente più `hidden_delta + routing_dynamics`.

Il run `automl_hidden_delta_routing_v1` ha completato 24 trial con budget
10/40/150 epoche e promozioni 8/3, senza leggere il test durante la ricerca.
Il vincitore è `trial_006`:

| Iperparametro | Valore |
|---|---:|
| state hidden units | 3072 |
| proiezione hidden corrente | 64 |
| proiezione hidden precedente | 96 |
| fusion hidden units | 3072 |
| dropout | 0,4 |
| Adam learning rate | `2e-4` |
| L2 | `1e-6` |
| ramo `hidden_delta` | 64 unità |
| ramo `routing_dynamics` | 128 unità |

Il miglior checkpoint validation è stato conservato in `best.weights.h5`
(non coincide necessariamente con l'ultima epoca). La sua valutazione finale
multi-Recall è:

| Split | R@6 | R@10 | R@20 | R@50 |
|---|---:|---:|---:|---:|
| validation | 0,548003 | 0,668154 | 0,790963 | 0,903837 |
| test | **0,551882** | **0,672929** | **0,795027** | **0,906925** |

Rispetto al Top-4 originale sul test (`0,542485 / 0,659771 / 0,779376 /
0,894476`), il guadagno è rispettivamente `+0,009397`, `+0,013158`,
`+0,015651` e `+0,012449`. I risultati sono in Drive:

```text
DS4_Colab/build7_results/automl_hidden_delta_routing_v1/
  models/best.weights.h5
  reports/final_ranking.json
  reports/final_topk_validation_test.json
```

Documentazione dei trial NN storici e correnti:

```text
ML_pipeline/neural_network/documentation/build7_two_previous_layers_experiments.md
```

## 7. Test di importanza feature con KNN/classical ML

Questi esperimenti aiutano a filtrare feature candidate prima di pagare il
costo di molti training NN. Non sostituiscono la valutazione neurale, perché
il KNN vede una geometria e un budget diversi.

Directory e script principali:

```text
ML_pipeline/classical_ml_preprocessing/run_progressive_knn_ablation.py
ML_pipeline/classical_ml_preprocessing/run_build5_knn_feature_importance.py
ML_pipeline/classical_ml_preprocessing/run_build6_knn_feature_importance.py
ML_pipeline/classical_ml_preprocessing/preprocess_build7_two_previous_layers_state.py
ML_pipeline/classical_ml_preprocessing/train_classical_models.py
```

Documentazione associata:

```text
ML_pipeline/classical_ml_preprocessing/documentation/doc_progressive_knn_ablation.md
ML_pipeline/classical_ml_preprocessing/documentation/build5_knn_feature_importance.md
ML_pipeline/classical_ml_preprocessing/documentation/build6_knn_feature_importance.md
ML_pipeline/classical_ml_preprocessing/documentation/doc_classical_preprocessing.md
```

Protocollo consigliato per una nuova feature: (1) materializzarla in una nuova
directory, (2) verificarne allineamento e causalità, (3) eseguire KNN
progressivo/permutation se il costo è sostenibile, (4) fare screening NN con
stesso split e early stopping, (5) ablation zero e permutation, (6) combinare
solo le feature con evidenza indipendente.

## 8. Benchmark runtime e record di speedup

### Configurazione record reale

Il record end-to-end attuale è **TEST_14 `output_2`**, non TEST_15:

| Parametro | Valore |
|---|---|
| Predictor | Build 7 Top-4 originale (`ffn.keras` indicato sopra) |
| Expert richiesti al predictor | Top-6 |
| Layer fisici nella finestra | 3–42 (40 layer) |
| Layer con predizione learned | 5–42 |
| Slot RAM pinned | 2 per layer/distanza futura |
| Worker I/O | 6, assegnazione `layer % 6` |
| Trasferimento SSD | batch Top-6 (`gate/up/down`) tramite bridge Windows |
| Ripetizioni | 3 coppie B/C alternate |
| B baseline medio | 103,170 s |
| C learned-prefetch medio | 72,588 s |
| Tempo risparmiato medio | 30,582 s |
| **Speedup medio** | **+29,65%** |

Il flusso record usa slot batch per layer: il bridge Windows legge gli expert
Top-6 dal GGUF direttamente nei buffer di **RAM pinned**; quando il router
reale raggiunge il layer, il decode confronta expert reali e prefetched,
trasferisce in VRAM gli expert coincidenti e applica fallback on-demand solo
per quelli assenti. RAM pinned è RAM host page-locked, non VRAM.

Documentazione e output:

```text
test_predictor/test_13/README.md        # record precedente Build 5/output_9
test_predictor/test_14/README.md        # confronto Build 7 e record output_2
test_predictor/test_14/output_2/benchmark_summary.json
```

### TEST_15: cache per `(layer, expert)`

TEST_15 ha esplorato cache globale/sharded di RAM pinned, riuso degli expert,
eviction e poi B.1 con riallocazione delle entry READY senza lease/ref future.
Ha risolto problemi di saturazione e bug di remap GPU, ma non ha superato il
record: la granularità per-expert e le chiamate bridge più piccole hanno
introdotto overhead rispetto al trasferimento batch Top-6.

Risultati utili: `output_6` è arrivato a circa +29,39% (molto vicino), ma il
record resta TEST_14/output_2 a +29,65%. Top-10, Top-20, più worker, B.1 e il
percorso ibrido sono stati misurati e non hanno superato in modo stabile il
batch Top-6 record.

Leggere in questo ordine:

```text
test_predictor/test_15/README.md
test_predictor/test_15/output_*/benchmark_summary.json
test_predictor/test_9/gestione_pinned_ram_expert_ds4.md
test_predictor/test_9/README.md
```

Il documento `gestione_pinned_ram_expert_ds4.md` spiega in dettaglio:

- thread di decode DS4, thread asincrono di prefetch, servizio modello Build 7;
- bridge Windows e trasferimento GGUF -> RAM pinned;
- stati e transizioni degli slot;
- soluzione B, B.1, shard `layer % N`, mutex/condition variable;
- `lease_count`, `future_refcount`, saturazione e riallocazione;
- misure/diagnostica da leggere nei JSON.

Non modificare TEST_14 per sperimentare: usare TEST_15 o una nuova directory
di test; preservare il record come controllo riproducibile.

## 9. Servizi e ambiente runtime

Il modello principale usato nelle run era:

```text
gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Il servizio del modello predictor Build 7 è distinto dal **bridge Windows**:

- il servizio predictor esegue la NN Python e restituisce i Top-K;
- il bridge Windows riceve offset GGUF e copia i byte dei pesi SSD nei buffer
  di RAM pinned esposti dal processo WSL/CUDA.

Implementazioni e runner correlati:

```text
test_predictor/test_14/runtime_predictor_build7_service.py
test_predictor/test_14/windows_native/test_14_windows_reader_service.py
test_predictor/test_14/test_14_runner.py
test_predictor/test_15/test_15_runner.py
test_predictor/test_15/test_15_runner.c
```

I dettagli operativi e le stringhe d'ambiente sono nei README dei test. Se il
GGUF risiede sull'SSD esterno Windows D:, WSL deve vedere `/mnt/d`; controllare
sempre prima della run con `findmnt /mnt/d` e `ls -lh` sul file GGUF.

## 10. Colab / Google Drive per il training

La GPU locale TensorFlow era utilizzabile ma lenta/instabile in alcuni run; i
training pesanti vengono eseguiti su Colab. Il remote rclone è `gdrive:` e la
struttura prevista in Drive è:

```text
DS4_Colab/build7_code/
DS4_Colab/build7_inputs/
DS4_Colab/build7_feature_candidates/priority_1_2_v1/
DS4_Colab/build7_results/
```

Nel runtime Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Poi sincronizzare in locale Colab (`/content`) codice, input e candidate con
`rsync`, eseguire un solo training per volta e copiare report/pesi di ritorno
in Drive. Prima di una nuova sessione verificare che siano presenti almeno:

```text
/content/build7_code/ffn_utils.py
/content/build7_code/train_build7_routing_dynamics_hidden_delta.py
/content/build7_code/tune_build7_hidden_delta_routing_dynamics.py
/content/build7_inputs/training/arrays/
/content/build7_inputs/current_fixed_nnz_state/arrays/
/content/build7_inputs/current_hidden_int8/arrays/
/content/build7_inputs/previous_hidden_int8/arrays/
/content/build7_inputs/expert_transition_probabilities_v1/arrays/
/content/build7_feature_candidates/priority_1_2_v1/
```

Il runtime deve usare GPU (controllare `!nvidia-smi`). Il primo epoch può essere
molto più lento per inizializzazione/compilazione; in seguito i run osservati
erano circa 17–27 s/epoca su Colab GPU.

### Procedura riproducibile: upload da WSL a Google Drive

I comandi seguenti presuppongono che `rclone` sia già configurato con il remote
`gdrive:`. Vanno eseguiti nella root della repository e copiano, senza
modificare gli artefatti locali, il minimo necessario ai run Colab.

```bash
# Directory remota per le nuove feature candidate.
rclone mkdir gdrive:DS4_Colab/build7_feature_candidates/priority_1_2_v1

# Feature candidate: circa 7,5 GiB per hidden_delta; --progress rende visibile
# l'avanzamento e rclone può riprendere copie interrotte.
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/feature_candidates/priority_1_2_v1/01_hidden_delta \
  gdrive:DS4_Colab/build7_feature_candidates/priority_1_2_v1/01_hidden_delta \
  --progress --transfers 4 --checkers 8 --low-level-retries 20

# Le altre feature candidate sono più piccole, ma servono per gli screening e
# per la combinazione hidden_delta + routing_dynamics.
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/feature_candidates/priority_1_2_v1/ \
  gdrive:DS4_Colab/build7_feature_candidates/priority_1_2_v1/ \
  --progress --transfers 4 --checkers 8 --low-level-retries 20
```

Caricare inoltre una sola volta i normali input Build 7 in
`gdrive:DS4_Colab/build7_inputs/`, mantenendo i nomi che Colab si aspetta:

```bash
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/training/ \
  gdrive:DS4_Colab/build7_inputs/training/ --progress
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/ablations/11_current_fixed_nnz_state/ \
  gdrive:DS4_Colab/build7_inputs/current_fixed_nnz_state/ --progress
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/ablations/05_current_hidden_int8/ \
  gdrive:DS4_Colab/build7_inputs/current_hidden_int8/ --progress
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/ablations/08_previous_hidden_int8/ \
  gdrive:DS4_Colab/build7_inputs/previous_hidden_int8/ --progress
rclone copy \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/transition_features/expert_transition_probabilities_v1/ \
  gdrive:DS4_Colab/build7_inputs/expert_transition_probabilities_v1/ --progress
```

Per aggiornare soltanto il codice necessario, senza inviare l'intera
repository, usare `copyto` per gli script e la utility condivisa:

```bash
rclone mkdir gdrive:DS4_Colab/build7_code
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/ffn_utils.py \
  gdrive:DS4_Colab/build7_code/ffn_utils.py
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_build7_priority_feature_candidate.py \
  gdrive:DS4_Colab/build7_code/train_build7_priority_feature_candidate.py
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/evaluate_build7_priority_feature_ablation.py \
  gdrive:DS4_Colab/build7_code/evaluate_build7_priority_feature_ablation.py
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/train_build7_routing_dynamics_hidden_delta.py \
  gdrive:DS4_Colab/build7_code/train_build7_routing_dynamics_hidden_delta.py
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/evaluate_build7_routing_dynamics_hidden_delta_ablation.py \
  gdrive:DS4_Colab/build7_code/evaluate_build7_routing_dynamics_hidden_delta_ablation.py
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/tune_build7_hidden_delta_routing_dynamics.py \
  gdrive:DS4_Colab/build7_code/tune_build7_hidden_delta_routing_dynamics.py
```

### Celle Colab: mount, staging locale e controlli

Nel notebook Colab, dopo avere selezionato un runtime GPU, eseguire queste
celle. I training leggono da `/content` (più veloce e meno fragile del mount
Drive diretto), mentre Drive resta il deposito persistente.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/build7_code /content/build7_inputs /content/build7_feature_candidates
!rsync -a --info=progress2 \
  /content/drive/MyDrive/DS4_Colab/build7_code/ /content/build7_code/
!rsync -a --info=progress2 \
  /content/drive/MyDrive/DS4_Colab/build7_inputs/ /content/build7_inputs/
!rsync -a --info=progress2 \
  /content/drive/MyDrive/DS4_Colab/build7_feature_candidates/priority_1_2_v1/ \
  /content/build7_feature_candidates/priority_1_2_v1/
!nvidia-smi
```

Se un runtime nuovo segnala input mancanti, verificare prima i nomi reali:

```bash
!find /content/build7_inputs -maxdepth 3 -type d | sort
!find /content/build7_inputs/training/arrays -maxdepth 1 -type f | head
```

### Celle Colab: screening, resume e ablation

Esempio per una famiglia singola (`hidden_delta`; sostituire il nome con
`routing_dynamics`, `layer_expert_ema` o `router_confidence`):

```bash
!python /content/build7_code/train_build7_priority_feature_candidate.py \
  --candidate-family hidden_delta \
  --candidate-features-dir /content/build7_feature_candidates/priority_1_2_v1 \
  --state-data-dir /content/build7_inputs/training/arrays \
  --current-fixed-nnz-state-data-dir /content/build7_inputs/current_fixed_nnz_state \
  --current-quantized-hidden-data-dir /content/build7_inputs/current_hidden_int8 \
  --previous-quantized-hidden-data-dir /content/build7_inputs/previous_hidden_int8 \
  --expert-transition-probabilities-data-dir /content/build7_inputs/expert_transition_probabilities_v1 \
  --output-dir /content/build7_priority_feature_screening_v1/hidden_delta_colab_v1 \
  --epochs 60 --early-stopping-patience 5 --batch-size 2048
```

Il resume usa il checkpoint già creato e amplia il budget; non riparte da
zero:

```bash
!python /content/build7_code/train_build7_priority_feature_candidate.py \
  --candidate-family hidden_delta \
  --candidate-features-dir /content/build7_feature_candidates/priority_1_2_v1 \
  --state-data-dir /content/build7_inputs/training/arrays \
  --current-fixed-nnz-state-data-dir /content/build7_inputs/current_fixed_nnz_state \
  --current-quantized-hidden-data-dir /content/build7_inputs/current_hidden_int8 \
  --previous-quantized-hidden-data-dir /content/build7_inputs/previous_hidden_int8 \
  --expert-transition-probabilities-data-dir /content/build7_inputs/expert_transition_probabilities_v1 \
  --output-dir /content/build7_priority_feature_screening_v1/hidden_delta_resume_ep60_patience10_v1 \
  --resume-weights /content/build7_priority_feature_screening_v1/hidden_delta_colab_v1/models/best.weights.h5 \
  --initial-epoch 60 --resume-best-validation-recall 0.530481 \
  --epochs 1000 --early-stopping-patience 10 --batch-size 2048
```

Per una candidate singola, salvare evidenza causale con zero/permutation:

```bash
!python /content/build7_code/evaluate_build7_priority_feature_ablation.py \
  --candidate-family hidden_delta \
  --candidate-features-dir /content/build7_feature_candidates/priority_1_2_v1 \
  --state-data-dir /content/build7_inputs/training/arrays \
  --current-fixed-nnz-state-data-dir /content/build7_inputs/current_fixed_nnz_state \
  --current-quantized-hidden-data-dir /content/build7_inputs/current_hidden_int8 \
  --previous-quantized-hidden-data-dir /content/build7_inputs/previous_hidden_int8 \
  --expert-transition-probabilities-data-dir /content/build7_inputs/expert_transition_probabilities_v1 \
  --checkpoint /content/build7_priority_feature_screening_v1/hidden_delta_resume_ep60_patience10_v1/models/best.weights.h5 \
  --output-report /content/build7_priority_feature_screening_v1/hidden_delta_resume_ep60_patience10_v1/reports/feature_ablation_validation.json \
  --splits validation --modes zero,permutation --batch-size 2048
```

### Celle Colab: migliore combinazione e AutoML

Il training della combinazione più promettente attuale usa entrambi i rami:

```bash
!python /content/build7_code/train_build7_routing_dynamics_hidden_delta.py \
  --candidate-features-dir /content/build7_feature_candidates/priority_1_2_v1 \
  --state-data-dir /content/build7_inputs/training/arrays \
  --current-fixed-nnz-state-data-dir /content/build7_inputs/current_fixed_nnz_state \
  --current-quantized-hidden-data-dir /content/build7_inputs/current_hidden_int8 \
  --previous-quantized-hidden-data-dir /content/build7_inputs/previous_hidden_int8 \
  --expert-transition-probabilities-data-dir /content/build7_inputs/expert_transition_probabilities_v1 \
  --output-dir /content/build7_priority_feature_screening_v1/hidden_delta_routing_dynamics_colab_v1 \
  --epochs 60 --early-stopping-patience 5 --batch-size 2048
```

Per la ricerca automatica successiva:

```bash
!python /content/build7_code/tune_build7_hidden_delta_routing_dynamics.py \
  --candidate-features-dir /content/build7_feature_candidates/priority_1_2_v1 \
  --state-data-dir /content/build7_inputs/training/arrays \
  --current-fixed-nnz-state-data-dir /content/build7_inputs/current_fixed_nnz_state \
  --current-quantized-hidden-data-dir /content/build7_inputs/current_hidden_int8 \
  --previous-quantized-hidden-data-dir /content/build7_inputs/previous_hidden_int8 \
  --expert-transition-probabilities-data-dir /content/build7_inputs/expert_transition_probabilities_v1 \
  --output-dir /content/build7_automl_hidden_delta_routing_v1 \
  --trials 24 --stage-epochs 10,40,150 --survivors 8,3 --batch-size 2048
```

### Salvataggio dei risultati dal runtime Colab a Drive

Eseguire sempre al termine di un run riuscito, prima di ricreare il runtime:

```bash
!mkdir -p /content/drive/MyDrive/DS4_Colab/build7_results/priority_feature_screening_v1
!rsync -a --info=progress2 \
  /content/build7_priority_feature_screening_v1/hidden_delta_resume_ep60_patience10_v1/ \
  /content/drive/MyDrive/DS4_Colab/build7_results/priority_feature_screening_v1/hidden_delta_resume_ep60_patience10_v1/

# Per l'AutoML cambia solo la sorgente/destinazione.
!rsync -a --info=progress2 \
  /content/build7_automl_hidden_delta_routing_v1/ \
  /content/drive/MyDrive/DS4_Colab/build7_results/automl_hidden_delta_routing_v1/
```

Per riportare in WSL un risultato selezionato:

```bash
rclone copy \
  gdrive:DS4_Colab/build7_results/automl_hidden_delta_routing_v1/ \
  test_predictor/prefetch_predictor_pipeline/build_dataset_7/ml/neural_network/automl_hidden_delta_routing_v1/ \
  --progress
```

## 11. Stato attuale e prossime mosse consigliate

1. Conservare e rendere riproducibile il checkpoint AutoML vincente, report
   trial, seed, architettura e pesi; il test è già stato consumato per la sua
   verifica finale.
2. Confrontare con un baseline identico per budget/epoche; non attribuire il
   guadagno a una feature se il controllo non ha ricevuto lo stesso training.
3. Integrare il checkpoint AutoML in una nuova variante isolata del servizio e
   ripetere TEST_14 con Top-6, 6 worker e due slot/layer. Solo il benchmark
   end-to-end può stabilire se la Recall offline superiore aumenta lo speedup.
5. Non presumere che Recall@10/@20 più alta dia speedup: finora Top-K grande
   aumentava il lavoro I/O/cache e non batteva il percorso batch Top-6.

L'ipotesi ML più promettente è migliorare davvero Recall@6 con feature causali
e capacità/regularizzazione adeguate. Le ottimizzazioni runtime più invasive
hanno già mostrato rendimenti decrescenti; mantenere il record TEST_14/output_2
come criterio end-to-end primario.

## 12. Checklist per un agente automatico

- Leggere questo file, `AGENT.md`, `features.md`, il documento Build 7 e i
  README di TEST_14/TEST_15 prima di modificare codice.
- Non sovrascrivere dataset, artefatti o output esperimento esistenti: creare
  una directory nuova e descrittiva per ogni materializzazione/training.
- Verificare causalità e row alignment per ogni nuova feature.
- Usare split per prompt già fissati; selezionare su validation, usare test una
  sola volta per il candidato finale.
- Salvare checkpoint, config, history, report multi-Recall, ablation e seed.
- Applicare le modifiche C/CUDA in modo minimo e compilare/testare la parte
  interessata; non far degradare CUDA/SSD streaming per ottimizzare una sola
  variante sperimentale.
- Per qualsiasi claim di speedup, confrontare A/B/C con stesse ripetizioni,
  modello GGUF, SSD, servizi e condizioni di sistema.

## 13. Documenti da usare come indice approfondito

```text
AGENT.md
test_predictor/prefetch_predictor_pipeline/dataset_definition/features.md
test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/documentation/build7_two_previous_layers_experiments.md
test_predictor/prefetch_predictor_pipeline/ML_pipeline/classical_ml_preprocessing/documentation/
test_predictor/test_13/README.md
test_predictor/test_14/README.md
test_predictor/test_15/README.md
test_predictor/test_9/README.md
test_predictor/test_9/gestione_pinned_ram_expert_ds4.md
```
