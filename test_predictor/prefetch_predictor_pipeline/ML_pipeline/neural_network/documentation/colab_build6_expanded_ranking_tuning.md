# Build 6: tuning ampliato con ranking loss su Google Colab

## Obiettivo

Questa guida esegue su Google Colab, preferibilmente con una NVIDIA A100 da
40 GB, un nuovo tuning separato della FFN Build 6. L'esperimento mantiene i
dati e i modelli storici invariati e usa:

$$
\mathcal{L}_{\mathrm{tot}} =
\mathcal{L}_{\mathrm{BCE}} + \lambda\mathcal{L}_{\mathrm{rank}},
\qquad \lambda = 1{,}0,\quad m = 0{,}05.
$$

Il nuovo script
`tune_build6_expanded_ranking_previous_layer_ffn.py` riusa il successive
halving riproducibile `30 -> 15 -> 5`, ma amplia le scelte per state e fusione
fino a 4096 neuroni. Non modifica il tuner storico, il checkpoint BCE migliore
o i risultati delle run precedenti.

## 1. Aggiornare il codice su Google Drive

Nel terminale WSL, dalla directory `~/ds4`, eseguire:

```bash
rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/tune_build6_expanded_ranking_previous_layer_ffn.py \
  gdrive:DS4_Colab/build6_code/tune_build6_expanded_ranking_previous_layer_ffn.py

rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/tune_build6_fixed_nnz_previous_layer_ffn.py \
  gdrive:DS4_Colab/build6_code/tune_build6_fixed_nnz_previous_layer_ffn.py

rclone copyto \
  test_predictor/prefetch_predictor_pipeline/ML_pipeline/neural_network/ffn_utils.py \
  gdrive:DS4_Colab/build6_code/ffn_utils.py
```

Il nuovo tuner importa quello storico per riusarne selezione a tre fasi,
checkpoint, pulizia dei checkpoint scartati e `--resume`. `ffn_utils.py`
contiene il builder della rete e la loss BCE + ranking pairwise: le tre copie
devono quindi rimanere coerenti. Gli input Build 6 sono gia' presenti in
`DS4_Colab/build6_inputs/` e non vanno ricaricati.

## 2. Allocare la GPU e montare Drive

In Colab scegliere **Runtime -> Change runtime type -> A100 GPU** e montare
Drive con una cella Python:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Google Drive e' lo storage persistente: conserva input, codice e risultati
anche quando il runtime viene eliminato. Il filesystem `/content`, invece, e'
veloce ma temporaneo.

## 3. Copiare input e codice nel disco locale di Colab

Eseguire una cella shell:

```bash
!mkdir -p /content/build6_inputs /content/build6_code

!rsync -a --info=progress2 \
  /content/drive/MyDrive/DS4_Colab/build6_inputs/ \
  /content/build6_inputs/

!rsync -a \
  /content/drive/MyDrive/DS4_Colab/build6_code/ \
  /content/build6_code/
```

L'addestramento legge continuamente hidden int8, KV, state e label. Leggerli
direttamente attraverso il mount FUSE di Drive rallenterebbe il training e
renderebbe i tempi meno stabili. `rsync` crea invece una copia sul disco locale
veloce del runtime. Va rieseguito quando si avvia un runtime nuovo; nello
stesso runtime e' incrementale e non ricopia i file gia' identici.

## 4. Avviare il tuning

In una cella Colab:

```bash
%cd /content/build6_code

!python tune_build6_expanded_ranking_previous_layer_ffn.py \
  --state-data-dir /content/build6_inputs/arrays \
  --fixed-nnz-state-data-dir /content/build6_inputs/02_current_previous_fixed_nnz_state \
  --current-quantized-hidden-data-dir /content/build6_inputs/05_current_hidden_int8 \
  --previous-quantized-hidden-data-dir /content/build6_inputs/08_previous_hidden_int8 \
  --current-kv-data-dir /content/build6_inputs/09_current_kv_f16 \
  --previous-kv-data-dir /content/build6_inputs/10_previous_kv_f16 \
  --token-position-data-dir /content/build6_inputs/token_position_inputs \
  --output-dir /content/build6_expanded_ranking_lambda1_margin005_v1 \
  --dropout 0.3 \
  --batch-size 4096 \
  --shuffle-block-size 65536 \
  --early-stopping-patience 3
```

Il wrapper fissa intenzionalmente `--ranking-loss-weight=1.0` e
`--ranking-loss-margin=0.05`: non aggiungere questi due argomenti al comando.
Questo evita che candidate o riprese della stessa run usino una loss diversa.

Il tuner campiona 30 configurazioni distinte con seed fisso, le porta a 3
epoche, promuove le migliori 15 fino a 7 epoche e infine le migliori 5 fino a
20 epoche. L'early stopping, con pazienza 3, puo' terminare prima una
candidata. La selezione resta basata su `val_recall_at_6`, poiche' e' la
metrica operativa che DS4 deve migliorare.

## 5. Salvare gli artefatti persistenti

Al termine della run, oppure prima di eliminare intenzionalmente il runtime,
eseguire:

```bash
!mkdir -p /content/drive/MyDrive/DS4_Colab/build6_results/expanded_ranking_lambda1_margin005_v1

!rsync -a --info=progress2 \
  /content/build6_expanded_ranking_lambda1_margin005_v1/ \
  /content/drive/MyDrive/DS4_Colab/build6_results/expanded_ranking_lambda1_margin005_v1/
```

Questo salva report, stato del tuning, progressi per epoca e checkpoint delle
candidate ancora rilevanti. Il tuner elimina automaticamente i checkpoint di
candidate scartate dopo ogni promozione; conserva quelli necessari a una
ripresa esatta e le cinque finaliste.

## Riprendere dopo un'interruzione

Prima copiare di nuovo gli artefatti persistiti da Drive nel nuovo runtime:

```bash
!rsync -a \
  /content/drive/MyDrive/DS4_Colab/build6_results/expanded_ranking_lambda1_margin005_v1/ \
  /content/build6_expanded_ranking_lambda1_margin005_v1/
```

Poi ripetere il comando della sezione 4 aggiungendo soltanto `--resume`:

```bash
  --resume
```

La configurazione, lo stato e i checkpoint devono coincidere con quelli
salvati; il tuner rifiuta appositamente una ripresa con parametri incompatibili.
