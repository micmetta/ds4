# Raccolta BUILD_DATASET_5: `kv_features_current`

## Obiettivo

BUILD_DATASET_5 raccoglie nuovamente gli stessi 1.000 prompt di BUILD_DATASET_4
senza modificare o sovrascrivere alcun artefatto esistente. La novità è la
cattura offline della riga KV corrente per ogni token e layer routed.

Il primo esperimento non usa PCA. I vettori raw da 512 dimensioni saranno
conservati per poter calcolare in seguito sia le statistiche current sia la
PCA, senza ripetere l'inferenza DS4.

```text
attn_norm -> kv_raw -> RMSNorm / RoPE -> kv -> KV cache -> attention
                                        |
                                        +-> cattura feature 5
                                        |
                                        +-> router del token corrente
```

La feature catturata è `kv(t,l)`, disponibile prima del router al token `t` e
layer `l`. Può quindi essere usata causalmente insieme al routing corrente per
predire gli expert del token `t+1`, senza usare informazioni future.

## Cosa viene scritto

Il collector usa un unico corpus e crea una sola directory raw:

```text
build_dataset_5/raw/
  routing/prompt_XXXX.routing.jsonl
  hidden/prompt_XXXX.ffn_norm.f32.bin
  kv/prompt_XXXX.kv.f32.bin
  logs/
  manifest.jsonl
```

Ogni record routed nel JSONL contiene un riferimento aggiuntivo:

```json
"kv_raw_ref": {
  "file": ".../prompt_XXXX.kv.f32.bin",
  "dtype": "float32",
  "shape": [512],
  "offset_bytes": 0,
  "source_tensor": "kv",
  "source_stage": "after_rmsnorm_rope"
}
```

Ogni vettore occupa `512 × 4 = 2.048 byte`; per circa 2,06 milioni di esempi
il volume KV raw è approssimativamente 3,9 GiB. I raw hidden `ffn_norm`
restano separati e invariati.

## Preparazione del binario

Da `~/ds4`, dopo avere terminato qualsiasi training che usa la GPU:

```bash
make ds4 dataset_runner
```

Questo ricompila `./ds4` con la cattura KV e il collector. Non avvia alcuna
inferenza e non cancella dataset o output precedenti.

## Corpus completo e raccolta dei 1.000 prompt

Il corpus unificato
`dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl` contiene i 1.000
prompt originali. V0, V1 e V2 rimangono invariati; ogni riga del nuovo file
aggiunge soltanto il campo `source_dataset`, usato poi dal dataset derivato per
conservare la provenienza originale.

La raccolta richiede un solo comando:

```bash
DS4_MODEL="gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" \
DS4_DATASET_PROMPTS_PATH="test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0_v1_v2_1000.jsonl" \
DS4_DATASET_CORPUS_ID="prefetch_prompts_v0_v1_v2_1000" \
DS4_DATASET_OUTPUT_DIR="test_predictor/prefetch_predictor_pipeline/build_dataset_5/raw" \
DS4_DATASET_BUILD_ID="build_dataset_5_kv" \
DS4_DATASET_CAPTURE_HIDDEN=1 \
DS4_DATASET_CAPTURE_KV=1 \
DS4_DATASET_ROUTER_MAX_SCORES=16 \
DS4_DATASET_MAX_TOKENS=64 \
./test_predictor/prefetch_predictor_pipeline/collector/dataset_runner
```

Il collector può essere rilanciato dopo un'interruzione nella stessa directory:
il manifest fa saltare i prompt già completati con successo. Non cancellare
directory parziali; bisogna rilanciare e correggere soltanto eventuali prompt
falliti.

## Verifica minima dopo la raccolta

Il risultato atteso è `failed=0`. I dataset derivati KV useranno
`--require-hidden --require-kv`: la pipeline si fermerà se manca anche un solo
riferimento per un evento routed.

## Passi successivi, dopo la raccolta

1. Creare il dataset derivato con `build_derived_dataset.py` e i flag
   `--require-hidden --require-kv`. Il campo `source_dataset` di ogni prompt
   viene conservato automaticamente.
2. Calcolare offline le statistiche `kv_features_current`: media, deviazione
   standard, RMS, norma L2 e massimo assoluto.
3. Integrare tali feature in una nuova ablation e confrontare la Recall@6 su
   validation con la configurazione senza KV.

Le stesse righe raw permetteranno poi di calcolare `kv_history_features`
confrontando token consecutivi dello stesso prompt e layer, senza una nuova
inferenza. PCA32/PCA64 resta una possibile ablation futura.
