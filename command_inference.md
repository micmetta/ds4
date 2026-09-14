# Comando di riferimento per l'inferenza CUDA con SSD streaming

```sh
DS4_CUDA_MMQ=0 \
  /home/michele/ds4/ds4 \
  --cuda \
  --ssd-streaming \
  --ssd-streaming-cache-experts 8GB \
  -c 1024 \
  -n 64 \
  --nothink \
  --temp 0 \
  -p "Explain what a binary search is in one short paragraph."
```

## Cosa fa ogni opzione

| Opzione | Significato |
|---|---|
| `DS4_CUDA_MMQ=0` | Disabilita il tier CUDA MMQ usato per prefill/batched verify. Il decode a token singolo del nostro benchmark non usa quel tier. |
| `/home/michele/ds4/ds4` | Eseguibile DS4 locale. |
| `--cuda` | Usa il backend CUDA sulla RTX 4080 Laptop. |
| `--ssd-streaming` | Non mantiene tutti i pesi MoE in VRAM: gli expert necessari vengono caricati dallo storage quando servono. |
| `-c 1024` | Alloca una context window massima di 1024 token, che comprende prompt e token generati. |
| `-n 64` | Genera al massimo 64 nuovi token. La generazione può terminare prima in caso di EOS. |
| `--nothink` | Usa la modalità di risposta diretta, senza thinking mode. |
| `--temp 0` | Sampling greedy: a ogni step seleziona il token con logit più alto. È essenziale per confronti riproducibili. |
| `-p "…"` | Prompt one-shot da eseguire. Per prompt lunghi o renderizzati conviene usare `--prompt-file FILE`. |

## `--ssd-streaming-cache-experts 8GB`

Questa opzione fornisce a DS4 un budget di memoria per la componente routed/MoE dello SSD streaming. In termini generali il budget contribuisce al piano memoria per:

- una riserva di expert necessaria durante il prefill;
- la gestione dinamica degli expert durante il decode.

Non significa che DS4 userà al massimo 8 GiB di VRAM totale, né che caricherà immediatamente 8 GiB di expert. Restano necessarie altre allocazioni GPU:

- pesi residenti del modello;
- KV cache;
- attivazioni, buffer e scratch CUDA;
- embedding e tensori non-routed;
- workspace temporanei del runtime.

Nella configurazione osservata da 8 GB, il piano memoria riportava approssimativamente:

| Componente | Memoria pianificata |
|---|---:|
| Modello residente | 0,99 GiB |
| Riserva expert per prefill | 3,38 GiB |
| Parte dinamica prevista per expert | 4,62 GiB |
| KV cache | 0,10 GiB |
| **Totale pianificato** | **circa 9,08 GiB** |

La RTX 4080 Laptop ha circa 12 GiB di VRAM. Per questo 8 GB è una baseline prudente: impostare automaticamente 12 GB per gli expert potrebbe lasciare troppo poco spazio alle altre allocazioni CUDA e causare OOM. Più budget non garantisce inoltre più prestazioni: eventuali hit aggiuntivi devono compensare la maggiore pressione sulla VRAM.

### Precisazione sul backend CUDA attuale

Il significato generale del flag è quello descritto sopra, ma nel backend CUDA attuale non esiste ancora una vera cache persistente LRU per chiave `(layer, expert)`, equivalente a quella del backend Metal.

Nel decode CUDA gli expert scelti vengono normalmente caricati in uno scratch compatto e riscritto layer per layer. Quindi questo flag partecipa al memory planning di DS4, ma non equivale ancora a una cache CUDA persistente da 8 GiB che produca hit fra token.

Il banco sperimentale dei TEST_4, TEST_5 e TEST_6 è separato: conserva gruppi futuri in RAM pinned e, in caso di hit, evita la lettura storage sul percorso critico; non è una cache VRAM persistente dimensionata da questa opzione.

## Perché questa è la configurazione di riferimento

Il prompt è corto, quindi `-c 1024` evita di riservare KV cache inutilmente grande. `--temp 0` mantiene stabile la sequenza generata e permette di attribuire differenze tra run a I/O, cache e scheduling, non alla casualità del sampling.

Per i dataset del predictor useremo la stessa base CUDA + SSD streaming + cache expert da 8 GB, ma un runner dedicato invocherà DS4 una volta per prompt e abiliterà il tracing del routing.
