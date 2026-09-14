# Dataset per il predictor di expert prefetch

Questa cartella contiene il corpus di input e, progressivamente, gli artefatti necessari per costruire il dataset supervisionato del predictor MoE di DS4.

L'obiettivo del predictor è stimare, durante il processamento del token `t`, gli expert che il router reale selezionerà nello stesso layer per il token successivo:

```text
E_l(t) -> E_l(t+1)
```

La previsione serve soltanto al prefetch: il router reale resta l'unica fonte autoritativa degli expert eseguiti dal modello.

## Corpus V0

`prefetch_prompts_v0.jsonl` è il corpus sorgente V0. È un file JSONL UTF-8: ogni riga è un prompt indipendente, identificato da un `id`, con una `category` e il testo `prompt`.

```json
{"id": 1, "category": "computer_science", "prompt": "Explain what a binary search is in one short paragraph."}
```

JSONL è la sorgente canonica perché resta leggibile, modificabile con Git e processabile una riga alla volta dal futuro runner senza caricare l'intero corpus in RAM.

## Corpus V1 — 400 prompt in inglese

`prefetch_prompts_v1_400_english.jsonl` contiene 400 prompt aggiuntivi in
inglese. Gli ID vanno da 101 a 500 e quindi non collidono con gli ID 1–100 del
corpus V0. Il file usa lo stesso schema JSONL (`id`, `category`, `prompt`) ed è
compatibile con il collector senza conversioni.

Le trace V1 devono essere raccolte in una nuova directory raw e con un
`DS4_DATASET_CORPUS_ID` distinto. La successiva fusione con V0 avverrà prima
di PCA, preprocessing e split, così tutti gli artefatti statistici saranno
appresi coerentemente sull'intero corpus da 500 prompt.

| Misura | Valore |
|---|---:|
| Prompt unici | 100 |
| Categorie | 10 |
| Prompt per categoria | 10 |
| Caratteri totali | 7.721 |
| Lunghezza mediana | 74 caratteri |
| Lunghezza massima | 166 caratteri |

Le categorie sono `computer_science`, `programming`, `debugging_code_reasoning`, `mathematics`, `logic_reasoning`, `science`, `general_knowledge`, `language_tasks`, `text_transformation` e `sql_structured_output`.

V0 è intenzionalmente piccolo, bilanciato e composto da prompt brevi. Non è il dataset finale: serve a validare la pipeline di raccolta, la struttura dei trace e la prevedibilità temporale del routing prima di includere corpus più grandi e long-context.

## Cosa implementeremo

Un runner dedicato sotto `test_predictor/prefetch_predictor_pipeline/` eseguirà una inferenza DS4 separata per ogni record. Il processo separato evita contaminazione di contesto tra prompt e permette resume, log e fallimenti attribuibili al singolo `id`.

La configurazione iniziale replichera il percorso CUDA + SSD streaming usato nei test di prefetch:

```text
DS4_CUDA_MMQ=0
--cuda
--ssd-streaming
--ssd-streaming-cache-experts 8GB
-c 1024
-n 64
--nothink
--temp 0
```

Il runner renderà il prompt nel formato chat DS4, lo salverà temporaneamente e invocherà `ds4 --prompt-file`. Il decode sarà greedy (`--temp 0`) per mantenere la raccolta riproducibile.

La prima versione raccoglierà soltanto segnali già disponibili e poco costosi:

```text
prompt id e categoria
sequence id
decode step e token id
layer
expert top-6 selezionati
router weights
top candidate diagnostici e cutoff margin
```

Hidden state, feature KV, indexer e attention non fanno parte della V0. Saranno aggiunti dopo avere misurato quanto routing history e score del router predicono già il token successivo.

## Artefatti previsti

```text
prefetch_prompts_v0.jsonl          corpus sorgente leggibile
run manifest / log per prompt      stato, configurazione e ripresa della raccolta
routing trace JSONL compresso      dati grezzi, ispezionabili evento per evento
examples Parquet                   dataset supervisionato per analisi/addestramento
```

Il trace grezzo conterrà un record per `(sequence_id, decode_step, layer)`. Routing history e label future saranno derivate offline dalle righe consecutive della stessa sequenza: questo evita disallineamenti e permette di cambiare la lunghezza della history senza rieseguire le inferenze.

Il Parquet sarà il formato ottimizzato per addestramento offline e query per colonna; il JSONL compresso resterà il formato umano e diagnostico.

## Volume stimato degli esempi V0

Nei test correnti, un run con `-n 64` ha prodotto 63 eventi di decode. L'ultimo evento di ogni layer non ha un token successivo e non può quindi produrre una label `E_l(t+1)`.

```text
100 prompt
x 62 transizioni decode valide per prompt
x 43 layer
= 266.600 esempi supervisionati E_l(t) -> E_l(t+1)
```

| Tipo di dato | Stima V0 |
|---|---:|
| Eventi di routing grezzi | 270.900 (`100 x 63 x 43`) |
| Esempi token-successivo complessivi | 266.600 (`100 x 62 x 43`) |
| Esempi per singolo layer | 6.200 (`100 x 62`) |
| Transizioni temporali se i 43 layer sono feature di un solo esempio | 6.200 |

Il numero effettivo dipende dai token veramente generati: un EOS anticipato riduce eventi e label. Le stime usano il decode osservato in TEST_6 con la stessa lunghezza massima di generazione.

Con una history di quattro token:

```text
[E_l(t-3), E_l(t-2), E_l(t-1), E_l(t)] -> E_l(t+1)
```

la stima diventa circa `100 x 59 x 43 = 253.700` esempi, cioè 5.900 per singolo layer.

## Tempi stimati della raccolta V0

La raccolta eseguirà un processo DS4 per prompt, senza Run A/B/C, oracle o
prefetch: ogni record richiede una sola inferenza CUDA + SSD streaming con
trace del routing attivo. La stima usa le misure reali dei TEST_5/6 sulla
stessa macchina e con la stessa configurazione base.

| Componente per prompt | Stima | Base della stima |
|---|---:|---|
| Decode di circa 63 eventi | 68--84 s | 63 token a 0,93--0,75 tok/s osservati in Run C/B. |
| Prefill, inizializzazione e scrittura trace | 10--35 s | Differenza e variabilità osservate nei run completi con prompt brevi. |
| Totale prudente per prompt | 90--120 s | Include il costo del processo separato e dell'I/O. |
| 100 prompt V0, sequenziale | circa 2,5--3,5 h | `100 x 90--120 s`, senza errori o ripetizioni. |

Questa è una stima per la raccolta iniziale routing-only: expert top-6, pesi,
top candidate e cutoff margin. Il primo run reale su 5 prompt dovrà misurare
il tempo medio effettivo e aggiornare il manifest; solo dopo conviene lanciare
tutto il corpus con resume.

Le feature dense non hanno ancora un tempo affidabile. Salvare hidden state,
KV cache o attention completi per ogni `(token, layer)` aggiunge readback GPU,
serializzazione e molti più byte; potrebbe aumentare sensibilmente il tempo e
la dimensione del dataset. Per questo la raccolta deve procedere in due fasi:

```text
V0 completa: routing + score diagnostici, circa 2,5--3,5 h stimate
V1 feature dense: piccola subset, prima misura overhead e volume
V2 feature dense: corpus esteso solo per feature che migliorano Recall@6
```

Il tempo di esecuzione è una conseguenza della scelta di usare un processo
separato per prompt: è più lento di una sessione persistente, ma garantisce
assenza di contesto residuo, ripresa sicura e log attribuibili a ciascun
record. Non si devono eseguire inferenze concorrenti con questo modello.

## A cosa servira V0

V0 è sufficiente per verificare che la pipeline sia corretta e per confrontare baseline leggere:

- riuso dell'ultimo expert-set;
- frequenza degli expert recenti;
- Markov-1 e history corta;
- un piccolo predictor globale condizionato anche dal layer;
- Recall@6, precision, set overlap e potenziale cache-hit rate.

I 266.600 record non sono altrettanti esempi completamente indipendenti: provengono da 100 sequenze e 43 layer correlati per token. V0 non basta quindi per modelli complessi basati su hidden state, KV o attention, ma è il dataset giusto per decidere se il segnale del routing history merita una raccolta più ampia.

## Regole per la raccolta futura

- Train, validation e test vanno divisi per prompt o sequenza, mai per singolo token, per evitare leakage temporale.
- Il trace deve distinguere chiaramente prefill e decode; il target iniziale riguarda solo il decode autoregressivo.
- Ogni run deve salvare modello, configurazione, build DS4 e hash del corpus.
- Un errore, un prompt saltato o un EOS anticipato deve essere registrato nel manifest, non nascosto o riempito artificialmente.
