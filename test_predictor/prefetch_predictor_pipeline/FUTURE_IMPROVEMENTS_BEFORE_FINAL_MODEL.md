# Miglioramenti da chiudere prima del modello finale

Questo file e' il registro unico dei potenziali miglioramenti alla definizione
dei prompt, alla raccolta, alla costruzione del dataset e alle feature. Gli
esperimenti intermedi possono proseguire senza modificare retroattivamente le
build gia' prodotte, purche' i risultati siano descritti come confronti tra
configurazioni intermedie. Prima dell'addestramento del modello finale, ogni
voce con stato **da fare** deve essere riesaminata e risolta, esplicitamente
scartata con motivazione, oppure mantenuta come limitazione nota.

## Regole del registro

- Non modificare silenziosamente una build gia' usata per una metrica: creare
  una nuova build e annotare qui cosa cambia.
- Una modifica ai prompt richiede una nuova inferenza DS4: non e' sufficiente
  tradurre il JSONL derivato, perche' token generati, routing e label dipendono
  dal testo effettivamente inferito.
- Per ogni nuova voce indicare: impatto atteso, intervento proposto, build o
  artefatti da rigenerare e criterio di verifica.

## 1. Rendere interamente inglese il corpus storico V0

**Stato:** da fare prima del modello finale.

Il corpus storico `prefetch_prompts_v0.jsonl` (100 prompt) contiene un prompt
che richiede esplicitamente una risposta in italiano. Tale prompt e' incluso
anche nel dataset combinato da 500 prompt (`V0 + V1`), pertanto il dataset
attuale non e' completamente inglese. Questa scelta e' intenzionalmente
mantenuta nelle build correnti per confrontare in modo equo i risultati gia'
ottenuti sul corpus da 100 prompt con quelli sul corpus da 500 prompt.

**Intervento proposto:** creare una nuova versione inglese dei prompt V0,
incluso il prompt che oggi richiede una risposta in italiano; poi rieseguire
l'inferenza DS4 e tutta la pipeline derivata per un corpus finale di 500
prompt interamente inglese.

**Da rigenerare:** raw collector, JSONL derivato, split per prompt (con una
politica di split documentata), PCA/artifacts se usati, preprocessing,
ablations, tuning e training finale.

**Verifica:** nessun prompt deve richiedere output in italiano; il manifest
deve identificare una nuova build e non sovrascrivere le build comparative
attuali.

## 2. Rendere fissa anche la codifica one-hot del layer

**Stato:** da fare prima del modello finale; nessun impatto numerico atteso
sulle matrici correnti.

Nella rappresentazione `multihot_256_layer_router_state`, i 16
`router_logit_expert_id_*` sono codificati correttamente con categorie fisse:
16 slot x 256 expert ID = 4.096 feature, indipendenti dagli ID osservati nel
train. Il `layer`, invece, usa ancora un `OneHotEncoder` fit sul train con
`handle_unknown="infrequent_if_exist"` e `min_frequency=41`.

Nelle build correnti tutti i 40 layer routed (3--42) sono molto frequenti nel
train: la matrice ha quindi gia' le attese 40 feature layer e nessun layer e'
stato raggruppato. Tuttavia, la semantica resta diversa da quella degli ID
router e dipende formalmente dalla distribuzione train.

**Intervento proposto:** sostituire l'encoder del layer con una codifica
one-hot a categorie fisse `3..42` (40 colonne), con gestione esplicita di
valori fuori intervallo come errore di validazione. Aggiornare summary e
documentazione per indicare che sia layer sia ID router hanno spazi categoriali
fissi.

**Da rigenerare:** le ablation che usano il layer (`02` e `03`) e i relativi
artefatti; quindi i training/tuning finali che li consumano.

**Verifica:** shape invariata: 296 feature per `routing_multihot + layer` e
4.408 per `routing_multihot + layer + router_state`; il summary non deve
attribuire `infrequent_if_exist` al layer o agli ID router.

## 3. Ablation del token ID con embedding GGUF congelato e proiezione appresa

**Stato:** input dell'ablation `05` preparati correttamente su BUILD_DATASET_3;
manca il training e la valutazione controllata.

Il campo `token_id` e' gia' disponibile nel record al tempo corretto: identifica
il token corrente `x_t` che DS4 sta elaborando mentre osserviamo `E_l(t)` e
prediciamo `E_l(t+1)`. La FFN `03` corrente non lo usa. Il token puo' portare
informazione lessicale utile a distinguere contesti con routing simile ma
prossimo token-layer diverso; e' inoltre disponibile a runtime senza dover
catturare nuovi tensori interni.

Non va pero' aggiunto come one-hot al vettore sparse da 4.408 feature: il
vocabolario e' grande e farebbe crescere inutilmente parametri e memoria del
primo layer Dense. Neppure una embedding appresa da zero sul solo train e'
ideale: validation e test contengono token ID mai osservati nel train, i cui
vettori sarebbero non allenati oppure ridotti a un OOV poco specifico.

La rappresentazione scelta usa l'embedding token pre-addestrato e congelato
del GGUF: `token_embd.weight` ha 129.280 righe da 4.096 dimensioni. Un layer
lineare della FFN, allenato solo sul train, proiettera' ogni vettore da 4.096
a 64 dimensioni. Il lookup GGUF esiste anche per token non presenti nel train
e quindi non richiede OOV o statistiche dagli split.

### Limite esplicito dell'artefatto offline e requisito runtime

Per contenere memoria e spazio disco, l'artefatto iniziale di BUILD_DATASET_3
non copia l'intera tabella da 129.280 token (circa 1,06 GB float16): conserva
soltanto gli embedding dei 4.637 `token_id` presenti nell'unione di train,
validation e test di quella build. Questo e' corretto per l'ablation offline,
perche' tutti gli esempi valutati hanno il proprio embedding congelato e non
vengono usate label o statistiche dei tre split.

Il lookup compatto non basta pero' da solo al predictor runtime finale: un
prompt futuro puo' contenere un `token_id` assente da BUILD_DATASET_3. Prima
del modello finale dovremo scegliere e implementare una delle due politiche:

1. caricare l'intera tabella token del GGUF, soluzione semplice ma con circa
   1,06 GB di memoria float16;
2. mantenere una cache compatta e recuperare dal GGUF l'embedding di un nuovo
   ID al primo utilizzo, soluzione piu' efficiente ma da progettare nel
   runtime DS4/predictor.

**Verifica richiesta:** il predictor runtime non deve mai sostituire
silenziosamente un `token_id` sconosciuto con un vettore casuale o OOV; deve
garantire un embedding GGUF valido per ogni ID del vocabolario supportato.

**Implementato:** BUILD_DATASET_3 conserva il lookup compatto in
`ml/token_embedding_artifacts/`. Lo script
`pipeline/token_features/prepare_token_embedding_inputs.py` prepara la nuova
ablation `05` senza duplicare le matrici sparse di `04`; gli script
`train_token_embedding_ffn.py` ed `evaluate_token_embedding_ffn.py` gestiscono
la FFN multi-input che mantiene invariato il ramo sparse corrente e aggiunge il ramo congelato
`token_id -> embedding GGUF 4096D -> Dense(64)`, piu' `token_position`
standardizzata. La proiezione 4096 -> 64 aggiunge 262.208 parametri
allenabili; il lookup originario resta float16 e congelato.

**Prossimo step registrato:** addestrare `05` con `hidden_layers=512`, ReLU,
dropout `0,3`, learning rate `0,00006997208` e batch size `4.096`; confrontare
la Recall@6 di validation con `0,4432`, ottenuta da `04` con gli stessi
iperparametri e la stessa batch size. Eseguire il test soltanto dopo aver
scelto la configurazione su validation.

**Da rigenerare:** non sono necessarie nuove inferenze DS4: `token_id` e' gia'
presente nel Parquet base. Servono il lookup GGUF, un dataset/input pipeline
dedicati, un modello FFN multi-input, tuning su validation e un output
sperimentale nuovo.

**Verifica:** confrontare Recall@6 validation contro la stessa FFN `03` senza
token ID, mantenendo identici split, router state e budget Top-6. Misurare
anche numero di parametri, latenza d'inferenza e memoria, perche' il predictor
finale deve restare economico.

## 4. Ablation delle `hidden_features` PCA-64

**Stato:** implementata e addestrata come ablation `04`; validation Recall@6
osservata: 0,4432 con batch size 4.096. Rimane da ripetere l'ablation `03` con
lo stesso batch per un confronto isolato.

BUILD_DATASET_3 contiene gia' `hidden_features.values`: una proiezione PCA
layer-wise a 64 dimensioni del vettore `ffn_norm` di dimensione 4.096,
disponibile al token corrente e al medesimo layer. La PCA e' fit soltanto sui
vettori train di ciascun layer e viene poi applicata invariata a validation e
test. La FFN `03` corrente non usa ancora questi 64 valori.

**Intervento proposto:** aggiungere le 64 componenti PCA come feature numeriche
standardizzate sul train al modello `routing + layer + router state`. Il totale
diventa 4.472 feature (4.408 + 64), evitando i circa 16,8 GB che servirebbero
per materializzare in float32 tutti i 4.096 valori raw su 1.025.400 esempi.

**Da rigenerare:** nessuna nuova inferenza; il JSONL PCA-64 e il Parquet base
esistono gia'. Servono una rappresentazione ML/ablation dedicata e training,
tuning e report in nuove directory.

**Verifica:** confrontare Recall@6 validation, parametri, memoria e latenza
con la FFN `03` da 4.408 feature; confermare che standardizzazione e PCA non
usino statistiche di validation o test.

## Nuove voci

### Proiezione supervisionata del raw hidden state: validazione da completare

**Stato:** implementata come nuova ablation separata, ma non ancora addestrata
né confrontata su validation. Usa `ffn_norm` raw 4096D, normalizzato con media
e deviazione standard calcolate esclusivamente sul train e separatamente per
ciascuno dei 40 layer. Una Dense lineare trainable comprime quindi il vettore
in una dimensione scelta tramite parametro. La stessa FFN conserva il lookup
GGUF congelato e una seconda proiezione trainable del token embedding.

**Motivazione:** la PCA mantiene direzioni ad alta varianza, non
necessariamente direzioni che migliorano la predizione degli expert. La
proiezione supervisionata può invece essere ottimizzata direttamente sulla
loss multi-label e sulla Recall@6.

**Verifica richiesta prima del modello finale:** confrontare su validation
almeno 32/64, 64/64 e 128/64 (hidden/token), mantenendo invariati dati, split,
batch e iperparametri. Valutare una sola configurazione selezionata sul test.
Misurare inoltre il costo runtime della lettura del raw hidden state e della
proiezione 4096D, perché il guadagno deve giustificare latenza e memoria.

**Ottimizzazione implementata per le run offline:** i raw hidden normalizzati
possono essere materializzati una volta in array contigui float16 (`07`) e
riusati da tutte le configurazioni. Gli epoch rimescolano l'ordine dei batch
contigui, non le singole righe: questo riduce l'I/O frammentato e mantiene
ogni esempio presente una volta per epoch. Per un confronto strettissimo con
le run storiche a row-shuffle, rieseguire una configurazione di controllo con
la medesima politica batch-shuffle.

**Variante INT8 da validare:** per evitare che i circa 15,8 GiB di hidden
float16 superino la RAM disponibile in WSL, è disponibile una materializzazione
simmetrica INT8 calibrata solo sul train. Gli input vengono dequantizzati in
float16 sulla GPU prima della proiezione Dense trainable, preservando
l'ottimizzazione supervisionata della matrice `4096 -> H_D`. Prima di adottarla
nel modello finale bisogna confrontare tempo per epoch, Recall@6, saturazione e
RMSE di ricostruzione con float16; INT8 riduce I/O e spazio, ma non garantisce
automaticamente maggiore velocità né assenza di perdita informativa.

Aggiungere qui ogni potenziale errore, leakage, ambiguita' di feature,
incoerenza tra train/validation/test o scelta che riduca semplicita',
riproducibilita' o confrontabilita' del modello finale.
