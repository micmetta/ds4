# Feature da raccogliere nel nostro dataset: stato semantico e computazionale

Questa è la parte centrale del dataset che vogliamo costruire. L'obiettivo non è limitarsi a salvare **quali expert sono stati scelti**, ma anche una fotografia compatta dello **stato interno del modello nel momento in cui quella scelta viene prodotta**.

La domanda che vogliamo poter studiare è:

$$
\boxed{
P(E_{t+1,l}\mid \text{routing passato},\ \text{router state},\ h_{t,l},\ \text{KV state},\ \text{indexer/attention state})
}
$$

Dove:

- $t$ = posizione/token corrente nel decoding;
- $l$ = layer corrente;
- $E_{t,l}$ = insieme degli expert selezionati al token $t$ nel layer $l$;
- $h_{t,l}$ = hidden state disponibile in quel punto del modello;
- `KV state` = feature estratte dalle rappresentazioni Key/Value già disponibili nel decoding;
- `indexer/attention state` = informazione su quali parti del contesto passato l'attention sta considerando rilevanti.

L'idea è confrontare progressivamente predictor sempre più ricchi e verificare se le feature semantico/computazionali aggiungono davvero informazione rispetto al semplice routing history.

---

## 0. `prompt_metadata`

Prima delle feature token-layer salviamo i metadati che permettono di ricondurre ogni trace al prompt originale del dataset V0. I record di `prefetch_prompts_v0.jsonl` contengono i campi `id`, `category` e `prompt`; il campo `id` è univoco nel V0 e costituisce la chiave primaria del mapping.

Nel JSONL finale questi metadati saranno rappresentati così:

```json
{
  "source_dataset": "prefetch_prompts_v0",
  "prompt_id": 42,
  "prompt_category": "computer_science",
  "sequence_id": "prefetch_prompts_v0_42_run_001"
}
```

- `source_dataset`: identifica il corpus sorgente e previene ambiguità quando aggiungeremo prompt provenienti da altri dataset;
- `prompt_id`: copia esatta del campo `id` del V0. Collega ogni record token-layer al suo prompt sorgente;
- `prompt_category`: copia del campo `category` del V0. È utile per analisi di copertura, statistiche e split per dominio;
- `sequence_id`: identifica una specifica esecuzione autoregressiva del modello su quel prompt. Se lo stesso prompt viene eseguito più volte, per esempio con seed o configurazioni differenti, ogni run ha un `sequence_id` diverso ma mantiene lo stesso `prompt_id`.

### Nota importante: metadati nel JSONL, non input automatici del predictor

`prompt_id`, `prompt_category`, `source_dataset` e `sequence_id` sono indispensabili per tracciabilità, controllo qualità e split corretti, ma **non saranno dati automaticamente in input al predictor**. In particolare, `prompt_id` non esisterebbe per un prompt nuovo e consentirebbe al modello di memorizzare esempi invece di imparare il routing.

Prima dell'addestramento eseguiremo un preprocessing che:

1. valida il mapping tra record JSONL e prompt V0 tramite `source_dataset` + `prompt_id`;
2. ricostruisce le finestre `routing_history` della dimensione $H$ scelta dal trace canonico;
3. seleziona soltanto le feature disponibili anche nel runtime del predictor e separa `target_experts_next_token` come label;
4. conserva i metadati per analisi e crea split train/validation/test raggruppati per `prompt_id`, evitando che run dello stesso prompt finiscano in split diversi.

`prompt_category` potrà essere provata in un ablation study soltanto se decidiamo di renderla realmente disponibile anche nel sistema finale; nella baseline non verrà usata come input predittivo.

---

## 1. `routing_history`

### Cosa rappresenta

È la cronologia degli expert selezionati nei token precedenti, **per uno specifico layer routed**. Non si devono confondere le posizioni dei token con i primi tre layer del modello: in DeepSeek-V4-Flash i layer hash $l=0,1,2$ usano una mappa deterministica `token → expert` per ogni token; i layer $l\geq3$ selezionano invece gli expert tramite il router MoE.

L'esempio seguente riguarda quindi il layer routed $l=17$ e una finestra di history $H=3$. Le righe rappresentano i routing reali già osservati; non sono i layer 0, 1 e 2.

```text
token 0, layer 17 -> [12, 31, 55, 81, 144, 203]
token 1, layer 17 -> [12, 31, 55, 90, 144, 211]
token 2, layer 17 -> [12, 44, 55, 90, 144, 211]
```

Quando l'inferenza raggiunge il layer 17 del token 3, la history disponibile è dunque quella dei token 0, 1 e 2. Il router reale calcola poi il routing del token corrente:

```text
router reale: token 3, layer 17 -> [12, 44, 55, 90, 144, 203]
```

Questo risultato viene aggiunto immediatamente alla history. A questo punto il `prefetch_predictor` può ricevere la history aggiornata e stimare gli expert del medesimo layer al token successivo:

```text
routing_history aggiornata: E(1,17), E(2,17), E(3,17)
predictor:                  stima E(4,17)
prefetch:                   avvia il trasferimento in RAM degli expert candidati di E(4,17)
```

Il predictor deve essere avviato appena il routing reale di $E_{t,l}$ è noto, quindi **durante** l'elaborazione del token $t$ e non soltanto alla sua conclusione. Il modello può nel frattempo proseguire con i layer successivi, produrre e campionare il token $t+1$, quindi iniziare a valutarlo. Quando raggiungerà il layer $l$, gli expert di $E_{t+1,l}$ che coincidono con la predizione potranno già essere presenti nella RAM del livello di prefetch.

Per i layer hash $l=0,1,2$ non serve invece un predictor: appena il token $t+1$ è stato campionato, la sua mappa deterministica fornisce esattamente gli expert richiesti. Conviene comunque tracciarli con il campo `is_hash`, ma trattarli separatamente dall'addestramento e dalla valutazione del predictor dei layer routed.

Formalmente, con una finestra di lunghezza $H$:

$$
E_{t-H+1,l},\ldots,E_{t,l}
$$

### Scelta iniziale: $H=1$

Per la prima versione del dataset conviene iniziare con $H=1$. Dopo avere osservato il routing reale del token 3 nel layer 17, l'input e la label del predictor sono semplicemente:

```text
routing_history = [E(3,17)]
target          = E(4,17)
```

Il predictor usa quindi soltanto il routing più recente per stimare gli expert del token successivo nello stesso layer. Questa è la baseline più semplice da raccogliere, addestrare e confrontare.

### Conservare il trace canonico per derivare history più lunghe

La `routing_history` non deve però essere l'unica informazione salvata. Il dataset base deve conservare una riga cronologica per ogni evento `(sequence_id, token_position, layer)`, con almeno `experts_t`; sono raccomandati anche `is_hash`, `token_id` e, quando disponibili, i pesi del router.

Con questo trace completo, una history più lunga può essere costruita in seguito con uno script offline, senza rieseguire il modello. Per esempio, dalla stessa sequenza sopra si può derivare una versione con $H=3$:

```text
routing_history = [E(1,17), E(2,17), E(3,17)]
target          = E(4,17)
```

Lo script dovrà raggruppare gli eventi per `sequence_id` e `layer`, ordinarli per `token_position` e costruire finestre di dimensione $H=1$, $H=3$, $H=8$ o altre. È essenziale non mescolare eventi appartenenti a prompt, sequenze o layer differenti.

### Perché è importante

È la baseline più naturale perché il routing MoE può avere **località temporale**: expert usati recentemente possono avere una probabilità maggiore di essere riutilizzati subito dopo.

Questa feature ci permette di rispondere alla prima domanda sperimentale:

> Quanto funziona il trace precedente come predictor?

È inoltre la feature più economica da raccogliere e usare.

### Formato JSONL della feature

Ogni record JSONL rappresenta un singolo evento `(sequence_id, token_position, layer)`. Il trace canonico mantiene sempre il routing reale corrente in `experts_t`; `routing_history` è invece la finestra derivata che verrà fornita al predictor. Per la baseline iniziale con $H=1$, il formato esatto della feature è:

```json
"routing_history": {
  "history_length": 1,
  "token_positions": [3],
  "expert_ids": [[12, 44, 55, 90, 144, 203]]
}
```

I singoli campi significano:

- `routing_history`: oggetto che contiene esclusivamente la finestra di routing usata come input dal predictor. È distinto da `experts_t`, che nel record canonico conserva il routing reale del token corrente.
- `history_length`: numero di eventi token-layer inclusi nella finestra. Qui vale `1`: il predictor vede soltanto il routing più recente. Il valore deve coincidere con la lunghezza di `token_positions` e di `expert_ids`.
- `token_positions`: posizioni assolute dei token a cui appartengono gli eventi della history. Qui `[3]` significa che la finestra contiene il routing reale prodotto dal modello durante l'elaborazione/decode del token 3. Per il medesimo layer, questo evento viene fornito al `prefetch_predictor` per stimare il routing che il modello produrrà durante l'elaborazione del token 4. I valori sono ordinati dal token più vecchio al più recente.
- `expert_ids`: routing reale associato a ciascuna posizione in `token_positions`. È una lista di liste: la lista interna all'indice `i` appartiene al token `token_positions[i]`. Qui `expert_ids[0]` è `[12, 44, 55, 90, 144, 203]`, cioè i sei expert selezionati dal router reale al token 3 per il layer del record.

Il `layer` non viene ripetuto nell'oggetto perché è già un campo del record JSONL esterno: tutta la history si riferisce sempre a quel solo layer. Per esempio, se il record ha `layer: 17`, questa struttura rappresenta $[E(3,17)]$ e servirà a predire $E(4,17)$.

Le liste in `token_positions` e `expert_ids` sono allineate per indice e ordinate dal token più vecchio al più recente. Con $H=1$ contengono un solo elemento; con una history $H=3$, per esempio, avrebbero tre posizioni e tre liste da sei ID. L'ordine degli expert nella lista viene mantenuto perché può contenere informazione aggiuntiva rispetto al solo insieme.

Poiché questa è la prima feature, il record JSONL completo fino a questo punto sarà:

```json
{
  "schema_version": "prefetch_dataset_v1",
  "source_dataset": "prefetch_prompts_v0",
  "prompt_id": 42,
  "prompt_category": "computer_science",
  "sequence_id": "prefetch_prompts_v0_42_run_001",
  "token_position": 3,
  "token_id": 12345,
  "layer": 17,
  "is_hash": false,
  "experts_t": [12, 44, 55, 90, 144, 203],
  "routing_history": {
    "history_length": 1,
    "token_positions": [3],
    "expert_ids": [[12, 44, 55, 90, 144, 203]]
  },
  "target_experts_next_token": [12, 44, 55, 90, 144, 211]
}
```

I campi del record esterno non ancora descritti sopra significano:

- `schema_version`: versione esplicita dello schema JSONL. Permette agli script di lettura di riconoscere senza ambiguità il formato del record e di rifiutare o migrare eventuali versioni incompatibili.
- `sequence_id`: identificatore della singola sequenza eseguita dal modello, normalmente una singola run di un prompt. Serve a costruire la history senza mescolare token provenienti da prompt o inferenze diverse.
- `token_position`: posizione assoluta del token attualmente elaborato nella sequenza. Qui `3` identifica il record relativo al decode del token 3; insieme a `sequence_id` e `layer` identifica l'evento di routing.
- `token_id`: ID nel vocabolario del token attualmente elaborato. Non sostituisce `token_position`: identifica il contenuto discreto del token, mentre la posizione identifica quando esso compare nella sequenza. È utile anche per ricostruire il routing deterministico dei layer hash.

  Nel record relativo alla posizione $t$, `token_id` è l'ID di $x_t$, cioè il token già disponibile all'inizio del decode corrente e che DS4 sta elaborando in tutti i layer. Durante questa elaborazione il router del layer $l$ produce `experts_t = E_l(t)`; al termine del passaggio il modello seleziona/genera il token successivo $x_{t+1}$. Nel decode seguente, $x_{t+1}$ diventerà il nuovo `token_id` corrente e il router produrrà $E_l(t+1)$, che è la label `target_experts_next_token` del record precedente.

  ```text
  token corrente già disponibile x_t  (token_id del record)
          |
          v
  elaborazione DS4 nel layer l  ->  experts_t = E_l(t)
          |
          v
  output head  ->  seleziona/genera x_(t+1)
          |
          v
  decode successivo su x_(t+1)  ->  E_l(t+1) = target_experts_next_token
  ```

  Quindi `token_id` **non** è l'ID del nuovo token che verrà generato al termine del record: quell'ID comparirà come `token_id` nel record alla `token_position` successiva. Nel primo decode dopo il prompt, il token corrente può provenire dal prompt renderizzato anziché essere stato appena generato dal modello; nei decode successivi è invece il token prodotto nel passo precedente.
- `layer`: indice del layer a cui appartiene questo evento. Qui `17`; tutte le liste in `routing_history` si riferiscono esclusivamente a questo stesso layer.
- `is_hash`: indica se il layer usa il routing deterministico token-to-expert (`true` per i layer hash 0–2) oppure il router MoE (`false` per i layer routed 3–42). Nel record mostrato è `false`, quindi gli expert sono prodotti dal router MoE.
- `experts_t`: i sei expert reali selezionati dal modello per il token e il layer del record, qui $E(3,17)$. È il trace canonico da cui si ricostruiscono history di qualunque lunghezza; nel caso $H=1$ coincide con l'unica lista dentro `routing_history.expert_ids`.
- `target_experts_next_token`: label supervisionata, ottenuta dal routing reale dello stesso layer al token successivo, qui $E(4,17)$. Non è una feature di input e non deve essere resa disponibile al predictor durante l'inferenza runtime.

Il trace canonico (`sequence_id`, `token_position`, `layer`, `is_hash`, `experts_t`) permette di ricostruire offline qualunque valore di $H$ senza rieseguire il modello.

---

## 2. `router_scores` / `router_logits`

### Cosa rappresentano

Il router non produce solamente gli ID degli expert scelti. Per ogni token e per ogni layer routed assegna prima uno **score** a tutti gli expert candidati; seleziona poi quelli con gli score più alti.

Concettualmente:

$$
h_{t,l}
\rightarrow
\text{Router}_l
\rightarrow
[r_1,r_2,\ldots,r_{N_e}]
\rightarrow
\text{Top-k expert}
\rightarrow
E_{t,l}
$$

Ogni elemento della formula ha un ruolo preciso:

- $t$ è la posizione del token corrente nella sequenza;
- $l$ è il layer corrente;
- $h_{t,l}$ è l'hidden state del token $t$ quando entra nella parte MoE del layer $l$: un vettore che riassume l'informazione elaborata fin qui sul token e sul suo contesto;
- $\text{Router}_l$ è la rete di gating del layer $l$, che trasforma $h_{t,l}$ in una valutazione di idoneità per ciascun expert;
- $N_e$ è il numero totale di routed expert candidati nel layer: nel nostro modello $N_e=256$;
- $[r_1,r_2,\ldots,r_{N_e}]$ è il vettore degli score, con $r_i$ associato all'expert $i$. Uno score più alto indica che il router considera quell'expert più adatto al token nel contesto corrente;
- $\text{Top-k}$ seleziona gli expert con i $k$ score più alti; nel nostro modello $k=6$;
- $E_{t,l}$ è il set, o la lista ordinata, dei sei expert che il router reale eseguirà effettivamente per il token $t$ nel layer $l$.

Con `router_logits` intendiamo preferibilmente gli score raw prodotti dal router prima di un'eventuale normalizzazione. Con `router_scores` possiamo invece indicare gli score o i pesi di gating associati agli expert selezionati: il formato esatto dovrà essere dichiarato nel trace per non confondere le due quantità.

Gli `expert_ids` ci dicono solamente **chi ha vinto**. Gli score ci dicono invece **con quanta convinzione** e quali altri expert erano vicini alla soglia di selezione.

### Raw logits e ranking finale non sono la stessa graduatoria

Nel trace BUILD_1, `router_logits` contiene la graduatoria dei raw logits.
Il routing reale non viene però deciso direttamente con quell'ordine: DS4
trasforma il valore del router in un valore pre-bias, applica il bias del layer
e ordina lo score finale:

$$
\text{raw logit}
\longrightarrow
\text{router value pre-bias}
\xrightarrow{+\,\text{router bias}}
\text{score finale}
\longrightarrow
\text{Top-6 / } E_{t,l}
$$

`top_candidates` nel trace raw è ordinato in base al quarto valore di ogni sua
lista; `experts` coincide con i suoi primi sei ID.
Ogni lista di `top_candidates` ha la struttura
`[ID expert, valore router pre-bias, bias, somma]`: questi sono valori per
posizione, non sotto-campi JSONL con nomi propri.
Non esiste una chiave JSONL chiamata `bias`: per una lista
`top_candidates[i]`, il bias è il terzo valore, cioè `top_candidates[i][2]`.
Il secondo valore è `top_candidates[i][1]`; il quarto,
`top_candidates[i][3]`, è la loro somma e determina l'ordine della lista.
`router_logits.expert_ids_ranked` è invece
ordinato solo per raw logit. I due ordini possono differire quando il bias
compensa o supera la differenza iniziale fra due expert.

Esempio reale osservato nel corpus BUILD_1:

```text
raw logits:
expert 15 = -2.57362795 > expert 90 = -2.61992455
quindi la graduatoria raw mette 15 prima di 90.

top_candidates:
expert 90 -> [90, 0.265103459, 9.00365162, 9.26875496]
expert 15 -> [15, 0.271091402, 8.98697281, 9.25806427]

calcolo per expert 90 (quinta lista, indice 4):
top_candidates[4][3] = top_candidates[4][1] + top_candidates[4][2]
                      = 0.265103459 + 9.00365162
                      = 9.268755079  (salvato come 9.26875496)

calcolo per expert 15 (sesta lista, indice 5):
top_candidates[5][3] = top_candidates[5][1] + top_candidates[5][2]
                      = 0.271091402 + 8.98697281
                      = 9.258064212  (salvato come 9.25806427)

Poiché 9.26875 > 9.25806, il quarto valore mette 90 prima di 15; quindi lo
stesso ordine compare in experts.
```

Questo è coerente e informativo: i raw logits descrivono lo stato interno
prima del bias, mentre `experts` è la decisione effettivamente eseguita dal
modello. Di conseguenza, con un limite come `MAX_SCORES=16`, **non è garantito
che tutti e sei gli expert selezionati siano inclusi nella Top-16 dei raw
logits**. Un expert oltre la sedicesima posizione raw può ricevere un bias
sufficiente a entrare nel Top-6 finale. Gli expert selezionati sono comunque
sempre disponibili nel campo separato `experts_t`; non va tentato di
ricostruire la label dai soli raw logits.

Il corpus BUILD_1 conferma il caso: nel decode, 15 eventi routed su 203.000
non contengono tutti i sei expert selezionati nella Top-16 raw; in 6 di tali
eventi solo uno dei sei compare nella Top-16, quindi gli altri cinque sono
oltre la sedicesima posizione raw. La loro posizione raw esatta non è
recuperabile dal trace con `MAX_SCORES=16`, ma il routing reale è conservato
interamente in `experts_t`. Inoltre `top_candidates` mantiene, per tutti i
sei selezionati, la lista `[ID expert, valore router pre-bias, bias, somma]`
che determina l'ordine finale.

Se in futuro vorremo fornire al predictor anche lo stato che determina
direttamente la scelta finale, aggiungeremo una feature separata basata su
`top_candidates`; non la confonderemo con `router_logits`, che resta la
feature dei soli valori raw.

Una mitigazione più semplice consiste nell'aumentare
`DS4_DATASET_ROUTER_MAX_SCORES` da `16` a `32`. In questo modo salviamo i raw
logits dei primi 32 expert e riduciamo la probabilità che un expert fuori
dalla Top-16 raw, ma promosso dal bias nel Top-6 finale, non abbia il proprio
logit raw disponibile al predictor. Non è una garanzia assoluta: il bias può
in teoria promuovere anche un expert oltre la posizione 32. Il compromesso è
il raddoppio della sola feature `router_logits` in termini di ID, valori,
serializzazione e spazio; routing reale e `top_candidates` non cambiano.

Esempio concettuale:

```text
Expert 12 -> 0.91
Expert 44 -> 0.87
Expert 55 -> 0.84
Expert 90 -> 0.81
Expert 144 -> 0.79
Expert 203 -> 0.76
Expert 31 -> 0.75   <- quasi selezionato
Expert 211 -> 0.74  <- quasi selezionato
```

### Perché è importante

Due token possono selezionare lo stesso set di expert ma trovarsi in stati del router molto diversi.

Caso A:

```text
expert scelti nettamente sopra tutti gli altri
```

Caso B:

```text
expert scelti con score quasi identici agli expert esclusi
```

Nel secondo caso il routing del token successivo potrebbe essere molto più instabile.

Gli score possono quindi fornire al predictor informazioni su:

- **confidenza del routing corrente**;
- expert quasi selezionati;
- possibili cambiamenti imminenti del routing;
- continuità o discontinuità dello stato del router.

### Formato JSONL della feature

La quantità di score memorizzata è controllata dal parametro di raccolta:

```text
DS4_DATASET_ROUTER_MAX_SCORES=16
```

Non è una soglia sul valore del logit: è il numero massimo di expert conservati, ordinati per score decrescente. Indicando con $M$ il valore configurato in `DS4_DATASET_ROUTER_MAX_SCORES`, il numero effettivo salvato è:

$$
K_{\mathrm{salvato}} = \max\bigl(6,\min(M,256)\bigr)
$$

Il limite viene forzato ad almeno sei per avere una rappresentazione minima
utile, ma questo **non** implica che i sei expert selezionati siano sempre
nella Top-K raw: la selezione usa lo score finale con bias descritto sopra.
Con $M=256$ ogni record routed conserva i logits di tutti i 256 expert;
valori come $M=16$ o $M=32$ mantengono molti expert con raw logit alto, ma
riducono lo spazio occupato.

Il formato esatto della sola feature 2 è:

```json
"router_logits": {
  "kind": "raw_logits",
  "max_scores_configured": 16,
  "stored_count": 16,
  "expert_ids_ranked": [12, 44, 55, 90, 144, 203, 31, 211, 7, 98, 16, 72, 33, 184, 5, 220],
  "values_ranked": [0.91, 0.87, 0.84, 0.81, 0.79, 0.76, 0.75, 0.74, 0.70, 0.68, 0.65, 0.63, 0.61, 0.59, 0.57, 0.55]
}
```

I singoli campi significano:

- `router_logits`: oggetto che conserva una porzione ordinata del vettore prodotto dal router per il token e il layer del record. È disponibile soltanto per i layer routed; per un layer hash il valore sarà `null`, perché non esiste un router MoE da cui estrarre logits.
- `kind`: semantica numerica di `values_ranked`. Nella prima versione vale `raw_logits`, cioè gli score prima della normalizzazione. Se in futuro salveremo pesi di gating normalizzati, il valore dovrà essere diverso, per esempio `normalized_gating_weights`: i due formati non devono essere mescolati nello stesso campo.
- `max_scores_configured`: valore $M$ richiesto dalla configurazione `DS4_DATASET_ROUTER_MAX_SCORES`. Qui è `16`: chiediamo di mantenere al massimo i 16 expert con score più alto.
- `stored_count`: numero di coppie ID-valore effettivamente presenti nel record. È il valore di $K_{\mathrm{salvato}}$ calcolato dalla formula precedente; qui vale `16`. Può differire dal valore configurato se quest'ultimo è minore di 6, perché il trace forza una capacità minima di sei, oppure se è maggiore di 256.
- `expert_ids_ranked`: ID degli expert salvati, ordinati dal raw logit più alto al più basso. Qui il primo ID è `12`, l'ultimo è `220`; con `max_scores_configured: 256` la lista contiene tutti gli ID degli expert, ma sempre nell'ordine di raw logit e non nell'ordine numerico degli ID. Con una Top-K minore di 256, la lista non coincide necessariamente con gli expert selezionati nel campo `experts_t`.
- `values_ranked`: valori di router associati agli ID nella stessa posizione. Qui `values_ranked[0] = 0.91` è il logit dell'expert `expert_ids_ranked[0] = 12`, mentre `values_ranked[6] = 0.75` appartiene all'expert `31`. La lunghezza deve coincidere esattamente con `stored_count` e con la lunghezza di `expert_ids_ranked`.

`expert_ids_ranked[i]` e `values_ranked[i]` sono quindi sempre allineati per indice. Questa rappresentazione conserva gli expert con raw logit più alto e gli expert appena esclusi in quella graduatoria; il routing realmente selezionato resta rappresentato in modo autoritativo e separato da `experts_t`.

Il record JSONL completo dopo l'aggiunta della feature 2 include anche la feature precedente `routing_history`:

```json
{
  "schema_version": "prefetch_dataset_v1",
  "source_dataset": "prefetch_prompts_v0",
  "prompt_id": 42,
  "prompt_category": "computer_science",
  "sequence_id": "prefetch_prompts_v0_42_run_001",
  "token_position": 3,
  "token_id": 12345,
  "layer": 17,
  "is_hash": false,
  "experts_t": [12, 44, 55, 90, 144, 203],
  "routing_history": {
    "history_length": 1,
    "token_positions": [3],
    "expert_ids": [[12, 44, 55, 90, 144, 203]]
  },
  "router_logits": {
    "kind": "raw_logits",
    "max_scores_configured": 16,
    "stored_count": 16,
    "expert_ids_ranked": [12, 44, 55, 90, 144, 203, 31, 211, 7, 98, 16, 72, 33, 184, 5, 220],
    "values_ranked": [0.91, 0.87, 0.84, 0.81, 0.79, 0.76, 0.75, 0.74, 0.70, 0.68, 0.65, 0.63, 0.61, 0.59, 0.57, 0.55]
  },
  "target_experts_next_token": [12, 44, 55, 90, 144, 211]
}
```

---

## 3. `token_embedding`

### Cosa rappresenta

È il vettore di embedding statico associato al `token_id` corrente. Il modello lo ottiene con una lookup nella propria tabella di embedding prima che il token venga trasformato dai layer successivi. A differenza dell'hidden state, lo stesso `token_id` produce sempre lo stesso embedding: non incorpora il contesto precedente né il comportamento dei router già osservati.

Questa feature può comunque essere utile al predictor come segnale semantico lessicale più ricco del solo ID numerico del token, in particolare come baseline per i primi layer. Non sostituisce `hidden_features`, che verrà invece calcolato layer per layer e sarà contestualizzato.

### Formato JSONL della feature

Per evitare di duplicare migliaia di valori uguali per ogni occorrenza dello stesso token, il trace JSONL conserva il `token_id` già presente nel record e una descrizione della lookup da eseguire nel preprocessing. La tabella degli embedding del modello verrà esportata una sola volta in un file separato, in formato binario efficiente.

```json
"token_embedding": {
  "kind": "model_token_embedding",
  "storage": "external_lookup",
  "embedding_table_id": "deepseek_v4_flash_token_embedding_gguf",
  "dimension": 4096,
  "token_id_source": "token_id"
}
```

- `kind`: dichiara che il vettore è l'embedding statico di input del modello, non un hidden state contestualizzato;
- `storage`: indica che i valori non sono duplicati nella riga JSONL ma sono recuperati da una tabella esterna;
- `embedding_table_id`: identifica in modo univoco la tabella esportata, inclusi modello e precisione usati;
- `dimension`: dimensione del vettore di embedding, pari a 4096 per questo modello;
- `token_id_source`: collega la lookup al campo `token_id` del record JSONL esterno.

Nel preprocessing, per ogni record, recupereremo il vettore dalla tabella tramite `token_id` e lo aggiungeremo alle feature del predictor. Questo evita di ripetere 4096 valori per record nel JSONL e preserva esattamente lo stesso embedding del modello. Se vorremo sperimentare una serializzazione completa nel JSONL, potremo aggiungere un'opzione dedicata, ma non è la scelta base.

### Nota: `token_id` nel preprocessing dei modelli classici

Nel primo confronto con i modelli classici di scikit-learn, prima di
materializzare il vettore embedding 4096D, useremo `token_id` come feature
**categorica** tramite one-hot encoding. La configurazione prevista è:

```python
OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=41,
    sparse_output=True,
)
```

La soglia `min_frequency=41` non è stata scelta osservando validation o test.
Deriva dalla struttura nota a priori di BUILD_1: per ciascun token di decoding,
il dataset contiene un record per ognuno dei 40 layer routed (layer 3–42).
Quindi un `token_id` osservato in una sola posizione di decoding del train
compare esattamente 40 volte nelle righe train:

```text
1 posizione di decoding × 40 layer routed = 40 righe train
```

Impostare la soglia a 41 significa pertanto raggruppare nel bucket
"infrequente" gli ID osservati in una sola posizione di decoding, evitando una
colonna one-hot dedicata a ciascuno di essi. La scelta è basata solo
sull'architettura fissa del modello e sulla struttura del trace, quindi non
introduce data leakage. L'encoder viene comunque sottoposto a `fit` solo sulle
righe train; validation, test e runtime usano esclusivamente `transform`.

### Record JSONL cumulativo dopo la feature 3

Questo è il formato del record non-hash nel primo dataset derivato, dopo avere aggiunto le feature 0–3. La raccolta raw conserva gli stessi metadati e il routing reale; `routing_history` e `target_experts_next_token` vengono costruiti in preprocessing ordinando gli eventi della stessa `sequence_id` e dello stesso `layer`.

```json
{
  "schema_version": "prefetch_dataset_v1",
  "source_dataset": "prefetch_prompts_v0",
  "prompt_id": 42,
  "prompt_category": "computer_science",
  "sequence_id": "prefetch_prompts_v0_42_run_001",
  "token_position": 3,
  "token_id": 12345,
  "layer": 17,
  "is_hash": false,
  "experts_t": [12, 44, 55, 90, 144, 203],
  "routing_history": {
    "history_length": 1,
    "token_positions": [3],
    "expert_ids": [[12, 44, 55, 90, 144, 203]]
  },
  "router_logits": {
    "kind": "raw_logits",
    "max_scores_configured": 16,
    "stored_count": 16,
    "expert_ids_ranked": [12, 44, 55, 90, 144, 203, 31, 211, 7, 98, 16, 72, 33, 184, 5, 220],
    "values_ranked": [0.91, 0.87, 0.84, 0.81, 0.79, 0.76, 0.75, 0.74, 0.70, 0.68, 0.65, 0.63, 0.61, 0.59, 0.57, 0.55]
  },
  "token_embedding": {
    "kind": "model_token_embedding",
    "storage": "external_lookup",
    "embedding_table_id": "deepseek_v4_flash_token_embedding_gguf",
    "dimension": 4096,
    "token_id_source": "token_id"
  },
  "target_experts_next_token": [12, 44, 55, 90, 144, 211]
}
```

`experts_t` e `routing_history` descrivono il routing reale del token 3 al layer 17; `target_experts_next_token` descrive il routing reale del token 4 nello stesso layer. Le liste di `router_logits` sono allineate per indice e ordinate per valore decrescente. Per i layer hash 0–2, `is_hash` sarà `true` e `router_logits` sarà `null`, perché la selezione deriva dalla mappa deterministica token-to-expert.

---

## 4. `hidden_features`

> **Versione del dataset:** questa feature viene introdotta in **BUILD_2**.
> BUILD_1 contiene le feature 0–3; BUILD_2 mantiene invariati prompt, target,
> routing, raw logits, token ID e split per prompt, aggiungendo solo `ffn_norm`
> e la sua PCA layer-wise.

### Cosa rappresentano

L'hidden state $h_{t,l}$ è la rappresentazione interna del token corrente nel layer $l$.

È molto più ricco del semplice token ID perché incorpora il **contesto elaborato fino a quel punto**.

Concettualmente:

```text
token t
   |
   v
layer precedenti
   |
   v
hidden state h(t,l)
   |
   +----> attention / altre trasformazioni
   |
   +----> router MoE
```

Il router decide gli expert proprio a partire da una rappresentazione interna di questo tipo. Per questo l'hidden state è probabilmente una delle feature più direttamente correlate alla scelta degli expert.

### Perché è importante

Il routing history dice:

> quali expert sono stati scelti prima?

L'hidden state può invece dire:

> che tipo di informazione semantica/computazionale sta elaborando il modello adesso?

Due token con lo stesso routing passato possono avere hidden state differenti e quindi richiedere expert differenti al passo successivo.

Questa feature è quindi fondamentale per verificare se il predictor può andare oltre la semplice località temporale.

### Perché confrontarlo con la KV cache

$K$ e $V$ sono proiezioni specializzate per l'attention. L'hidden state è invece una rappresentazione più generale.

Per questo conviene fare un'ablation del tipo:

```text
routing only
routing + hidden
routing + KV
routing + hidden + KV
```

Se `routing + hidden + KV` migliora rispetto a `routing + hidden`, avremo evidenza che la KV cache contiene informazione predittiva incrementale e non solo una trasformazione ridondante dello stesso stato.

### Punto di cattura nel percorso CUDA di DS4

Per ogni token $t$ e layer routed $l$, il percorso CUDA di DS4 è essenzialmente:

```text
stato dopo attention
→ ffn_cur
→ RMSNorm
→ ffn_norm
→ proiezione del router
→ router_logits[256]
→ Top-6 expert
```

`ffn_norm` è il candidato migliore per `hidden_features`: è esattamente il vettore che il router usa per decidere gli expert del token corrente. Catturarlo permette quindi di studiare se lo stato che determina $E(t,l)$ contiene anche segnale utile per predire $E(t+1,l)$.

### Raccolta raw nel modello

Nel backend CUDA il candidato da catturare è `ffn_norm`: il vettore normalizzato che entra direttamente nella proiezione del router. Per ogni evento token-layer routed è un tensore GPU contiguo:

$$
h^{\mathrm{router}}_{t,l}\in\mathbb{R}^{4096}
$$

Nel backend attuale è rappresentato come `float[4096]`, quindi occupa 16 KiB per evento. Deve essere copiato o proiettato subito dopo la normalizzazione e prima che il buffer venga riutilizzato dal layer successivo.

Per il V0 possiamo conservare l'intero trace raw offline in un file binario, per esempio `hidden_raw.f32.bin`, e salvare nel JSONL solo un riferimento alla riga binaria:

```json
"hidden_raw_ref": {
  "file": "hidden_raw.f32.bin",
  "dtype": "float32",
  "shape": [4096],
  "offset_bytes": 4128768,
  "source_tensor": "ffn_norm"
}
```

Il raw trace completo ci permette di ripetere qualunque esperimento di proiezione senza rieseguire l'inferenza. Non è invece il formato da passare direttamente al predictor runtime, perché un worker CPU non dovrebbe ricevere 16 KiB per ogni token e layer.

### Proiezione PCA per il dataset derivato

La prima compressione da sperimentare sarà una PCA separata per ogni layer routed. Gli hidden state hanno distribuzioni e significati differenti tra layer iniziali e finali; una PCA globale li mescolerebbe inutilmente.

Per ogni layer $l$, usando soltanto i record dello split train, calcoliamo:

- la media $\mu_l\in\mathbb{R}^{4096}$;
- le prime $d$ componenti principali $W_l\in\mathbb{R}^{4096\times d}$, con $d=32$ oppure $d=64$.

Ogni vettore raw viene poi trasformato in:

$$
z_{t,l}=W_l^\top\bigl(h^{\mathrm{router}}_{t,l}-\mu_l\bigr),
\qquad
z_{t,l}\in\mathbb{R}^{d}
$$

### Come funziona la PCA

**Questa sezione documenta soltanto i passaggi necessari per il nostro dataset. Creeremo in seguito un file di documentazione separato dedicato alla PCA, nel quale descriveremo e ripeteremo insieme la procedura completa, passaggio per passaggio, con la motivazione di ogni operazione.**

L'esempio seguente usa un solo layer, vettori artificiali di 4 dimensioni e una proiezione a 2 dimensioni. Nel caso reale sostituiremo 4 con 4096 e 2 con 32 oppure 64.

Supponiamo che, per il solo layer 17 e usando esclusivamente gli esempi train, siano stati raccolti questi quattro vettori:

$$
H_{\mathrm{train}}=
\begin{bmatrix}
5 & 4 & 2 & 0 \\
5 & 4 & 0 & 2 \\
1 & 0 & 2 & 0 \\
1 & 0 & 0 & 2
\end{bmatrix}
$$

Ogni riga è un hidden state; ogni colonna è una dimensione del vettore. La PCA calcola prima la media colonna per colonna:

$$
\mu_{17}=[3,2,1,1]
$$

Per un nuovo hidden state $h=[4,3,2,0]$, il primo passo è sottrarre la media:

$$
h-\mu_{17}=[1,1,1,-1]
$$

Per calcolare $W_{17}$, la PCA esegue concettualmente questi passaggi su **tutti** i vettori train del layer 17, non soltanto sul nuovo vettore $h$:

$$
H_c=H_{\mathrm{train}}-\mu_{17}
$$

$H_c$ è la matrice degli esempi train centrati: la media viene sottratta da ogni riga, componente per componente.

$$
C=\frac{1}{N-1}H_c^\top H_c
$$

$C$ è la matrice di covarianza: riassume quali dimensioni dell'hidden state tendono a variare insieme nei dati train.

La PCA calcola poi autovalori e autovettori di $C$, ordina gli autovettori per autovalore decrescente e usa i primi $d$ come colonne di $W_{17}$. Gli autovettori con autovalore più alto corrispondono alle direzioni in cui i dati variano maggiormente.

In questo esempio semplificato immaginiamo che le prime due direzioni trovate siano:

$$
W_{17}=
\begin{bmatrix}
0.707 & 0.000 \\
0.707 & 0.000 \\
0.000 & 0.707 \\
0.000 & -0.707
\end{bmatrix}
$$

La prima colonna descrive una variazione congiunta delle prime due dimensioni; la seconda una variazione opposta tra terza e quarta. La proiezione è il prodotto scalare del vettore centrato con ciascuna colonna:

$$
z=
\begin{bmatrix}
0.707 & 0.707 & 0.000 & 0.000 \\
0.000 & 0.000 & 0.707 & -0.707
\end{bmatrix}
\begin{bmatrix}
1 \\
1 \\
1 \\
-1
\end{bmatrix}
=
\begin{bmatrix}
(0.707\cdot1)+(0.707\cdot1)+(0\cdot1)+(0\cdot-1) \\
(0\cdot1)+(0\cdot1)+(0.707\cdot1)+(-0.707\cdot-1)
\end{bmatrix}
=
\begin{bmatrix}
1.414 \\
1.414
\end{bmatrix}
$$

In altre parole, ogni riga di $W_{17}^\top$ viene moltiplicata componente per componente per l'hidden state centrato e poi sommata: ogni somma produce un solo valore di $z$. Con 64 componenti PCA avremo 64 righe e quindi 64 prodotti scalari, uno per ciascun valore del vettore compatto.

Quindi il prodotto

$$
W_{17}^\top
\left(h^{\mathrm{router}}_{t,17}-\mu_{17}\right)
$$

trasforma l'hidden state centrato nello **spazio delle direzioni principali**
imparate dalla PCA. Non stiamo più conservando i quattro valori originali del
token: stiamo ottenendo le sue *coordinate PCA*, cioè quanto il token corrente
si discosta dal comportamento medio del layer lungo ciascuna direzione
importante.

Nel nostro esempio, il risultato $z=[1.414,\ 1.414]$ significa precisamente:

- il primo valore `1.414` misura una forte variazione congiunta positiva nelle
  prime due dimensioni rispetto alla media del layer;
- il secondo valore `1.414` misura una forte variazione lungo il contrasto tra
  terza e quarta dimensione rispetto alla media.

Questi due numeri sostituiscono quindi il vettore centrato di quattro valori:
mantengono le variazioni più informative per la popolazione di hidden state del
layer 17 e scartano le direzioni meno variabili. Nel caso reale, 64 coordinate
PCA riassumeranno il vettore raw `ffn_norm` di 4096 valori.

Abbiamo quindi trasformato:

```text
hidden raw:        [4, 3, 2, 0]     # 4 numeri
meno media layer:  [1, 1, 1, -1]    # 4 numeri
PCA:               [1.414, 1.414]  # 2 numeri
```

Nel caso reale non calcoleremo a mano né $\mu_l$ né $W_l$. Useremo una libreria PCA durante il preprocessing:

```python
from sklearn.decomposition import PCA

# Solo hidden state train del layer 17: shape [N, 4096]
H_train = hidden_raw_train_layer_17

pca = PCA(n_components=64)
pca.fit(H_train)

# Parametri che la libreria ha calcolato automaticamente.
mu_17 = pca.mean_                 # shape: [4096]
W_17 = pca.components_.T          # shape: [4096, 64]

# Proiezione di un nuovo hidden state del layer 17.
z = pca.transform(h.reshape(1, -1))  # shape: [1, 64]
```

Il `fit` della PCA e il `transform` che genera il dataset derivato vengono eseguiti **dopo** la raccolta, nel preprocessing. Durante l'esecuzione dei prompt DS4 deve soltanto catturare e salvare il vettore raw `ffn_norm`; non deve né calcolare la PCA né conoscere $\mu_l$ e $W_l$. Solo se decideremo di usare questa feature nel predictor runtime, applicheremo in seguito gli stessi parametri PCA fissi anche durante il decoding online.

La formula della proiezione PCA a cui si riferiscono i punti seguenti è:

$$
z_{t,l}=W_l^\top\left(h^{\mathrm{router}}_{t,l}-\mu_l\right),
\qquad z_{t,l}\in\mathbb{R}^{d}.
$$

I componenti della formula significano:

- $t$ è la posizione del token corrente;
- $l$ è il layer routed corrente;
- $h^{\mathrm{router}}_{t,l}$ è il vettore raw `ffn_norm` del token $t$ nel layer $l$, composto da 4096 valori;
- $\mu_l$ è il vettore medio, anch'esso di 4096 valori, calcolato sui soli hidden state dello split train del layer $l$;
- $h^{\mathrm{router}}_{t,l}-\mu_l$ è il vettore centrato: rimuove il comportamento medio del layer e conserva ciò che è caratteristico del token corrente;
- $W_l$ è la matrice PCA specifica del layer $l$. Le sue colonne sono le $d$ direzioni principali trovate nei dati del layer, cioè le direzioni lungo cui gli hidden state variano maggiormente;
- $d$ è la dimensione scelta per la feature compressa, inizialmente 32 oppure 64;
- $z_{t,l}$ è il vettore compatto finale. Ogni suo valore misura quanto l'hidden state corrente si trova lungo una delle direzioni principali della PCA.

Se scegliamo $d=64$, allora:

$$
W_l\in\mathbb{R}^{4096\times64},
\qquad
W_l^\top\in\mathbb{R}^{64\times4096}
$$

La trasformazione si legge quindi così:

```text
hidden raw:        4096 numeri
meno media layer:  4096 numeri
proiezione con W_l: 64 numeri
```

$W_l$ non è inizialmente una matrice imparata dal predictor. Viene calcolata una volta tramite PCA sui dati train del layer $l$, poi resta fissa e viene usata allo stesso modo per validation, test e runtime.

La PCA è la proiezione lineare che, a parità di dimensione $d$, conserva la massima varianza dei dati in senso di errore quadratico di ricostruzione. Va addestrata solo sui prompt dello split train: validation e test non devono contribuire né alla media né alle componenti principali.

Conservare varianza non garantisce automaticamente di conservare tutta l'informazione utile al routing futuro. Per questo PCA-32 e PCA-64 saranno baseline: se migliorano routing+logits, confronteremo in seguito una proiezione lineare supervisionata, ottimizzata direttamente per predire $E(t+1,l)$.

### Formato JSONL della feature derivata

Nel dataset derivato per training salveremo il vettore PCA compatto, non il raw vector:

```json
"hidden_features": {
  "source_tensor": "ffn_norm",
  "projection": "pca",
  "projection_id": "pca_layerwise_64_train_v1",
  "input_dimension": 4096,
  "output_dimension": 64,
  "values": [0.184, -1.021, 0.337, ..., 0.912]
}
```

`values` contiene esattamente 64 numeri nel dataset reale: nell'esempio sono mostrati i primi 3, i 60 valori intermedi abbreviati con `...` e l'ultimo valore. L'ellissi è solo illustrativa e non fa parte della riga JSONL effettivamente prodotta. `projection_id` identifica in modo univoco le medie e le matrici PCA usate, che verranno salvate una sola volta in un artefatto separato. Il runtime applicherà la stessa proiezione al tensor `ffn_norm` e passerà al predictor soltanto il vettore compatto.

---

## 5. `kv_features_current`: informazioni del token corrente dalla KV cache

### Cosa rappresentano

Durante il decoding, per ogni nuovo token il blocco di attention costruisce nuove rappresentazioni Key e Value e le aggiunge allo stato di cache utilizzato dai token successivi.

Nel caso classico possiamo pensare a:

$$
K_t = h_tW_K
$$

$$
V_t = h_tW_V
$$

Quindi $K_t$ e $V_t$ sono trasformazioni contestualizzate dello stato corrente, specializzate per l'attention.

Nel codice reale di DeepSeek-V4-Flash/DS4 l'attention può usare rappresentazioni compresse e strutture specifiche invece di una KV cache classica pura; per il dataset dovremo quindi mappare questi concetti sui **tensori effettivamente materializzati da DS4**.

### Il tensor effettivamente disponibile in DS4

Nel percorso CUDA di DS4, per ogni token $t$ e layer $l$, l'attention produce una riga KV temporanea lungo il percorso:

```text
attn_norm
→ kv_raw
→ RMSNorm / RoPE
→ kv
→ scrittura nella raw KV cache
→ attention
→ ffn_norm
→ router
```

Il candidato pratico da catturare è `kv`, cioè la riga KV del token corrente già normalizzata e preparata per la cache. La indichiamo con:

$$
u_{t,l}=\mathrm{kv}(t,l)\in\mathbb{R}^{D_{\mathrm{kv}}}
$$

`kv_raw` è invece la rappresentazione prima della normalizzazione. Per il primo esperimento raccoglieremo soltanto `kv`. Non assumeremo che il suo layout sia la semplice concatenazione standard $[K_t;V_t]$: DeepSeek-V4-Flash/DS4 usa attenzione e cache compresse, quindi `kv` va trattato come una riga KV interna del backend finché non avremo verificato una corrispondenza separata tra Key e Value.

#### Ruolo di `kv` durante il decoding autoregressivo

Il vettore `kv` non è un segnale diagnostico aggiuntivo: è la riga di cache effettivamente riutilizzata dall'attention di DS4. Per ogni nuovo token e per ogni layer, il backend calcola soltanto la nuova riga $u_{t,l}$; le righe dei token precedenti non vengono ricalcolate. La nuova riga viene inserita nella cache del layer e sarà disponibile ai token successivi:

$$
\mathrm{KVCache}^{(t)}_l =
[u_{1,l}, u_{2,l}, \ldots, u_{t,l}].
$$

Quando viene elaborato il token $t+1$, la sua query viene confrontata con le righe già memorizzate nella cache e l'attention recupera l'informazione rilevante. DS4 può inoltre comprimere o riassumere porzioni meno recenti della cache; ciò non modifica il fatto che $u_{t,l}$ sia il payload interno che il backend aggiunge e riusa.

Di conseguenza, i dati raccolti sono una sequenza temporale di righe KV compatte per ciascuna coppia *(token, layer)*, non un unico vettore per prompt e non una cache completa duplicata a ogni passo. Nel modello Flash usato qui ogni riga catturata ha 512 componenti float32. Questa è una variante architetturale della classica cache $K/V$: conserva il medesimo principio di riuso causale, ma non espone necessariamente due tensori separati chiamati Key e Value.

#### Confronto con la cache K/V classica

In un Transformer classico le due proiezioni sono distinte:

$$
K_t = h_t W_K,
\qquad
V_t = h_t W_V.
$$

Le matrici $W_K$ e $W_V$ producono vettori con ruoli diversi: la Key serve al confronto con la query corrente, mentre la Value è l'informazione che viene pesata e aggregata. Per questo il caso classico memorizza due rappresentazioni per token nella cache.

Nel percorso DeepSeek-V4-Flash osservato in DS4, invece, una sola proiezione `attn_kv` trasforma lo stato di attention normalizzato da 4096 dimensioni in un vettore compatto:

$$
\underbrace{h^{\mathrm{attn}}_{t,l}}_{4096\mathrm{D}}
\xrightarrow{\;W_{KV}\;}
\underbrace{kv\_raw_{t,l}}_{512\mathrm{D}}
\xrightarrow{\mathrm{RMSNorm+RoPE}}
\underbrace{u_{t,l}}_{512\mathrm{D}}.
$$

Non è quindi corretto interpretare le 512 componenti come due metà $[K_t;V_t]$. Il modello è progettato e addestrato affinché una singola rappresentazione latente compatta sia sufficiente al proprio meccanismo di attention. L'ottimizzazione non è soltanto l'uso di un unico contenitore di cache: riduce soprattutto la quantità di dati conservati per token e per layer.

All'aumentare del contesto, la cache cresce nel **numero di righe** memorizzate, non nella dimensionalità della singola riga. Ridurre il payload per riga consente quindi di mantenere più token passati a parità di memoria e di ridurre il traffico memoria dell'attention. In forma concettuale:

$$
\text{cache classica} \sim [K_t,V_t]
\qquad\longrightarrow\qquad
\text{cache DS4} \sim u_{t,l}\in\mathbb{R}^{512}.
$$

Questa differenza è un adattamento architetturale rispetto alla formulazione classica della tesi: il principio causale rimane invariato, ma le feature vengono estratte dalla rappresentazione KV compatta effettivamente usata dal backend.

#### Origine della rappresentazione compatta e ruolo di DS4

La proiezione unica `attn_kv` e la sua dimensionalità 4096D $\rightarrow$ 512D sono una scelta dell'architettura DeepSeek-V4-Flash e dei pesi contenuti nel checkpoint GGUF. Il backend non può sostituirla liberamente con due proiezioni classiche $W_K$ e $W_V$ senza modificare il modello e quindi il suo comportamento in inferenza.

DS4, invece, è responsabile dell'implementazione efficiente di questa architettura: calcola la proiezione, conserva le righe nella cache e le fornisce ai kernel di attention. Le ulteriori politiche di esecuzione — per esempio il formato della cache, la gestione di parti raw o compresse e le fusioni CUDA/Metal — appartengono al backend DS4. In sintesi:

- **DeepSeek-V4-Flash** definisce la rappresentazione KV compatta e la matrice $W_{KV}$ appresa durante il pre-training;
- **DS4** implementa il percorso di inferenza che calcola, memorizza e riusa quella rappresentazione, applicando ottimizzazioni senza alterarne il significato architetturale.

### Significato intuitivo di K e V nel caso classico

La Key descrive, in modo molto semplificato, **come il token corrente può essere riconosciuto/confrontato dalle query di attention future**.

È quindi una rappresentazione utile per stabilire la rilevanza del token all'interno del contesto.

### Significato intuitivo di V

La Value rappresenta il **contenuto informativo che può essere recuperato** quando l'attention assegna peso a quella posizione.

K e V hanno quindi ruoli differenti:

```text
K -> "quando questa posizione è rilevante?"
V -> "quale informazione porta questa posizione?"
```

### Perché la rappresentazione KV può aiutare il predictor

Il routing futuro può dipendere dal contenuto semantico che il modello sta elaborando.

Nel caso classico, K e V catturano una parte di questo stato con una prospettiva diversa dall'hidden state:

- K descrive la collocazione del token nello spazio usato per il matching dell'attention;
- V descrive l'informazione che il token rende disponibile al contesto futuro.

Nel caso DS4 queste due funzioni non vengono osservate come feature separate: analizziamo invece la riga `kv` compatta, che è la rappresentazione interna effettivamente conservata e riusata dall'attention.

La domanda di ricerca diventa:

> la rappresentazione KV interna dell'attention al token $t$ contiene informazione utile a prevedere gli expert del token $t+1$?

### Vincolo temporale fondamentale

Per predire gli expert del token $t+1$ possiamo usare solo informazioni già disponibili al token $t$. Nella notazione classica:

$$
K_{\leq t},V_{\leq t}
$$

Nel backend DS4, lo stesso vincolo equivale a poter usare le righe `kv` fino a $u_{t,l}$ ma non la riga futura. Non possiamo usare:

$$
u_{t+1,l}
$$

perché a quel punto il token successivo è già in elaborazione e il prefetch arriverebbe troppo tardi.

### Feature current da calcolare

La prima versione può calcolare direttamente dalla riga $u_{t,l}$ poche statistiche molto economiche:

$$
\lVert u_{t,l}\rVert_2,
\qquad
\mathrm{RMS}(u_{t,l}),
\qquad
\mathrm{mean}(u_{t,l}),
\qquad
\mathrm{std}(u_{t,l}),
\qquad
\max\lvert u_{t,l}\rvert
$$

Queste quantità descrivono l'intensità e la distribuzione della rappresentazione KV corrente. Da sole sono una baseline leggera; per conservare più informazione, raccoglieremo anche la riga raw `kv` in un file binario offline e costruiremo una PCA separata per layer, con la stessa procedura usata per `hidden_features`:

$$
z^{\mathrm{KV}}_{t,l}=P_l^\top\bigl(u_{t,l}-\nu_l\bigr),
\qquad
z^{\mathrm{KV}}_{t,l}\in\mathbb{R}^{32\ \mathrm{o}\ 64}
$$

$\nu_l$ e $P_l$ sono rispettivamente media e componenti PCA calcolate sui soli record train del layer $l$. Questo ci permette di confrontare in modo pulito `routing + hidden` contro `routing + hidden + KV`.

### Vincolo temporale e uso nel prefetch

La riga `kv(t,l)` esiste prima del router del token $t$ nello stesso layer. Può quindi alimentare una predizione molto anticipata di $E(t+1,l)$ usando la routing history disponibile fino a $t-1$. Dopo che il router ha prodotto $E(t,l)$, la predizione principale viene aggiornata con il routing corrente e può usare anche la feature KV già raccolta.

### Formato JSONL della feature

Nel trace raw salveremo il riferimento binario alla riga `kv`; nel dataset derivato conserveremo statistiche e/o proiezione PCA:

```json
"kv_features_current": {
  "source_tensor": "kv",
  "source_stage": "after_rmsnorm_rope",
  "raw_ref": {
    "file": "kv_raw.f32.bin",
    "dtype": "float32",
    "shape": ["D_kv"],
    "offset_bytes": 0
  },
  "stats_current": {
    "l2_norm": 18.42,
    "rms": 0.77,
    "mean": -0.01,
    "std": 0.77,
    "max_abs": 3.12
  },
  "projection": "pca",
  "projection_id": "kv_pca_layerwise_32_train_v1",
  "values": [0.18, -0.42, 0.07, ..., 0.31]
}
```

Per il primo esperimento del `build_dataset_5`, il dataset derivato conserva `kv_raw_ref` e aggiunge soltanto `stats_current`; non viene eseguita alcuna PCA. La riga raw resta disponibile per future proiezioni o feature storiche, mentre il predictor usera' le cinque statistiche dopo una normalizzazione fit esclusivamente sul train nel preprocessing ML. La PCA KV sara' valutata e documentata separatamente.

---

## 6. `kv_history_features`: feature ricavate dalla storia della KV cache

### Cosa rappresentano

La KV cache contiene il contesto già elaborato, ma usarla integralmente come input sarebbe troppo costoso. Questa feature è collegata alla sezione 5: invece di guardare soltanto la riga KV corrente $u_{t,l}$, misura come tale riga cambia rispetto alle righe KV dei token precedenti nello stesso layer.

Nel DS4 attuale la statistica direttamente fattibile è la similarità coseno tra le righe `kv` realmente materializzate:

$$
\mathrm{cos}^{\mathrm{KV}}_{t,l}=
\frac{u_{t,l}\cdot u_{t-1,l}}
{\lVert u_{t,l}\rVert_2\,\lVert u_{t-1,l}\rVert_2+\varepsilon}
$$

e la distanza tra token consecutivi:

$$
\Delta^{\mathrm{KV}}_{t,l}=\lVert u_{t,l}-u_{t-1,l}\rVert_2
$$

Possiamo calcolare le stesse quantità su una breve finestra $H_{KV}$, per esempio gli ultimi 4 token, e salvarne media, minimo, massimo e ultimo valore. Non serve leggere o serializzare l'intera cache, che in DS4 comprende raw sliding window degli ultimi 128 token e storia compressa più vecchia.

### Cosa ci dicono

- `kv_cosine_prev`: quanto la riga KV corrente è simile alla riga del token precedente nello stesso layer;
- `kv_delta_l2_prev`: quanto cambia in valore assoluto la riga KV corrente rispetto alla precedente;
- `kv_relative_delta`: la stessa variazione normalizzata per l'energia della riga corrente;
- statistiche sulla breve finestra: se il cambiamento è transitorio, stabile o improvviso.

Una variazione brusca di queste quantità potrebbe segnalare un cambiamento del tipo di informazione elaborata e quindi anticipare un cambio di expert.

### Possiamo calcolare separatamente $\cos(K_t,K_{t-1})$ e $\cos(V_t,V_{t-1})$?

Concettualmente sì: sarebbero feature interessanti perché distinguerebbero stabilità nello spazio di matching (Key) e stabilità nel contenuto recuperabile (Value). Tuttavia non è corretto calcolarle subito dividendo arbitrariamente `kv` in due metà. Nel percorso DeepSeek-V4-Flash/DS4 la riga `kv` è una rappresentazione interna compatta e il suo layout non è stato ancora verificato come coppia esplicita $[K;V]$.

Quindi, per la prima implementazione:

- calcoliamo `kv_cosine_prev` e `kv_delta_l2_prev` direttamente su `kv`: sono fattibili con il codice attuale;
- lasciamo `cos(K_t,K_{t-1})` e `cos(V_t,V_{t-1})` come estensione condizionata;
- li aggiungeremo solo se individueremo nel backend tensori Key e Value separati, oppure documenteremo con certezza come espandere `kv` in K e V senza introdurre una ricostruzione costosa.

Espandere artificialmente K e V solo per il predictor rischierebbe infatti di aggiungere più overhead del beneficio di prefetch.

### Perché sono interessanti

Consentono di catturare **dinamica semantica** senza salvare migliaia di valori della cache.

Per esempio:

```text
routing quasi invariato
ma kv_cosine_prev cala e kv_delta_l2_prev cresce
        |
        v
possibile cambio di expert al prossimo token
```

Questa è esattamente una situazione che un predictor basato solo sul routing potrebbe non vedere.

### Formato JSONL della feature

```json
"kv_history_features": {
  "source_tensor": "kv",
  "history_length": 4,
  "cosine_prev": 0.91,
  "delta_l2_prev": 4.86,
  "relative_delta": 0.26,
  "cosine_window": [0.88, 0.94, 0.91],
  "k_v_separate_available": false
}
```

`k_v_separate_available: false` rende esplicito che questo primo schema usa la riga `kv` reale di DS4 e non pretende di fornire cosine separate per Key e Value. Se in futuro il backend esporrà K e V separati, potremo aggiungere campi distinti senza cambiare il significato dei dati già raccolti.

---

## 7. `indexer_features`

### Cosa rappresentano

DeepSeek-V4-Flash/DS4 usa meccanismi di attention compressa/selettiva in cui una componente di **indexing/selezione** può individuare quali porzioni della storia compressa sono più rilevanti per il token corrente.

Concettualmente:

```text
KV history compressa
        |
        v
     indexer
        |
        +--> posizione/row A
        +--> posizione/row B
        +--> posizione/row C
        +--> ...
```

Queste selezioni indicano **quali parti del contesto passato il modello sta cercando di recuperare in quel momento**.

### Perché sono importanti

Sono potenzialmente una feature molto forte perché non descrivono solo il token corrente, ma anche **quale parte della memoria contestuale è diventata rilevante per esso**.

Due token con hidden state simili potrebbero interrogare regioni molto diverse del contesto precedente.

Quindi l'indexer potrebbe aggiungere informazione del tipo:

> il modello sta tornando a parlare di una parte di contesto associata a codice, matematica, un tool result, una vecchia istruzione, ecc.

Questa variazione del contesto recuperato potrebbe anticipare un cambiamento nel routing MoE.

### Come salvarle

Possibili feature:

- ID/indici delle righe selezionate;
- ranking delle righe selezionate;
- score associati alla selezione, se disponibili;
- overlap fra selezioni al token $t$ e $t-1$;
- rappresentazione compressa dell'insieme selezionato.

Per evitare dipendenza eccessiva da indici assoluti, potrebbe essere utile derivare anche feature relative, come il grado di overlap tra i set selezionati consecutivamente.

---

## 8. `attention_features`

### Cosa rappresentano

Questa categoria raccoglie feature riassuntive dello stato dell'attention che non coincidono necessariamente con K/V o con l'indexer.

L'obiettivo non è salvare matrici di attention complete, ma estrarre segnali economici che dicano **quanto e dove il modello sta concentrando l'attenzione**.

Nel nostro progetto questa categoria va considerata opzionale e dipendente da ciò che DS4 materializza senza overhead eccessivo.

### Perché potrebbero aiutare

Il routing MoE viene eseguito dopo che il token è stato trasformato dai blocchi precedenti del layer. Se l'attention modifica in modo significativo la rappresentazione del token, un riassunto del suo comportamento può anticipare il tipo di expert richiesto dalla successiva FFN/MoE.

È quindi una famiglia di feature da testare solo dopo routing, router scores, hidden e KV/indexer, che sono più direttamente motivate.

---

## 9. `token_id`, `token_position`, `layer` e metadati del prompt

Queste non sono feature semantiche interne in senso stretto, ma sono indispensabili per interpretare correttamente il dataset.

### `token_id`

Identifica il token corrente.

Può essere usato come baseline per capire quanto del routing sia prevedibile semplicemente dal token lessicale rispetto allo stato contestualizzato.

### `token_position`

Indica la posizione nel decoding.

È utile per distinguere l'inizio della generazione da fasi più avanzate e per studiare se la prevedibilità cambia con la lunghezza del contesto.

### `layer`

È indispensabile perché la distribuzione degli expert e il significato delle rappresentazioni interne cambiano tra layer.

### `category`, `mode`, `source`

Provengono soprattutto dal corpus DS4 di Sanfilippo.

Ci permettono di fare split e analisi per:

- coding;
- agentic/tool use;
- reasoning;
- language;
- translation;
- think/nothink;
- long context.

Questi campi sono importanti per verificare che il predictor non impari semplicemente pattern specifici di un dominio.

---

## 10. Target principale: `target_experts_next_token`

### Cosa rappresenta

È la label principale del predictor token-ahead:

$$
Y_{t,l}=E_{t+1,l}
$$

Il record associato allo stato del token $t$ contiene quindi gli expert realmente usati al token successivo nello stesso layer.

### Perché è la label corretta per il prefetch token-ahead

Se al token $t$ riusciamo a predire gli expert di $t+1$, possiamo iniziare il caricamento mentre il modello sta ancora completando il lavoro corrente.

```text
stato al token t
      |
      v
  predictor
      |
      v
expert previsti per t+1
      |
      v
prefetch SSD/RAM -> VRAM
```

---

## 11. Target alternativo: prediction layer-ahead

Oltre alla prediction token-ahead, abbiamo discusso una seconda strategia:

$$
(l,t)\rightarrow(l+k,t)
$$

cioè usare lo stato disponibile in un layer precedente dello **stesso token** per predire gli expert richiesti da un layer successivo.

Esempio:

```text
token t

Layer 10
  |
  +--> hidden/KV/indexer features
             |
             v
         predictor
             |
             v
    expert previsti Layer 15
             |
             v
         prefetch

Layer 11
Layer 12
Layer 13
Layer 14
Layer 15 -> expert già pronto
```

Per supportare anche questo esperimento il dataset dovrebbe poter generare label del tipo:

```text
"target_layer_offset": 5
"target_experts_same_token": [...]
```

### Perché è importante

Il layer-ahead predictor potrebbe avere **più tempo reale per nascondere l'I/O** rispetto alla sola prediction del token successivo.

Per questo vale la pena progettare il dataset in modo da non escludere questa seconda formulazione.

---

## 12. Feature operative del runtime: utili per il prefetch, ma da separare dal predictor semantico

Lo stato del sistema può includere:

```text
expert già in VRAM?
expert in RAM?
expert solo su SSD?
prefetch già in corso?
```

Queste feature sono molto importanti per decidere **quale expert conviene prefetchare**, ma non descrivono quale expert il modello sceglierà.

Per questo conviene separare due problemi:

### Predictor di routing

$$
\text{stato semantico/computazionale}
\rightarrow
P(E_{future})
$$

### Policy di prefetch

$$
P(E_{future})
+
\text{stato VRAM/RAM/SSD}
\rightarrow
\text{azione di prefetch}
$$

Questa separazione è importante per non confondere l'accuratezza del predictor con l'efficacia della policy di caching.

---

# Ablation study consigliata sulle feature

La maniera più pulita di capire quali informazioni aggiungano davvero potere predittivo è addestrare una serie incrementale di modelli.

| Predictor | Feature disponibili | Domanda a cui risponde |
|---|---|---|
| **P0** | routing corrente | Quanto vale la semplice località? |
| **P1** | routing history | Quanto aiuta una finestra temporale? |
| **P2** | P1 + router scores | La confidenza/competizione del router anticipa cambi futuri? |
| **P3** | P2 + hidden features | Lo stato interno generale aggiunge informazione? |
| **P4** | P2 + KV current features | K/V contengono segnale predittivo utile senza hidden? |
| **P5** | P3 + KV current/history | La KV aggiunge informazione oltre l'hidden state? |
| **P6** | P5 + indexer features | Quali parti del contesto recuperate aiutano la prediction? |
| **P7** | P6 + attention features | Ulteriori segnali dell'attention migliorano ancora? |

La comparazione più importante è:

$$
\text{routing + hidden}
\quad\text{vs}\quad
\text{routing + hidden + KV}
$$

Se il secondo modello migliora in modo consistente, allora avremo evidenza che **la KV cache contiene informazione incrementale sul routing futuro**.

---

# Principio fondamentale: feature utili ma economiche

Il predictor deve accelerare DS4, non diventare un nuovo collo di bottiglia.

Per questo non basta chiedersi:

> questa feature aumenta Recall@k?

Bisogna chiedersi:

> aumenta Recall@k abbastanza da compensare il costo per estrarla, copiarla e processarla?

Il percorso ideale è quindi:

```text
segnale interno già disponibile in DS4
              |
              v
feature extraction molto economica
              |
              v
rappresentazione compatta
              |
              v
piccolo predictor
              |
              v
prefetch anticipato
```

Le feature raw possono essere raccolte su una subset offline per capire cosa funziona; il predictor finale dovrebbe invece usare solo feature che possano essere ottenute con overhead molto basso.

---

# Schema consigliato del nostro record di training

Una struttura più completa, capace di supportare gli ablation study descritti sopra, potrebbe essere:

```json
{
  "prompt_id": "...",
  "category": "source",
  "mode": "nothink",
  "source": "ds4.c:14",

  "sequence_id": 27,
  "token_position": 413,
  "token_id": 12345,
  "layer": 17,

  "routing_history": [
    [12, 31, 55, 81, 144, 203],
    [12, 31, 55, 90, 144, 211],
    [12, 44, 55, 90, 144, 211],
    [12, 44, 55, 90, 144, 203]
  ],

  "router_scores": [],
  "hidden_features": [],

  "kv_features_current": {
    "k_projection": [],
    "v_projection": [],
    "k_norm": 0.0,
    "v_norm": 0.0,
    "k_cos_prev": 0.0,
    "v_cos_prev": 0.0
  },

  "kv_history_features": [],

  "indexer_features": {
    "selected_rows": [],
    "selected_scores": [],
    "overlap_with_previous": 0.0
  },

  "attention_features": [],

  "target_experts_next_token": [12, 44, 55, 90, 144, 211],

  "target_layer_offset": 5,
  "target_experts_same_token": []
}
```

Questo è uno **schema concettuale**, non necessariamente il formato fisico finale. Per dataset molto grandi potrebbe essere preferibile usare Parquet/Arrow o array binari separati per le feature dense.

## Versione minima consigliata per il primo tracer DS4

Per evitare di complicare subito l'instrumentazione, partirei da:

```text
prompt_id
category
mode
sequence_id
token_position
token_id
layer
experts_t
routing_history
router_scores_t
target_experts_next_token
```

Poi aggiungerei in quest'ordine:

```text
1. hidden_features
2. kv_features_current
3. kv_history_features
4. indexer_features
5. attention_features
```

In questo modo ogni nuova famiglia di feature può essere valutata con un ablation study separato.

---

# Sviluppo futuro — contesto del layer precedente sullo stesso token

Il predictor Build 5 corrente usa le feature disponibili per il token `t` e
per il layer MoE `l` per predire i sei expert del token successivo nello stesso
layer, cioe' `E_l(t+1)`. Una possibile estensione e' aggiungere anche le
informazioni gia' osservate nel **layer MoE precedente dello stesso token**,
`(t, l-1)`:

```text
feature di (t, l-1) ─┐
                     ├─> predizione di E_l(t+1)
feature di (t, l)   ─┘
```

Queste feature sono causalmente disponibili durante l'inferenza: il layer
`l-1` del token `t` e' stato gia' completato prima di arrivare al layer `l`.
Non introducono quindi informazioni dal futuro, né dal token `t+1`.

La prima ablation consigliata deve aggiungere soltanto informazioni compatte:

```text
previous_layer_experts_same_token : i 6 ID expert realmente scelti in (t, l-1)
previous_layer_router_scores      : opzionale, score/logit del router di (t, l-1)
```

Gli expert selezionati e gli score del router del layer precedente possono
avere correlazioni locali con il routing del layer corrente e quindi fornire
segnale aggiuntivo per `E_l(t+1)`. L'effetto atteso e' un possibile aumento di
Recall@6, ma deve essere dimostrato con un ablation study sullo stesso split
Build 5: non e' garantito.

Non e' invece consigliabile come prima prova aggiungere l'intero raw hidden
4096D del layer precedente. Il raw hidden corrente di `(t, l)` puo' gia'
contenere molta informazione prodotta dai layer precedenti; duplicare anche il
vettore precedente potrebbe essere ridondante, aumentare i parametri della
rete e richiedere una copia GPU->CPU piu' costosa durante il predictor runtime.

Per il primo layer MoE non esiste un layer MoE precedente. In quel caso il
dataset e il predictor runtime devono inserire una rappresentazione sentinella
esplicita, per esempio sei ID sentinella e score zero, affinche' la forma
dell'input resti identica per tutti i layer.

Ordine proposto di sperimentazione:

1. aggiungere i soli 6 ID degli expert di `(t, l-1)`;
2. confrontare validation Recall@6 con il record Build 5 attuale, `0,502358287`;
3. se il segnale e' utile, aggiungere anche gli score/router logits compatti;
4. valutare costo CPU, latenza e Recall@6 online prima di aggiungere feature
   dense del layer precedente.
