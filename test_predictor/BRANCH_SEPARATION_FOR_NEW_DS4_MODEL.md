# Separazione del lavoro di ricerca dal nuovo DS4 ufficiale

## Perché separiamo i due rami

Il lavoro svolto finora sul prefetch predittivo è stato sviluppato e misurato
su una specifica versione di DS4 e del modello DeepSeek V4 Flash usata nelle
Build 1--5. Il risultato migliore ottenuto finora è:

- validation Recall@6: **0,502358**;
- test Recall@6: **0,506056**.

Nel frattempo il repository ufficiale di Salvatore Sanfilippo (`antirez/ds4`)
ha ricevuto nuovi aggiornamenti, inclusi quelli necessari per sperimentare il
nuovo modello DeepSeek V4.1 Flash. Prima di adattare il progetto a quella
versione, manteniamo immutabile e recuperabile il lavoro precedente.

Questa separazione evita di confondere due esperimenti diversi: i risultati
del modello precedente non sono direttamente confrontabili con quelli del
nuovo modello finché non saranno raccolti nuovamente dati, feature e metriche
con lo stesso protocollo.

## Ramo di ricerca già consolidato

Ramo Git:

```text
predictive-expert-prefetch
```

Repository remoto privato:

```text
https://github.com/micmetta/ds4-prefetch.git
```

Questo ramo contiene il lavoro finora realizzato:

- test da `test_0` a `test_8`;
- raccolta dati e pipeline delle Build 1--5;
- definizione delle feature;
- rete neurale fixed-NNZ;
- ricerca Hyperband, riaddestramento e valutazione del modello migliore;
- integrazione sperimentale del predictor CPU e del learned prefetch;
- documentazione e report leggeri.

Non vengono versionati i file pesanti o rigenerabili, quali dataset grezzi,
array NumPy, checkpoint, modelli `.keras`/`.h5`, trace TensorBoard e output
delle run. Il codice e i report necessari per rigenerarli restano invece
versionati.

Questo ramo **non deve essere aggiornato con `upstream/main`** durante la
nuova sperimentazione: rimane la fotografia del lavoro fatto sul modello e
sull'implementazione DS4 precedenti.

## Nuovo ramo per il DS4 ufficiale aggiornato

Il nuovo ramo verrà creato direttamente dall'ultimo commit di:

```text
upstream/main  =  https://github.com/antirez/ds4.git, ramo main
```

Nome del ramo:

```text
main
```

Directory di lavoro separata prevista:

```text
/home/michele/ds4-v41-flash
```

Il ramo `main` locale verrà prima allineato al ramo `main` ufficiale di
Sanfilippo (`upstream/main`). La nuova directory sarà quindi creata con un
*Git worktree* che usa proprio quel ramo:

```bash
cd /home/michele/ds4
git worktree add ../ds4-v41-flash main
```

Il worktree del ramo `main` partirà quindi dal DS4 ufficiale aggiornato, senza
trascinarsi modifiche locali, dataset o sorgenti del vecchio esperimento.
Potrà ricevere nuove modifiche dedicate al DeepSeek V4.1 Flash.

## Differenza pratica tra i due rami

| Aspetto | `predictive-expert-prefetch` | `main` |
|---|---|---|
| Punto di partenza | DS4 usato negli esperimenti già completati | Ultimo `upstream/main` ufficiale |
| Modello di riferimento | DeepSeek V4 Flash precedente | DeepSeek V4.1 Flash / versione aggiornata |
| Dataset e feature | Build 1--5 già prodotte | Da raccogliere nuovamente |
| Rete neurale | Predictor migliore già addestrato | Da valutare e riaddestrare sul nuovo dataset |
| Obiettivo | Conservare, riprodurre e confrontare i risultati esistenti | Verificare se il learned prefetch funziona sul nuovo modello |
| Rischio di alterare il lavoro precedente | Nessuno, perché il ramo resta separato | Nessuno, perché usa un worktree distinto |

## Perché non riutilizziamo direttamente il modello neurale migliore

Il predictor migliore usa feature legate alla struttura e alla distribuzione
del modello precedente: routing history, identificativi e punteggi del router,
stati hidden raw, embedding dei token, stato Key-Value e posizione del token.

Con un nuovo modello possono cambiare, tra le altre cose, il numero e
l'organizzazione dei layer MoE, gli esperti, il vocabolario, gli embedding e
la distribuzione dei valori hidden e Key-Value. Per questo i pesi del predictor
precedente non vanno considerati automaticamente validi sul nuovo modello.

L'approccio corretto è:

1. compilare ed eseguire il DS4 aggiornato nel nuovo worktree;
2. ricreare il protocollo di raccolta routing per il nuovo modello;
3. generare un nuovo dataset con train, validation e test;
4. controllare le dimensioni e il significato delle feature;
5. riaddestrare e valutare un nuovo predictor;
6. ripetere i test runtime di prefetch e confrontarli con i risultati del ramo
   precedente.

## Garanzia di recuperabilità

Le due directory possono coesistere:

```text
/home/michele/ds4            -> ricerca già consolidata
/home/michele/ds4-v41-flash  -> ramo `main`, nuova sperimentazione su DS4 aggiornato
```

Una modifica, un commit o un test eseguito in una directory non modifica
l'altra. Se il nuovo percorso non desse risultati utili, sarà sempre possibile
tornare al ramo `predictive-expert-prefetch` e proseguire esattamente dal punto
in cui si è lasciato il lavoro precedente.
