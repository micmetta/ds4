# TEST_13 — Bridge Windows reader nel prefetch DS4

TEST_13 sostituisce solo la lettura GGUF del worker learned-prefetch: DS4 invia
al servizio Windows il layer e i sei expert, riceve gate/up/down direttamente
nello slot CUDA pinned e mantiene invariati predictor, stati e upload VRAM.

Avvio del servizio Windows:
```powershell
$env:DS4_TEST_13_WINDOWS_MODEL='D:\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
py '\\wsl.localhost\Debian\home\michele\ds4\test_predictor\test_13\windows_native\test_13_windows_reader_service.py'
```

Il runner abilita internamente `DS4_TEST_13_WINDOWS_BRIDGE=1`. Nella
configurazione CLion devi fornire solo `DS4_TEST_13_WINDOWS_HOST=172.23.208.1`
e `DS4_TEST_13_WINDOWS_PORT=38113`. Senza il flag interno, DS4 conserva il
precedente percorso SSD locale.

Il primo prototipo apre una connessione TCP per ogni gruppo di sei expert. È
una scelta deliberatamente semplice per verificare prima correttezza, hit e
telemetria. Dopo questa verifica potremo riusare connessioni persistenti per
ridurre ulteriormente la latenza, senza cambiare i buffer pinned né il formato
dei dati.

## Runner A/B/C

Il runner riusa l'orchestratore verificato di TEST_9, ma salva l'output in
`test_predictor/test_13/output*`. Per la prima prova usa una ripetizione e un
solo layer, cosi' verifichiamo prima correttezza e stati READY:

```bash
DS4_MODEL=/home/michele/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
DS4_TEST_13_PREFETCH_MODEL=/mnt/d/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
DS4_TEST_13_CHECKPOINT=/home/michele/ds4/test_predictor/prefetch_predictor_pipeline/build_dataset_5/ml/neural_network/retrain_best_hyperband_trial_0024_EP1000_PAT5/models/best.weights.h5 \
DS4_TEST_13_WINDOWS_HOST=172.23.208.1 DS4_TEST_13_WINDOWS_PORT=38113 \
DS4_TEST_13_PREFETCH_LAYERS=1 DS4_TEST_13_PREFETCH_SLOTS_PER_LAYER=1 \
DS4_TEST_13_PREFETCH_NUM_WORKERS=1 DS4_TEST_13_REPETITIONS=1 \
uv run --project test_predictor/prefetch_predictor_pipeline/ML_pipeline python test_predictor/test_13/test_13_runner.py
```

Il servizio Windows resta in ascolto dopo il completamento del test: fermalo
con `Ctrl+C` nella finestra PowerShell solo dopo che TEST_13 ha scritto il suo
report.

## Primo risultato verificato: `output`

La prima esecuzione ha usato un layer MoE, uno slot futuro per layer, un worker
e una coppia B poi C. Il report è
`output/benchmark_summary.json`; la trace che dimostra il trasferimento è
`output/repetition_01_B_then_C/run_C.prefetch.jsonl`.

| Misura | Baseline B | Learned prefetch C con bridge Windows |
|---|---:|---:|
| Tempo totale | 105,731 s | 96,758 s |
| Differenza C - B | | -8,972 s |
| Speedup della singola coppia | | +8,49% |

La fase C ha creato 63 richieste learned. Il worker Windows ha iniziato 39
letture e ne ha completate 38 in stato `READY`; non ci sono stati eventi
`prefetch_worker_failed`. Le prime letture completate dei sei expert, pari a
40,50 MiB per gruppo, hanno richiesto circa 70--81 ms fra l'accodamento e lo
stato `READY` nello slot CUDA pinned.

`full_prefetch_hits` vale zero, ma non significa che il prefetch non sia stato
usato. Un full hit richiederebbe che tutti e sei gli expert predetti coincidano
con quelli realmente richiesti. Il predictor ha invece riutilizzato
parzialmente gruppi di expert:

- `partial_prefetch_reuse_groups`: 38;
- `experts_reused_from_pinned_ram`: 111;
- `experts_loaded_from_ssd_at_predicted_layers`: 267.

Quindi DS4 ha trovato già in RAM pinned alcuni expert corretti e ha letto solo
gli expert rimanenti. Il +8,49% è incoraggiante, ma deriva da una sola coppia:
la configurazione con `DS4_TEST_13_REPETITIONS=3` serve a verificare se il
vantaggio medio resta positivo quando B e C vengono alternati.

## Risultato consolidato: `output_2`

La seconda esecuzione ha mantenuto un layer, uno slot e un worker, ma ha
eseguito tre coppie B/C con ordine alternato. Questo riduce l'effetto di cache,
temperatura e altre variazioni fra due singole generazioni.

| Ripetizione | Ordine | B baseline | C bridge Windows | Speedup di C |
|---|---|---:|---:|---:|
| 1 | B poi C | 102,345 s | 104,484 s | -2,09% |
| 2 | C poi B | 97,792 s | 95,410 s | +2,44% |
| 3 | B poi C | 104,767 s | 101,190 s | +3,41% |
| **Media** | alternato | **101,635 s** | **100,361 s** | **+1,25%** |

Due coppie su tre sono più veloci con C. Il vantaggio medio è quindi positivo,
ma la deviazione standard dello speedup è 2,94 punti percentuali: con solo un
layer prefetchato il rumore della misura è ancora comparabile al beneficio.
Non possiamo quindi leggere il +1,25% come uno speedup garantito in ogni
singola generazione.

La parte I/O è invece verificata in modo netto. Nelle tre fasi C il bridge ha
completato rispettivamente 36, 39 e 37 letture `READY`, senza alcun evento
`prefetch_worker_failed`. Il tempo medio di una lettura Windows--TCP--buffer
CUDA pinned per un gruppo di sei expert da 40,50 MiB è stato:

| Ripetizione C | Media | Mediana | P95 | Massimo |
|---|---:|---:|---:|---:|
| 1 | 53,409 ms | 54,375 ms | 70,998 ms | 72,966 ms |
| 2 | 55,075 ms | 53,445 ms | 87,937 ms | 96,236 ms |
| 3 | 54,554 ms | 54,410 ms | 80,233 ms | 90,753 ms |

Le richieste learned sono sempre 63 per fase C. I gruppi riutilizzati
parzialmente sono stati 35, 39 e 36; gli expert corretti già nella RAM pinned
sono stati rispettivamente 103, 112 e 109. Anche qui non ci sono full hit:
la Recall@6 del predictor produce soprattutto riusi parziali, non coincidenze
perfette di tutti i sei expert.

## Risultato con otto layer e tre worker: `output_3`

`output_3` applica il bridge Windows alla prima configurazione con un carico
significativo:

```text
layer learned             = 8, cioè layer fisici DS4 3--10
slot futuri per layer     = 2
worker DS4                = 3, ripartiti per layer modulo 3
ripetizioni B/C           = 3, ordine B-C, C-B, B-C
```

| Ripetizione | Ordine | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---|---:|---:|---:|---:|---:|
| 1 | B poi C | 103,940 s | 91,450 s | +12,02% | 52,833 ms | 61,796 ms |
| 2 | C poi B | 105,233 s | 90,130 s | +14,35% | 9,910 ms | 62,030 ms |
| 3 | B poi C | 106,771 s | 89,906 s | +15,80% | 49,076 ms | 61,049 ms |
| **Media** | alternato | **105,315 s** | **90,496 s** | **+14,05%** | **37,273 ms** | **61,625 ms** |

Tutte e tre le coppie sono positive. Ogni fase C ha completato fra 493 e 504
letture `READY`, senza alcun fallimento del worker. I gruppi con riuso
parziale sono 455, 474 e 455; gli expert corretti già disponibili nella RAM
pinned sono 1.238, 1.296 e 1.238. Il tempo di lettura medio di un gruppo da
40,50 MiB è rimasto circa 61--62 ms, con P95 fra 80,6 e 83,6 ms.

### Confronto corretto con TEST_9

Il riferimento storico con **otto layer e due slot** è
`TEST_9/output_2`, non `TEST_9/output_10`: quest'ultimo usa sedici layer.
`output_2` ha misurato +7,99% medio, mentre questo TEST_13 misura +14,05%.
Il confronto è molto incoraggiante, ma non dimostra ancora da solo che il
guadagno di 6,07 punti percentuali sia causato esclusivamente dal bridge:
`output_2` è stato prodotto con una versione precedente della politica worker,
prima dell'attuale parametro esplicito da tre worker.

Il controllo sperimentale definitivo è rieseguire TEST_9 con il codice
attuale, otto layer, due slot e tre worker, lasciando invariato ogni altro
parametro. Solo allora B/C locale e B/C con bridge differiranno soltanto per il
percorso di lettura degli expert.

## Confronto con il miglior carico storico: `output_4`

`output_4` usa il carico che ha prodotto il miglior risultato storico di
TEST_9/output_10:

```text
layer learned             = 16, cioè layer fisici DS4 3--18
slot futuri per layer     = 2
worker DS4                = 3, ripartiti per layer modulo 3
ripetizioni B/C           = 3, ordine B-C, C-B, B-C
```

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 103,325 s | 85,009 s | +17,73% | 62,457 ms | 67,599 ms |
| 2 | 104,117 s | 90,101 s | +13,46% | 16,508 ms | 63,708 ms |
| 3 | 103,219 s | 84,600 s | +18,04% | 65,167 ms | 68,061 ms |
| **Media** | **103,553 s** | **86,570 s** | **+16,41%** | **48,044 ms** | **66,456 ms** |

Tutte e tre le coppie sono positive. Il +16,41% medio è sostanzialmente
uguale al miglior valore storico di TEST_9/output_10, +16,63%; la differenza
è di soli 0,22 punti percentuali, inferiore alla variabilità misurata fra le
tre ripetizioni di TEST_13 (deviazione standard 2,56 punti).

| Misura media per fase C | TEST_9/output_10, lettura WSL | TEST_13/output_4, bridge Windows |
|---|---:|---:|
| Gruppi con riuso parziale | ~877 | ~932 |
| Expert corretti riusati dalla RAM pinned | ~2.385 | ~2.535 |
| Expert ancora caricati quando richiesti | ~3.663 | ~3.447 |
| Letture `READY` | ~954 | ~995 |
| Full hit | 0 | 0 |

Il bridge completa quindi leggermente più richieste utili e lascia meno expert
da recuperare al momento della domanda reale. Le sue letture del gruppo da
40,50 MiB restano nell'intervallo medio 63,7--68,1 ms, con P95 82,5--86,2 ms.

Le ultime due colonne derivano direttamente dalla trace:

- **Attesa media prima della lettura** = `queue_to_worker_ms`: tempo trascorso
  fra l'enqueue DS4 e l'istante in cui un worker prende realmente in carico la
  richiesta;
- **Lettura e trasferimento medi** = `worker_read_ms`: per TEST_13 include la
  lettura nativa di Windows dal SSD, l'invio TCP e la ricezione nel buffer CUDA
  pinned di WSL. È quindi una misura più completa della sola lettura SSD.

Il confronto con TEST_9/output_10 rende visibile l'effetto del bypass di
DrvFS. Entrambi gli esperimenti usano sedici layer, due slot per layer e tre
worker. Il percorso storico WSL ha misurato circa **372,95 ms** di attesa
media e **191,94 ms** di lettura locale per gruppo; TEST_13/output_4 misura
invece **48,04 ms** e **66,46 ms**.

| Percorso | Layer | Slot per layer | Worker | Attesa media prima della lettura | Lettura / trasferimento |
|---|---:|---:|---:|---:|---:|
| TEST_9, lettura WSL tramite `/mnt/d` | 16 | 2 | 3 | 372,95 ms | 191,94 ms |
| TEST_13, bridge Windows | 16 | 2 | 3 | 48,04 ms | 66,46 ms |

Il bridge riduce quindi sia l'accodamento sia il tempo end-to-end di
riempimento dello slot pinned, anche se il speedup complessivo resta limitato
dagli expert non predetti correttamente e dal poco anticipo disponibile.

Il fatto che questo miglioramento I/O non superi nettamente +16,63% mostra che
la sola banda di lettura non è più il limite dominante con 16 layer, due slot
e tre worker: molti expert restano mancanti perché la previsione non è esatta
oppure non ha abbastanza anticipo per diventare `READY`. Il confronto più
rigoroso per isolare solo il bridge resta una nuova TEST_9, eseguita oggi con
la stessa configurazione e lo stesso codice, ma con il vecchio percorso WSL.

## Miglior risultato attuale: `output_5`

`output_5` confronta direttamente il bridge Windows con TEST_9/output_14:
entrambi usano 24 layer learned (fisici 3--26), 2 slot per layer, 4 worker
modulo 4 e tre coppie B/C alternate.

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 106,901 s | 83,038 s | +22,32% | 43,72 ms | 71,65 ms |
| 2 | 104,145 s | 82,212 s | +21,06% | 7,72 ms | 69,01 ms |
| 3 | 104,469 s | 81,858 s | +21,64% | 43,22 ms | 68,64 ms |
| **Media** | **105,172 s** | **82,369 s** | **+21,68%** | **31,55 ms** | **69,77 ms** |

Questo è il miglior risultato misurato finora: supera TEST_9/output_14
(+13,85%) di 7,83 punti percentuali e il precedente record assoluto
TEST_9/output_10 (+16,63%) di 5,05 punti. La deviazione standard dello speedup
è soltanto 0,63 punti percentuali, quindi il vantaggio è anche molto più
stabile delle misure precedenti.

| Misura media per fase C | TEST_9/output_14: WSL `/mnt/d` | TEST_13/output_5: bridge Windows |
|---|---:|---:|
| Layer learned | 24 | 24 |
| Slot per layer | 2 | 2 |
| Worker | 4 | 4 |
| Gruppi con riuso parziale | ~935 | ~1.394 |
| Expert corretti riusati dalla RAM pinned | ~2.554 | ~3.883 |
| Expert ancora caricati quando richiesti | ~6.518 | ~5.189 |
| Letture `READY` | ~1.084 | ~1.494 |
| Attesa media prima della lettura | ~498,57 ms | ~31,55 ms |
| Lettura / trasferimento medio | ~271,44 ms | ~69,77 ms |
| Full hit | 0 | 0 |

Il bridge riduce l'attesa in coda di circa 467 ms per gruppo e il tempo
end-to-end di lettura/trasferimento di circa 202 ms. Questo permette a molte
più richieste di diventare `READY` prima del punto in cui DS4 domanda gli
expert; di conseguenza crescono i riusi parziali e lo speedup complessivo.
Anche in questa configurazione non ci sono full hit: il limite residuo è la
qualità temporale e la Recall@6 del predictor, non l'affidabilità del bridge.

## Scalabilità oltre il punto ottimale: `output_6`

`output_6` aumenta soltanto i layer learned da 24 a 32; restano invariati due
slot per layer, quattro worker modulo 4 e tre ripetizioni. I layer fisici
prefetchati diventano quindi 3--34 e ogni worker deve servire otto layer.

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 94,639 s | 79,566 s | +15,93% | 56,76 ms | 73,43 ms |
| 2 | 104,042 s | 89,046 s | +14,41% | 11,48 ms | 74,66 ms |
| 3 | 98,537 s | 80,960 s | +17,84% | 69,27 ms | 74,16 ms |
| **Media** | **99,073 s** | **83,191 s** | **+16,06%** | **45,84 ms** | **74,08 ms** |

I layer aggiuntivi aumentano le richieste che arrivano in tempo:

| Misura media per fase C | 24 layer, `output_5` | 32 layer, `output_6` |
|---|---:|---:|
| Gruppi con riuso parziale | ~1.394 | ~1.871 |
| Expert corretti riusati dalla RAM pinned | ~3.883 | ~5.273 |
| Expert ancora caricati quando richiesti | ~5.189 | ~6.823 |
| Letture `READY` | ~1.494 | ~1.997 |
| Attesa media prima della lettura | ~31,55 ms | ~45,84 ms |
| Lettura / trasferimento medio | ~69,77 ms | ~74,08 ms |
| Speedup medio | **+21,68%** | **+16,06%** |

Quindi il bridge non è saturato: le letture `READY` crescono da circa 1.494 a
circa 1.997. Tuttavia ogni layer aggiunto richiede anche una nuova inferenza
del predictor, nuove richieste, eventuali upload e ancora molti expert non
predetti correttamente. Il costo marginale dei layer 27--34 è maggiore del
beneficio marginale dei loro riusi. Con le configurazioni misurate finora,
**24 layer, 2 slot e 4 worker (`output_5`) resta il punto ottimale**.

### Estensione: cinque worker modulo cinque

TEST_13 accetta ora anche `DS4_TEST_13_PREFETCH_NUM_WORKERS=5`. La
partizione resta identica nella forma: ogni worker gestisce solo i layer fisici
per cui `layer % 5` coincide con il proprio ID. Con 32 layer learned (fisici
3--34), i cinque worker ricevono quindi sei o sette layer ciascuno. Questa
estensione serve a misurare se un quinto flusso Windows--WSL migliora ancora lo
speedup rispetto alla configurazione a quattro worker; non implica che cinque
worker siano già migliori.

## Risultati verificati — `output_7` (32 layer, cinque worker)

La directory [`output_7`](output_7) usa gli stessi 32 layer learned e due
slot per layer di `output_6`, ma aumenta i worker del bridge Windows da quattro
a cinque. I layer fisici 3--34 sono assegnati con `layer % 5`: i worker 0--2
ricevono sei layer, mentre i worker 3--4 ne ricevono sette.

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 104,647 s | 79,295 s | +24,23% | 51,71 ms | 74,37 ms |
| 2 | 111,934 s | 84,441 s | +24,56% | 4,98 ms | 74,56 ms |
| 3 | 103,427 s | 86,539 s | +16,33% | 59,17 ms | 71,17 ms |
| **Media** | **106,669 s** | **83,425 s** | **+21,71%** | **38,62 ms** | **73,37 ms** |

| Misura media per fase C | 32 layer, 4 worker `output_6` | 32 layer, 5 worker `output_7` |
|---|---:|---:|
| Letture `READY` | ~1.997 | ~1.997 |
| Gruppi con riuso parziale | ~1.871 | ~1.870 |
| Expert riusati dalla RAM pinned | ~5.273 | ~5.271 |
| Attesa media prima della lettura | ~45,84 ms | **~38,62 ms** |
| Lettura / trasferimento medio | ~74,08 ms | ~73,37 ms |
| Speedup medio | +16,06% | **+21,71%** |

Il quinto worker non aumenta in modo materiale il numero di richieste che
diventano `READY`: il bridge era già abbastanza veloce per leggere quasi tutti
i gruppi necessari. Riduce però leggermente l'attesa prima della lettura e,
in questa esecuzione, porta il risultato medio a **+21,71%**. Il valore supera
di soli 0,03 punti percentuali il record di `output_5` (+21,68%, 24 layer e
quattro worker): i due risultati sono quindi praticamente equivalenti entro la
variabilità del benchmark. Il confronto certo è invece con `output_6`: a parità
di 32 layer, passare da quattro a cinque worker migliora nettamente lo speedup
medio da +16,06% a +21,71%.

## Risultati verificati — `output_8` (40 layer, cinque worker)

La directory [`output_8`](output_8) porta il predictor alla massima copertura
supportata dalla banca pinned attuale: 40 layer learned, cioè i layer fisici
3--42. Rimangono due slot per layer, cinque worker modulo 5 e tre coppie B/C
alternate. I 40 layer sono distribuiti in modo uniforme: otto layer per
worker.

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 104,932 s | 77,210 s | +26,42% | 72,53 ms | 86,05 ms |
| 2 | 115,188 s | 82,608 s | +28,28% | 4,90 ms | 80,86 ms |
| 3 | 108,780 s | 85,074 s | +21,79% | 73,86 ms | 68,39 ms |
| **Media** | **109,633 s** | **81,631 s** | **+25,50%** | **50,43 ms** | **78,43 ms** |

| Misura media per fase C | 32 layer, 5 worker `output_7` | 40 layer, 5 worker `output_8` |
|---|---:|---:|
| Letture `READY` | ~1.997 | ~2.501 |
| Gruppi con riuso parziale | ~1.870 | ~2.363 |
| Expert riusati dalla RAM pinned | ~5.271 | ~6.825 |
| Attesa media prima della lettura | ~38,62 ms | ~50,43 ms |
| Lettura / trasferimento medio | ~73,37 ms | ~78,43 ms |
| Speedup medio | +21,71% | **+25,50%** |

Questo è il nuovo miglior risultato misurato: tutte le tre coppie sono
positive e lo speedup medio raggiunge **+25,50%**. Rispetto a `output_7`, i
layer aggiuntivi aumentano di circa 500 le letture che arrivano `READY` e di
circa 1.550 gli expert riusati dalla RAM pinned per fase C. Il tempo medio di
lettura cresce leggermente, ma il beneficio dei riusi aggiuntivi è superiore al
costo: includere i layer fisici 35--42 migliora lo speedup di 3,79 punti
percentuali. Per questa macchina e per il bridge Windows--WSL, `output_8` è
quindi il riferimento attuale per la copertura massima dei layer learned.

### Estensione: sei worker modulo sei

TEST_13 accetta anche `DS4_TEST_13_PREFETCH_NUM_WORKERS=6`. Il criterio non
cambia: ciascun worker serve i layer fisici per cui `layer % 6` coincide con il
proprio ID. Con 40 layer learned fisici 3--42, ogni worker riceve sei o sette
layer. Il test con sei worker deve mantenere invariati 40 layer e due slot per
layer: così isola l'effetto del sesto flusso concorrente rispetto a `output_8`.

### Estensione: sette worker modulo sette

TEST_13 accetta anche `DS4_TEST_13_PREFETCH_NUM_WORKERS=7`. Con 40 layer
learned fisici 3--42 la regola `layer % 7` assegna cinque o sei layer a ogni
worker. Il confronto mantiene due slot per layer e tutte le altre impostazioni
di `output_9`, per isolare l'effetto del settimo flusso concorrente.

## Risultati verificati — `output_9` (40 layer, sei worker)

La directory [`output_9`](output_9) è il confronto diretto con `output_8`:
mantiene 40 layer learned fisici 3--42, due slot per layer e tre coppie B/C
alternate, cambiando soltanto il numero di worker da cinque a sei. La politica
è `layer % 6`, per cui ogni worker riceve sei o sette layer fisici.

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 104,144 s | 75,546 s | +27,46% | 65,93 ms | 80,51 ms |
| 2 | 102,169 s | 77,324 s | +24,32% | 5,31 ms | 77,29 ms |
| 3 | 106,589 s | 75,006 s | +29,63% | 57,61 ms | 79,90 ms |
| **Media** | **104,300 s** | **75,959 s** | **+27,14%** | **42,95 ms** | **79,23 ms** |

| Misura media per fase C | 40 layer, 5 worker `output_8` | 40 layer, 6 worker `output_9` |
|---|---:|---:|
| Letture `READY` | ~2.501 | ~2.498 |
| Gruppi con riuso parziale | ~2.363 | ~2.359 |
| Expert riusati dalla RAM pinned | ~6.825 | ~6.810 |
| Attesa media prima della lettura | ~50,43 ms | **~42,95 ms** |
| Lettura / trasferimento medio | ~78,43 ms | ~79,23 ms |
| Speedup medio | +25,50% | **+27,14%** |

Il sesto worker non fa crescere il numero di gruppi `READY`: quasi tutti i
gruppi utili arrivavano già in tempo con cinque worker. Riduce però l'attesa
media in coda di circa 7,5 ms senza aumentare significativamente il tempo di
lettura/trasferimento. In questa configurazione il piccolo miglioramento di
latenza si traduce in un nuovo record di **+27,14%** medio, con tutte le tre
ripetizioni positive. `output_9` è quindi la configurazione di riferimento
attuale: massima copertura dei 40 layer learned, due slot per layer e sei
worker modulo 6.

### Cosa dimostra `output_9` sul collo di bottiglia I/O

Con 40 layer learned, sei worker e due slot per layer, ogni fase C emette 2.520
richieste learned. In `output_9` il worker rende `READY` 2.487--2.516 gruppi
per ripetizione: circa il 98,7--99,8% delle richieste viene quindi letta dal
GGUF Windows e trasferita nella RAM pinned WSL prima del punto di domanda.

Anche i layer fisici più alti sono serviti in tempo. Nelle tre ripetizioni, per
i layer 35--42 si osservano 62 o 63 gruppi `READY` su 63 richieste per layer;
non c'è più la starvation dei layer alti osservata con il vecchio percorso di
lettura WSL `/mnt/d`.

Questo non equivale però a dire che tutti gli expert realmente domandati dal
router siano pronti. `READY` indica solo che sono pronti i sei expert
**predetti**. Con una Recall@6 online di circa 0,45, in media solo circa 2,7
dei sei expert reali appartengono al gruppo predetto. È per questo che i full
hit restano zero e parte degli expert deve ancora essere caricata alla domanda
reale. Dopo il bridge Windows, migliorare Recall@6 è quindi la leva principale
per aumentare ulteriormente lo speedup; I/O, banca pinned e layer alti non sono
più il collo di bottiglia dominante in questa configurazione.

## Risultati verificati — `output_10` (40 layer, sette worker)

La directory [`output_10`](output_10) mantiene 40 layer learned fisici 3--42,
due slot per layer e tre coppie B/C alternate. Rispetto a `output_9` cambia
solo la partizione da sei a sette worker, applicando `layer % 7`: ogni worker
serve cinque o sei layer.

| Ripetizione | B baseline | C bridge Windows | Speedup di C | Attesa media prima della lettura | Lettura e trasferimento medi |
|---|---:|---:|---:|---:|---:|
| 1 | 100,100 s | 76,244 s | +23,83% | 71,43 ms | 83,89 ms |
| 2 | 110,715 s | 79,075 s | +28,58% | 6,39 ms | 73,23 ms |
| 3 | 100,048 s | 75,382 s | +24,65% | 74,50 ms | 78,32 ms |
| **Media** | **103,621 s** | **76,900 s** | **+25,69%** | **50,77 ms** | **78,48 ms** |

| Misura media per fase C | 40 layer, 6 worker `output_9` | 40 layer, 7 worker `output_10` |
|---|---:|---:|
| Letture `READY` | ~2.498 | ~2.499 |
| Gruppi con riuso parziale | ~2.359 | ~2.361 |
| Expert riusati dalla RAM pinned | ~6.810 | ~6.819 |
| Attesa media prima della lettura | **~42,95 ms** | ~50,77 ms |
| Lettura / trasferimento medio | ~79,23 ms | ~78,48 ms |
| Speedup medio | **+27,14%** | +25,69% |

Il settimo worker non aumenta la quantità di lavoro utile: gruppi `READY`,
riusi parziali ed expert riusati restano sostanzialmente invariati rispetto a
sei worker. Il tempo di lettura resta analogo, ma l'attesa media prima della
lettura cresce di circa 7,8 ms; il guadagno medio scende quindi di 1,45 punti
percentuali. Il punto di equilibrio osservato è **sei worker**: `output_9`
resta il record con +27,14% e ha anche una struttura più semplice di sette
flussi concorrenti.
