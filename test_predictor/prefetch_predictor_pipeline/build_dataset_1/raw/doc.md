# Raw trace BUILD_1

Questa cartella contiene gli output diretti del collector DS4 per i 100 prompt
del corpus V0. È la sorgente primaria da cui è stato costruito il dataset
derivato BUILD_1.

- `manifest.jsonl`: una riga per ogni tentativo di raccolta; registra prompt,
  categoria, percorso del trace, stato ed exit code. L'ultima entry `success`
  di ogni `prompt_id` è quella valida per la ripresa o il preprocessing.
- `routing/`: trace JSONL del routing per singolo prompt.
- `logs/`: stdout e stderr prodotti dalla corrispondente inferenza DS4.

I dati raw non devono essere modificati manualmente: servono per audit,
riproducibilità e per rigenerare i dataset derivati.
