# Log delle inferenze BUILD_1

Questa cartella contiene 200 file di log: per ciascuno dei 100 prompt esistono
un file `prompt_XXXX.stdout.log` e uno `prompt_XXXX.stderr.log`.

- `stdout.log` conserva l'output testuale dell'inferenza;
- `stderr.log` conserva diagnostica DS4, caricamento modello e possibili errori.

I log permettono di diagnosticare un prompt senza dover rieseguire l'intera
raccolta. Non sono feature date ai modelli ML.
