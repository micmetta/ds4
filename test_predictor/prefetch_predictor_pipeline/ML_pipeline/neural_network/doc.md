# Neural network

Questa cartella contiene gli script Keras per il predictor neurale multi-label. I dati di input sono le matrici sparse create dal preprocessing classico: non duplica dataset o artefatti delle singole build.

- `ffn_utils.py`: helper condivisi per caricamento CSR, mini-batch sparse, architettura FFN e Recall@6.
- `train_ffn.py`: addestra una singola FFN configurabile e salva modello/report.
- `tune_ffn.py`: cerca gli iperparametri con KerasTuner RandomSearch e salva il miglior modello.
- `evaluate_ffn.py`: valuta un modello salvato una sola volta sul test set.
- `documentation/`: documentazione operativa degli script.
