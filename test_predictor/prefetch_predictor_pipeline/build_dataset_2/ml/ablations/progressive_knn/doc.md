# Screening progressivo KNN

Questa cartella contiene il confronto rapido dei gruppi di feature di BUILD_DATASET_2 con KNN. Tutte le configurazioni usano lo stesso routing multi-hot H=1 e lo stesso validation split; cambiano soltanto le feature aggiunte.

- `artifacts/`: configurazioni, gruppi di feature e parametri KNN;
- `reports/`: Recall@6 train/validation ordinata per validation.

Non vengono salvati modelli o matrici permanenti: KNN è usato solo per lo screening. La documentazione completa è in `ML_pipeline/classical_ml_preprocessing/documentation/doc_progressive_knn_ablation.md`.
