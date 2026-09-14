#!/usr/bin/env python3
"""
Prepara feature sparse per i modelli classici di scikit-learn.

Input:
    A BUILD_DATASET training directory containing one Parquet file and
    ``arrays/y_<split>.npy`` labels.
    ../training/arrays/y_<split>.npy

Output:
    arrays/X_classic_<split>.npz
    arrays/y_<split>.npy
    artifacts/categorical_encoder.joblib
    artifacts/numeric_scaler.joblib
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPLITS = ("train", "validation", "test")
TOP_K = 6
NUM_EXPERTS = 256

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PIPELINE_ROOT = PIPELINE_DIR.parent
DEFAULT_TRAINING_DIR = PIPELINE_ROOT / "build_dataset_1" / "ml" / "training"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "build_dataset_1" / "ml" / "classical_ml"

CATEGORICAL_COLUMNS = [
    "token_id",
    "layer",
    *[f"routing_history_expert_{index}" for index in range(TOP_K)],
    *[f"router_logit_expert_id_{index}" for index in range(16)],
]

BASE_NUMERIC_COLUMNS = [
    "token_position",
    *[f"router_logit_value_{index}" for index in range(16)],
]


def leggi_argomenti() -> argparse.Namespace:
    """
    Legge i percorsi di input e output dalla riga di comando.

    Args:
        Nessun argomento Python. La configurazione arriva da ``--training-dir``
        e ``--output-dir``.

    Returns:
        Namespace con la cartella del dataset base e la cartella di output.
    """
    parser = argparse.ArgumentParser(
        description="Crea matrici sparse per i modelli classici di scikit-learn."
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help="Cartella che contiene Parquet e arrays prodotti dal preprocessing della build.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella classical_ml della build in cui salvare arrays e artifacts.",
    )
    return parser.parse_args()


def trova_parquet_base(training_dir: Path) -> Path:
    """
    Find the single Parquet table produced by a dataset preprocessing step.

    Args:
        training_dir: Build training directory containing a ``prefetch_training``
            Parquet file.

    Returns:
        Path of the only compatible Parquet table.

    Raises:
        FileNotFoundError: If no compatible Parquet table exists.
        ValueError: If more than one compatible Parquet table exists.
    """
    candidates = sorted(training_dir.glob("prefetch_training*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun Parquet prefetch_training*.parquet trovato in: {training_dir}"
        )
    if len(candidates) != 1:
        names = [path.name for path in candidates]
        raise ValueError(
            "Trovati più Parquet compatibili; conserva un solo dataset nella "
            f"training directory: {names}"
        )
    return candidates[0]


def trova_colonne_hidden_pca(table: pd.DataFrame) -> list[str]:
    """
    Return optional PCA hidden-feature columns in their numeric order.

    Args:
        table: Base Parquet table loaded from one dataset build.

    Returns:
        Ordered names ``hidden_pca_0`` through ``hidden_pca_(d-1)``. Returns
        an empty list for BUILD_DATASET_1, which has no feature 4.

    Raises:
        ValueError: If PCA column names are malformed or their indices are not
            consecutive from zero.
    """
    indices: list[int] = []
    for column in table.columns:
        if not column.startswith("hidden_pca_"):
            continue
        suffix = column.removeprefix("hidden_pca_")
        if not suffix.isdigit():
            raise ValueError(f"Nome colonna PCA non valido: {column}")
        indices.append(int(suffix))

    indices.sort()
    if indices != list(range(len(indices))):
        raise ValueError(
            "Le colonne hidden_pca devono essere consecutive da hidden_pca_0: "
            f"trovate {indices}"
        )
    return [f"hidden_pca_{index}" for index in indices]


def carica_tabella_base(training_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Carica il Parquet base e verifica le colonne necessarie al preprocessing.

    Args:
        training_dir: Cartella ``build_dataset_N/ml/training`` prodotta dal
            preprocessing della build.

    Returns:
        Tupla con la tabella completa e le colonne numeriche da standardizzare.
        Le feature PCA ``hidden_pca_*`` vengono incluse automaticamente quando
        sono presenti nel Parquet.

    Raises:
        FileNotFoundError: Se il Parquet base non esiste.
        ValueError: Se mancano colonne necessarie o gli split non sono validi.
    """
    parquet_path = trova_parquet_base(training_dir)
    table = pd.read_parquet(parquet_path)
    numeric_columns = [*BASE_NUMERIC_COLUMNS, *trova_colonne_hidden_pca(table)]
    required_columns = {"split", *CATEGORICAL_COLUMNS, *numeric_columns}
    missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        raise ValueError(f"Nel Parquet mancano queste colonne: {missing_columns}")

    found_splits = set(table["split"].unique())
    if found_splits != set(SPLITS):
        raise ValueError(f"Split non validi: trovati {sorted(found_splits)}")

    return table, numeric_columns


def carica_e_verifica_label(training_dir: Path, split: str, table: pd.DataFrame) -> np.ndarray:
    """
    Carica le label multi-hot di uno split e ne verifica la coerenza col Parquet.

    Args:
        training_dir: Cartella ``build_dataset_1/ml/training`` che contiene ``arrays/``.
        split: Nome dello split, cioe' ``train``, ``validation`` o ``test``.
        table: Sotto-tabella Parquet contenente soltanto le righe dello split.

    Returns:
        Array ``uint8`` con shape ``[N_split, 256]``. Ogni riga contiene sei
        valori uguali a 1, corrispondenti agli expert target.

    Raises:
        ValueError: Se shape, tipo, numero di label attive o target non
        corrispondono ai dati presenti nel Parquet.
    """
    label_path = training_dir / "arrays" / f"y_{split}.npy"
    if not label_path.is_file():
        raise FileNotFoundError(f"Label non trovate: {label_path}")

    labels = np.load(label_path)
    if labels.shape != (len(table), NUM_EXPERTS):
        raise ValueError(
            f"{split}: shape label inattesa {labels.shape}; "
            f"attesa {(len(table), NUM_EXPERTS)}"
        )
    if labels.dtype != np.uint8:
        raise ValueError(f"{split}: label dtype inatteso {labels.dtype}")
    if not np.all(labels.sum(axis=1) == TOP_K):
        raise ValueError(f"{split}: ogni label deve avere esattamente {TOP_K} valori attivi")

    target_columns = [f"target_expert_{index}" for index in range(TOP_K)]
    targets = table[target_columns].to_numpy(dtype=np.int16)
    for row_index, (label, target) in enumerate(zip(labels, targets)):
        if not np.array_equal(np.flatnonzero(label), np.sort(target)):
            raise ValueError(f"{split}: label incoerente con il Parquet alla riga {row_index}")

    return labels


def crea_encoder_categorico() -> OneHotEncoder:
    """
    Crea l'encoder one-hot per token, layer e ID expert discreti.

    Args:
        Nessun argomento Python. La configurazione BUILD_1 e' fissata nello
        script per essere riproducibile.

    Returns:
        OneHotEncoder sparse che raggruppa le categorie viste una sola volta e
        manda le categorie sconosciute nel bucket infrequente, quando presente.
    """
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=41,
        sparse_output=True,
        dtype=np.float32,
    )


def trasforma_feature(
    table: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    numeric_columns: list[str],
) -> sparse.csr_matrix:
    """
    Trasforma feature categoriche e numeriche in una matrice sparse CSR unica.

    Args:
        table: Sotto-tabella relativa a un singolo split.
        encoder: OneHotEncoder gia' fit sul train.
        scaler: StandardScaler gia' fit sul train.
        numeric_columns: Colonne numeriche, comprese le eventuali coordinate
            PCA di BUILD_DATASET_2.

    Returns:
        Matrice CSR float32: prima le colonne one-hot, poi ``token_position``
        e tutte le feature numeriche standardizzate.
    """
    categorical = encoder.transform(table[CATEGORICAL_COLUMNS])
    numeric = scaler.transform(table[numeric_columns]).astype(np.float32)
    numeric_sparse = sparse.csr_matrix(numeric)
    return sparse.hstack((categorical, numeric_sparse), format="csr", dtype=np.float32)


def salva_artefatti(
    output_dir: Path,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    matrices: dict[str, sparse.csr_matrix],
    labels: dict[str, np.ndarray],
    numeric_columns: list[str],
) -> None:
    """
    Salva matrici sparse, label e trasformatori appresi sul train.

    Args:
        output_dir: Cartella ``classical_ml_preprocessing`` di destinazione.
        encoder: Encoder categoriale gia' fit sul train.
        scaler: Scaler numerico gia' fit sul train.
        matrices: Mapping ``split -> X_classic`` in formato CSR.
        labels: Mapping ``split -> y`` multi-hot verificato.
        numeric_columns: Colonne standardizzate nello stesso ordine dello
            scaler e della parte numerica delle matrici sparse.

    Returns:
        ``None``. I risultati sono scritti nelle sottocartelle ``arrays/`` e
        ``artifacts/`` di ``output_dir``.
    """
    arrays_dir = output_dir / "arrays"
    artifacts_dir = output_dir / "artifacts"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        sparse.save_npz(arrays_dir / f"X_classic_{split}.npz", matrices[split])
        np.save(arrays_dir / f"y_{split}.npy", labels[split])

    joblib.dump(encoder, artifacts_dir / "categorical_encoder.joblib")
    joblib.dump(scaler, artifacts_dir / "numeric_scaler.joblib")

    feature_names = [
        *encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist(),
        *numeric_columns,
    ]
    (artifacts_dir / "feature_names.json").write_text(
        json.dumps({"feature_names": feature_names}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": numeric_columns,
        "hidden_pca_dimension": sum(
            column.startswith("hidden_pca_") for column in numeric_columns
        ),
        "one_hot_configuration": {
            "handle_unknown": "infrequent_if_exist",
            "min_frequency": 41,
        },
        "feature_count": len(feature_names),
        "splits": {},
    }
    for split in SPLITS:
        matrix = matrices[split]
        summary["splits"][split] = {
            "X_shape": list(matrix.shape),
            "X_nonzero_values": int(matrix.nnz),
            "X_density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            "y_shape": list(labels[split].shape),
        }
    (artifacts_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """
    Esegue l'intero preprocessing per i modelli classici di scikit-learn.

    Args:
        Nessun argomento Python. Percorsi e configurazione arrivano da
        ``leggi_argomenti()`` e dalla struttura del Parquet scelto. BUILD_DATASET_2
        aggiunge automaticamente le colonne ``hidden_pca_*``.

    Returns:
        ``None``. Le matrici CSR e gli artefatti vengono salvati nella cartella
        ``classical_ml_preprocessing``.
    """
    args = leggi_argomenti()

    print("1) Carico il Parquet base...")
    table, numeric_columns = carica_tabella_base(args.training_dir)
    split_tables = {
        split: table.loc[table["split"] == split].reset_index(drop=True)
        for split in SPLITS
    }
    print(f"   Esempi totali: {len(table):,}")
    hidden_dimension = sum(
        column.startswith("hidden_pca_") for column in numeric_columns
    )
    print(
        "   Feature numeriche: "
        f"{len(BASE_NUMERIC_COLUMNS)} base + {hidden_dimension} PCA"
    )

    print("2) Carico e verifico le label multi-label...")
    labels = {
        split: carica_e_verifica_label(args.training_dir, split, split_tables[split])
        for split in SPLITS
    }

    print("3) Fit di OneHotEncoder e StandardScaler sul solo train...")
    encoder = crea_encoder_categorico()
    scaler = StandardScaler()
    encoder.fit(split_tables["train"][CATEGORICAL_COLUMNS])
    scaler.fit(split_tables["train"][numeric_columns])

    print("4) Trasformo train, validation e test in matrici sparse CSR...")
    matrices = {
        split: trasforma_feature(
            split_tables[split], encoder, scaler, numeric_columns
        )
        for split in SPLITS
    }

    print("5) Salvo matrici e artefatti riutilizzabili...")
    salva_artefatti(
        args.output_dir, encoder, scaler, matrices, labels, numeric_columns
    )

    print("\nPreprocessing classico completato.")
    for split in SPLITS:
        matrix = matrices[split]
        print(f"{split}: X={matrix.shape}, nnz={matrix.nnz:,}, y={labels[split].shape}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
