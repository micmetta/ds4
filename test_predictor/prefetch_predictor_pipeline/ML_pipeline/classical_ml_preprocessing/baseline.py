#!/usr/bin/env python3
"""Valuta la baseline deterministica di persistenza del routing."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOP_K = 6

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PIPELINE_ROOT = PIPELINE_DIR.parent
DEFAULT_INPUT = PIPELINE_ROOT / "build_dataset_1" / "ml" / "training" / "prefetch_training_build_1.parquet"
DEFAULT_REPORT = PIPELINE_ROOT / "build_dataset_1" / "ml" / "classical_ml" / "reports" / "baseline.json"


def leggi_argomenti() -> argparse.Namespace:
    """
    Legge i percorsi e l'eventuale richiesta di valutare il test set.

    Args:
        Nessun argomento Python. I valori arrivano da ``--input``, ``--report``
        e ``--evaluate-test`` nella riga di comando.

    Returns:
        Namespace con il Parquet da leggere, il report JSON da scrivere e il
        flag che abilita la valutazione del test set.
    """
    parser = argparse.ArgumentParser(
        description="Calcola Recall@6 della baseline E_l(t) -> E_l(t+1)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Calcola Recall@6 anche sul test. Usare solo dopo le scelte su validation.",
    )
    return parser.parse_args()


def carica_tabella(percorso_input: Path) -> pd.DataFrame:
    """
    Carica il Parquet base e verifica le colonne necessarie alla baseline.

    Args:
        percorso_input: File Parquet prodotto da ``preprocessing.py``.

    Returns:
        Tabella contenente split, expert correnti e target del token successivo.

    Raises:
        FileNotFoundError: Se il file Parquet non esiste.
        ValueError: Se mancano colonne richieste oppure il dataset non contiene
            i tre split previsti.
    """
    if not percorso_input.is_file():
        raise FileNotFoundError(f"Parquet non trovato: {percorso_input}")

    current_columns = [f"experts_t_{index}" for index in range(TOP_K)]
    history_columns = [f"routing_history_expert_{index}" for index in range(TOP_K)]
    target_columns = [f"target_expert_{index}" for index in range(TOP_K)]
    required_columns = ["split", *current_columns, *history_columns, *target_columns]

    table = pd.read_parquet(percorso_input, columns=required_columns)
    missing_splits = {"train", "validation", "test"} - set(table["split"])
    if missing_splits:
        raise ValueError(f"Split mancanti nel Parquet: {sorted(missing_splits)}")
    if not np.array_equal(
        table[current_columns].to_numpy(), table[history_columns].to_numpy()
    ):
        raise ValueError(
            "La routing history H=1 deve coincidere con gli expert correnti E_l(t)."
        )
    return table


def calcola_recall_persistenza(tabella: pd.DataFrame) -> float:
    """
    Calcola Recall@6 copiando gli expert correnti come predizione del passo dopo.

    Args:
        tabella: Righe di un solo split con le sei colonne ``experts_t_*`` e le
            sei colonne ``target_expert_*``.

    Returns:
        Recall@6 media della regola ``predizione = E_l(t)``. Ogni riga vale il
        numero di expert condivisi tra ``E_l(t)`` e ``E_l(t+1)``, diviso sei.

    Raises:
        ValueError: Se una riga non contiene sei expert distinti e validi nel
            routing corrente o nel target.
    """
    current_columns = [f"experts_t_{index}" for index in range(TOP_K)]
    target_columns = [f"target_expert_{index}" for index in range(TOP_K)]

    current = tabella[current_columns].to_numpy(dtype=np.int16)
    target = tabella[target_columns].to_numpy(dtype=np.int16)
    invalid_current = np.any(np.diff(np.sort(current, axis=1), axis=1) == 0)
    invalid_target = np.any(np.diff(np.sort(target, axis=1), axis=1) == 0)
    if (
        np.any(current < 0)
        or np.any(current >= 256)
        or np.any(target < 0)
        or np.any(target >= 256)
        or invalid_current
        or invalid_target
    ):
        raise ValueError("Ogni routing e target deve contenere sei expert distinti tra 0 e 255.")

    # Confronto riga per riga: non confrontare un routing con i target delle altre righe.
    matches = np.equal(current[:, :, None], target[:, None, :]).any(axis=2).sum(axis=1)
    return float(np.mean(matches / TOP_K))


def valuta_split(tabella: pd.DataFrame, split: str) -> dict[str, Any]:
    """
    Calcola il report compatto della baseline per uno split del dataset.

    Args:
        tabella: Tabella Parquet completa già validata.
        split: Nome dello split da selezionare: train, validation oppure test.

    Returns:
        Dizionario serializzabile in JSON con numero di esempi e Recall@6.
    """
    split_table = tabella.loc[tabella["split"] == split]
    return {
        "examples": int(len(split_table)),
        "recall_at_6": calcola_recall_persistenza(split_table),
    }


def salva_report(report: dict[str, Any], percorso_report: Path) -> None:
    """
    Scrive il risultato della baseline in un JSON leggibile e riutilizzabile.

    Args:
        report: Dizionario prodotto dalle valutazioni degli split.
        percorso_report: File JSON di destinazione.

    Returns:
        ``None``. Il report viene scritto con indentazione di due spazi.
    """
    percorso_report.parent.mkdir(parents=True, exist_ok=True)
    percorso_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """
    Esegue la baseline di persistenza e salva Recall@6 su train e validation.

    Args:
        Nessun argomento Python. La configurazione è letta da
        ``leggi_argomenti()``.

    Returns:
        ``None``. Stampa le metriche e crea ``baseline.json``.
    """
    args = leggi_argomenti()
    print("1) Carico il Parquet base...")
    table = carica_tabella(args.input)

    print("2) Copio E_l(t) come predizione di E_l(t+1)...")
    report: dict[str, Any] = {
        "baseline": "routing_persistence",
        "prediction_rule": "predicted_experts = experts_t",
        "top_k": TOP_K,
        "test_evaluated": args.evaluate_test,
        "splits": {},
    }
    for split in ("train", "validation"):
        split_report = valuta_split(table, split)
        report["splits"][split] = split_report
        print(f"   {split} Recall@6: {split_report['recall_at_6']:.4f}")

    if args.evaluate_test:
        split_report = valuta_split(table, "test")
        report["splits"]["test"] = split_report
        print(f"   test Recall@6: {split_report['recall_at_6']:.4f}")

    salva_report(report, args.report)
    print("\nBaseline completata.")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
