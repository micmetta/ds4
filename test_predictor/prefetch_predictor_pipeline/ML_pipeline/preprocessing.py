#!/usr/bin/env python3
"""Prepara i file tabellari e i tensori di training di BUILD_1.

Input:
    derived/prefetch_dataset_build_1.jsonl

Output:
    training/prefetch_training_build_1.parquet  (tabella ispezionabile)
    training/arrays/X_<split>.npy               (feature numeriche)
    training/arrays/y_<split>.npy               (target multi-hot, 256 expert)

Questo script non addestra alcun modello: prepara soltanto un dataset pulito,
con split separati per prompt, che uno script di training potra' usare.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import pyarrow  # richiesto da pandas per scrivere Parquet
except ImportError as error:
    raise SystemExit(
        "Dipendenza mancante. Installa i requisiti con:\n"
        "  python3 -m pip install -r "
        "test_predictor/prefetch_predictor_pipeline/build_dataset_1/requirements.txt\n\n"
        f"Dettaglio: {error}"
    ) from error


# Il router DeepSeek-V4 Flash ha 256 expert routed e seleziona Top-6.
NUM_EXPERTS = 256
TOP_K = 6
ROUTER_SCORES = 16
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PIPELINE_ROOT / "build_dataset_1" / "derived" / "prefetch_dataset_build_1.jsonl"
DEFAULT_OUTPUT = PIPELINE_ROOT / "build_dataset_1" / "ml" / "training"


def leggi_argomenti() -> argparse.Namespace:
    """
    Legge gli argomenti opzionali della riga di comando.

    Args:
        Nessun argomento Python. I valori arrivano dalla riga di comando:
        ``--input``, ``--output`` e ``--seed``.

    Returns:
        Namespace con i percorsi di input/output e il seed scelto per lo split.
    """
    parser = argparse.ArgumentParser(
        description="Prepara Parquet e tensori .npy dal JSONL derivato BUILD_1."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def carica_record_jsonl(percorso_input: Path) -> list[dict[str, Any]]:
    """
    Legge e valida tutti i record supervisionati del JSONL derivato.

    Args:
        percorso_input: File ``prefetch_dataset_build_1.jsonl`` da leggere.

    Returns:
        Lista di record JSON validati. Ogni elemento rappresenta una coppia
        ``(token t, layer l)`` con il target al token successivo.

    Raises:
        ValueError: Se il file non contiene record validi o un record non
        rispetta lo schema BUILD_1.
    """
    record = []

    with percorso_input.open("r", encoding="utf-8") as file_input:
        for numero_riga, riga in enumerate(file_input, start=1):
            riga = riga.strip()
            if not riga:
                continue

            esempio = json.loads(riga)
            controlla_record(esempio, numero_riga)
            record.append(esempio)

    if not record:
        raise ValueError(f"Il file {percorso_input} non contiene record validi.")

    return record


def controlla_record(esempio: dict[str, Any], numero_riga: int) -> None:
    """
    Verifica che un record JSONL sia compatibile con le ipotesi di BUILD_1.

    Args:
        esempio: Record JSON decodificato dalla riga corrente del dataset.
        numero_riga: Numero della riga nel file JSONL, usato nei messaggi di errore.

    Returns:
        ``None``. Se il record e' valido, la funzione termina senza modifiche.

    Raises:
        ValueError: Se mancano campi obbligatori, la history non ha lunghezza 1,
        routing/target non sono Top-6 oppure i raw logits non sono 16.
    """
    campi_obbligatori = [
        "prompt_id",
        "prompt_category",
        "token_position",
        "token_id",
        "layer",
        "routing_history",
        "router_logits",
        "target_experts_next_token",
    ]

    for campo in campi_obbligatori:
        if campo not in esempio:
            raise ValueError(f"Riga {numero_riga}: manca il campo '{campo}'.")

    history = esempio["routing_history"]
    logits = esempio["router_logits"]
    target = esempio["target_experts_next_token"]
    if history["history_length"] != 1:
        raise ValueError(
            f"Riga {numero_riga}: BUILD_1 supporta per ora solo history_length=1."
        )

    if len(history["expert_ids"][0]) != TOP_K or len(target) != TOP_K:
        raise ValueError(f"Riga {numero_riga}: attesi {TOP_K} expert per routing/target.")

    if logits["stored_count"] != ROUTER_SCORES:
        raise ValueError(
            f"Riga {numero_riga}: attesi {ROUTER_SCORES} raw logits, "
            f"trovati {logits['stored_count']}."
        )


def crea_split_per_prompt(
    record: list[dict[str, Any]], seed: int
) -> dict[int, str]:
    """
    Crea lo split 80/10/10 per prompt, mantenendo separate le categorie.

    Args:
        record: Lista dei record JSONL gia' validati.
        seed: Seed pseudo-casuale che rende lo split riproducibile.

    Returns:
        Dizionario ``prompt_id -> split``, dove lo split e' ``train``,
        ``validation`` oppure ``test``. Tutte le righe dello stesso prompt
        ricevono lo stesso valore.
    """
    prompt_per_categoria = defaultdict(set)
    for esempio in record:
        prompt_per_categoria[esempio["prompt_category"]].add(esempio["prompt_id"])

    generatore = random.Random(seed)
    split_per_prompt = {}
    for categoria, prompt_ids in sorted(prompt_per_categoria.items()):
        prompt_ids = sorted(prompt_ids)
        generatore.shuffle(prompt_ids)

        numero_prompt = len(prompt_ids)
        numero_train = int(numero_prompt * 0.8)
        numero_validation = int(numero_prompt * 0.1)

        for prompt_id in prompt_ids[:numero_train]:
            split_per_prompt[prompt_id] = "train"
        for prompt_id in prompt_ids[numero_train : numero_train + numero_validation]:
            split_per_prompt[prompt_id] = "validation"
        for prompt_id in prompt_ids[numero_train + numero_validation :]:
            split_per_prompt[prompt_id] = "test"

    return split_per_prompt


def crea_colonne_tabellari_e_tensori(
    record: list[dict[str, Any]], split_per_prompt: dict[int, str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trasforma i record JSONL in una tabella e nelle matrici numeriche X/y.

    Args:
        record: Lista dei record JSONL gia' validati.
        split_per_prompt: Dizionario ``prompt_id -> split`` prodotto da
        ``crea_split_per_prompt()``.

    Returns:
        Tupla composta da:

        - tabella Pandas con tutte le colonne ispezionabili nel Parquet;
        - matrice ``X`` float32, shape ``[N, 41]``;
        - matrice ``y`` uint8 multi-hot, shape ``[N, 256]``;
        - array int16 dei target Top-6 originali, shape ``[N, 6]``;
        - array con lo split assegnato a ogni riga.
    """
    colonne = defaultdict(list)
    righe_x = []
    righe_y = []
    target_ids = []
    split_per_riga = []

    for esempio in record:
        history = esempio["routing_history"]
        router_logits = esempio["router_logits"]
        history_expert_ids = history["expert_ids"][0]
        experts_t = esempio["experts_t"]
        target = esempio["target_experts_next_token"]
        split = split_per_prompt[esempio["prompt_id"]]

        # Colonne descrittive: utili per esplorare il dataset in PyCharm.
        colonne["split"].append(split)
        colonne["prompt_id"].append(esempio["prompt_id"])
        colonne["prompt_category"].append(esempio["prompt_category"])
        colonne["sequence_id"].append(esempio["sequence_id"])
        colonne["token_position"].append(esempio["token_position"])
        colonne["token_id"].append(esempio["token_id"])
        colonne["layer"].append(esempio["layer"])
        colonne["history_length"].append(history["history_length"])
        colonne["history_token_position_0"].append(history["token_positions"][0])

        for indice in range(TOP_K):
            colonne[f"experts_t_{indice}"].append(experts_t[indice])
            colonne[f"routing_history_expert_{indice}"].append(history_expert_ids[indice])
            colonne[f"target_expert_{indice}"].append(target[indice])

        for indice in range(ROUTER_SCORES):
            colonne[f"router_logit_expert_id_{indice}"].append(
                router_logits["expert_ids_ranked"][indice]
            )
            colonne[f"router_logit_value_{indice}"].append(
                router_logits["values_ranked"][indice]
            )

        # Matrice X: ordine dichiarato in feature_columns.json.
        feature = [
            esempio["token_position"],
            esempio["token_id"],
            esempio["layer"],
            *history_expert_ids,
            *router_logits["expert_ids_ranked"],
            *router_logits["values_ranked"],
        ]
        righe_x.append(feature)

        # Multi-label: y[e] = 1 se l'expert e appartiene a E_l(t+1).
        label = np.zeros(NUM_EXPERTS, dtype=np.uint8)
        label[target] = 1
        righe_y.append(label)
        target_ids.append(target)
        split_per_riga.append(split)

    return (
        pd.DataFrame(colonne),
        np.asarray(righe_x, dtype=np.float32),
        np.asarray(righe_y, dtype=np.uint8),
        np.asarray(target_ids, dtype=np.int16),
        np.asarray(split_per_riga),
    )


def salva_output(
    tabella: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    target_ids: np.ndarray,
    split_per_riga: np.ndarray,
    output_dir: Path,
    split_per_prompt: dict[int, str],
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    """
    Salva il Parquet, i metadati JSON e gli array NumPy separati per split.

    Args:
        tabella: DataFrame completo da scrivere in formato Parquet.
        x: Matrice di feature ``X`` con shape ``[N, 41]``.
        y: Matrice multi-label ``y`` con shape ``[N, 256]``.
        target_ids: Target Top-6 originali con shape ``[N, 6]``.
        split_per_riga: Nome dello split assegnato a ciascuna riga.
        output_dir: Cartella ``training/`` in cui salvare gli output.
        split_per_prompt: Mapping riproducibile ``prompt_id -> split``.
        seed: Seed da registrare insieme al mapping degli split.

    Returns:
        Tupla con il percorso del Parquet creato e il riepilogo di conteggi e
        shape per train, validation e test.
    """
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    percorso_parquet = output_dir / "prefetch_training_build_1.parquet"
    tabella.to_parquet(percorso_parquet, engine="pyarrow", index=False)

    nomi_feature = [
        "token_position",
        "token_id",
        "layer",
        *[f"routing_history_expert_{i}" for i in range(TOP_K)],
        *[f"router_logit_expert_id_{i}" for i in range(ROUTER_SCORES)],
        *[f"router_logit_value_{i}" for i in range(ROUTER_SCORES)],
    ]

    descrizione_feature = {
        "feature_columns": nomi_feature,
        "shape_X": ["numero_esempi", len(nomi_feature)],
        "shape_y": ["numero_esempi", NUM_EXPERTS],
        "target": "y[e] = 1 se l'expert e e' in target_experts_next_token",
        "note": (
            "Gli ID di token ed expert sono indici discreti: un futuro modello "
            "dovra' trattarli come categorie/embedding, non come grandezze continue. "
            "La feature token_embedding 4096D non e' ancora materializzata in BUILD_1: "
            "il JSONL conserva il token_id per recuperarla dal GGUF in una build successiva."
        ),
    }
    (output_dir / "feature_columns.json").write_text(
        json.dumps(descrizione_feature, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "split_by_prompt.json").write_text(
        json.dumps({"seed": seed, "split_per_prompt": split_per_prompt}, indent=2) + "\n",
        encoding="utf-8",
    )

    riepilogo = {}
    for split in ("train", "validation", "test"):
        indici = np.flatnonzero(split_per_riga == split)
        np.save(arrays_dir / f"X_{split}.npy", x[indici])
        np.save(arrays_dir / f"y_{split}.npy", y[indici])
        np.save(arrays_dir / f"target_expert_ids_{split}.npy", target_ids[indici])

        riepilogo[split] = {
            "examples": int(len(indici)),
            "prompts": int(sum(1 for valore in split_per_prompt.values() if valore == split)),
            "X_shape": list(x[indici].shape),
            "y_shape": list(y[indici].shape),
        }

    riepilogo["total_examples"] = int(len(tabella))
    riepilogo["parquet"] = str(percorso_parquet)
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(riepilogo, indent=2) + "\n", encoding="utf-8"
    )

    return percorso_parquet, riepilogo


def main() -> None:
    """
    Esegue l'intero preprocessing BUILD_1 nell'ordine definito dallo script.

    Args:
        Nessun argomento Python. Configurazione e percorsi sono letti da
        ``leggi_argomenti()``.

    Returns:
        ``None``. I risultati persistenti sono i file creati nella cartella
        di output e il riepilogo stampato sul terminale.
    """
    argomenti = leggi_argomenti()
    if not argomenti.input.is_file():
        raise SystemExit(f"Input non trovato: {argomenti.input}")

    print("1) Leggo il dataset JSONL derivato...")
    record = carica_record_jsonl(argomenti.input)
    print(f"   Record letti: {len(record):,}")

    print("2) Creo lo split train/validation/test per prompt e categoria...")
    split_per_prompt = crea_split_per_prompt(record, argomenti.seed)

    print("3) Creo la tabella Parquet e i tensori X/y...")
    tabella, x, y, target_ids, split_per_riga = crea_colonne_tabellari_e_tensori(
        record, split_per_prompt
    )

    print("4) Salvo i file di training...")
    percorso_parquet, riepilogo = salva_output(
        tabella,
        x,
        y,
        target_ids,
        split_per_riga,
        argomenti.output,
        split_per_prompt,
        argomenti.seed,
    )

    print("\nPreprocessing BUILD_1 completato.")
    print(f"Parquet: {percorso_parquet}")
    for split in ("train", "validation", "test"):
        dati = riepilogo[split]
        print(
            f"{split}: {dati['examples']:,} esempi, {dati['prompts']} prompt, "
            f"X={dati['X_shape']}, y={dati['y_shape']}"
        )


if __name__ == "__main__":
    main()
