#!/usr/bin/env python3
"""Addestra e valuta i primi modelli classici per il prefetch predictor."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TOP_K = 6
DEFAULT_LOGISTIC_MAX_ITER = 1_000
MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PIPELINE_ROOT / "build_dataset_1" / "ml" / "classical_ml" / "arrays"
DEFAULT_MODELS_DIR = PIPELINE_ROOT / "build_dataset_1" / "ml" / "classical_ml" / "models"
DEFAULT_REPORTS_DIR = PIPELINE_ROOT / "build_dataset_1" / "ml" / "classical_ml" / "reports"


def addestra_classificatore_binario(
    estimator: LogisticRegression,
    x: sparse.csr_matrix,
    target: np.ndarray,
    expert_id: int,
) -> tuple[int, LogisticRegression]:
    """
    Addestra un singolo classificatore binario per un expert.

    Questa funzione è separata dal wrapper per poter essere eseguita anche da
    un worker joblib in modalità parallela.

    Args:
        estimator: Modello base Logistic Regression da clonare.
        x: Matrice CSR delle feature train.
        target: Colonna multi-hot del solo expert da apprendere.
        expert_id: ID della colonna target; viene restituito per ricomporre
            i modelli nell'ordine corretto.

    Returns:
        Tupla ``(expert_id, modello_addestrato)``.
    """
    binary_model = clone(estimator)
    binary_model.fit(x, target)
    return expert_id, binary_model


class SequentialOneVsRestClassifier:
    """
    Addestra un classificatore binario per label, in sequenza e con progresso.

    La classe conserva l'idea di One-vs-Rest, ma evita il parallelismo per
    rendere esplicito nel terminale quale expert è stato appena addestrato.

    Args:
        estimator: Classificatore binario scikit-learn non ancora addestrato,
            usato come modello base e clonato una volta per expert.

    Returns:
        Un oggetto compatibile con ``fit()`` e ``predict_proba()`` usati dal
        resto dello script.
    """

    def __init__(self, estimator: LogisticRegression) -> None:
        self.estimator = estimator
        self.estimators_: list[LogisticRegression] = []

    def fit(self, x: sparse.csr_matrix, y: np.ndarray) -> "SequentialOneVsRestClassifier":
        """
        Addestra un classificatore binario per ciascuna colonna target.

        Args:
            x: Matrice CSR delle feature train.
            y: Matrice multi-hot con shape ``[N, numero_expert]``.

        Returns:
            L'istanza stessa dopo l'addestramento dei classificatori.

        Raises:
            ValueError: Se ``y`` non è una matrice bidimensionale coerente con
            il target multi-label.
        """
        if y.ndim != 2:
            raise ValueError("y deve avere shape [N, numero_expert].")

        total_experts = y.shape[1]
        self.estimators_ = []
        for expert_id in range(total_experts):
            binary_model = clone(self.estimator)
            binary_model.fit(x, y[:, expert_id])
            self.estimators_.append(binary_model)
            print(
                f"   Logistic Regression: expert {expert_id + 1}/{total_experts} completato",
                flush=True,
            )
        return self

    def predict_proba(self, x: sparse.csr_matrix) -> np.ndarray:
        """
        Calcola la probabilità positiva di ogni expert per ciascun esempio.

        Args:
            x: Matrice CSR degli esempi da valutare.

        Returns:
            Matrice float64 con shape ``[N, numero_expert]``. La colonna i
            contiene la probabilità della classe positiva dell'expert i.

        Raises:
            RuntimeError: Se il metodo viene invocato prima di ``fit()``.
        """
        if not self.estimators_:
            raise RuntimeError("Il modello deve essere addestrato prima di predict_proba().")

        scores = np.zeros((x.shape[0], len(self.estimators_)), dtype=np.float64)
        for expert_id, binary_model in enumerate(self.estimators_):
            positive_index = np.flatnonzero(binary_model.classes_ == 1)
            if len(positive_index) == 1:
                probabilities = binary_model.predict_proba(x)
                scores[:, expert_id] = probabilities[:, positive_index[0]]
        return scores


class ParallelProgressOneVsRestClassifier:
    """
    Addestra classificatori One-vs-Rest in parallelo mostrando il progresso.

    Gli expert completano in un ordine non deterministico. Il messaggio nel
    processo principale conserva però un contatore globale affidabile.

    Args:
        estimator: Classificatore binario Logistic Regression da clonare per
            ogni expert.
        n_jobs: Numero di worker joblib. Il valore -1 usa tutti i core.

    Returns:
        Un oggetto compatibile con ``fit()`` e ``predict_proba()``.
    """

    def __init__(self, estimator: LogisticRegression, n_jobs: int) -> None:
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.estimators_: list[LogisticRegression] = []

    def fit(self, x: sparse.csr_matrix, y: np.ndarray) -> "ParallelProgressOneVsRestClassifier":
        """
        Addestra un modello per expert e stampa ogni completamento ricevuto.

        Args:
            x: Matrice CSR delle feature train.
            y: Matrice multi-hot con shape ``[N, numero_expert]``.

        Returns:
            L'istanza stessa dopo l'addestramento completo.

        Raises:
            ValueError: Se ``y`` non è una matrice bidimensionale coerente.
        """
        if y.ndim != 2:
            raise ValueError("y deve avere shape [N, numero_expert].")

        total_experts = y.shape[1]
        completed_models: list[LogisticRegression | None] = [None] * total_experts
        completed_count = 0
        tasks = (
            joblib.delayed(addestra_classificatore_binario)(
                self.estimator,
                x,
                y[:, expert_id],
                expert_id,
            )
            for expert_id in range(total_experts)
        )

        # ``generator_unordered`` yields every result immediately after its
        # worker finishes, rather than waiting for lower expert IDs first.
        results = joblib.Parallel(
            n_jobs=self.n_jobs,
            return_as="generator_unordered",
        )(tasks)
        for expert_id, binary_model in results:
            completed_models[expert_id] = binary_model
            completed_count += 1
            print(
                "   Logistic Regression: "
                f"expert {expert_id + 1}/{total_experts} completato "
                f"({completed_count}/{total_experts} totali)",
                flush=True,
            )

        if any(model is None for model in completed_models):
            raise RuntimeError("Il training parallelo non ha prodotto tutti i modelli.")
        self.estimators_ = [model for model in completed_models if model is not None]
        return self

    def predict_proba(self, x: sparse.csr_matrix) -> np.ndarray:
        """
        Calcola la probabilità positiva di ogni expert per ciascun esempio.

        Args:
            x: Matrice CSR degli esempi da valutare.

        Returns:
            Matrice float64 con shape ``[N, numero_expert]``.

        Raises:
            RuntimeError: Se il metodo viene invocato prima di ``fit()``.
        """
        if not self.estimators_:
            raise RuntimeError("Il modello deve essere addestrato prima di predict_proba().")

        scores = np.zeros((x.shape[0], len(self.estimators_)), dtype=np.float64)
        for expert_id, binary_model in enumerate(self.estimators_):
            positive_index = np.flatnonzero(binary_model.classes_ == 1)
            if len(positive_index) == 1:
                probabilities = binary_model.predict_proba(x)
                scores[:, expert_id] = probabilities[:, positive_index[0]]
        return scores


def leggi_argomenti() -> argparse.Namespace:
    """
    Legge i modelli e i percorsi scelti dalla riga di comando.

    Args:
        Nessun argomento Python. I valori arrivano da ``--models``,
        ``--data-dir``, ``--evaluate-test``, ``--knn-max-train-samples`` e
        ``--output-suffix``, ``--logistic-max-iter`` e
        ``--logistic-n-jobs``.

    Returns:
        Namespace con la configurazione del training corrente.
    """
    parser = argparse.ArgumentParser(
        description="Addestra modelli classici multi-label e calcola Recall@6."
    )
    parser.add_argument(
        "--models",
        default="logistic_regression,decision_tree,random_forest",
        help=(
            "Elenco separato da virgole. Valori ammessi: "
            + ", ".join(MODEL_NAMES)
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffisso opzionale per modello, report e summary. Per un run con "
            "la sola Logistic Regression, se omesso viene generato "
            "automaticamente da --logistic-max-iter."
        ),
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=DEFAULT_LOGISTIC_MAX_ITER,
        help=(
            "Numero massimo di iterazioni per ogni classificatore binario della "
            "Logistic Regression. Default: 1000."
        ),
    )
    parser.add_argument(
        "--logistic-n-jobs",
        type=int,
        default=-1,
        help=(
            "Numero di classificatori One-vs-Rest Logistic eseguiti in "
            "parallelo. -1 usa tutti i core; 1 li addestra in sequenza e "
            "stampa il progresso per expert. Default: -1."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Calcola Recall@6 anche sul test. Usare solo dopo la scelta del modello.",
    )
    parser.add_argument(
        "--knn-max-train-samples",
        type=int,
        default=20_000,
        help=(
            "Massimo numero di righe train usate da KNN. 0 usa tutto il train, "
            "ma può richiedere molto tempo."
        ),
    )
    return parser.parse_args()


def leggi_nomi_modello(testo_modelli: str) -> list[str]:
    """
    Converte l'elenco testuale di modelli in nomi validati.

    Args:
        testo_modelli: Stringa separata da virgole ricevuta da ``--models``.

    Returns:
        Lista ordinata di nomi modello senza duplicati.

    Raises:
        ValueError: Se l'elenco è vuoto oppure contiene un nome non supportato.
    """
    names = [name.strip() for name in testo_modelli.split(",") if name.strip()]
    if not names:
        raise ValueError("Devi indicare almeno un modello.")
    invalid_names = sorted(set(names) - set(MODEL_NAMES))
    if invalid_names:
        raise ValueError(f"Modelli non supportati: {invalid_names}")
    return list(dict.fromkeys(names))


def crea_nome_output(nome_base: str, suffisso: str, estensione: str) -> str:
    """
    Crea un nome file opzionalmente versionato per un run di training.

    Args:
        nome_base: Nome del modello oppure ``training_summary``.
        suffisso: Etichetta opzionale del run, senza estensione.
        estensione: Estensione finale del file, inclusivo del punto.

    Returns:
        Nome nella forma ``<nome_base>.<estensione>`` oppure
        ``<nome_base>_<suffisso>.<estensione>``.

    Raises:
        ValueError: Se il suffisso contiene caratteri diversi da lettere,
        cifre, trattino e underscore.
    """
    if not suffisso:
        return f"{nome_base}{estensione}"
    if not all(character.isalnum() or character in "_-" for character in suffisso):
        raise ValueError("--output-suffix accetta solo lettere, cifre, '_' e '-'.")
    return f"{nome_base}_{suffisso}{estensione}"


def scegli_suffisso_output(args: argparse.Namespace, selected_models: list[str]) -> str:
    """
    Sceglie un suffisso coerente con la configurazione del training.

    Args:
        args: Argomenti già letti dalla riga di comando.
        selected_models: Nomi dei modelli che verranno addestrati.

    Returns:
        Il suffisso esplicito dell'utente, se presente. Per un run con la sola
        Logistic Regression genera invece ``max_iter_<valore>``. Per run con
        più modelli senza suffisso restituisce una stringa vuota.
    """
    if args.output_suffix:
        return args.output_suffix
    if selected_models == ["logistic_regression"]:
        return f"max_iter_{args.logistic_max_iter}"
    return ""


def carica_split(data_dir: Path, split: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Carica la matrice sparse e le label multi-label di uno split.

    Args:
        data_dir: Cartella ``classical_ml_preprocessing/arrays``.
        split: Nome dello split da caricare: train, validation oppure test.

    Returns:
        Tupla ``(X, y)``: matrice CSR float32 e target multi-hot uint8.

    Raises:
        FileNotFoundError: Se uno dei file richiesti non esiste.
        ValueError: Se shape o contenuto delle label non sono coerenti.
    """
    x_path = data_dir / f"X_classic_{split}.npz"
    y_path = data_dir / f"y_{split}.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"File mancanti per lo split {split}: {x_path}, {y_path}")

    x = sparse.load_npz(x_path).tocsr()
    y = np.load(y_path)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: X ha {x.shape[0]} righe, y ne ha {y.shape[0]}")
    if y.ndim != 2 or y.shape[1] != 256:
        raise ValueError(f"{split}: y deve avere shape [N, 256], trovata {y.shape}")
    if not np.all(y.sum(axis=1) == TOP_K):
        raise ValueError(f"{split}: ogni target deve avere esattamente {TOP_K} expert")
    return x, y


def crea_modello(
    nome_modello: str,
    logistic_max_iter: int,
    logistic_n_jobs: int,
) -> Any:
    """
    Crea un classificatore con parametri iniziali riproducibili.

    Args:
        nome_modello: Uno dei nomi ammessi in ``MODEL_NAMES``.
        logistic_max_iter: Limite di iterazioni da usare esclusivamente per
            ``logistic_regression``.
        logistic_n_jobs: Numero di classifier Logistic addestrati in parallelo.
            Il valore 1 abilita la stampa sequenziale del progresso; -1 usa
            tutti i core disponibili.

    Returns:
        Classificatore scikit-learn non ancora addestrato.

    Raises:
        ValueError: Se il nome richiesto non corrisponde a un modello disponibile.
    """
    if nome_modello == "logistic_regression":
        if logistic_max_iter < 1:
            raise ValueError("--logistic-max-iter deve essere un intero positivo.")
        if logistic_n_jobs == 0 or logistic_n_jobs < -1:
            raise ValueError("--logistic-n-jobs deve essere -1 oppure un intero positivo.")
        base_model = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=logistic_max_iter,
            random_state=RANDOM_STATE,
        )
        if logistic_n_jobs == 1:
            return SequentialOneVsRestClassifier(base_model)
        return ParallelProgressOneVsRestClassifier(base_model, n_jobs=logistic_n_jobs)
    if nome_modello == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if nome_modello == "random_forest":
        return RandomForestClassifier(
            # Il primo tentativo con 100 alberi illimitati ha esaurito la RAM.
            # Questi limiti mantengono il baseline utilizzabile sul dataset BUILD_1.
            n_estimators=30,
            max_depth=16,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if nome_modello == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    raise ValueError(f"Modello non supportato: {nome_modello}")


def seleziona_train_knn(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    max_samples: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Seleziona in modo riproducibile un sottoinsieme train per KNN, se richiesto.

    Args:
        x_train: Matrice CSR train completa.
        y_train: Label multi-hot train complete.
        max_samples: Numero massimo di righe da usare. Il valore 0 usa tutto.

    Returns:
        Tupla ``(X_train, y_train)`` completa oppure ridotta allo stesso insieme
        di indici casuali riproducibili.
    """
    if max_samples == 0 or max_samples >= x_train.shape[0]:
        return x_train, y_train
    if max_samples < 1:
        raise ValueError("--knn-max-train-samples deve essere 0 oppure positivo.")

    generator = np.random.default_rng(RANDOM_STATE)
    indices = np.sort(generator.choice(x_train.shape[0], size=max_samples, replace=False))
    return x_train[indices], y_train[indices]


def estrai_score_positivi(modello: Any, x: sparse.csr_matrix) -> np.ndarray:
    """
    Converte ``predict_proba`` in uno score positivo per ciascuno dei 256 expert.

    Args:
        modello: Classificatore multi-label già addestrato.
        x: Matrice CSR degli esempi da valutare.

    Returns:
        Matrice float64 con shape ``[N, 256]``; ogni colonna è lo score della
        probabilità che il relativo expert appartenga al target.

    Raises:
        ValueError: Se il classificatore restituisce una struttura di probabilità
        non compatibile con un target multi-label da 256 expert.
    """
    probabilities = modello.predict_proba(x)
    if isinstance(probabilities, np.ndarray):
        if probabilities.shape[1] != 256:
            raise ValueError(f"Probabilità con shape inattesa: {probabilities.shape}")
        return probabilities

    if not isinstance(probabilities, list) or len(probabilities) != 256:
        raise ValueError("predict_proba deve restituire 256 colonne o 256 array.")

    scores = np.zeros((x.shape[0], 256), dtype=np.float64)
    for expert_id, (expert_probabilities, classes) in enumerate(
        zip(probabilities, modello.classes_)
    ):
        positive_index = np.flatnonzero(classes == 1)
        if len(positive_index) == 1:
            scores[:, expert_id] = expert_probabilities[:, positive_index[0]]
    return scores


def calcola_recall_at_6(scores: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calcola la media della Recall@6 a partire dagli score di tutti gli expert.

    Args:
        scores: Score predetti con shape ``[N, 256]``.
        y_true: Target multi-hot con shape ``[N, 256]`` e sei label attive.

    Returns:
        Recall@6 media, cioè la frazione media di expert reali presenti nei sei
        expert con score più alto.
    """
    predicted_top6 = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
    hits = y_true[np.arange(y_true.shape[0])[:, None], predicted_top6].sum(axis=1)
    return float(np.mean(hits / TOP_K))


def addestra_e_valuta(
    nome_modello: str,
    modello: Any,
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_validation: sparse.csr_matrix,
    y_validation: np.ndarray,
    x_test: sparse.csr_matrix | None,
    y_test: np.ndarray | None,
    knn_max_train_samples: int,
) -> tuple[Any, dict[str, Any]]:
    """
    Addestra un modello e calcola Recall@6 su train, validation e, se richiesto, test.

    Args:
        nome_modello: Nome usato per gestire il sottoinsieme KNN e il report.
        modello: Classificatore scikit-learn non ancora addestrato.
        x_train: Feature sparse del train.
        y_train: Label multi-hot del train.
        x_validation: Feature sparse della validation.
        y_validation: Label multi-hot della validation.
        x_test: Feature sparse test oppure ``None`` se il test è disabilitato.
        y_test: Label test oppure ``None`` se il test è disabilitato.
        knn_max_train_samples: Limite di righe train riservato al solo KNN.

    Returns:
        Tupla con il modello addestrato e un dizionario contenente tempi,
        dimensione train usata e Recall@6 disponibile per gli split valutati.
    """
    x_fit, y_fit = x_train, y_train
    if nome_modello == "knn":
        x_fit, y_fit = seleziona_train_knn(x_train, y_train, knn_max_train_samples)

    start_time = time.perf_counter()
    modello.fit(x_fit, y_fit)
    training_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    train_scores = estrai_score_positivi(modello, x_fit)
    train_evaluation_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    validation_scores = estrai_score_positivi(modello, x_validation)
    validation_seconds = time.perf_counter() - start_time
    report: dict[str, Any] = {
        "train_examples_used": int(x_fit.shape[0]),
        "training_seconds": training_seconds,
        "train_evaluation_seconds": train_evaluation_seconds,
        "train_recall_at_6": calcola_recall_at_6(train_scores, y_fit),
        "validation_seconds": validation_seconds,
        "validation_recall_at_6": calcola_recall_at_6(validation_scores, y_validation),
    }

    if x_test is not None and y_test is not None:
        start_time = time.perf_counter()
        test_scores = estrai_score_positivi(modello, x_test)
        report["test_seconds"] = time.perf_counter() - start_time
        report["test_recall_at_6"] = calcola_recall_at_6(test_scores, y_test)

    return modello, report


def salva_report(
    reports_dir: Path,
    nome_modello: str,
    report: dict[str, Any],
    output_suffix: str,
) -> None:
    """
    Salva il report JSON di un singolo modello.

    Args:
        reports_dir: Cartella in cui memorizzare i report JSON.
        nome_modello: Nome del modello valutato.
        report: Metriche e tempi prodotti da ``addestra_e_valuta()``.
        output_suffix: Etichetta opzionale che distingue run con parametri diversi.

    Returns:
        ``None``. Il report viene scritto in ``<nome_modello>.json`` oppure in
        ``<nome_modello>_<output_suffix>.json``.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / crea_nome_output(nome_modello, output_suffix, ".json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """
    Esegue il training dei modelli richiesti e salva modelli e metriche.

    Args:
        Nessun argomento Python. I parametri arrivano da ``leggi_argomenti()``.

    Returns:
        ``None``. Ogni modello viene salvato in ``models/`` e ogni report in
        ``reports/``.
    """
    args = leggi_argomenti()
    selected_models = leggi_nomi_modello(args.models)
    output_suffix = scegli_suffisso_output(args, selected_models)

    print("1) Carico le matrici sparse e le label...")
    x_train, y_train = carica_split(args.data_dir, "train")
    x_validation, y_validation = carica_split(args.data_dir, "validation")
    x_test, y_test = (None, None)
    if args.evaluate_test:
        x_test, y_test = carica_split(args.data_dir, "test")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "top_k": TOP_K,
        "test_evaluated": args.evaluate_test,
        "logistic_max_iter": args.logistic_max_iter,
        "logistic_n_jobs": args.logistic_n_jobs,
        "models": {},
    }

    for nome_modello in selected_models:
        print(f"2) Addestro {nome_modello}...")
        modello = crea_modello(
            nome_modello,
            args.logistic_max_iter,
            args.logistic_n_jobs,
        )
        modello, report = addestra_e_valuta(
            nome_modello,
            modello,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            args.knn_max_train_samples,
        )
        if nome_modello == "logistic_regression":
            report["logistic_max_iter"] = args.logistic_max_iter
            report["logistic_n_jobs"] = args.logistic_n_jobs
        model_filename = crea_nome_output(nome_modello, output_suffix, ".joblib")
        joblib.dump(modello, args.models_dir / model_filename)
        salva_report(args.reports_dir, nome_modello, report, output_suffix)
        all_reports["models"][nome_modello] = report
        print(f"   train Recall@6: {report['train_recall_at_6']:.4f}")
        print(f"   validation Recall@6: {report['validation_recall_at_6']:.4f}")
        if "test_recall_at_6" in report:
            print(f"   test Recall@6: {report['test_recall_at_6']:.4f}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_base = "training_summary"
    if len(selected_models) == 1:
        summary_base = f"{summary_base}_{selected_models[0]}"
    summary_filename = crea_nome_output(summary_base, output_suffix, ".json")
    (args.reports_dir / summary_filename).write_text(
        json.dumps(all_reports, indent=2) + "\n", encoding="utf-8"
    )
    print("\nTraining completato.")
    print(f"Modelli: {args.models_dir}")
    print(f"Report: {args.reports_dir}")


if __name__ == "__main__":
    main()
