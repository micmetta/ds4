"""Small shared helpers for sparse Keras feed-forward networks."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import sparse


# Hide TensorFlow C++ informational messages while preserving warnings and
# errors emitted by Python or TensorFlow itself. Users may override it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


TOP_K = 6
NUM_EXPERTS = 256
RANDOM_STATE = 42
SPLITS = ("train", "validation", "test")


def require_tensorflow():
    """
    Import TensorFlow only when a neural-network command is executed.

    Returns:
        The imported TensorFlow module.

    Raises:
        RuntimeError: If TensorFlow is not installed in the ML environment.
    """
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow non e' installato. Esegui 'uv sync' dalla cartella "
            "ML_pipeline prima di lanciare gli script neural_network."
        ) from error
    return tf


def parse_hidden_layers(value: str, argument_name: str = "--hidden-layers") -> list[int]:
    """
    Parse a comma-separated list of positive hidden-layer sizes.

    Args:
        value: Text such as ``\"512,256,128\"``.
        argument_name: Nome dell'argomento CLI da usare nei messaggi di errore.

    Returns:
        Ordered list of positive layer sizes.

    Raises:
        ValueError: If the list is empty or contains a non-positive value.
    """
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(
            f"{argument_name} deve contenere interi separati da virgole."
        ) from error
    if not layers or any(units < 1 for units in layers):
        raise ValueError(f"{argument_name} deve contenere almeno un intero positivo.")
    return layers


def require_new_output_directory(output_dir: Path) -> None:
    """
    Refuse an existing experiment directory before any output is written.

    Args:
        output_dir: Requested directory for one training or tuning run.

    Returns:
        None. The caller may create the directory after this validation.

    Raises:
        FileExistsError: If the requested experiment directory already exists.
    """
    if output_dir.exists():
        raise FileExistsError(
            f"La cartella di output esiste gia': {output_dir}. "
            "Scegli una nuova --output-dir per non sovrascrivere l'esperimento."
        )


def load_split(data_dir: Path, split: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Load one sparse feature matrix and its multi-hot targets.

    Args:
        data_dir: Directory containing ``X_<split>.npz`` or the compatible
            classical name ``X_classic_<split>.npz``, plus ``y_<split>.npy``.
        split: One of train, validation or test.

    Returns:
        Tuple ``(features, labels)`` with CSR float32 features and uint8 labels.

    Raises:
        FileNotFoundError: If one required input file is missing.
        ValueError: If the feature and label shapes are inconsistent.
    """
    features_path = data_dir / f"X_{split}.npz"
    if not features_path.is_file():
        features_path = data_dir / f"X_classic_{split}.npz"
    labels_path = data_dir / f"y_{split}.npy"
    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"{split}: attesi {features_path.name} e {labels_path.name} in {data_dir}"
        )

    features = sparse.load_npz(features_path).tocsr().astype(np.float32)
    labels = np.load(labels_path)
    if labels.shape != (features.shape[0], NUM_EXPERTS):
        raise ValueError(
            f"{split}: y ha shape {labels.shape}; attesa "
            f"{(features.shape[0], NUM_EXPERTS)}."
        )
    if labels.dtype != np.uint8 or not np.all(labels.sum(axis=1) == TOP_K):
        raise ValueError(f"{split}: ogni target deve essere uint8 con {TOP_K} expert attivi.")
    return features, labels


def load_labels(data_dir: Path, split: str) -> np.ndarray:
    """Load and validate labels without loading the CSR state matrix."""
    labels_path = data_dir / f"y_{split}.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(f"{split}: label non trovate in {data_dir}")
    labels = np.load(labels_path, mmap_mode="r")
    if labels.ndim != 2 or labels.shape[1] != NUM_EXPERTS:
        raise ValueError(f"{split}: y ha shape {labels.shape}; attesa (*, {NUM_EXPERTS}).")
    if labels.dtype != np.uint8 or not np.all(labels.sum(axis=1) == TOP_K):
        raise ValueError(f"{split}: ogni target deve essere uint8 con {TOP_K} expert attivi.")
    return labels


def load_fixed_nnz_state_inputs(
    state_input_dir: Path,
    split: str,
    expected_examples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load contiguous fixed-NNZ state slots materialized from a CSR matrix."""
    arrays_dir = state_input_dir / "arrays"
    metadata_path = state_input_dir / "artifacts" / "fixed_nnz_state_metadata.json"
    indices_path = arrays_dir / f"state_feature_indices_{split}.npy"
    values_path = arrays_dir / f"state_feature_values_{split}.npy"
    if not indices_path.is_file() or not values_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{split}: input state fixed-NNZ mancanti in {state_input_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_count = metadata.get("feature_count")
    nonzero_features = metadata.get("nonzero_features_per_example")
    input_slots = metadata.get("input_slots")
    if not all(isinstance(value, int) and value > 0 for value in (feature_count, nonzero_features, input_slots)):
        raise ValueError("Metadata state fixed-NNZ non valida.")
    if input_slots != nonzero_features + 1:
        raise ValueError("Metadata state fixed-NNZ: atteso uno slot bias aggiuntivo.")
    indices = np.load(indices_path, mmap_mode="r")
    values = np.load(values_path, mmap_mode="r")
    expected_shape = (expected_examples, input_slots)
    if indices.shape != expected_shape or values.shape != expected_shape:
        raise ValueError(f"{split}: state fixed-NNZ ha shape incompatibile; attesa {expected_shape}.")
    if indices.dtype != np.uint16 or values.dtype != np.float32:
        raise ValueError(f"{split}: state fixed-NNZ richiede indici uint16 e valori float32.")
    if np.any(indices[:, -1] != feature_count) or not np.all(values[:, -1] == 1.0):
        raise ValueError(f"{split}: slot bias state fixed-NNZ non valido.")
    if np.any(indices[:, :-1] >= feature_count):
        raise ValueError(f"{split}: indice feature state fuori intervallo.")
    return indices, values, {
        "feature_count": feature_count,
        "nonzero_features_per_example": nonzero_features,
        "input_slots": input_slots,
    }


def csr_to_sparse_tensor(matrix: sparse.csr_matrix, tf):
    """
    Convert one CSR mini-batch into a TensorFlow SparseTensor.

    Args:
        matrix: CSR feature mini-batch.
        tf: Imported TensorFlow module.

    Returns:
        Reordered float32 ``tf.SparseTensor`` with the same matrix shape.
    """
    coo = matrix.tocoo()
    indices = np.column_stack((coo.row, coo.col)).astype(np.int64, copy=False)
    tensor = tf.SparseTensor(
        indices=indices,
        values=coo.data.astype(np.float32, copy=False),
        dense_shape=np.asarray(coo.shape, dtype=np.int64),
    )
    return tf.sparse.reorder(tensor)


def make_sparse_dataset(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    tf,
):
    """
    Create a streaming tf.data.Dataset without densifying the CSR matrix.

    Args:
        features: Full CSR matrix for one split.
        labels: Matching multi-hot targets.
        batch_size: Number of rows yielded by each batch.
        shuffle: Whether to shuffle row order once per epoch.
        tf: Imported TensorFlow module.

    Returns:
        Dataset yielding sparse feature batches and float32 label batches.
    """
    if batch_size < 1:
        raise ValueError("--batch-size deve essere un intero positivo.")

    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle else None

    def batches() -> Iterator[tuple[object, np.ndarray]]:
        row_ids = np.arange(features.shape[0])
        if random_generator is not None:
            random_generator.shuffle(row_ids)
        for start in range(0, len(row_ids), batch_size):
            selected = row_ids[start : start + batch_size]
            yield csr_to_sparse_tensor(features[selected], tf), labels[selected].astype(np.float32)

    signature = (
        tf.SparseTensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def load_token_inputs(token_data_dir: Path, split: str, expected_examples: int) -> tuple[np.ndarray, np.ndarray]:
    """Load compact token embedding indices and standardized positions for one split."""
    indices_path = token_data_dir / f"token_embedding_index_{split}.npy"
    positions_path = token_data_dir / f"token_position_{split}.npy"
    if not indices_path.is_file() or not positions_path.is_file():
        raise FileNotFoundError(f"{split}: input token mancanti in {token_data_dir}")
    indices = np.load(indices_path, mmap_mode="r")
    positions = np.load(positions_path, mmap_mode="r")
    if indices.shape != (expected_examples,) or positions.shape != (expected_examples, 1):
        raise ValueError(f"{split}: shape input token incompatibili con le feature sparse.")
    if not np.isfinite(positions).all() or np.any(indices < 0):
        raise ValueError(f"{split}: input token non validi.")
    return indices, positions


def load_raw_hidden_inputs(raw_hidden_data_dir: Path, split: str, expected_examples: int):
    """Load raw hidden references and train-only layer-wise normalization statistics."""
    arrays_dir = raw_hidden_data_dir / "arrays"
    artifacts_dir = raw_hidden_data_dir / "artifacts"
    paths_path = artifacts_dir / "hidden_file_paths.json"
    mean_path = artifacts_dir / "hidden_train_mean_by_layer.npy"
    std_path = artifacts_dir / "hidden_train_std_by_layer.npy"
    required = [
        arrays_dir / f"hidden_file_index_{split}.npy",
        arrays_dir / f"hidden_offset_bytes_{split}.npy",
        arrays_dir / f"hidden_layer_index_{split}.npy",
        paths_path, mean_path, std_path,
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"{split}: input raw hidden mancanti in {raw_hidden_data_dir}")
    file_indices = np.load(required[0], mmap_mode="r")
    offsets = np.load(required[1], mmap_mode="r")
    layer_indices = np.load(required[2], mmap_mode="r")
    file_paths = json.loads(paths_path.read_text(encoding="utf-8"))
    means = np.load(mean_path, mmap_mode="r")
    stds = np.load(std_path, mmap_mode="r")
    if (file_indices.shape != (expected_examples,) or offsets.shape != (expected_examples,)
            or layer_indices.shape != (expected_examples,)):
        raise ValueError(f"{split}: riferimenti raw hidden incompatibili con le feature state.")
    if means.shape != (40, 4096) or stds.shape != (40, 4096) or not isinstance(file_paths, list):
        raise ValueError("Artefatti di normalizzazione raw hidden non validi.")
    if np.any(file_indices < 0) or np.any(file_indices >= len(file_paths)) or np.any(layer_indices < 0) or np.any(layer_indices >= 40):
        raise ValueError(f"{split}: indici raw hidden fuori intervallo.")
    return file_indices, offsets, layer_indices, file_paths, means, stds


def load_materialized_hidden_inputs(hidden_data_dir: Path, split: str, expected_examples: int) -> np.ndarray:
    """Load one contiguous float16 normalized-hidden array as a memory map."""
    path = hidden_data_dir / "arrays" / f"raw_hidden_normalized_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{split}: hidden materializzati mancanti in {hidden_data_dir}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (expected_examples, 4096) or values.dtype != np.float16:
        raise ValueError(f"{split}: shape o dtype degli hidden materializzati non validi.")
    return values


def load_quantized_hidden_inputs(hidden_data_dir: Path, split: str, expected_examples: int) -> tuple[np.ndarray, float]:
    """Load one contiguous int8 hidden array and its train-calibrated scale."""
    path = hidden_data_dir / "arrays" / f"raw_hidden_normalized_int8_{split}.npy"
    summary_path = hidden_data_dir / "artifacts" / "quantization_summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"{split}: hidden int8 o summary mancanti in {hidden_data_dir}")
    values = np.load(path, mmap_mode="r")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scale = float(summary["dequantization_scale"])
    if values.shape != (expected_examples, 4096) or values.dtype != np.int8 or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{split}: shape, dtype o scala degli hidden int8 non validi.")
    return values, scale


def load_materialized_kv_inputs(kv_data_dir: Path, split: str, expected_examples: int) -> np.ndarray:
    """Load one contiguous train-normalized 512D KV array as a memory map."""
    path = kv_data_dir / "arrays" / f"kv_normalized_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{split}: KV materializzati mancanti in {kv_data_dir}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (expected_examples, 512) or values.dtype != np.float16:
        raise ValueError(f"{split}: shape o dtype dei KV materializzati non validi.")
    return values


class RawHiddenReader:
    """Read raw hidden vectors through a small LRU cache of binary files."""

    def __init__(self, file_paths: list[str], max_open_files: int = 64) -> None:
        self.file_paths = file_paths
        self.max_open_files = max_open_files
        self.files: OrderedDict[int, object] = OrderedDict()

    def read_batch(self, file_indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        vectors = np.empty((len(file_indices), 4096), dtype=np.float32)
        for row, (file_index, offset) in enumerate(zip(file_indices, offsets)):
            index = int(file_index)
            handle = self.files.pop(index, None)
            if handle is None:
                handle = Path(self.file_paths[index]).open("rb")
                if len(self.files) >= self.max_open_files:
                    self.files.popitem(last=False)[1].close()
            self.files[index] = handle
            handle.seek(int(offset))
            vector = np.fromfile(handle, dtype=np.float32, count=4096)
            if vector.shape != (4096,):
                raise ValueError(f"Vettore hidden incompleto: {self.file_paths[index]}:{offset}")
            vectors[row] = vector
        return vectors

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()
        self.files.clear()


def make_token_embedding_dataset(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    token_indices: np.ndarray,
    token_positions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    tf,
):
    """Create a streaming dataset with sparse state features and two dense token inputs."""
    if batch_size < 1:
        raise ValueError("--batch-size deve essere un intero positivo.")
    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle else None

    def batches() -> Iterator[tuple[dict[str, object], np.ndarray]]:
        row_ids = np.arange(features.shape[0])
        if random_generator is not None:
            random_generator.shuffle(row_ids)
        for start in range(0, len(row_ids), batch_size):
            selected = row_ids[start : start + batch_size]
            yield {
                "state_features": csr_to_sparse_tensor(features[selected], tf),
                "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
                "token_position": token_positions[selected].astype(np.float32, copy=False),
            }, labels[selected].astype(np.float32)

    signature = (
        {
            "state_features": tf.SparseTensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
            "token_embedding_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "token_position": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def make_raw_hidden_token_embedding_dataset(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    token_indices: np.ndarray,
    token_positions: np.ndarray,
    hidden_file_indices: np.ndarray,
    hidden_offsets: np.ndarray,
    hidden_layer_indices: np.ndarray,
    hidden_file_paths: list[str],
    hidden_means: np.ndarray,
    hidden_stds: np.ndarray,
    batch_size: int,
    shuffle: bool,
    tf,
):
    """Create batches with state, frozen-token inputs and standardized raw hidden vectors."""
    if batch_size < 1:
        raise ValueError("--batch-size deve essere un intero positivo.")
    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle else None

    def batches() -> Iterator[tuple[dict[str, object], np.ndarray]]:
        reader = RawHiddenReader(hidden_file_paths)
        try:
            row_ids = np.arange(features.shape[0])
            if random_generator is not None:
                random_generator.shuffle(row_ids)
            for start in range(0, len(row_ids), batch_size):
                selected = row_ids[start : start + batch_size]
                layers = hidden_layer_indices[selected]
                raw_hidden = reader.read_batch(hidden_file_indices[selected], hidden_offsets[selected])
                normalized_hidden = (raw_hidden - hidden_means[layers]) / hidden_stds[layers]
                yield {
                    "state_features": csr_to_sparse_tensor(features[selected], tf),
                    "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
                    "token_position": token_positions[selected].astype(np.float32, copy=False),
                    "raw_hidden_state": normalized_hidden.astype(np.float32, copy=False),
                }, labels[selected].astype(np.float32)
        finally:
            reader.close()

    signature = (
        {
            "state_features": tf.SparseTensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
            "token_embedding_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "token_position": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "raw_hidden_state": tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def make_materialized_hidden_token_embedding_dataset(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    token_indices: np.ndarray,
    token_positions: np.ndarray,
    normalized_hidden: np.ndarray,
    batch_size: int,
    shuffle_batches: bool,
    tf,
    shuffle_block_size: int | None = None,
):
    """Create contiguous batches with reproducibly shuffled macro-block order."""
    if batch_size < 1:
        raise ValueError("--batch-size deve essere un intero positivo.")
    if shuffle_block_size is None:
        shuffle_block_size = batch_size
    if shuffle_block_size < batch_size:
        raise ValueError("Lo shuffle block deve contenere almeno un batch.")
    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle_batches else None

    def batches() -> Iterator[tuple[dict[str, object], np.ndarray]]:
        starts = np.arange(0, features.shape[0], shuffle_block_size)
        if random_generator is not None:
            random_generator.shuffle(starts)
        for block_start in starts:
            block_stop = min(int(block_start) + shuffle_block_size, features.shape[0])
            for start in range(int(block_start), block_stop, batch_size):
                stop = min(start + batch_size, block_stop)
                selected = slice(start, stop)
                yield {
                    "state_features": csr_to_sparse_tensor(features[selected], tf),
                    "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
                    "token_position": token_positions[selected].astype(np.float32, copy=False),
                    "raw_hidden_state": normalized_hidden[selected].astype(np.float16, copy=False),
                }, labels[selected].astype(np.float32)

    signature = (
        {
            "state_features": tf.SparseTensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
            "token_embedding_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "token_position": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "raw_hidden_state": tf.TensorSpec(shape=(None, 4096), dtype=tf.float16),
        },
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def make_quantized_hidden_token_embedding_dataset(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    token_indices: np.ndarray,
    token_positions: np.ndarray,
    quantized_hidden: np.ndarray,
    batch_size: int,
    shuffle_blocks: bool,
    tf,
    shuffle_block_size: int = 65536,
):
    """Yield contiguous int8 hidden batches with reproducibly shuffled macro-blocks."""
    if batch_size < 1 or shuffle_block_size < batch_size:
        raise ValueError("Batch o shuffle block non validi.")
    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle_blocks else None

    def batches() -> Iterator[tuple[dict[str, object], np.ndarray]]:
        starts = np.arange(0, features.shape[0], shuffle_block_size)
        if random_generator is not None:
            random_generator.shuffle(starts)
        for block_start in starts:
            block_stop = min(int(block_start) + shuffle_block_size, features.shape[0])
            for start in range(int(block_start), block_stop, batch_size):
                stop = min(start + batch_size, block_stop)
                selected = slice(start, stop)
                yield {
                    "state_features": csr_to_sparse_tensor(features[selected], tf),
                    "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
                    "token_position": token_positions[selected].astype(np.float32, copy=False),
                    "raw_hidden_state": quantized_hidden[selected],
                }, labels[selected].astype(np.float32)

    signature = (
        {
            "state_features": tf.SparseTensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
            "token_embedding_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "token_position": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "raw_hidden_state": tf.TensorSpec(shape=(None, 4096), dtype=tf.int8),
        },
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def make_quantized_hidden_token_embedding_kv_dataset(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    token_indices: np.ndarray,
    token_positions: np.ndarray,
    quantized_hidden: np.ndarray,
    normalized_kv: np.ndarray,
    batch_size: int,
    shuffle_blocks: bool,
    tf,
    shuffle_block_size: int = 65536,
):
    """Yield aligned state, int8 hidden, token and direct 512D KV batches."""
    if batch_size < 1 or shuffle_block_size < batch_size:
        raise ValueError("Batch o shuffle block non validi.")
    if normalized_kv.shape != (features.shape[0], 512) or normalized_kv.dtype != np.float16:
        raise ValueError("KV materializzati incompatibili con le feature state.")
    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle_blocks else None

    def batches() -> Iterator[tuple[dict[str, object], np.ndarray]]:
        starts = np.arange(0, features.shape[0], shuffle_block_size)
        if random_generator is not None:
            random_generator.shuffle(starts)
        for block_start in starts:
            block_stop = min(int(block_start) + shuffle_block_size, features.shape[0])
            for start in range(int(block_start), block_stop, batch_size):
                stop = min(start + batch_size, block_stop)
                selected = slice(start, stop)
                yield {
                    "state_features": csr_to_sparse_tensor(features[selected], tf),
                    "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
                    "token_position": token_positions[selected].astype(np.float32, copy=False),
                    "raw_hidden_state": quantized_hidden[selected],
                    "kv_current_state": normalized_kv[selected].astype(np.float16, copy=False),
                }, labels[selected].astype(np.float32)

    signature = (
        {
            "state_features": tf.SparseTensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
            "token_embedding_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "token_position": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "raw_hidden_state": tf.TensorSpec(shape=(None, 4096), dtype=tf.int8),
            "kv_current_state": tf.TensorSpec(shape=(None, 512), dtype=tf.float16),
        },
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
    state_feature_indices: np.ndarray,
    state_feature_values: np.ndarray,
    labels: np.ndarray,
    token_indices: np.ndarray,
    token_positions: np.ndarray,
    quantized_hidden: np.ndarray,
    normalized_kv: np.ndarray,
    batch_size: int,
    shuffle_blocks: bool,
    tf,
    shuffle_block_size: int = 65536,
):
    """Yield contiguous fixed-NNZ state slots with the remaining FFN inputs."""
    if batch_size < 1 or shuffle_block_size < batch_size:
        raise ValueError("Batch o shuffle block non validi.")
    examples, input_slots = state_feature_indices.shape
    if (state_feature_values.shape != (examples, input_slots)
            or labels.shape[0] != examples
            or token_indices.shape != (examples,)
            or token_positions.shape != (examples, 1)
            or quantized_hidden.shape != (examples, 4096)
            or normalized_kv.shape != (examples, 512)):
        raise ValueError("Input fixed-NNZ non allineati.")
    random_generator = np.random.default_rng(RANDOM_STATE) if shuffle_blocks else None

    def batches() -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
        starts = np.arange(0, examples, shuffle_block_size)
        if random_generator is not None:
            random_generator.shuffle(starts)
        for block_start in starts:
            block_stop = min(int(block_start) + shuffle_block_size, examples)
            for start in range(int(block_start), block_stop, batch_size):
                stop = min(start + batch_size, block_stop)
                selected = slice(start, stop)
                yield {
                    "state_feature_indices": state_feature_indices[selected].astype(np.int32, copy=False),
                    "state_feature_values": state_feature_values[selected],
                    "token_embedding_index": token_indices[selected].astype(np.int32, copy=False),
                    "token_position": token_positions[selected].astype(np.float32, copy=False),
                    "raw_hidden_state": quantized_hidden[selected],
                    "kv_current_state": normalized_kv[selected].astype(np.float16, copy=False),
                }, labels[selected].astype(np.float32)

    signature = (
        {
            "state_feature_indices": tf.TensorSpec(shape=(None, input_slots), dtype=tf.int32),
            "state_feature_values": tf.TensorSpec(shape=(None, input_slots), dtype=tf.float32),
            "token_embedding_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "token_position": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "raw_hidden_state": tf.TensorSpec(shape=(None, 4096), dtype=tf.int8),
            "kv_current_state": tf.TensorSpec(shape=(None, 512), dtype=tf.float16),
        },
        tf.TensorSpec(shape=(None, NUM_EXPERTS), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def build_ffn(
    input_features: int,
    hidden_layers: list[int],
    activation: str,
    dropout: float,
    learning_rate: float,
    l2_regularization: float,
    tf,
):
    """
    Build and compile a multi-label sparse-input feed-forward network.

    Args:
        input_features: Number of columns in the sparse feature matrix.
        hidden_layers: Ordered unit count for each hidden Dense layer.
        activation: Keras activation name used by hidden layers.
        dropout: Dropout rate after every hidden layer.
        learning_rate: Adam learning rate.
        l2_regularization: L2 coefficient applied to Dense kernels.
        tf: Imported TensorFlow module.

    Returns:
        Compiled Keras model with 256 sigmoid outputs.

    Raises:
        ValueError: If a numeric hyperparameter is outside its valid range.
    """
    if dropout < 0 or dropout >= 1:
        raise ValueError("--dropout deve essere compreso tra 0 e 1 escluso.")
    if learning_rate <= 0 or l2_regularization < 0:
        raise ValueError("Learning rate deve essere positivo e L2 non negativo.")

    regularizer = tf.keras.regularizers.l2(l2_regularization) if l2_regularization else None
    inputs = tf.keras.Input(shape=(input_features,), sparse=True, name="features")
    values = inputs
    for index, units in enumerate(hidden_layers, start=1):
        values = tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            name=f"hidden_{index}",
        )(values)
        if dropout:
            values = tf.keras.layers.Dropout(dropout, name=f"dropout_{index}")(values)
    # Keep probabilities, loss and Recall@6 numerically stable even when the
    # caller enables the global mixed_float16 policy.
    outputs = tf.keras.layers.Dense(
        NUM_EXPERTS, activation="sigmoid", dtype="float32", name="expert_scores"
    )(values)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="prefetch_ffn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[RecallAtSix(tf)],
        # Keras may enable XLA automatically. Sparse TensorFlow operations
        # used by the first Dense layer are not fully supported by XLA GPU.
        jit_compile=False,
    )
    return model


def build_token_embedding_ffn(
    input_features: int,
    frozen_token_embeddings: np.ndarray,
    hidden_layers: list[int],
    activation: str,
    dropout: float,
    learning_rate: float,
    l2_regularization: float,
    tf,
    token_projection_dimension: int = 64,
):
    """Build a sparse-state FFN with a frozen GGUF token lookup and learned 64D projection."""
    if frozen_token_embeddings.ndim != 2 or frozen_token_embeddings.shape[1] != 4096:
        raise ValueError("Il lookup token deve avere shape [numero_token, 4096].")
    if not hidden_layers:
        raise ValueError("Serve almeno un layer hidden per la branca sparse.")
    if dropout < 0 or dropout >= 1 or learning_rate <= 0 or l2_regularization < 0:
        raise ValueError("Iperparametri FFN non validi.")
    regularizer = tf.keras.regularizers.l2(l2_regularization) if l2_regularization else None
    state_inputs = tf.keras.Input(shape=(input_features,), sparse=True, name="state_features")
    token_indices = tf.keras.Input(shape=(), dtype="int32", name="token_embedding_index")
    token_position = tf.keras.Input(shape=(1,), dtype="float32", name="token_position")
    state_values = tf.keras.layers.Dense(
        hidden_layers[0], activation=activation, kernel_regularizer=regularizer, name="state_hidden_1"
    )(state_inputs)
    if dropout:
        state_values = tf.keras.layers.Dropout(dropout, name="state_dropout_1")(state_values)
    token_values = tf.keras.layers.Embedding(
        input_dim=frozen_token_embeddings.shape[0],
        output_dim=frozen_token_embeddings.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(frozen_token_embeddings.astype(np.float32)),
        trainable=False,
        name="frozen_gguf_token_embedding",
    )(token_indices)
    token_values = tf.keras.layers.Dense(token_projection_dimension, activation="linear", name="token_embedding_projection")(token_values)
    values = tf.keras.layers.Concatenate(name="state_token_fusion")(
        [state_values, token_values, token_position]
    )
    for index, units in enumerate(hidden_layers[1:], start=2):
        values = tf.keras.layers.Dense(
            units, activation=activation, kernel_regularizer=regularizer, name=f"fusion_hidden_{index}"
        )(values)
        if dropout:
            values = tf.keras.layers.Dropout(dropout, name=f"fusion_dropout_{index}")(values)
    outputs = tf.keras.layers.Dense(NUM_EXPERTS, activation="sigmoid", name="expert_scores")(values)
    model = tf.keras.Model(
        inputs={
            "state_features": state_inputs,
            "token_embedding_index": token_indices,
            "token_position": token_position,
        },
        outputs=outputs,
        name="prefetch_token_embedding_ffn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[RecallAtSix(tf)],
        jit_compile=False,
    )
    return model


def build_raw_hidden_token_embedding_ffn(
    input_features: int,
    frozen_token_embeddings: np.ndarray,
    hidden_layers: list[int],
    activation: str,
    dropout: float,
    learning_rate: float,
    l2_regularization: float,
    tf,
    hidden_projection_dimension: int = 64,
    token_projection_dimension: int = 64,
    raw_hidden_dequantization_scale: float | None = None,
):
    """Build a multi-input FFN with trainable projections of raw hidden state and frozen token embeddings."""
    if frozen_token_embeddings.ndim != 2 or frozen_token_embeddings.shape[1] != 4096:
        raise ValueError("Il lookup token deve avere shape [numero_token, 4096].")
    if not hidden_layers or hidden_projection_dimension < 1 or token_projection_dimension < 1:
        raise ValueError("Dimensioni hidden non valide.")
    if dropout < 0 or dropout >= 1 or learning_rate <= 0 or l2_regularization < 0:
        raise ValueError("Iperparametri FFN non validi.")
    regularizer = tf.keras.regularizers.l2(l2_regularization) if l2_regularization else None
    state_inputs = tf.keras.Input(shape=(input_features,), sparse=True, name="state_features")
    raw_hidden_dtype = "int8" if raw_hidden_dequantization_scale is not None else "float16"
    raw_hidden = tf.keras.Input(shape=(4096,), dtype=raw_hidden_dtype, name="raw_hidden_state")
    token_indices = tf.keras.Input(shape=(), dtype="int32", name="token_embedding_index")
    token_position = tf.keras.Input(shape=(1,), dtype="float32", name="token_position")
    state_values = tf.keras.layers.Dense(hidden_layers[0], activation=activation, kernel_regularizer=regularizer, name="state_hidden_1")(state_inputs)
    if dropout:
        state_values = tf.keras.layers.Dropout(dropout, name="state_dropout_1")(state_values)
    hidden_values = raw_hidden
    if raw_hidden_dequantization_scale is not None:
        if not np.isfinite(raw_hidden_dequantization_scale) or raw_hidden_dequantization_scale <= 0:
            raise ValueError("Scala di dequantizzazione raw hidden non valida.")
        hidden_values = tf.keras.layers.Rescaling(
            scale=raw_hidden_dequantization_scale,
            offset=0.0,
            dtype="mixed_float16",
            name="raw_hidden_dequantization",
        )(hidden_values)
    hidden_values = tf.keras.layers.Dense(
        hidden_projection_dimension,
        activation="linear",
        dtype="mixed_float16",
        name="raw_hidden_projection",
    )(hidden_values)
    token_values = tf.keras.layers.Embedding(
        input_dim=frozen_token_embeddings.shape[0], output_dim=frozen_token_embeddings.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(frozen_token_embeddings.astype(np.float32)),
        trainable=False, name="frozen_gguf_token_embedding",
    )(token_indices)
    token_values = tf.keras.layers.Dense(token_projection_dimension, activation="linear", name="token_embedding_projection")(token_values)
    values = tf.keras.layers.Concatenate(name="state_hidden_token_fusion")([state_values, hidden_values, token_values, token_position])
    for index, units in enumerate(hidden_layers[1:], start=2):
        values = tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=regularizer, name=f"fusion_hidden_{index}")(values)
        if dropout:
            values = tf.keras.layers.Dropout(dropout, name=f"fusion_dropout_{index}")(values)
    outputs = tf.keras.layers.Dense(NUM_EXPERTS, activation="sigmoid", name="expert_scores")(values)
    model = tf.keras.Model(
        inputs={"state_features": state_inputs, "raw_hidden_state": raw_hidden, "token_embedding_index": token_indices, "token_position": token_position},
        outputs=outputs, name="prefetch_raw_hidden_token_embedding_ffn",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[RecallAtSix(tf)], jit_compile=False)
    return model


def build_raw_hidden_token_embedding_kv_ffn(
    input_features: int,
    frozen_token_embeddings: np.ndarray,
    hidden_layers: list[int],
    activation: str,
    dropout: float,
    learning_rate: float,
    l2_regularization: float,
    tf,
    hidden_projection_dimension: int = 64,
    token_projection_dimension: int = 64,
    raw_hidden_dequantization_scale: float | None = None,
):
    """Build an FFN with direct normalized 512D KV input and no KV projection."""
    if frozen_token_embeddings.ndim != 2 or frozen_token_embeddings.shape[1] != 4096:
        raise ValueError("Il lookup token deve avere shape [numero_token, 4096].")
    if not hidden_layers or hidden_projection_dimension < 1 or token_projection_dimension < 1:
        raise ValueError("Dimensioni hidden non valide.")
    if dropout < 0 or dropout >= 1 or learning_rate <= 0 or l2_regularization < 0:
        raise ValueError("Iperparametri FFN non validi.")
    if raw_hidden_dequantization_scale is None or not np.isfinite(raw_hidden_dequantization_scale) or raw_hidden_dequantization_scale <= 0:
        raise ValueError("Questo modello richiede una scala int8 raw hidden valida.")

    regularizer = tf.keras.regularizers.l2(l2_regularization) if l2_regularization else None
    state_inputs = tf.keras.Input(shape=(input_features,), sparse=True, name="state_features")
    raw_hidden = tf.keras.Input(shape=(4096,), dtype="int8", name="raw_hidden_state")
    kv_current = tf.keras.Input(shape=(512,), dtype="float16", name="kv_current_state")
    token_indices = tf.keras.Input(shape=(), dtype="int32", name="token_embedding_index")
    token_position = tf.keras.Input(shape=(1,), dtype="float32", name="token_position")

    state_values = tf.keras.layers.Dense(
        hidden_layers[0], activation=activation, kernel_regularizer=regularizer, name="state_hidden_1"
    )(state_inputs)
    if dropout:
        state_values = tf.keras.layers.Dropout(dropout, name="state_dropout_1")(state_values)
    hidden_values = tf.keras.layers.Rescaling(
        scale=raw_hidden_dequantization_scale,
        offset=0.0,
        dtype="mixed_float16",
        name="raw_hidden_dequantization",
    )(raw_hidden)
    hidden_values = tf.keras.layers.Dense(
        hidden_projection_dimension,
        activation="linear",
        dtype="mixed_float16",
        name="raw_hidden_projection",
    )(hidden_values)
    token_values = tf.keras.layers.Embedding(
        input_dim=frozen_token_embeddings.shape[0],
        output_dim=frozen_token_embeddings.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(frozen_token_embeddings.astype(np.float32)),
        trainable=False,
        name="frozen_gguf_token_embedding",
    )(token_indices)
    token_values = tf.keras.layers.Dense(
        token_projection_dimension, activation="linear", name="token_embedding_projection"
    )(token_values)
    values = tf.keras.layers.Concatenate(name="state_hidden_token_kv_fusion")(
        [state_values, hidden_values, token_values, kv_current, token_position]
    )
    for index, units in enumerate(hidden_layers[1:], start=2):
        values = tf.keras.layers.Dense(
            units, activation=activation, kernel_regularizer=regularizer, name=f"fusion_hidden_{index}"
        )(values)
        if dropout:
            values = tf.keras.layers.Dropout(dropout, name=f"fusion_dropout_{index}")(values)
    # Keep probabilities, loss and Recall@6 numerically stable even when the
    # caller enables the global mixed_float16 policy.
    outputs = tf.keras.layers.Dense(
        NUM_EXPERTS, activation="sigmoid", dtype="float32", name="expert_scores"
    )(values)
    model = tf.keras.Model(
        inputs={
            "state_features": state_inputs,
            "raw_hidden_state": raw_hidden,
            "kv_current_state": kv_current,
            "token_embedding_index": token_indices,
            "token_position": token_position,
        },
        outputs=outputs,
        name="prefetch_raw_hidden_token_embedding_kv_ffn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[RecallAtSix(tf)],
        jit_compile=False,
    )
    return model


def build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(
    state_feature_count: int,
    state_input_slots: int,
    frozen_token_embeddings: np.ndarray,
    state_hidden_units: int,
    fusion_hidden_layers: list[int],
    activation: str,
    dropout: float,
    learning_rate: float,
    l2_regularization: float,
    tf,
    hidden_projection_dimension: int,
    token_projection_dimension: int,
    raw_hidden_projection_activation: str,
    token_projection_activation: str,
    kv_projection_dimension: int | None,
    kv_projection_activation: str,
    raw_hidden_dequantization_scale: float,
    raw_hidden_dequantization_dtype: str = "mixed_float16",
    raw_hidden_projection_dtype: str = "mixed_float16",
    kv_input_dtype: str = "float16",
):
    """Costruisce la rete neurale feed-forward multi-input mostrata qui sotto.

    state 4.413D
      (routing history, layer, router ID/score, 5 statistiche KV)
            |
            v
    Dense(512) -> ReLU -> Dropout(0.3) ------------------+
                                                           |
    hidden raw 4.096D -> Dense(64) -----------------------+
                                                           |
    token embedding congelato 4.096D -> Dense(64) --------+
                                                           +--> Concatenate(1.153D) --> Dense(256) --> sigmoid
    KV completo normalizzato 512D -> identita' -----------+       (expert_scores)
                                                           |
    token_position standardizzata 1D ---------------------+

    Il primo ramo e' logicamente un Dense(512) applicato allo stato sparso
    4.413D. Per evitare di ricostruire ogni riga sparsa come vettore denso,
    l'implementazione usa direttamente i 44 valori diversi da zero e uno slot
    sentinella per il termine additivo costante; il risultato matematico e' lo
    stesso di ``state @ kernel + bias``. Con ``state_hidden_units=512`` e
    ``fusion_hidden_layers=[]``, usati nella run corrente, non esistono
    ulteriori strati di fusione fra la
    concatenazione e Dense(256).

    Argomenti ricevuti nella configurazione Build 5 corrente:

    - state_feature_count = 4413: numero di feature possibili del ramo state.
    - state_input_slots = 45: 44 feature memorizzate per esempio piu' uno slot
      sentinella che implementa il termine additivo costante del Dense state.
    - frozen_token_embeddings = array NumPy memory-mapped, shape (7157, 4096),
      dtype float16: una riga per ogni token distinto osservato nel Build 5;
      la tabella viene passata alla rete come embedding non addestrabile.
    - state_hidden_units = 512: numero di neuroni del Dense del solo ramo
      state fixed-NNZ.
    - fusion_hidden_layers = []: nessun Dense aggiuntivo dopo la
      concatenazione; il vettore fuso entra direttamente nel classificatore.
    - activation = "relu": funzione applicata dopo il Dense(512) state.
    - dropout = 0.3: quota di attivazioni state azzerate solo in training.
    - learning_rate = 0.00006997208: tasso di apprendimento di Adam.
    - l2_regularization = 0.0: non applica regolarizzazione L2 ai pesi.
    - tf = modulo TensorFlow, versione 2.21.0: fornisce Keras e le operazioni
      TensorFlow usate per creare e compilare la rete.
    - hidden_projection_dimension = 64: proiezione raw hidden 4096D -> 64D.
    - token_projection_dimension = 64: proiezione token embedding 4096D -> 64D.
    - raw_hidden_projection_activation = "linear" nella baseline; puo' essere
      "relu" per applicare ReLU dopo la proiezione raw hidden.
    - token_projection_activation = "linear" nella baseline; puo' essere
      "relu" per applicare ReLU dopo la proiezione token.
    - kv_projection_dimension = None nella baseline: il KV 512D entra diretto;
      puo' essere 64, 128 o 512 per una proiezione apprendibile del KV.
    - kv_projection_activation = "relu" quando e' presente la proiezione KV.
    - raw_hidden_dequantization_scale = 0.0306656: scala che riconverte gli
      hidden int8 quantizzati in valori in virgola mobile approssimati.
    """
    # Verifica che la rappresentazione fixed-NNZ contenga almeno una feature e
    # uno slot reale piu' lo slot sentinella usato per simulare il bias.
    if state_feature_count < 1 or state_input_slots < 2:
        raise ValueError("Dimensioni state fixed-NNZ non valide.")

    # Il lookup congelato estratto dal GGUF deve contenere un embedding 4096D
    # per ogni token compatto osservato nel dataset.
    if frozen_token_embeddings.ndim != 2 or frozen_token_embeddings.shape[1] != 4096:
        raise ValueError("Il lookup token deve avere shape [numero_token, 4096].")

    # La larghezza del ramo state, gli eventuali layer di fusione e le due
    # dimensioni di proiezione devono sempre essere positive.
    if (
        state_hidden_units < 1
        or any(units < 1 for units in fusion_hidden_layers)
        or hidden_projection_dimension < 1
        or token_projection_dimension < 1
    ):
        raise ValueError("Dimensioni hidden non valide.")

    if raw_hidden_projection_activation not in {"linear", "relu"}:
        raise ValueError("Attivazione proiezione raw hidden non valida.")
    if token_projection_activation not in {"linear", "relu"}:
        raise ValueError("Attivazione proiezione token non valida.")
    if kv_projection_dimension not in {None, 64, 128, 512}:
        raise ValueError("Dimensione proiezione KV non valida.")
    if kv_projection_activation != "relu":
        raise ValueError("La proiezione KV richiede l'attivazione relu.")

    # Controlla gli iperparametri prima di creare i layer Keras.
    if dropout < 0 or dropout >= 1 or learning_rate <= 0 or l2_regularization < 0:
        raise ValueError("Iperparametri FFN non validi.")

    # La scala e' necessaria per riconvertire gli hidden salvati come int8 al
    # loro valore float approssimato prima della proiezione apprendibile.
    if not np.isfinite(raw_hidden_dequantization_scale) or raw_hidden_dequantization_scale <= 0:
        raise ValueError("Questo modello richiede una scala int8 raw hidden valida.")
    if raw_hidden_dequantization_dtype not in {"mixed_float16", "float32"}:
        raise ValueError("Tipo numerico per la dequantizzazione raw hidden non valido.")
    if raw_hidden_projection_dtype not in {"mixed_float16", "float32"}:
        raise ValueError("Tipo numerico per la proiezione raw hidden non valido.")
    if kv_input_dtype not in {"float16", "float32"}:
        raise ValueError("Tipo numerico per l'input KV non valido.")

    # Applica L2 alle matrici apprendibili soltanto quando il coefficiente e'
    # diverso da zero; altrimenti lascia i kernel non regolarizzati.
    regularizer = tf.keras.regularizers.l2(l2_regularization) if l2_regularization else None

    ### CREAZIONE DEI DUE INPUT KERAS DEL RAMO STATE ###
    # Durante model.fit(...), questi input riceveranno le feature state reali
    # del batch corrente, fornite da train_dataset.
    #
    # state_input_slots = 45: 44 feature state attive per esempio piu' uno
    # slot sentinella finale, che implementa il termine additivo costante.
    # Keras definisce qui la forma di un esempio: (45,).
    # Con batch_size=4096, i tensori runtime sono (4096, 45): una riga e' un esempio, una colonna
    # è uno slot della sua rappresentazione state.
    #
    # state_indices: contiene gli INDICI delle feature state.
    # Questo è l'esempio reale della prima riga del training_set (state_indices[0, :]):
    #   - 6 esperti history: 15, 90, 96, 112, 119, 202;
    #   - feature categorica layer_3: 256;
    #   - 16 ID router: 392, 671, 1010, 1154, 1432, 1591, 1950, 2189, 2499, 2671, 3096, 3336, 3615, 3630, 3948, 4389;
    #   - 16 indici router logits: 392, 671, 1010, 1154, 1432, 1591, 1950, 2189, 2499, 2671, 3096, 3336, 3615, 3630, 3948, 4389;
    #   - 5 indici statistiche Key-Value: 4408, 4409, 4410, 4411, 4412;
    #   - indice sentinella del bias: 4413
    #
    # state_feature_values: contiene il VALORE associato all'indice nella stessa colonna (matching esatto con state_indices).
    # Questo è l'esempio reale della prima riga del training_set (state_feature_values[0, :]):
    #   - 6 esperti history, layer_3 e 16 ID router: valore 1;
    #   - 16 router logits standardizzati: -1.61500335, -1.66757727,
    #     -2.11209822, -2.00715661, -1.93500447, -1.81196034, -1.93749273,
    #     -1.83856726, -1.71391559, -1.85004640, -1.84914732, -1.78769433,
    #     -1.74574232, -1.69826663, -1.67387772, -1.67472768;
    #   - 5 statistiche Key-Value: -0.802221656, -0.802221656, -0.060272869, -0.797440290, 1.572362300;
    #   - sentinella: valore 1, quindi aggiunge il vettore bias.
    state_indices = tf.keras.Input(
        shape=(state_input_slots,), dtype="int32", name="state_feature_indices"
    )
    state_feature_values = tf.keras.Input(
        shape=(state_input_slots,), dtype="float32", name="state_feature_values"
    )

    raw_hidden = tf.keras.Input(shape=(4096,), dtype="int8", name="raw_hidden_state")
    kv_current = tf.keras.Input(shape=(512,), dtype=kv_input_dtype, name="kv_current_state")
    token_indices = tf.keras.Input(shape=(), dtype="int32", name="token_embedding_index")
    token_position = tf.keras.Input(shape=(1,), dtype="float32", name="token_position")

    # Crea una tabella di pesi con forma (4414, 512): a ogni possibile
    # feature state, incluso il bias, corrisponde una riga di 512 pesi
    # apprendibili. Durante il forward pass, il layer di embedding qui sotto, riceve state_indices e,
    # per ciascun indice presente nei 45 slot di ogni esempio del batch,
    # recupera dalla tabella la riga di pesi corrispondente.
    #
    # Per esempio, nella prima riga del training set:
    #   indice 15   -> recupera i 512 pesi dell'esperto 15 nella routing history
    #   indice 90   -> recupera i 512 pesi dell'esperto 90 nella routing history
    #   ...
    #   indice 4392 -> recupera la riga dei pesi di router_logit_value_0
    #   ...
    #   indice 4412 -> recupera la riga dei pesi di kv_current_max_abs
    #   indice 4413 -> recupera i 512 pesi della sentinella del bias
    state_weight_rows = tf.keras.layers.Embedding(
        input_dim=state_feature_count + 1,      # 4413 + 1 = 4413 features state + 1 bias righe di pesi
        output_dim=state_hidden_units,          # 512 pesi per riga (param: --state-hidden-units 512)
        embeddings_regularizer=regularizer,     # None: (param: --l2-regularization non passato, default 0.0)
        name="state_fixed_nnz_linear_weights",  # Nome del layer nel riepilogo Keras
    )(state_indices)

    # Reshape di state_feature_values: (4096, 45) -> (4096, 45, 1)
    state_value_column = tf.keras.layers.Reshape(
        (state_input_slots, 1), name="state_fixed_nnz_value_column"
    )(state_feature_values)

    # Calcola il contributo pesato di ogni feature dello route state.
    # INPUT shapes:
    #   - state_weight_rows: (4096, 45, 512)
    #   - state_value_column: (4096, 45, 1)
    # OPERATION:
    #   state_weight_rows x state_value_column (z = xW - in questa fase viene eseguita solo la molti)
    # Output shape:
    #   - state_weighted_rows: (4096, 45, 512)
    state_weighted_rows = tf.keras.layers.Multiply(name="state_fixed_nnz_weighted_rows")(
        [state_weight_rows, state_value_column]
    )

    # Passo 3: Nel Passo 3, per ogni feature attiva i e per ogni neurone j,
    # viene calcolato il singolo prodotto x_i × W_i,j.
    #
    # Per il primo esempio, il valore -1.61500335 della feature router_logit_value_0, per esempio,
    # scala i 512 pesi della sua riga nella matrice W
    state_linear = tf.keras.layers.Lambda(
        lambda values: tf.reduce_sum(values, axis=1),
        output_shape=(state_hidden_units,),
        name="state_fixed_nnz_linear_sum",
    )(state_weighted_rows)

    # Passo 4: Il Passo 4 non esegue una nuova moltiplicazione,
    # i termini x_i × W_{i,j} sono già stati tutti calcolati nel Passo 3.
    # Il suo unico compito è completare il prodotto scalare sommando quei termini sulle feature.
    #
    # Per ciascun neurone j = 1, ..., 512, il risultato concettuale di un normale Dense(512) è:
    # z_j = ∑_i (xi · Wi,j) + bj; per i = 1,..., 14413; j = 1, ..., 512
    #
    # Passo 5: Completamento del calcolo di ogni singolo neurone j aggiungendo la funzione
    # di attivazione ReLU:
    # aj = ReLU(zj) = max(0, zj)
    state_values = tf.keras.layers.Activation(activation, name="state_hidden_1")(state_linear)

    # Sempre Passo 5 doc: Applica il dropout=0.3, in questo modo circa il 30 per cento delle attivazioni ReLU
    # viene temporaneamente azzerato a ogni batch di training. Durante validation, test e inferenza
    # non verrà azzerata nessuna attivazione.
    if dropout:
        state_values = tf.keras.layers.Dropout(dropout, name="state_dropout_1")(state_values)

    # Passo 6: Dequantizzazione degli hidden raw: int8 → float16 mediante la scala calibrata offline
    hidden_values = tf.keras.layers.Rescaling(
        scale=raw_hidden_dequantization_scale,
        offset=0.0,
        dtype=raw_hidden_dequantization_dtype,
        name="raw_hidden_dequantization",
    )(raw_hidden)

    # Passo 7: Il vettore hidden dequantizzato viene poi proiettato da 4.096 a 64 valori.
    # Con activation="linear" esegue soltanto una trasformazione affine; con
    # activation="relu" azzera i valori negativi prima della concatenazione.
    hidden_values = tf.keras.layers.Dense(
        hidden_projection_dimension,
        activation=raw_hidden_projection_activation,
        dtype=raw_hidden_projection_dtype,
        name="raw_hidden_projection",
    )(hidden_values)

    # Passo 8: Recupera l'embedding di 4.096 valori dalla tabella GGUF congelata. I suoi
    # pesi non vengono aggiornati dai gradienti dell'addestramento.
    token_values = tf.keras.layers.Embedding(
        input_dim=frozen_token_embeddings.shape[0],
        output_dim=frozen_token_embeddings.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(frozen_token_embeddings.astype(np.float32)),
        trainable=False,
        name="frozen_gguf_token_embedding",
    )(token_indices)

    # Passo 9: Proiezione lineare apprendibile dell'embedding token da 4.096 valori a
    # 64 valori, senza alterare la tabella di embedding congelata.
    # Con activation="linear" esegue soltanto una trasformazione affine; con
    # activation="relu" azzera i valori negativi prima della concatenazione.
    token_values = tf.keras.layers.Dense(
        token_projection_dimension,
        activation=token_projection_activation,
        name="token_embedding_projection"
    )(token_values)

    # Nella baseline il KV normalizzato 512D entra direttamente. Le varianti
    # del tuning possono applicare Dense(64|128|512) seguito da ReLU.
    kv_values = kv_current
    if kv_projection_dimension is not None:
        kv_values = tf.keras.layers.Dense(
            kv_projection_dimension,
            activation=kv_projection_activation,
            name="kv_current_projection",
        )(kv_values)

    # Concatenazione dei cinque rami. Nella baseline unisce 512 state + 64
    # hidden + 64 token + 512 KV diretto + 1 posizione = 1153 valori. Nel
    # tuning la dimensione KV puo' invece essere 64, 128 o 512 dopo ReLU.
    values = tf.keras.layers.Concatenate(name="state_hidden_token_kv_fusion")(
        [state_values, hidden_values, token_values, kv_values, token_position]
    )

    # Gli strati di fusione sono indipendenti dal Dense del ramo state. Con
    # --fusion-hidden-layers omesso (lista vuota), questo ciclo e' vuoto: il
    # vettore concatenato passa direttamente al classificatore finale.
    for index, units in enumerate(fusion_hidden_layers, start=1):
        values = tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            name=f"fusion_hidden_{index}"
        )(values)
        if dropout:
            values = tf.keras.layers.Dropout(dropout, name=f"fusion_dropout_{index}")(values)

    # ARCHITETTURA: Concatenate(1.153D) -> Dense(256) -> sigmoid.
    # Produce 256 probabilita' indipendenti, una per esperto. float32 mantiene
    # stabile il calcolo della funzione sigmoide, della loss e di Recall@6.
    outputs = tf.keras.layers.Dense(
        NUM_EXPERTS, activation="sigmoid", dtype="float32", name="expert_scores"
    )(values)

    # Collega esplicitamente i sei input ai rispettivi rami e definisce il
    # modello Keras completo che verra' addestrato dal trainer.
    model = tf.keras.Model(
        inputs={
            "state_feature_indices": state_indices,
            "state_feature_values": state_feature_values,
            "raw_hidden_state": raw_hidden,
            "kv_current_state": kv_current,
            "token_embedding_index": token_indices,
            "token_position": token_position,
        },
        outputs=outputs,
        name="prefetch_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn",
    )

    # COMPILAZIONE. Adam aggiorna soltanto i parametri trainable; la binary
    # cross-entropy confronta le 256 sigmoid con le label multi-label; Recall@6
    # misura se gli esperti corretti sono tra i sei score piu' alti.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[RecallAtSix(tf)],
        # Mantiene XLA disattivato: alcune operazioni e/o pipeline del progetto
        # non hanno mostrato un beneficio affidabile con compilazione XLA GPU.
        jit_compile=False,
    )
    # Restituisce il modello compilato, pronto per model.fit oppure profiling.
    return model


class RecallAtSix:
    """Create the Keras metric class lazily, after TensorFlow is imported."""

    def __new__(cls, tf):
        """
        Return a TensorFlow Metric that matches the project Recall@6 definition.

        Args:
            tf: Imported TensorFlow module.

        Returns:
            Keras metric instance named ``recall_at_6``.
        """
        class _RecallAtSix(tf.keras.metrics.Metric):
            def __init__(self) -> None:
                super().__init__(name="recall_at_6")
                self.hit_sum = self.add_weight(name="hit_sum", initializer="zeros")
                self.example_count = self.add_weight(name="example_count", initializer="zeros")

            def update_state(self, y_true, y_pred, sample_weight=None) -> None:
                del sample_weight
                top_indices = tf.math.top_k(y_pred, k=TOP_K).indices
                top_mask = tf.reduce_sum(
                    tf.one_hot(top_indices, depth=NUM_EXPERTS, dtype=tf.float32), axis=1
                )
                hits = tf.reduce_sum(tf.cast(y_true, tf.float32) * top_mask, axis=1)
                self.hit_sum.assign_add(tf.reduce_sum(hits / TOP_K))
                self.example_count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

            def result(self):
                return tf.math.divide_no_nan(self.hit_sum, self.example_count)

            def reset_state(self) -> None:
                self.hit_sum.assign(0.0)
                self.example_count.assign(0.0)

        return _RecallAtSix()


def calculate_recall_at_6(scores: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate the project Recall@6 from all output scores at once.

    Args:
        scores: Predicted float scores with shape ``[N, 256]``.
        targets: Matching multi-hot target matrix with shape ``[N, 256]``.

    Returns:
        Mean fraction of the six target experts found in the predicted Top-6.
    """
    top_indices = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
    rows = np.arange(scores.shape[0])[:, None]
    return float(targets[rows, top_indices].sum(axis=1).mean() / TOP_K)


def save_json(path: Path, content: dict) -> None:
    """
    Write a readable JSON file and create its parent directory when needed.

    Args:
        path: Destination JSON path.
        content: JSON-serializable dictionary.

    Returns:
        None. Writes the file with two-space indentation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
