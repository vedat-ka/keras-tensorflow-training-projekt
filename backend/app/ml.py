from __future__ import annotations

import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any

# Suppress TensorFlow and oneDNN info/warning noise before importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")  # keep oneDNN on but silence its log
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras import Input
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, RandomRotation, RandomTranslation, RandomZoom
from keras.models import Sequential, load_model
from keras.utils import to_categorical

try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None

try:
    import openvino as ov
except ImportError:
    ov = None

if tfds is not None:
    tfds.disable_progress_bar()
    logging.getLogger("tensorflow_datasets").setLevel(logging.ERROR)


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_STATE: dict[str, Any] = {
    "status": "idle",
    "dataset": "digits",
    "training_phase": "idle",
    "message": "Noch kein Training gestartet.",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "last_result": None,
}

STATE_LOCK = Lock()
RUNTIME_LOCK = Lock()
_CANCEL_EVENT = Event()
_OV_CACHE: dict[str, Any] = {} if ov is not None else {}
_KERAS_CACHE: dict[str, Any] = {}
_OV_LOCK = Lock()

DATASET_CONFIGS = {
    "digits": {
        "labels": [str(index) for index in range(10)],
        "display_name": "MNIST Ziffern",
        "num_classes": 10,
        "description": "Handgeschriebene Ziffern 0-9 mit dem MNIST-Datensatz.",
        "model_file": "digits_model.keras",
        "metadata_file": "digits_metadata.json",
    },
    "letters": {
        "labels": [
            *[chr(code) for code in range(ord("A"), ord("Z") + 1)],
            *[chr(code) for code in range(ord("a"), ord("z") + 1)],
        ],
        "display_name": "EMNIST Buchstaben",
        "num_classes": 52,
        "description": "Handgeschriebene Buchstaben A-Z und a-z mit dem EMNIST-ByClass-Datensatz.",
        "model_file": "letters_v2_model.keras",
        "metadata_file": "letters_v2_metadata.json",
    },
}


class TrainingProgressCallback(Callback):
    def __init__(self, dataset_name: str, total_epochs: int, cancel_event: Event) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.total_epochs = total_epochs
        self.started_at = 0.0
        self.cancel_event = cancel_event

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.started_at = time.perf_counter()
        set_training_state(
            training_phase="training",
            current_epoch=0,
            total_epochs=self.total_epochs,
            progress_percent=0,
            epoch_metrics=None,
            message="Training gestartet. Erste Epoche wird berechnet.",
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        current_epoch = epoch + 1
        elapsed_seconds = time.perf_counter() - self.started_at
        average_epoch_seconds = elapsed_seconds / current_epoch if current_epoch else 0.0
        remaining_seconds = max(0.0, average_epoch_seconds * (self.total_epochs - current_epoch))

        set_training_state(
            training_phase="training",
            current_epoch=current_epoch,
            total_epochs=self.total_epochs,
            progress_percent=round((current_epoch / self.total_epochs) * 100, 1),
            epoch_metrics={
                "loss": float(logs.get("loss", 0.0)),
                "accuracy": float(logs.get("accuracy", 0.0)),
                "val_loss": float(logs.get("val_loss", 0.0)),
                "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            },
            estimated_seconds_remaining=round(remaining_seconds, 1),
            message=(
                f"Epoche {current_epoch}/{self.total_epochs} abgeschlossen. "
                f"Validierungsgenauigkeit: {float(logs.get('val_accuracy', 0.0)):.4f}"
            ),
        )
        if self.cancel_event.is_set():
            self.model.stop_training = True


def cancel_training() -> None:
    """Signal the running training to stop after the current epoch."""
    _CANCEL_EVENT.set()


def _recommended_thread_config() -> tuple[int, int]:
    cpu_count = os.cpu_count() or 4
    # For Intel Arrow Lake / Meteor Lake (16+ cores without hyperthreading on
    # P-cores) use most cores for intra-op parallelism (within a single op,
    # e.g. matrix multiply) and a few for inter-op (between independent ops).
    intra_op = max(1, cpu_count - 2)
    inter_op = min(4, max(1, cpu_count // 4))
    return intra_op, inter_op


def configure_runtime() -> dict[str, Any]:
    intra_op, inter_op = _recommended_thread_config()
    threading_configured = True
    with RUNTIME_LOCK:
        tf.config.set_soft_device_placement(True)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra_op)
            tf.config.threading.set_inter_op_parallelism_threads(inter_op)
        except RuntimeError:
            threading_configured = False

    gpu_devices = tf.config.list_physical_devices("GPU")
    inference_device = _get_inference_device() if ov is not None else "CPU (openvino nicht installiert)"
    ov_devices: list[str] = []
    if ov is not None:
        try:
            ov_devices = ov.Core().available_devices
        except Exception:  # noqa: BLE001
            pass
    return {
        "device": "GPU" if gpu_devices else "CPU",
        "cpu_count": os.cpu_count() or 1,
        "intra_op_threads": intra_op,
        "inter_op_threads": inter_op,
        "gpu_count": len(gpu_devices),
        "gpus": [device.name for device in gpu_devices],
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "onednn_enabled": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "1") != "0",
        "threading_configured": threading_configured,
        "inference_device": inference_device,
        "openvino_devices": ov_devices,
    }


def set_training_state(**updates: Any) -> dict[str, Any]:
    with STATE_LOCK:
        TRAINING_STATE.update(updates)
        TRAINING_STATE["updated_at"] = datetime.now(timezone.utc).isoformat()
        return dict(TRAINING_STATE)


def get_training_state() -> dict[str, Any]:
    with STATE_LOCK:
        return dict(TRAINING_STATE)


def get_model_path(dataset_name: str) -> Path:
    return MODELS_DIR / DATASET_CONFIGS[dataset_name]["model_file"]


def get_metadata_path(dataset_name: str) -> Path:
    return MODELS_DIR / DATASET_CONFIGS[dataset_name]["metadata_file"]


def _fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    # EMNIST images are stored transposed relative to the drawing orientation.
    # The only fix needed is a transpose of the spatial axes.
    # rot90_ccw(fliplr(img)) == img.T for square images — so the net correction is just a transpose.
    if images.ndim == 3:
        return np.transpose(images, (0, 2, 1))
    if images.ndim == 4:
        return np.transpose(images, (0, 2, 1, 3))
    raise ValueError(f"Unerwartete EMNIST-Bilddimensionen: {images.shape}")


def _load_digits_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    return train_x, train_y, test_x, test_y


def _load_letters_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tfds is None:
        raise RuntimeError(
            "Buchstaben-Training benoetigt tensorflow-datasets. Installiere das Paket zuerst in der Backend-Umgebung."
        )

    set_training_state(
        training_phase="dataset",
        message="EMNIST ByClass-Datensatz wird geladen oder aus lokalem Cache vorbereitet.",
    )

    train_split = tfds.as_numpy(
        tfds.load("emnist/byclass", split="train", batch_size=-1, as_supervised=True)
    )
    test_split = tfds.as_numpy(
        tfds.load("emnist/byclass", split="test", batch_size=-1, as_supervised=True)
    )
    train_x, train_y = train_split
    test_x, test_y = test_split
    train_x = _fix_emnist_orientation(train_x)
    test_x = _fix_emnist_orientation(test_x)

    # EMNIST ByClass label mapping is 0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z.
    train_mask = train_y >= 10
    test_mask = test_y >= 10
    train_x = train_x[train_mask]
    train_y = train_y[train_mask].astype("int32") - 10
    test_x = test_x[test_mask]
    test_y = test_y[test_mask].astype("int32") - 10

    # Build a class-balanced subset so weak classes (e.g. x/X) are not drowned
    # by frequent classes. This keeps training manageable and far more robust.
    # 3500 × 52 classes = 182k samples.
    train_x, train_y = _balance_samples_per_class(train_x, train_y, samples_per_class=3500)

    return train_x, train_y, test_x, test_y


def _reshape_and_normalize(images: np.ndarray) -> np.ndarray:
    return images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0


def _balance_samples_per_class(
    images: np.ndarray, labels: np.ndarray, samples_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a class-balanced dataset with exactly `samples_per_class` items per class."""
    rng = np.random.default_rng(42)
    indices: list[np.ndarray] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        replace = len(cls_idx) < samples_per_class
        selected_cls = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.append(selected_cls)

    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return images[selected], labels[selected]


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unbekannter Datensatz: {dataset_name}")

    set_training_state(
        training_phase="dataset",
        message=f"Datensatz fuer {DATASET_CONFIGS[dataset_name]['display_name']} wird vorbereitet.",
    )

    if dataset_name == "digits":
        train_x, train_y, test_x, test_y = _load_digits_dataset()
    else:
        train_x, train_y, test_x, test_y = _load_letters_dataset()

    config = DATASET_CONFIGS[dataset_name]
    train_x = _reshape_and_normalize(train_x)
    test_x = _reshape_and_normalize(test_x)
    train_y = to_categorical(train_y, num_classes=config["num_classes"])
    test_y = to_categorical(test_y, num_classes=config["num_classes"])
    return train_x, train_y, test_x, test_y


def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            # Augmentation layers — only active during training (training=True).
            # Small random rotation/zoom/shift bridges the gap between the
            # EMNIST drawing style and freehand canvas strokes.
            RandomRotation(0.12),
            RandomZoom(0.15),
            RandomTranslation(0.12, 0.12),
            Conv2D(32, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def _build_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.cache().shuffle(min(len(images), 10000), reshuffle_each_iteration=True)
    else:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model(dataset_name: str, epochs: int, batch_size: int) -> dict[str, Any]:
    _CANCEL_EVENT.clear()
    tf.keras.backend.clear_session()
    runtime = configure_runtime()
    set_training_state(
        status="training",
        dataset=dataset_name,
        training_phase="startup",
        current_epoch=0,
        total_epochs=epochs,
        progress_percent=0,
        estimated_seconds_remaining=None,
        epoch_metrics=None,
        message=(
            f"Training fuer {DATASET_CONFIGS[dataset_name]['display_name']} gestartet "
            f"auf {runtime['device']}."
        ),
    )

    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    config = DATASET_CONFIGS[dataset_name]

    set_training_state(
        training_phase="model",
        message="Modellarchitektur wird erstellt.",
    )

    model = build_model(config["num_classes"])
    train_dataset = _build_tf_dataset(train_x, train_y, batch_size=batch_size, training=True)
    test_dataset = _build_tf_dataset(test_x, test_y, batch_size=batch_size, training=False)

    set_training_state(
        training_phase="training",
        message="Training laeuft. Fortschritt wird nach jeder Epoche aktualisiert.",
    )

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        verbose=2,
        callbacks=[
            TrainingProgressCallback(dataset_name=dataset_name, total_epochs=epochs, cancel_event=_CANCEL_EVENT),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        ],
    )
    epochs_done = len(history.history.get("loss", []))
    was_cancelled = _CANCEL_EVENT.is_set()
    score = model.evaluate(test_dataset, verbose=0)

    model_path = get_model_path(dataset_name)
    metadata_path = get_metadata_path(dataset_name)
    model.save(model_path)

    result = {
        "dataset": dataset_name,
        "display_name": config["display_name"],
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": float(score[0]),
        "accuracy": float(score[1]),
        "labels": config["labels"],
        "description": config["description"],
        "runtime": runtime,
        "history": {
            "loss": [float(value) for value in history.history.get("loss", [])],
            "accuracy": [float(value) for value in history.history.get("accuracy", [])],
            "val_loss": [float(value) for value in history.history.get("val_loss", [])],
            "val_accuracy": [float(value) for value in history.history.get("val_accuracy", [])],
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.name),
    }

    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Invalidate cached OpenVINO model so it is rebuilt from the new weights.
    with _OV_LOCK:
        _OV_CACHE.pop(dataset_name, None)
        _KERAS_CACHE.pop(dataset_name, None)
    # Remove persisted OV IR so it is rebuilt from the new keras model.
    _ov_ir_path(dataset_name).with_suffix(".xml").unlink(missing_ok=True)
    _ov_ir_path(dataset_name).with_suffix(".bin").unlink(missing_ok=True)

    set_training_state(
        status="ready",
        dataset=dataset_name,
        training_phase="abgebrochen" if was_cancelled else "completed",
        current_epoch=epochs_done,
        total_epochs=epochs,
        progress_percent=round(epochs_done / epochs * 100, 1),
        estimated_seconds_remaining=0,
        message=(
            f"Training abgebrochen nach {epochs_done} Epochen. Accuracy: {result['accuracy']:.4f}"
            if was_cancelled else
            f"Training abgeschlossen. Accuracy: {result['accuracy']:.4f}. OpenVINO GPU wird vorbereitet..."
        ),
        last_result=result,
    )

    # Pre-warm OpenVINO compilation in background so the first predict is instant.
    def _warm_ov() -> None:
        compiled = _get_ov_compiled(dataset_name)
        if compiled is not None:
            dummy = np.zeros((1, 28, 28, 1), dtype="float32")
            compiled(dummy)
            set_training_state(
                message=(
                    f"Training abgebrochen nach {epochs_done} Epochen. Accuracy: {result['accuracy']:.4f}. OpenVINO GPU bereit."
                    if was_cancelled else
                    f"Training abgeschlossen. Accuracy: {result['accuracy']:.4f}. OpenVINO GPU bereit."
                ),
            )

    from threading import Thread as _Thread
    _Thread(target=_warm_ov, daemon=True).start()

    return result


def load_metadata(dataset_name: str) -> dict[str, Any] | None:
    metadata_path = get_metadata_path(dataset_name)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def get_model_status() -> dict[str, Any]:
    state = get_training_state()
    datasets: dict[str, Any] = {}
    for dataset_name, config in DATASET_CONFIGS.items():
        metadata = load_metadata(dataset_name)
        model_path = get_model_path(dataset_name)
        datasets[dataset_name] = {
            "display_name": config["display_name"],
            "description": config["description"],
            "ready": model_path.exists(),
            "metadata": metadata,
        }
    return {
        "training": state,
        "runtime": configure_runtime(),
        "datasets": datasets,
    }


def _prepare_pixels(pixels: list[float]) -> np.ndarray:
    image = np.array(pixels, dtype="float32")
    if image.size != 28 * 28:
        raise ValueError("Es werden genau 784 Pixelwerte erwartet.")

    max_value = float(np.max(image)) if image.size else 0.0
    if max_value > 1.0:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)
    image = image.reshape(1, 28, 28, 1)
    return image


def _ov_ir_path(dataset_name: str) -> Path:
    """Return the base path (without extension) for the persisted OV IR model."""
    stem = Path(DATASET_CONFIGS[dataset_name]["model_file"]).stem
    return MODELS_DIR / f"{stem}_ov"


def _get_inference_device() -> str:
    """Return the best available OpenVINO device: GPU > NPU > CPU."""
    if ov is None:
        return "CPU"
    core = ov.Core()
    for device in ("GPU", "NPU", "CPU"):
        if device in core.available_devices:
            return device
    return "CPU"


def _get_ov_compiled(dataset_name: str) -> Any:
    """Return a cached OpenVINO compiled model, building it on first call.

    The OV IR (xml/bin) is persisted to MODELS_DIR so a server restart only
    needs to compile (fast) instead of re-converting (slow).
    Falls back to None on any error so the Keras cache is used instead.
    """
    if ov is None:
        return None
    with _OV_LOCK:
        if dataset_name in _OV_CACHE:
            return _OV_CACHE[dataset_name]

        ir_xml = _ov_ir_path(dataset_name).with_suffix(".xml")
        device = _get_inference_device()
        logger = logging.getLogger(__name__)

        try:
            if ir_xml.exists():
                # Fast path: IR already on disk, just compile.
                ov_model = ov.Core().read_model(str(ir_xml))
                logger.info("OpenVINO: IR geladen von Disk (%s)", ir_xml)
            else:
                # Slow path: Keras → SavedModel → OV IR → save to disk.
                import tempfile
                keras_model = load_model(get_model_path(dataset_name))
                tmp_dir = tempfile.mkdtemp()
                saved_model_path = os.path.join(tmp_dir, "saved_model")
                try:
                    keras_model.export(saved_model_path, format="tf_saved_model")
                    ov_model = ov.convert_model(saved_model_path)
                finally:
                    # Release TF file locks before cleanup; ignore remaining errors on Windows.
                    import gc; gc.collect()
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                ov.save_model(ov_model, str(ir_xml))
                logger.info("OpenVINO: IR gespeichert nach %s", ir_xml)

            compiled = ov.Core().compile_model(ov_model, device)
            _OV_CACHE[dataset_name] = compiled
            logger.info("OpenVINO: Modell kompiliert fuer %s auf %s", dataset_name, device)
            return compiled

        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenVINO fehlgeschlagen (%s) — Keras-Cache wird genutzt.", exc)
            _KERAS_CACHE[dataset_name] = load_model(get_model_path(dataset_name))
            return None


def _get_keras_cached(dataset_name: str) -> Any:
    """Return a cached Keras model (used when OpenVINO is unavailable)."""
    with _OV_LOCK:
        if dataset_name not in _KERAS_CACHE:
            _KERAS_CACHE[dataset_name] = load_model(get_model_path(dataset_name))
        return _KERAS_CACHE[dataset_name]


def predict_pixels(dataset_name: str, pixels: list[float]) -> dict[str, Any]:
    model_path = get_model_path(dataset_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Fuer '{dataset_name}' wurde noch kein Modell trainiert."
        )

    config = DATASET_CONFIGS[dataset_name]
    image = _prepare_pixels(pixels)

    compiled = _get_ov_compiled(dataset_name)
    if compiled is not None:
        raw = compiled(image)
        prediction = np.array(list(raw.values())[0][0], dtype="float32")
    else:
        prediction = _get_keras_cached(dataset_name).predict(image, verbose=0)[0]

    if len(prediction) != config["num_classes"]:
        raise RuntimeError(
            "Das gespeicherte Modell passt nicht mehr zur aktuellen Klassenkonfiguration. Bitte neu trainieren."
        )
    winner_index = int(np.argmax(prediction))

    ranked = np.argsort(prediction)[::-1][:5]
    top_predictions = [
        {
            "index": int(index),
            "label": config["labels"][int(index)],
            "probability": float(prediction[int(index)]),
        }
        for index in ranked
    ]

    return {
        "dataset": dataset_name,
        "prediction_index": winner_index,
        "prediction_label": config["labels"][winner_index],
        "confidence": float(prediction[winner_index]),
        "top_predictions": top_predictions,
    }
