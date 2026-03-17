from __future__ import annotations

from threading import Lock, Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import DATASET_CONFIGS, cancel_training, get_model_status, get_training_state, predict_pixels, set_training_state, train_model


app = FastAPI(title="Keras TensorFlow Handwriting API")

TRAINING_THREAD: Thread | None = None
TRAINING_THREAD_LOCK = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset: str = Field(default="digits")
    epochs: int = Field(default=15, ge=1, le=50)
    batch_size: int = Field(default=64, ge=16, le=256)


class PredictRequest(BaseModel):
    dataset: str = Field(default="digits")
    pixels: list[float]


def _validate_dataset(dataset_name: str) -> None:
    if dataset_name not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail="Unbekannter Datensatz.")


def _get_active_training_thread() -> Thread | None:
    with TRAINING_THREAD_LOCK:
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            return TRAINING_THREAD
        return None


def _set_active_training_thread(thread: Thread | None) -> None:
    global TRAINING_THREAD
    with TRAINING_THREAD_LOCK:
        TRAINING_THREAD = thread


def _clear_stale_training_state() -> None:
    current_state = get_training_state()
    if current_state.get("status") == "training" and _get_active_training_thread() is None:
        set_training_state(
            status="idle",
            training_phase="idle",
            progress_percent=0,
            current_epoch=0,
            estimated_seconds_remaining=None,
            epoch_metrics=None,
            message="Vorheriges Training ist nicht mehr aktiv. Ein neues Training kann gestartet werden.",
        )


def _run_training(dataset_name: str, epochs: int, batch_size: int) -> None:
    try:
        train_model(dataset_name=dataset_name, epochs=epochs, batch_size=batch_size)
    except Exception as error:
        set_training_state(
            status="error",
            dataset=dataset_name,
            message=f"Training fehlgeschlagen: {error}",
        )
    finally:
        _set_active_training_thread(None)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, object]:
    return {"datasets": DATASET_CONFIGS}


@app.get("/api/status")
def status() -> dict[str, object]:
    return get_model_status()


@app.post("/api/train/cancel")
def cancel_train() -> dict[str, str]:
    if _get_active_training_thread() is None:
        raise HTTPException(status_code=409, detail="Kein laufendes Training zum Abbrechen.")
    cancel_training()
    return {"message": "Abbruch wurde signalisiert. Training stoppt nach der aktuellen Epoche."}


@app.post("/api/train")
def train(request: TrainRequest) -> dict[str, object]:
    _validate_dataset(request.dataset)
    _clear_stale_training_state()
    current_state = get_training_state()
    if current_state["status"] == "training" and _get_active_training_thread() is not None:
        raise HTTPException(status_code=409, detail="Es laeuft bereits ein Training.")

    worker = Thread(
        target=_run_training,
        kwargs={
            "dataset_name": request.dataset,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
        },
        daemon=True,
    )
    _set_active_training_thread(worker)
    worker.start()
    return {
        "message": "Training wurde im Hintergrund gestartet.",
        "dataset": request.dataset,
    }


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    _validate_dataset(request.dataset)
    try:
        return predict_pixels(request.dataset, request.pixels)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
