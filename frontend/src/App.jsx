import { useEffect, useRef, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';
const CANVAS_SIZE = 280;
const MODEL_INPUT_SIZE = 28;
const MODEL_CONTENT_SIZE = 22;

const fallbackConfig = {
  digits: {
    labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    display_name: 'MNIST Ziffern',
    description: 'Handgeschriebene Ziffern 0-9',
  },
  letters: {
    labels: [
      ...Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index)),
      ...Array.from({ length: 26 }, (_, index) => String.fromCharCode(97 + index)),
    ],
    display_name: 'EMNIST Buchstaben',
    description: 'Handgeschriebene Buchstaben A-Z und a-z',
  },
};

function App() {
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const [datasets, setDatasets] = useState(fallbackConfig);
  const [status, setStatus] = useState(null);
  const [dataset, setDataset] = useState('digits');
  const [epochs, setEpochs] = useState(15);
  const [batchSize, setBatchSize] = useState(64);
  const [isSubmittingTrain, setIsSubmittingTrain] = useState(false);
  const [isSubmittingPredict, setIsSubmittingPredict] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('Starte ein Training und teste danach dein gezeichnetes Symbol.');

  useEffect(() => {
    prepareCanvas();
  }, []);

  useEffect(() => {
    let active = true;

    const loadConfig = async () => {
      try {
        const configResponse = await fetch(`${API_BASE}/api/config`);
        if (!configResponse.ok) {
          throw new Error('Konfiguration konnte nicht geladen werden.');
        }

        const configData = await configResponse.json();
        if (active) {
          setDatasets(configData.datasets);
        }
      } catch (requestError) {
        if (active) {
          setError(requestError.message);
        }
      }
    };

    loadConfig();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    setPrediction(null);
    setError('');
    setInfo('Starte ein Training und teste danach dein gezeichnetes Symbol.');
  }, [dataset]);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const statusResponse = await fetch(`${API_BASE}/api/status`);

        if (!statusResponse.ok) {
          throw new Error('Backend ist nicht erreichbar.');
        }

        const statusData = await statusResponse.json();

        if (!active) {
          return;
        }

        setStatus(statusData);
      } catch (requestError) {
        if (active) {
          setError(requestError.message);
        }
      }
    };

    load();
    const intervalId = window.setInterval(load, 3000);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const prepareCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext('2d');
    context.fillStyle = '#000000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = 20;
    context.strokeStyle = '#ffffff';
  };

  const getCanvasPoint = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const handlePointerDown = (event) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const point = getCanvasPoint(event);
    isDrawingRef.current = true;
    context.beginPath();
    context.moveTo(point.x, point.y);
  };

  const handlePointerMove = (event) => {
    if (!isDrawingRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const point = getCanvasPoint(event);
    context.lineTo(point.x, point.y);
    context.stroke();
  };

  const stopDrawing = () => {
    isDrawingRef.current = false;
  };

  const clearCanvas = () => {
    prepareCanvas();
    setPrediction(null);
    setInfo('Zeichenflaeche geleert.');
    setError('');
  };

  const extractPixels = () => {
    const sourceCanvas = canvasRef.current;
    const sourceContext = sourceCanvas.getContext('2d');
    const sourceData = sourceContext.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);

    let minX = sourceCanvas.width;
    let minY = sourceCanvas.height;
    let maxX = 0;
    let maxY = 0;
    let hasInk = false;

    for (let y = 0; y < sourceCanvas.height; y += 1) {
      for (let x = 0; x < sourceCanvas.width; x += 1) {
        const offset = (y * sourceCanvas.width + x) * 4;
        const value = sourceData.data[offset];
        if (value > 5) {
          hasInk = true;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    if (!hasInk) {
      return new Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE).fill(0);
    }

    // Keep a thin margin but let the glyph use most of the 28x28 area.
    const padding = 8;
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    const squareSize = Math.max(width, height) + padding * 2;

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = squareSize;
    cropCanvas.height = squareSize;
    const cropContext = cropCanvas.getContext('2d');
    cropContext.fillStyle = '#000000';
    cropContext.fillRect(0, 0, squareSize, squareSize);

    const drawX = (squareSize - width) / 2;
    const drawY = (squareSize - height) / 2;
    cropContext.drawImage(
      sourceCanvas,
      minX,
      minY,
      width,
      height,
      drawX,
      drawY,
      width,
      height
    );

    const fitCanvas = document.createElement('canvas');
    fitCanvas.width = MODEL_INPUT_SIZE;
    fitCanvas.height = MODEL_INPUT_SIZE;
    const fitContext = fitCanvas.getContext('2d');
    fitContext.fillStyle = '#000000';
    fitContext.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    fitContext.drawImage(
      cropCanvas,
      0,
      0,
      squareSize,
      squareSize,
      (MODEL_INPUT_SIZE - MODEL_CONTENT_SIZE) / 2,
      (MODEL_INPUT_SIZE - MODEL_CONTENT_SIZE) / 2,
      MODEL_CONTENT_SIZE,
      MODEL_CONTENT_SIZE
    );

    const fitted = fitContext.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const pixels = [];
    for (let index = 0; index < fitted.data.length; index += 4) {
      pixels.push(fitted.data[index] / 255);
    }
    return pixels;
  };

  const refreshStatus = async () => {
    const response = await fetch(`${API_BASE}/api/status`);
    if (!response.ok) {
      throw new Error('Status konnte nicht geladen werden.');
    }
    const data = await response.json();
    setStatus(data);
  };

  const handleTrain = async () => {
    setError('');
    setInfo('Training wird gestartet.');
    setIsSubmittingTrain(true);

    try {
      const response = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset,
          epochs: Number(epochs),
          batch_size: Number(batchSize),
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || 'Training konnte nicht gestartet werden.');
      }

      setInfo('Training laeuft im Hintergrund. Der Status aktualisiert sich automatisch.');
      await refreshStatus();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmittingTrain(false);
    }
  };

  const handleCancelTrain = async () => {
    try {
      await fetch(`${API_BASE}/api/train/cancel`, { method: 'POST' });
      setInfo('Abbruch wurde gesendet. Training stoppt nach der aktuellen Epoche.');
    } catch {
      setError('Abbruch konnte nicht gesendet werden.');
    }
  };

  const handlePredict = async () => {
    setError('');
    setPrediction(null);
    setIsSubmittingPredict(true);

    try {
      const pixels = extractPixels();
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset, pixels }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || 'Vorhersage fehlgeschlagen.');
      }

      const predictionData = await response.json();
      setPrediction(predictionData);
      setInfo(`Erkannt: ${predictionData.prediction_label}`);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmittingPredict(false);
    }
  };

  const currentTraining = status?.training;
  const runtime = status?.runtime;
  const selectedDatasetStatus = status?.datasets?.[dataset];
  const currentPhase = currentTraining?.training_phase ?? 'idle';
  const progressPercent = Number(currentTraining?.progress_percent ?? 0);
  const progressLabel = currentTraining?.status === 'training'
    ? `${currentTraining?.current_epoch ?? 0} / ${currentTraining?.total_epochs ?? epochs} Epochen`
    : 'Noch kein laufendes Training';
  const showIndeterminateProgress = currentTraining?.status === 'training' && progressPercent === 0;

  return (
    <div className="page-shell">
      <div className="page-glow page-glow-left" />
      <div className="page-glow page-glow-right" />
      <main className="app-layout">
        <section className="hero-panel panel">
          <p className="eyebrow">React Dashboard + Python Backend + Keras/TensorFlow</p>
          <h1>Handgeschriebene Zeichen trainieren und live erkennen.</h1>
          <p className="hero-copy">
            Trainiere ein CNN fuer Ziffern oder Buchstaben und teste das Modell direkt im Browser
            mit einer Zeichenflaeche. Das Backend speichert das trainierte Modell als .keras-Datei.
          </p>

          <div className="hero-metrics">
            <div>
              <span>Aktiver Datensatz</span>
              <strong>{datasets[dataset]?.display_name ?? dataset}</strong>
            </div>
            <div>
              <span>Trainingsstatus</span>
              <strong>{currentTraining?.status ?? 'unbekannt'}</strong>
            </div>
            <div>
              <span>Modell bereit</span>
              <strong>{selectedDatasetStatus?.ready ? 'ja' : 'nein'}</strong>
            </div>
            <div>
              <span>Hardware</span>
              <strong>{runtime?.inference_device ? `OpenVINO · ${runtime.inference_device}` : (runtime?.device ?? 'unbekannt')}</strong>
            </div>
          </div>
        </section>

        <section className="panel control-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Training</p>
              <h2>Modell konfigurieren</h2>
            </div>
            <p className="muted-text">{datasets[dataset]?.description}</p>
          </div>

          <div className="form-grid">
            <label>
              <span>Datensatz</span>
              <select value={dataset} onChange={(event) => setDataset(event.target.value)}>
                {Object.entries(datasets).map(([key, value]) => (
                  <option key={key} value={key}>
                    {value.display_name}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Epochen</span>
              <input
                type="number"
                min="1"
                max="30"
                value={epochs}
                onChange={(event) => setEpochs(event.target.value)}
              />
            </label>

            <label>
              <span>Batch Size</span>
              <input
                type="number"
                min="16"
                max="256"
                step="16"
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
              />
            </label>
          </div>

          <div className="button-row">
            <button onClick={handleTrain} disabled={isSubmittingTrain || currentTraining?.status === 'training'}>
              {isSubmittingTrain || currentTraining?.status === 'training' ? 'Training laeuft...' : 'Training starten'}
            </button>
            {currentTraining?.status === 'training' && (
              <button className="btn-cancel" onClick={handleCancelTrain}>Abbrechen</button>
            )}
          </div>

          <div className="progress-block">
            <div className="progress-header">
              <span>Trainingsfortschritt</span>
              <strong>{currentTraining?.status === 'training' ? `${progressPercent.toFixed(0)}%` : '0%'}</strong>
            </div>
            <div className="progress-track" aria-hidden="true">
              <div
                className={`progress-fill${showIndeterminateProgress ? ' progress-fill-indeterminate' : ''}`}
                style={{ width: `${currentTraining?.status === 'training' ? progressPercent : 0}%` }}
              />
            </div>
            <p className="progress-copy">{progressLabel}</p>
            {currentTraining?.status === 'training' && (
              <p className="phase-copy">Phase: {currentPhase}</p>
            )}
          </div>

          <div className="status-box">
            <p>{currentTraining?.message ?? 'Noch kein Training gestartet.'}</p>
            {runtime && (
              <div className="metric-strip hardware-strip">
                <span>Training: CPU ({runtime.intra_op_threads} Threads)</span>
                <span>Inferenz: {runtime.inference_device ?? runtime.device} (OpenVINO)</span>
                <span>oneDNN: {runtime.onednn_enabled ? 'aktiv' : 'aus'}</span>
                <span>TensorFlow: {runtime.tensorflow_version}</span>
              </div>
            )}
            {currentTraining?.epoch_metrics && (
              <div className="metric-strip">
                <span>Epoch Loss: {currentTraining.epoch_metrics.loss.toFixed(4)}</span>
                <span>Epoch Accuracy: {(currentTraining.epoch_metrics.accuracy * 100).toFixed(2)}%</span>
                <span>Val Accuracy: {(currentTraining.epoch_metrics.val_accuracy * 100).toFixed(2)}%</span>
                <span>Restzeit: {Math.round(currentTraining.estimated_seconds_remaining ?? 0)}s</span>
              </div>
            )}
            {selectedDatasetStatus?.metadata && (
              <div className="metric-strip">
                <span>Accuracy: {(selectedDatasetStatus.metadata.accuracy * 100).toFixed(2)}%</span>
                <span>Loss: {selectedDatasetStatus.metadata.loss.toFixed(4)}</span>
                <span>Epochs: {selectedDatasetStatus.metadata.epochs}</span>
              </div>
            )}
          </div>
        </section>

        <section className="panel canvas-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Live Test</p>
              <h2>Zeichenflaeche</h2>
            </div>
            <p className="muted-text">Mit weissem Strich auf schwarzem Hintergrund zeichnen.</p>
          </div>

          <canvas
            ref={canvasRef}
            className="draw-canvas"
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={stopDrawing}
            onPointerLeave={stopDrawing}
          />

          <div className="button-row">
            <button
              onClick={handlePredict}
              disabled={isSubmittingPredict || !selectedDatasetStatus?.ready}
            >
              {isSubmittingPredict ? 'Erkenne...' : 'Vorhersage senden'}
            </button>
            <button className="button-secondary" onClick={clearCanvas}>
              Zeichenflaeche loeschen
            </button>
          </div>

          <p className="info-text">{info}</p>
          {error && <p className="error-text">{error}</p>}

          {prediction && (
            <div className="prediction-box">
              <div className="prediction-main">
                <span>Ergebnis</span>
                <strong>{prediction.prediction_label}</strong>
                <small>{(prediction.confidence * 100).toFixed(2)}% Sicherheit</small>
              </div>

              <div className="prediction-list">
                {prediction.top_predictions.map((item) => (
                  <div className="prediction-item" key={`${item.label}-${item.index}`}>
                    <span>{item.label}</span>
                    <div className="prediction-bar">
                      <div style={{ width: `${item.probability * 100}%` }} />
                    </div>
                    <strong>{(item.probability * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
