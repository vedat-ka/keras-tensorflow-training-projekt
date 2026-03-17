# Keras TensorFlow Handwriting Dashboard

Dieses Projekt kombiniert ein Python-Backend mit FastAPI und ein React-Dashboard mit Vite. Das Backend trainiert ein CNN mit Keras/TensorFlow fuer handgeschriebene Ziffern oder Buchstaben. Im Dashboard kannst du das Training starten und danach direkt im Browser ein Zeichen malen, das vom trainierten Modell erkannt wird.

## Python-Version

TensorFlow laeuft in diesem Projekt nicht mit Python 3.13+ und auch nicht mit Alpha-Versionen wie Python 3.15. Verwende fuer das Backend eine stabile Python-Version 3.10, 3.11 oder 3.12.



## Projektstruktur

```text
kerasTensorflow/
|- backend/
|  |- app/
|  |  |- main.py
|  |  \- ml.py
|  \- requirements.txt
|- frontend/
|  |- src/
|  |  |- App.jsx
|  |  |- main.jsx
|  |  \- styles.css
|  \- package.json
\- README.md
```

## Features

- React-Dashboard fuer Training, Status und Live-Test
- FastAPI-Backend fuer Training und Vorhersage
- CNN-Modell mit Keras/TensorFlow
- Hardware-Status im Dashboard mit CPU- oder GPU-Erkennung
- CPU-optimiertes Training fuer Windows mit TensorFlow-Threading und tf.data-Pipeline
- Trainingsfortschritt mit Epochenanzeige und Fortschrittsbalken
- MNIST fuer Ziffern 0-9
- EMNIST Letters fuer Buchstaben A-Z
- Canvas mit Vorverarbeitung auf 28x28 Pixel

## Backend starten

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Das Backend ist danach unter `http://localhost:8000` erreichbar.


## Fehlerbehebung bei TensorFlow-Installation

Falls du bereits eine `.venv` mit Python 3.14 oder 3.15 erstellt hast, loesche sie und lege sie neu mit Python 3.12 an:

```powershell
cd backend
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Die Ausgabe von `python --version` muss hier `Python 3.12.x` zeigen, bevor du `pip install -r requirements.txt` startest.

## Frontend starten

```powershell
cd frontend
npm install
npm run dev
```

Das Dashboard ist danach unter `http://localhost:5173` erreichbar.

## Ablauf

1. Backend starten.
2. Frontend starten.
3. Im Dashboard einen Datensatz auswaehlen.
4. Training starten.
5. Nach Abschluss in der Zeichenflaeche eine Zahl oder einen Buchstaben zeichnen.
6. Vorhersage senden und das Ergebnis mit Wahrscheinlichkeiten ansehen.

## API-Endpunkte

- `GET /api/health` prueft, ob das Backend laeuft.
- `GET /api/config` liefert Datensatz-Konfigurationen.
- `GET /api/status` liefert Trainingsstatus und Modellinformationen.
- `POST /api/train` startet ein Training im Hintergrund.
- `POST /api/predict` sendet 28x28 Pixelwerte an das Modell.

## Hinweise

- Beim ersten EMNIST-Training laedt TensorFlow Datasets den Datensatz herunter. Das kann etwas dauern.
- Die trainierten Modelle werden in `backend/models/` gespeichert.
- Fuer gute Erkennung sollte das Zeichen moeglichst gross und mittig gezeichnet werden.


Wenn `pip` Meldungen wie `numpy-...-cp314`, `dm-tree`, `CMake must be installed` oder `No matching distribution found for tensorflow` ausgibt, dann wurde die virtuelle Umgebung mit Python 3.14 oder 3.15 erstellt und muss neu aufgebaut werden.