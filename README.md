# README – Lung Sound Viewer & Annotation Tool

## English version

### Introduction
This project is a **Lung Sound Viewer & Annotation Tool** built with **PySide6** and **PyQtGraph**.
Developed as part of the **HeartGuard / LungInsight project**, it enables loading, visualizing, annotating, and analyzing lung sound recordings. You can view waveforms and spectrograms, select intervals, label segments (e.g., “Inhalation”, “Exhalation”, “Wheeze”), and export annotations for research or machine learning.

![Screenshot of the Current Tool](images/screenshot_tool.png)

### What is a lung sound?
**Lung sounds** are audio signals recorded from the chest, reflecting airflow and respiratory events.
Common types include:
- **Inhalation**: air entering the lungs
- **Exhalation**: air leaving the lungs
- **Wheeze**: continuous high-pitched sounds
- **Crackle**: brief, discontinuous popping sounds

### Features
- Load `.wav` files and view **waveform** and **spectrogram**
- Create, edit, and delete **labeled time segments**
- Apply a **band-pass filter** to highlight frequency ranges
- **Auto-segment** recordings into overlapping windows
- Export all labels to a **CSV file**
- Save per-file annotations as **JSON sidecars**
- Optional **audio playback** via `sounddevice`

### Requirements
- Python 3.10+
- PySide6
- pyqtgraph
- numpy
- pandas
- soundfile
- scipy
- sounddevice

### Installation
Install dependencies via pip:
```bash
pip install PySide6 pyqtgraph numpy pandas soundfile scipy sounddevice
```

### Usage
Run the application:
```bash
cd viewer_app
python app.py
```
1. Choose a folder containing `.wav` files
2. Optionally fill in metadata (gender, age, recording location)
3. Start annotating by selecting intervals and assigning labels
4. Export annotations to CSV from the right-hand panel

Each `.wav` file will have a corresponding `.json` sidecar with labels and metadata.

## Code description

## File Structure
- `app.py` — entrypoint
- `src/` — package with modules:
  - `config.py` — constants & paths
  - `models.py` — dataclasses `Segment`, `FileState`
  - `utils.py` — helpers, label color map
  - `audio.py` — `bandpass_filter`, `compute_stft_db`, `Player`
  - `dialogs.py` — `StartDialog`, `AutoSegmentDialog`
  - `widgets.py` — custom PyQtGraph `ClickableRegion`
  - `mainwindow.py` — `App` main window

> `labels_dataset.json` will be created next to `app.py` if not present.

### Key components
- **AudioLoader:** loads and parses `.wav` files
- **LabelStore:** manages labeled intervals and exports to JSON/CSV
- **App (Qt MainWindow):** main application with waveform/spectrogram plots, controls, and event-handlers

### Signal processing
- Band-pass filtering to isolate frequency ranges
- Spectrogram computation for visual analysis
- Auto-segmentation into fixed-length windows

### User Interface (UI)
- Waveform and spectrogram visualization
- Interactive interval selection and labeling
- Metadata entry (gender, age, location)
- Right panel with label controls and export options

## License
Free to use and modify for research purposes.
