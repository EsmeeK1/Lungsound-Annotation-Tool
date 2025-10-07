# README – Lung Sound Viewer & Annotatie Tool

## Nederlandse versie

### Introductie
Dit project is een **Lung Sound Viewer & Annotatie Tool** gebouwd met **PySide6** en **PyQtGraph**.
Ontwikkeld als onderdeel van het **HeartGuard / LungInsight project**, maakt het laden, visualiseren, annoteren en analyseren van longgeluid-opnames mogelijk. Je kunt golfvormen en spectrogrammen bekijken, intervallen selecteren, segmenten labelen (zoals “Inademing”, “Uitademing”, “Wheeze”) en annotaties exporteren voor onderzoek of machine learning.

### Wat is een longgeluid?
**Longgeluiden** zijn audiosignalen opgenomen vanaf de borst, die luchtstroom en ademhalingsgebeurtenissen weergeven.
Veelvoorkomende typen zijn:
- **Inademing**: lucht die de longen binnenkomt
- **Uitademing**: lucht die de longen verlaat
- **Wheeze**: continue, hoge tonen
- **Crackle**: korte, onderbroken plopgeluiden

### Functies
- Laad `.wav`-bestanden en bekijk **golfvorm** en **spectrogram (STFT)**
- Maak, bewerk en verwijder **gelabelde tijdsegmenten**
- Pas een **bandpass-filter** toe om frequentiegebieden te accentueren
- **Auto-segmenteer** opnames in overlappende vensters
- Exporteer alle labels naar een **CSV-bestand**
- Sla annotaties per bestand op als **JSON-sidecar**
- En **audio-afspelen** via een playback functie

### Vereisten
- Python 3.10+
- PySide6
- pyqtgraph
- numpy
- pandas
- soundfile
- scipy
- sounddevice

### Installatie
Installeer afhankelijkheden via pip:
```bash
pip install PySide6 pyqtgraph numpy pandas soundfile scipy sounddevice matplotlib
python app.py
```

### Gebruik
Start de applicatie:
```bash
python lungsound_viewer_refactored.py
```
1. Kies een map met `.wav`-bestanden
2. Vul optioneel metadata in (geslacht, leeftijd, locatie) --> handig voor latere data analyses
3. Begin met annoteren door intervallen te selecteren en labels toe te wijzen
4. Exporteer annotaties naar CSV via het rechterpaneel

Elk `.wav`-bestand krijgt een bijbehorende `.json` sidecar met labels en metadata.

## Codebeschrijving

### Belangrijkste componenten
- **AudioLoader:** laadt en verwerkt `.wav`-bestanden
- **LabelStore:** beheert gelabelde intervallen en exporteert naar JSON/CSV
- **App (Qt MainWindow):** hoofdapplicatie met golfvorm/spectrogram, bedieningselementen en event-handlers

### Signaalverwerking
- Bandpass-filtering om frequentiegebieden te isoleren
- Spectrogramberekening voor visuele analyse
- Auto-segmentatie in vensters van vaste lengte

### Gebruikersinterface (UI)
- Visualisatie van golfvorm en spectrogram
- Interactieve intervalselectie en labeling
- Metadata-invoer (geslacht, leeftijd, locatie)
- Rechterpaneel met labelbediening en exportopties

## Bestandsstructuur
- `app.py` — startpunt
- `src/` — pakket met modules:
  - `config.py` — constanten en paden
  - `models.py` — dataclasses `Segment`, `FileState`
  - `utils.py` — hulpfuncties, kleurtoewijzing voor labels
  - `audio.py` — `bandpass_filter`, `compute_stft_db`, `Player`
  - `dialogs.py` — `StartDialog`, `AutoSegmentDialog`
  - `widgets.py` — aangepaste PyQtGraph `ClickableRegion`
  - `mainwindow.py` — `App` hoofdvenster

> `labels_dataset.json` wordt aangemaakt naast `app.py` als het nog niet bestaat.

## Licentie
Vrij te gebruiken en aan te passen voor onderzoeksdoeleinden.
