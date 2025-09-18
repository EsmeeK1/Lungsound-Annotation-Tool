# README – ECG Viewer Annotatie Tool

## 🇳🇱 Nederlandse versie

### Introductie
Dit project is een **ECG Viewer en Annotatie Tool** gebouwd met **PySide6** en **PyQtGraph**.
De tool is ontwikkeld als onderdeel van het **HeartGuard / LungInsight-project**, dat zich richt op slimme monitoring van hart- en longsignalen. Met deze app kun je ECG-data inladen, visualiseren, inzoomen, intervallen selecteren en labels exporteren naar **CSV** voor verder gebruik in machine learning of onderzoek.

### Wat is een ECG-signaal?
Een **elektrocardiogram (ECG)** registreert de elektrische activiteit van het hart over de tijd.
Belangrijkste componenten:

- **P-golf**: depolarisatie van de boezems
- **QRS-complex**: depolarisatie van de kamers (ventrikels)
- **T-golf**: repolarisatie van de kamers

### ECG-leads
De tool ondersteunt de 6 klassieke **limb leads**:

- Lead I, II, III
- aVR, aVL, aVF

Daarnaast wordt een **V1-extended** signaal berekend als gemiddelde van alle leads.

### Features
- Laden van ECG-data uit **.txt-bestanden**
- Automatische berekening van 6 leads en V1
- Interactieve plotweergave met zoom en pan
- Selectie van tijdsintervallen (0.01s resolutie)
- Labels toevoegen aan intervallen, per lead of V1
- Export van annotaties naar **CSV**
- Live readout van geselecteerde interval

### Requirements
- Python 3.10+
- PySide6
- pyqtgraph
- numpy
- pandas

Installatie:
```bash
pip install PySide6 pyqtgraph numpy pandas
```

### Usage
Run de applicatie:

```bash
python ecg_viewer_pyqt.py
```

1. Klik Open .txt om een ECG-bestand te laden
2. Gebruik muis en scroll om te in- en uitzoomen
3. Selecteer een interval in V1 of een individuele lead
4. Kies een label en een target lead → klik Add
5. Exporteer resultaten via Export CSV

## Codebeschrijving
### Kerncomponenten
* **load_ecg_txt:** laad en parse ECG-bestanden, bereken leads
* **LabelStore:** beheer labels en exporteer naar DataFrame/CSV
* **App (Qt MainWindow):** hoofdapplicatie met plots, controls en event-handlers

### Signaalverwerking
* Parse van tijdstempels en resampling
* Berekening van leads III, aVR, aVL en aVF uit Lead I & II
* Extended V1: gemiddelde van alle leads

### User Interface (UI)
* 6-leads grid + V1-avg plot
* Cursor en interactieve interval-selectie
* Rechts paneel met knoppen, labels en export

### Belangrijkste functies
* `open_txt():` data laden en tekenen
* `on_move():` cursor bewegen en waarde tonen
* `_on_any_region_changed():` interval synchroniseren tussen alle leads
* `add_label():` labels toevoegen aan lijst en visualisatie
* `export_csv():` exporteren van alle intervallen


## Credits (Nederlands)

De originele **Tkinter-app** (te vinden in de map `experiments` van deze repository) is afgeleid van het volgende project:
[https://github.com/rediet-getnet/ECG-Signal-Viewer-and-Annotation-Tool](https://github.com/rediet-getnet/ECG-Signal-Viewer-and-Annotation-Tool)

Alle credits gaan naar de oorspronkelijke auteur voor het bouwen van deze versie.
Na het experimenteren met de Tkinter-implementatie is er gekozen om verder te gaan met een **PySide6 + PyQtGraph** applicatie vanwege de verbeterde snelheid en bruikbaarheid. Ook daarna hebben we de applicatie aangepast met specifiek functionaliteiten voor het **HeartGuard** project.