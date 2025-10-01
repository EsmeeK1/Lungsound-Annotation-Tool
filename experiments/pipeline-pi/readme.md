# README — Model Bundles maken voor LongPi
Dit project gebruikt model-bundles. Een bundle bevat alles wat nodig is: het TFLite-model, de preprocessing, labels en een manifest met metadata.\

## 1. Bundlestructuur
Maak een map in models/bundles/ met een duidelijke naam en versie, bijvoorbeeld:

models/bundles/project-2.2.0/
├─ model.tflite
├─ preprocess.py
├─ labels.json
└─ manifest.json

## 2. Bestandsspecificaties

``model.tflite``
- Het TensorFlow Lite-model dat je exporteert.
- Eén input-tensor, shape en dtype moeten kloppen met manifest.json.

``preprocess.py``
- Plugin die raw audio → modelinput maakt.
- Voorbeeld zie ``preprocess_example.py``
- De ``callable`` die terugkomt moet een numpy-array opleveren met exact shape en dtype zoals in manifest.

``labels.json``
- JSON-array met één label per klasse.
- Voorbeeld zie ``label_example.py``

``manifest.json``
- Metadata over model, preprocessing en controle.
- Voorbeeld zie ``manifest_example.json``
- Optioneel: "model_sha256": "<hash>" (SHA-256 van model.tflite). Gebruik hiervoor ``hash_model.py``


## 3. Versiebeheer
- Elke wijziging → nieuwe map (bijv. project-2.2.1).
- Oude bundles blijven staan voor rollback.
- Welke bundle actief is, bepaalt deployment.json op de Pi (daar hoef jij niets mee).