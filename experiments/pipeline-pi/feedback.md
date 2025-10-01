# 1-10-2025

# Sterke punte
Elk model zit netjes self-contained in zijn eigen map (model, labels, manifest, preprocess). Dit met het versiebeheer maakt het overzichtelijk hoe elk model opgesteld is.

Via deployment.json kun je op de Pi snel wisselen tussen modellen zonder codeaanpassingen. Dat zou goed moeten werken.

Alle logica staat centraal in preprocess.py en leest parameters uit manifest.json → makkelijk onderhoudbaar. Als je hetzelfde principe hebt maar andere labels zou je ook maar 1 algemeen bestand kunnen hebben, maar ik heb alles apart gehouden voor overzicht.

# Verbeterpunten
Eén gedeelde preprocess.py (of preprocess_common.py) op een centrale plek en in elke bundle alleen een mini-shim; voorkomt code drift wat ik nu heb. Dus kijken of dat een optie is of hoe we dat omzeilen met modellen met dezelfde preprocessing maar andere labels.

Ik kan zo snel niet meer verzinnen totdat ik het ga proberen te implementeren in de pi.