# Motorcycle Helmet & Plate Tracking

Application Streamlit de détection et suivi (tracking) de motards à partir d'images ou de vidéos, avec YOLOv8 et ByteTrack. Détecte le port du casque, la présence de plaque d'immatriculation et suit chaque motard sur la durée d'une vidéo.

## Classes détectées
- `NoHelmet`
- `PlateNumber`
- `Rider`
- `WithHelmet`

## Fonctionnalités
- Upload d'une ou plusieurs images → détection YOLOv8
- Upload d'une vidéo → détection + tracking ByteTrack (IDs persistants)
- Téléchargement automatique des poids (`best.pt` / `last.pt`) depuis Google Drive au premier lancement
- Résultats annotés affichables et téléchargeables

## Installation

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app/Home.py
```

Au premier lancement, les poids du modèle seront téléchargés automatiquement dans `models/`.

## Structure du projet

```
CV_Project/
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Image_Detection.py
│   │   └── 2_Video_Tracking.py
│   └── utils/
│       ├── model_loader.py
│       ├── inference.py
│       ├── tracker.py
│       ├── visualization.py
│       └── io_utils.py
├── models/            # poids téléchargés au runtime (non versionnés)
├── data/samples/       # exemples pour tester
├── notebooks/          # expés Colab
└── tests/
```

## Notes
- Les fichiers `.pt` ne sont pas commités dans le repo (voir `.gitignore`). Ils sont téléchargés automatiquement depuis Google Drive via `gdown`.
- Si Google Drive limite le téléchargement (fichiers volumineux), basculer vers une GitHub Release avec les poids attachés.
