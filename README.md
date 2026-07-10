# CV Project - Motorcycle Detection & Tracking

Application Streamlit pour detecter des motos sur images et suivre les objets dans une video avec Ultralytics YOLO.

Auteur: Vina Raharitsifa - M1 I2AD, INSI

## Structure

```text
CV_Project/
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Image_Detection.py
│   │   └── 2_Video_Tracking.py
│   └── utils/
│       ├── inference.py
│       ├── tracker.py
│       ├── visualization.py
│       ├── io_utils.py
│       └── model_loader.py
├── models/
├── data/samples/
├── notebooks/
└── tests/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Modele

Place ton modele YOLO dans `models/best.pt` ou `models/last.pt`.

Si le modele n'est pas present localement, l'application telecharge automatiquement les poids depuis Google Drive:

- `best.pt`: https://drive.google.com/file/d/1zOpySTI7k6Rcwc13aUw271D1V22-sEe6/view?usp=sharing
- `last.pt`: https://drive.google.com/file/d/1DWn9li9PkObgXZzO1vWIVsLzAQ2XP6IN/view?usp=sharing

Option runtime: definir une URL avant de lancer Streamlit pour telecharger le modele automatiquement si le fichier n'existe pas:

```powershell
$env:CV_BEST_MODEL_URL="https://.../best.pt"
```

Variables supportees: `CV_BEST_MODEL_URL`, `CV_LAST_MODEL_URL`, `CV_MODEL_URL`.

## Lancer l'application

```bash
streamlit run app/Home.py
```

## Tests

```bash
pytest
```

## GitHub

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Vina771/CV_Project.git
git push -u origin main
```
