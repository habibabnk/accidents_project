# RoadSight — French Road Accident Analytics

A data analysis and machine learning web app for exploring French road accident
patterns and estimating the severity of accidents based on contextual conditions.

Built on the official BAAC (Bulletins d'Analyse des Accidents Corporels) dataset
published by the French Ministry of the Interior on [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/).

> **Screenshot placeholder** — add `docs/screenshot.png` after first run

---

## What the model predicts

The Random Forest classifier estimates **P(serious | accident occurred)** — the
probability that a reported accident involves at least one fatality or hospitalised
injury, given a set of contextual inputs (time, location, weather, lighting).

It does **not** predict whether an accident will happen. All outputs are statistical
averages over historical data; they do not reflect real-time road conditions.

---

## Features

- **Overview dashboard** — yearly trends, geographic distribution by department,
  time-of-day heatmap, weather breakdown
- **Interactive filtering** — filter by year, department, hour range and weather
  with live chart updates
- **Severity prediction** — Random Forest with feature importance display;
  inputs: month, hour, day-of-week, department, weather, lighting

---

## Data coverage

| Detail | Value |
|--------|-------|
| Source | BAAC (Ministère de l'Intérieur) via data.gouv.fr |
| Years  | 2021 – 2024 |
| Rows   | ~221,000 accidents |
| Tables | `caracteristiques` + `usagers` per year |
| Train / test split | 2021–2023 / 2024 (time-based) |

Earlier years (2015–2020) use a different schema and are not included.

---

## Model metrics (2024 held-out test set)

| Metric | Value |
|--------|-------|
| Accuracy | ~0.76 |
| AUC-ROC  | ~0.72 |
| Serious rate (train) | ~26.7% |

*Run the app and click "Entraîner le modèle" to reproduce these numbers.*

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Interface | Streamlit |
| Data processing | Pandas |
| Machine learning | Scikit-learn (Random Forest) |
| Visualisation | Plotly |

---

## Installation

```bash
git clone https://github.com/habibabnk/accidents_project
cd accidents_project
pip install -r requirements.txt
```

---

## Running the app

The repo ships with a **5,000-row demo sample** (`data/sample/`) so the app
launches immediately without downloading full data:

```bash
streamlit run app.py
```

Available at `http://localhost:8501`. The sidebar shows "SAMPLE MODE" when
running on demo data.

---

## Downloading the full dataset

```bash
python download_data.py          # all years 2021-2024
python download_data.py --year 2024   # single year
```

Files are saved to `2021/`, `2022/`, `2023/`, `2024/` and total ~250 MB.
These directories are gitignored; only the sample subset is committed.

---

## Project structure

```
accidents_project/
├── app.py               # Streamlit application
├── data_loader.py       # CSV loading, normalisation, gravity decoding
├── modeling.py          # Random Forest pipeline
├── download_data.py     # Fetch full BAAC CSVs from data.gouv.fr
├── data_dictionary.md   # BAAC field codes and gravity format notes
├── requirements.txt
├── data/
│   └── sample/          # 5k-row demo subset (committed)
│       ├── caract-2024.csv
│       └── usagers-2024.csv
├── 2021/ … 2024/        # Full BAAC data (gitignored, download separately)
└── accidents.ipynb      # Exploratory notebook
```

---

## Limitations & methodology

- The feature set is limited to fields present in the `caracteristiques` table
  (time, location, weather, lighting, road type). Speed, driver profile, and
  vehicle condition are not available.
- The model is a conditional severity estimator, not an accident predictor.
  A high score means "accidents in these conditions tend to be serious", not
  "an accident is likely here".
- Class imbalance (~73% non-serious) is handled with `class_weight='balanced'`
  in the Random Forest.
- The dataset covers mainland France and overseas departments only. DOM-TOM
  department codes may appear but coverage is partial.
- Historical patterns may not reflect current road infrastructure or enforcement.

---

## Author

**Habiba Benkemouche** — [LinkedIn](https://www.linkedin.com/in/habiba-benkemouche-56b168264) · [GitHub](https://github.com/habibabnk)

---

## License

MIT License — see [LICENSE](LICENSE).
