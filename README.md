# French Road Accident Risk Explorer

A data analysis and machine learning web application for exploring French road 
accident patterns and predicting serious accident risk, based on national data 
from 2015 to 2024.

---

## Features

- **Overview dashboard**: yearly trends, geographic distribution by department,
  temporal patterns (day/hour), environmental factors (weather, lighting)
- **Interactive filtering**: filter by year range, department, time, weather 
  and lighting conditions with real-time chart updates
- **Risk prediction**: Random Forest classifier estimating the likelihood of 
  serious accidents based on selected conditions, with feature importance display

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Interface | Streamlit |
| Data processing | Python, Pandas |
| Machine learning | Scikit-learn (Random Forest) |
| Visualization | Plotly |

---

## Installation

### Prerequisites
- Python 3.8+
- 8GB RAM recommended

### Setup
```bash
git clone https://github.com/habibabnk/accidents_project
cd accidents_project
pip install -r requirements.txt
streamlit run app.py
```

Available at `http://localhost:8501`

---

## Project Structure
```
accidents_project/
├── app.py                # Main Streamlit application
├── data_loader.py        # Data loading and preprocessing
├── modeling.py           # Random Forest model
├── requirements.txt
└── data/
    ├── 2015/ ... 2020/   # caracteristiques CSV per year
    └── 2021/ ... 2024/   # caracteristiques, usagers, lieux, vehicules
```

---

## Data

French national road accident open data (2015–2024).  
Key fields: department, date/time, weather, lighting, road type, collision type, gravity.  
The loader automatically handles separator variations (`;` / `,`) and encoding differences across years.

---

## Disclaimer

This project is for educational and analytical purposes only.  
Predictions are statistical estimates based on historical data and should not 
be used for real-time safety decisions.

---

## Author

**Habiba Benkemouche** — [LinkedIn](https://www.linkedin.com/in/habiba-benkemouche-56b168264) · [GitHub](https://github.com/habibabnk)
