# BAAC Data Dictionary

Field codes used from the official French road-accident database
(Bulletins d'Analyse des Accidents Corporels — BAAC).

Source: [Notice descriptive BAAC](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/)

---

## Table: `caracteristiques` — accident circumstances

| Field | App column | Description |
|-------|-----------|-------------|
| `Num_Acc` / `Accident_Id` | `accident_id` | Unique accident identifier (join key with other tables) |
| `jour` | `day` | Day of month (1–31) |
| `mois` | `month` | Month (1–12) |
| `an` | `year_src` | Year (4-digit) |
| `hrmn` | `time` | Time as HHMM string; parsed to `hour` (0–23) |
| `lum` | `lighting` | Lighting conditions (see below) |
| `dep` | `department` | Department code (01–976) |
| `com` | `commune` | INSEE commune code |
| `agg` | `localization` | In built-up area: 1=hors agglomération, 2=en agglomération |
| `int` | `intersection` | Intersection type (see below) |
| `atm` | `weather` | Atmospheric conditions (see below) |
| `col` | `collision_type` | Collision type (see below) |
| `lat` / `long` | `latitude` / `longitude` | GPS coordinates (WGS84, decimal degrees) |

---

## Table: `usagers` — persons involved

| Field | Description |
|-------|-------------|
| `Num_Acc` | Accident identifier (join key) |
| `grav` | Severity of injury for this person (see gravity format below) |
| `catu` | User category: 1=driver, 2=passenger, 3=pedestrian |
| `sexe` | Sex: 1=male, 2=female |
| `an_nais` | Year of birth |
| `trajet` | Journey purpose: 0=non-renseigné, 1=domicile–travail, etc. |

---

## Gravity format — critical nuance

The `grav` field changed meaning in 2018. The app auto-detects the format
by comparing counts of each value (fatalities are always rarer than minor injuries).

### Post-2018 format (BAAC 2018+, all data in this app)

| `grav` | Meaning |
|--------|---------|
| 1 | Indemne (uninjured) |
| **2** | **Tué (killed)** |
| **3** | **Blessé hospitalisé (hospitalised)** |
| 4 | Blessé léger (minor injury) |

**`is_serious = 1` when `grav` ∈ {2, 3}** (at least one killed or hospitalised person in the accident).

### Pre-2018 format (not present in this dataset, documented for reference)

| `grav` | Meaning |
|--------|---------|
| 1 | Indemne |
| 2 | Blessé léger |
| 3 | Blessé hospitalisé |
| 4 | Tué |

Detection logic: if count(`grav`=2) < count(`grav`=4), it is the post-2018 format
(2 = killed, which is rare). Otherwise it is the old format (4 = killed).

---

## Code tables

### `lum` — Lighting conditions

| Code | Label |
|------|-------|
| 1 | Plein jour |
| 2 | Crépuscule ou aube |
| 3 | Nuit sans éclairage public |
| 4 | Nuit avec éclairage public non allumé |
| 5 | Nuit avec éclairage public allumé |

### `atm` — Atmospheric conditions

| Code | Label |
|------|-------|
| -1 | Non renseigné |
| 1 | Normale |
| 2 | Pluie légère |
| 3 | Pluie forte |
| 4 | Neige – grêle |
| 5 | Brouillard – fumée |
| 6 | Vent fort – tempête |
| 7 | Temps éblouissant |
| 8 | Temps couvert |
| 9 | Autre |

### `col` — Collision type

| Code | Label |
|------|-------|
| -1 | Non renseigné |
| 1 | Deux véhicules — frontale |
| 2 | Deux véhicules — par l'arrière |
| 3 | Deux véhicules — par le côté |
| 4 | Trois véhicules et plus — en chaîne |
| 5 | Trois véhicules et plus — collisions multiples |
| 6 | Autre collision |
| 7 | Sans collision |

### `agg` — Localisation (in/out of built-up area)

| Code | Label |
|------|-------|
| 1 | Hors agglomération |
| 2 | En agglomération |

### `int` — Intersection type

| Code | Label |
|------|-------|
| -1 | Non renseigné |
| 1 | Hors intersection |
| 2 | Intersection en X |
| 3 | Intersection en T |
| 4 | Intersection en Y |
| 5 | Intersection à plus de 4 branches |
| 6 | Giratoire |
| 7 | Place |
| 8 | Passage à niveau |
| 9 | Autre intersection |

### `dep` — Department

Standard French department codes (01–19 mainland, 2A–2B Corsica, 971–976 overseas).
Zero-padded to 2 characters in the app (`01`, `75`, `2A`, etc.).
