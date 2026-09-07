# HealthConnect Clinic — No-Show Prediction

**AnalystLab Africa · Experience Lab Internship Programme**
**Track:** Data Science
**Status:** 🟡 In Progress — Week 4 of the Experience Lab (Problem Understanding & Initial EDA)

---

## Project Overview

HealthConnect Clinic is a fictional healthcare provider struggling with a high rate of missed
appointments ("no-shows"), which wastes clinical capacity, lengthens waiting lists, and adds
administrative overhead.

**Central project question:**
> How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the
> patient support experience?

This repository contains my work as the **Data Science intern** on the project: defining the
machine learning problem, assessing whether the available data can support a no-show prediction
model, and — from Week 5 onward — building and evaluating that model.

> This is one track of a larger, multi-discipline project. Project Management, Data Analytics,
> Machine Learning Engineering, and Generative AI interns are contributing their own workstreams
> to the same HealthConnect Clinic case study.

---

## Repository Structure

```
├── README.md                                    # This file
├── data/
│   └── HealthConnect_Appointment_Data.csv        # Raw appointment dataset (5,000 records, 18 fields)
├── notebooks/
│   └── HealthConnect_DS_Week4_EDA.ipynb          # Week 4 EDA — data quality checks + feature exploration
└── docs/
    ├── HealthConnect_ML_Problem_Definition_Week4.pdf   # ML problem definition (target, features, approach, risks)
    └── HealthConnect_Week4_Project_Summary.pdf         # Concise Week 4 summary + Week 5 plan
```
---

## Week 4 Summary: Problem Definition & Exploratory Data Analysis

### The Data
- **5,000 appointment records**, 18 fields: patient demographics, booking/appointment details,
  prior appointment & no-show history, reminder information, distance to clinic, waiting time, and
  the final outcome.
- **1,696 unique patients**, averaging ~3 appointments each.
- Data quality is good: no duplicate records, no logical inconsistencies (e.g. `previous_no_shows`
  never exceeds `previous_appointments`, appointment dates never precede booking dates), and only
  1–2% genuine missing values (`distance_to_clinic_km`, `waiting_time_minutes`).

### The Target
`appointment_outcome` splits into:

| Outcome   | Share |
|-----------|-------|
| No-Show   | 48.5% |
| Attended  | 46.3% |
| Cancelled | 5.3%  |

The classes are close to balanced — a more favourable starting point than the heavily-imbalanced
scenario often assumed for no-show prediction problems.

### Key Findings

- **`booking_lead_days` is the strongest predictor found.** No-show rate rises from **~28%** for
  appointments booked 0–7 days ahead to **~68%** for appointments booked 45–60 days ahead — a
  clear, monotonic relationship.
- **`previous_no_shows` is the second strongest signal.** No-show rate rises from ~44% (no prior
  no-shows) to ~68% (3+ prior no-shows).
- Demographic and scheduling-context fields (age, gender, appointment type, day of week, time of
  day, reminder channel) show only **weak, flat relationships** with the outcome individually.
- `waiting_time_minutes` was excluded as a candidate feature — it's only known once an appointment
  is already underway, so using it would leak information not available at prediction time.
- Because patients appear multiple times in the dataset, a **patient-grouped or chronological
  train/test split** is required for modelling, rather than a random row-level split, to avoid
  leaking patient-specific behaviour between train and test sets.

Full code, statistics, and charts are in
[`HealthConnect_DS_EDA.ipynb`](HealthConnect_DS_EDA.ipynb).

### Proposed ML Approach

- **Problem type:** Supervised binary classification (No-Show vs. Attended; Cancelled appointments
  modelled as a separate category).
- **Baseline:** Logistic regression on the strongest features (`booking_lead_days`,
  `previous_no_shows`) for an interpretable starting point.
- **Candidate models:** Random Forest / Gradient Boosting (e.g. XGBoost, LightGBM) for tabular
  performance gains.
- **Evaluation:** Precision, recall, F1, and ROC-AUC, with particular attention to recall on the
  no-show class.
- **Interpretability:** Feature importance / SHAP so results are explainable to clinic staff.

Full details, including handling of the `Cancelled` category and a full assumptions/risks register,
are documented in
[`docs/HealthConnect_ML_Problem_Definition_Week4.pdf`](docs/HealthConnect_ML_Problem_Definition_Week4.pdf).

---

## Getting Started

### Requirements
```
python >= 3.10
pandas
numpy
matplotlib
scikit-learn
jupyter
```

### Setup
```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt   # or: pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook notebooks/HealthConnect_DS_Week4_EDA.ipynb
```

The notebook expects `HealthConnect_Appointment_Data.csv` to be in the same working directory (or
update the file path in the first code cell to point at `data/HealthConnect_Appointment_Data.csv`).

---

## Notes & Disclaimers

- All data is **fictional and synthetic**, provided for internship training purposes by
  AnalystLab Africa. No real patient data is used anywhere in this project.
- This repository reflects work in progress as part of a structured internship programme; content
  will be updated weekly as the project develops.

---

## Author

**Ayden Demanou**
Data Science Intern — AnalystLab Africa Experience Lab
[LinkedIn](#) · [X / Twitter](#)

*Built as part of the AnalystLab Africa Experience Lab Internship Programme. #AnalystLabAfrica*
