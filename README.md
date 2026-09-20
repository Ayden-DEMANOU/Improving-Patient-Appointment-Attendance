# HealthConnect Clinic — No-Show Prediction

**AnalystLab Africa · Experience Lab Internship Programme**
**Track:** Data Science

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
├── HealthConnect_Appointment_Data.csv        # Raw appointment dataset (5,000 records, 18 fields)
├── HealthConnect_DS_Week4_EDA.ipynb                      # Week 4 EDA — data quality checks + feature exploration
│   └── HealthConnect_DS_Week5_Baseline_Modelling.ipynb   # Week 5 — data prep, feature engineering, baseline model
│   └── HealthConnect_DS_Week6_Baseline_Modelling.ipynb   # Week 6 — error analysis, feature refinement, model comparison
└── docs/
    ├── HealthConnect_ML_Problem_Definition_Week4.docx    # ML problem definition (target, features, approach, risks)
    ├── HealthConnect_Week4_Project_Summary.pdf         # Concise Week 4 summary + Week 5 plan
    └── HealthConnect_Week5_Project_Summary.pdf          # Concise Week 5 summary + Week 6 plan
    └── HealthConnect_Week6_Project_Summary.pdf          # Concise Week 6 summary + Week 7 plan
    └── HealthConnect_Week6_CrossTrack_Integration_Evidence.md      # Data Analytics → Data Science integration evidence
    └── WK7_HealthConnect_HC-POD_Cross-Track_Evidence_ Ayden_Nancy_Lee      # Data Analytics → Data Science integration evidence
    └── HealthConnect_Week7_Project_Summary.pdf          # Concise Week 7 summary + Week 8 plan



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
[`HealthConnect_DS_Week4_EDA.ipynb`](HealthConnect_DS_EDA.ipynb).

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

## Week 5 Summary: Data Preparation, Feature Engineering & Baseline Model

### Data Preparation
- Filtered to **Attended vs. No-Show** appointments (4,737 of 5,000 records), excluding the
  small, behaviourally distinct `Cancelled` group (5.3%) per the Week 4 target definition.
- Imputed missing `distance_to_clinic_km` with the median; filled missing `reminder_channel`
  with an explicit `"None"` category (missingness there is structural, not a data issue).
- Dropped `waiting_time_minutes` entirely — it's only known once an appointment is already
  underway, so it would leak information not available at prediction time.
- Dropped `age_group` (redundant with `age`) and merged the rare `"Prefer not to say"` gender
  category into `"Other"` to avoid an unstable dummy variable.

### Feature Engineering
| Feature | Description |
|---|---|
| `previous_no_show_rate` | `previous_no_shows / previous_appointments`, capturing reliability independent of visit volume |
| `is_new_patient` | Flags patients with no appointment history |
| `has_reminder` | Binary version of `reminder_sent` |
| `gender_grp` | `gender` with the rare category consolidated |

### Train/Test Strategy
A **patient-grouped split** (`GroupShuffleSplit`, 80/20) was used instead of a random row split,
since each patient appears ~3 times on average — verified to produce **zero patient overlap**
between train and test sets.

### Baseline Model & Results

| Metric | Score |
|---|---|
| Accuracy | 0.627 (vs. 0.50 naive majority-class baseline) |
| Precision | 0.622 |
| Recall | 0.650 |
| F1-score | 0.636 |
| ROC-AUC | 0.678 |

**Logistic Regression** was chosen as the baseline for its interpretability. Feature coefficients
confirm the Week 4 findings: `booking_lead_days` and `previous_no_shows` are the strongest drivers
of predicted no-show risk. A multicollinearity issue between the three patient-history features
(`previous_appointments`, `previous_no_shows`, `previous_no_show_rate`) was identified and flagged
for Week 6.

Full code, all five decision-supporting visualisations (target distribution, distribution/outlier
checks, correlation heatmap, feature-target relationship plots), and the confusion matrix / ROC
curve are in
[`HealthConnect_DS_Week5_Baseline_Modelling.ipynb`](HealthConnect_DS_Week5_Baseline_Modelling.ipynb).

---

## Week 6 Summary: Model Improvement, Error Analysis & Validation
 
Week 6 does **not** repeat the Week 5 baseline. It analyses its weaknesses, fixes a known issue,
adds evidence-based features, and validates whether an improved model is actually better, not just
numerically, but reliably and operationally.
 
### Error Analysis
Profiling the Week 5 baseline's false positives/negatives found a clear pattern:
- **False negatives** (missed no-shows): short lead time (~19 days), low prior no-show count — the
  model has little signal to catch these.
- **False positives** (predicted no-show, actually attended): long lead time (~39 days), moderate
  history — the model is reading its available signals correctly; they're just probabilistic.
- **Accuracy varied by appointment type** — notably lower for Specialist Consultation (58.3%) than
  Diagnostic Test (69.7%), prompting the segment investigation below.
### Cross-Track Integration (Data Analytics → Data Science)
A segment-level analysis (no-show rate by `appointment_type` × `booking_lead_days`) found that
**Follow-up appointments are the most lead-time-sensitive type** — no-show rate rises from 31.9%
(0–7 days) to 75.2% (45–60 days), a steeper climb than any other type. This finding was turned into
a new feature (`long_lead_followup`), which then showed a measurable, non-trivial contribution
(~5% of Random Forest feature importance) — real evidence of integration, not just communication.
 
### Feature Refinement
- Resolved the Week 5 multicollinearity issue by dropping the redundant `previous_appointments`
  feature.
- Added `booking_lead_days_sq` to let the linear model represent the non-linear (convex) lead-time
  effect found in earlier EDA.
- Added `long_lead_followup` from the cross-track finding above.
### Model Comparison
 
| Model | Accuracy | ROC-AUC (test) | ROC-AUC (5-fold CV mean) |
|---|---|---|---|
| Week 5 baseline (Logistic Regression) | 0.628 | 0.678 | — |
| Logistic Regression v2 (refined features) | 0.635 | 0.686 | 0.680 |
| Random Forest | 0.648 | 0.691 | 0.679 |
| Gradient Boosting | 0.619 | 0.662 | 0.655 |
 
Cross-validation showed Logistic Regression v2 and Random Forest are **statistically
indistinguishable** — the single-split edge for Random Forest was within normal fold-to-fold noise.
Gradient Boosting consistently underperformed, an honestly-reported negative result rather than a
tuned comparison. **Both Logistic Regression v2 and Random Forest are carried forward as joint
candidates** for Week 7 rather than forcing a single "winner" on a marginal difference.
 
### Business-Relevance Check
Beyond accuracy: if HealthConnect staff prioritised reminder outreach for the top-scoring 20% of
appointments by predicted risk, **~72–73% of those would actually be no-shows**, versus a 50%
baseline — a **~1.44–1.45x lift**. This confirms the model is operationally useful for prioritising
limited outreach effort, which is the actual HealthConnect use case.
 
Full code, error analysis, cross-track integration detail, and all charts are in
[`HealthConnect_DS_Week6_Model_Improvement.ipynb`](HealthConnect_DS_Week6_Model_Improvement.ipynb).
 

## Week 7 Summary: Model Testing, Refinement & End-to-End Validation
 
Week 7 does **not** repeat Week 6's model comparison. It formally tests the Week 6 candidates,
checks whether last week's "improvement" actually holds up statistically, retests a cross-track
finding on data it was never derived from, and makes one targeted, evidence-based refinement.
 
### Formal Test Suite
 
| Test | Result | Pass/Fail |
|---|---|---|
| Reproducibility (retrain twice) | Identical ROC-AUC across runs | ✅ Pass |
| Overfitting (train vs. test gap) | LogReg v2: 0.003 (negligible); Random Forest: 0.035 (borderline) | ✅ Pass / ⚠️ Marginal |
| Cross-validation stability | Mean ROC-AUC 0.680, std ≈ 0.007 across 5 folds | ✅ Pass |
| Statistical significance (Week 5 → Week 6 improvement) | Bootstrap 95% CI [0.0026, 0.0145] — excludes zero | ✅ Pass |
| Cross-track feature retest (held-out data) | 68.4% no-show rate for flagged appointments vs. 50.0% overall (1.37x) | ✅ Pass |
 
**Key result:** the Random Forest-vs-LogReg choice doesn't matter much (tied in Week 6), but the
Week 6 **feature refinements** demonstrably do — confirmed via bootstrap significance testing this
week, and Random Forest's larger overfitting gap further supports **Logistic Regression v2 as the
primary Week 8 candidate**.
 
### Error Analysis & Refinement
Error analysis on the refined candidate confirmed the Week 6 blind spot persists: no-shows with
short lead time and clean prior history are still under-detected. This is a **threshold problem, not
a feature problem** — these cases simply carry no strong risk signal. Lowering the decision threshold
from 0.50 to 0.40:
 
| Metric | Threshold 0.50 | Threshold 0.40 |
|---|---|---|
| Overall recall | 65.2% | 81.8% |
| Overall precision | 63.0% | 58.4% |
| Blind-spot segment recall | 8.2% | **27.6%** |
 
A real, more-than-3x improvement on the hardest-to-detect segment — though most of those cases are
still missed, which is documented honestly rather than overstated.
 
### Cross-Track Testing (Data Analytics → Data Science)
The Week 6 `long_lead_followup` finding was retested on the **held-out test set only** — data it
was never derived from or trained on — and replicated (1.37x lift, consistent with the original
1.44–1.45x finding). This confirms the feature reflects a genuine pattern rather than a training-set
artefact. Full record in
[`docs/HealthConnect_Week7_CrossTrack_Testing_Evidence.md`](docs/HealthConnect_Week7_CrossTrack_Testing_Evidence.md).
 
Full code, all tests, and charts are in
[`HealthConnect_DS_Week7_Testing_Refinement.ipynb`](HealthConnect_DS_Week7_Testing_Refinement.ipynb).
 
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
 
## Roadmap
 
- [x] **Week 4** — Problem understanding, data quality assessment, exploratory data analysis,
      ML problem definition
- [x] **Week 5** — Data preparation, feature engineering, patient-grouped train/test split,
      baseline logistic regression model (Accuracy 0.627, ROC-AUC 0.678)
- [x] **Week 6** — Error analysis, feature refinement, cross-track integration with Data Analytics,
      model comparison (Logistic Regression v2 & Random Forest as joint candidates, ROC-AUC ≈ 0.68–0.69)
- [x] **Week 7** — Formal test suite, statistical significance testing, cross-track retest,
      decision-threshold refinement (recommended candidate: Logistic Regression v2 @ threshold 0.40)
- [ ] **Week 8** — Fairness/bias review, final integration, and HealthConnect presentation
- [ ] **Week 7** — Testing, refinement, and fairness/bias review across patient subgroups
- [ ] **Final** — Presentation and portfolio write-up
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
