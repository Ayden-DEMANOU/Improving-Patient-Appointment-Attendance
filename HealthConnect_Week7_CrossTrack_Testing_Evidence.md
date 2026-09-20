# HealthConnect Clinic — Week 7 HC-POD Cross-Track Testing & Validation Evidence

**Track:** Data Science
**Testing partner track:** Data Analytics
**Week:** 7

---

## Note on This Version

This supersedes the earlier draft of this file. The original was written before a genuine request
arrived from the Data Analytics counterpart; this version documents that **real exchange** instead,
which is a stronger and more authentic piece of evidence for the Week 7 requirement.

---

## Required Documentation

**1. Track collaborated with:**
Data Analytics

**2. Project dependency:**
Per the Week 7 HealthConnect Testing & Validation Matrix: "Data Analytics → Data Science: Validate
analytical findings that influence model features or modelling decisions." The Data Analytics
track completed its own Week 7 individual testing and, based on those results, directly requested
that Data Science validate two specific features using the working model rather than standalone
correlation.

**3. Component/output being tested:**
Two Week 6 model features: `long_lead_followup` (built from a Data-Analytics-style finding about
Follow-up appointments) and `distance_to_clinic_km` (an original dataset field with weak standalone
correlation to the outcome).

**4. Information/output received (from Data Analytics):**
Their validated Week 7 findings:
- Booking lead time: 39.13% no-show rate (0–30 days) vs. 63.95% (31–60 days), a +24.82pp gap.
- Previous no-shows: 46.30% (0 prior) vs. 57.83% (≥1 prior).
- Distance to clinic: very weak standalone association (Spearman ρ ≈ 0.055) — Analytics explicitly
  does not support using distance as a standalone prioritisation factor.
- A Lead Time × Appointment Type interaction test: Follow-up showed the largest descriptive gap
  (+28.73pp) but the interaction was **not statistically significant** (p ≈ 0.326).

Along with this, a direct request: test whether `long_lead_followup` provides incremental
out-of-sample predictive value beyond the broader lead-time signal, and whether `distance_to_clinic_km`
provides meaningful incremental value in the multivariable model despite its weak standalone
correlation.

**5. Information/output provided (to Data Analytics):**
The results of both tests — see Testing Activity below — including the reply message summarising
findings, and the full notebook with reproducible code.

**6. Testing activity completed:**
A **with-vs-without ablation test** for each feature: the same Logistic Regression pipeline (same
patient-grouped split, same preprocessing) trained once with the feature included and once with it
excluded, compared on ROC-AUC (single-split and 5-fold grouped cross-validation), a bootstrap 95%
confidence interval on the difference, and — for distance — the top-20%-risk business lift metric.

**7. Issue or finding identified:**
- **`long_lead_followup`**: no statistically meaningful incremental value. Single-split AUC
  difference +0.0017 (bootstrap 95% CI [-0.0008, 0.0041], includes zero); 5-fold CV mean difference
  +0.0009, winning only 4 of 5 folds. This is consistent with Data Analytics' own finding that the
  underlying interaction was not statistically significant. Week 6's justification for the feature
  (Random Forest feature importance ≈ 5%) measured how often the model *used* the feature, not how
  much *unique* signal it added — those turned out to be different things.
- **`distance_to_clinic_km`**: a small but genuinely consistent incremental contribution. The
  single-split bootstrap CI was inconclusive ([-0.0089, 0.0096]), but 5-fold cross-validation showed
  "with distance" beating "without" in **5 out of 5 folds** (mean difference +0.0043), and the
  top-20%-highest-risk business lift dropped from 1.44x to 1.40x when the feature was removed.

**8. Refinement/action taken:**
- Removed `long_lead_followup` from the candidate feature set going into Week 8 / the Machine
  Learning Engineering hand-off — it added complexity without a defensible accuracy benefit.
- Kept `distance_to_clinic_km` in the model — its small multivariable contribution, while modest, is
  more consistent than its near-zero standalone correlation would suggest.

**9. Retest result:**
For `long_lead_followup`, the "without" run in the ablation test **is** the retest: performance
without the feature is statistically indistinguishable from performance with it (and marginally
higher on accuracy/F1 in the single-split comparison). For `distance_to_clinic_km`, the 5-fold
cross-validation served as an independent retest of the inconclusive single-split result, and it
strengthened rather than weakened the case for keeping the feature.

**10. What changed as a result:**
The Week 8 candidate feature set is now simpler (`long_lead_followup` removed) without any
measurable loss of performance, and `distance_to_clinic_km` is confirmed — not just assumed — to
belong in the model. Data Analytics also now has a concrete, tested example to cite when explaining
to non-technical stakeholders that a weak *standalone* correlation does not automatically mean a
feature has no role in a full predictive model.

**11. Evidence:**
- `HealthConnect_Week7_CrossTrack_DA_DS_Validation.ipynb` — full ablation test code, bootstrap
  significance tests, 5-fold cross-validation, business-lift comparison, and the per-fold chart for
  distance.
- The reply message sent back to the Data Analytics counterpart, summarising both results.

**12. How this activity improved the overall HealthConnect solution:**
This is the clearest instance yet of the Week 7 requirement working as intended: a real request from
another track, tested with a defensible method (ablation + significance testing, not opinion), that
changed an actual decision — one feature removed, one feature's inclusion confirmed with stronger
evidence than before. It also produced a piece of shared understanding (weak standalone correlation
≠ no multivariable value) that is useful to both tracks' future work, not just to Data Science's
model.

---

## Note on Authenticity

Unlike the Week 6 cross-track integration and the first draft of this document, this exchange is
based on an actual message from the Data Analytics counterpart, provided as-is, with a specific,
falsifiable request. The tests above were run without knowing the answer in advance, and the result
for `long_lead_followup` (recommending its removal) was not the outcome that would have been
convenient for the Week 6 narrative — it's reported anyway, because that's what the test showed.
