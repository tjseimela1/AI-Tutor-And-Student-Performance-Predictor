# Dataset and Methods - school-focused xAPI model

## Authentic source and scope
- `data/xAPI-Edu-Data.csv` was copied **byte-for-byte unchanged** from the user-provided CSV (SHA-256 `ca48b2373f389b32d57b62293f3930f82a3d93eb7f84457940df69208e2edade`).
- Source: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- 480 rows and 17 CSV columns: 16 explanatory attributes and `Class` label (L=127; M=211; H=142).
- School stages: `lowerlevel` 199, `MiddleSchool` 248, `HighSchool` 33. Semesters: F 245, S 235.
- The dataset is cross-sectional; it **does not** contain names, persistent learner IDs, dated weekly observations, exact marks, dropout outcomes, or real-time LMS integration. Generated `XAPI-0001` row labels are not source identities.

## Prediction features and evaluation
- Nine classifier inputs: `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`, `StageID`, `GradeID`, `Topic`, `Semester`, `StudentAbsenceDays`.
- The target `Class` is excluded from inputs. Demographic/parent fields are excluded from scoring (`gender`, `NationalITy`, `PlaceofBirth`, `SectionID`, `Relation`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction`). Their exclusion does **not** prove that the model is fair.
- A seeded stratified split allocates 288 training, 96 validation and 96 held-out test rows. Training-only pipelines impute numeric/categorical features, standardise numeric measures when needed and one-hot encode categorical data.
- Compare Logistic Regression, Random Forest and a multilayer perceptron (48,24,12 units). Choose by validation macro-F1 and evaluate the winner once on held-out test rows.
- Selected Random Forest: test accuracy 0.6979, macro-F1 0.7059, macro recall 0.7037. Low-class recall is 0.8000 (20/25). Metrics vary if inputs or training code change; re-run `python train_models.py` and inspect `models/metrics.json`.
- Review-priority levels from the predicted P(L) are uncalibrated demonstration thresholds, not measured risks of failing or grounds for automatic decisions.

## Time-series teaching simulation
- The actual xAPI CSV includes `Semester` F/S, which allows a **cross-sectional comparison** of 245 first-semester and 235 second-semester records. It is not a historical progression of the same learners.
- `xapi_simulation.py` creates eight *hypothetical* weeks per row from its original activity counter using deterministic seeded multinomial allocation. Every eight-week total exactly equals the corresponding original counter; the allocation in time is invented.
- Three demonstration forecasts (prior-week baseline, three-week moving average, four-week linear trend) are compared using expanding-window one-step tests for simulated weeks 5-8. The lowest simulated MAE model predicts illustrative weeks 9-11.
- For the default full-cohort resource-visits example, the simulated MAE of the selected moving average is 0.437 source-activity counts per learner per simulated week. The simulated week-9 forecast is 7.783. These numbers **are not validated historical or future school observations**, and the simulation is entirely separate from the Low/Middle/High classifier.
- Do not use this simulation as evidence of real seasonal trends, genuine individual progression or external forecast accuracy. Ask the lecturer whether an openly simulated illustrative sample is acceptable for the time-series criterion.

## The study tutor
- `chatbot.py` uses a simple TF-IDF text matcher over prepared school-level educational explanations. The chatbot is an NLP demonstration rather than a model trained on the xAPI CSV; there are no open-ended LLM answers, live web sources or claims of guaranteed subject accuracy.

## Source reference
Amrieh, E. A., Hamtini, T. & Aljarah, I. (2016). Mining Educational Data to Predict Student's Academic Performance Using Ensemble Methods. *International Journal of Database Theory and Application*, 9(8), 119-136. https://doi.org/10.14257/ijdta.2016.9.8.13
