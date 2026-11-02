# Business Objectives — School-Based Learning Analytics

## Business background
Schools maintain information about subjects, class participation, digital learning activities, absences and academic outcomes. Teachers need understandable summaries and timely support tools rather than opaque automated decisions. The xAPI-Edu-Data benchmark supports a school-level **research demonstration** of these capabilities; it does not represent enrolled learners at a particular school.

## Objectives

1. Classify benchmark learner records into **Low (L), Middle (M), High (H)** academic performance using school-stage, grade, subject, and engagement variables.
2. Display comparative counts and participation indicators by **educational stage, grade, topic and semester**.
3. Provide **human-review priorities** and simple school support suggestions, without equating estimated Low class with a confirmed failure.
4. Offer a safe, **retrieval-based school-study chatbot** for revision, homework and common subject concepts.
5. Generate reproducible evaluation reports, confusion matrices, CSV exports and model comparisons.
6. Demonstrate the need for real dated school observations before producing time-series forecasts.

## Success criteria
- All 480 source records are loaded without changing the original data.
- Low/Middle/High outputs are assessed using separately held-out benchmark data, reporting class-level and macro performance.
- Grade/stage categories and counts match the source dataset; parent fields are used for description only.
- The interface works on desktop and mobile; all available tests pass.
- The tutor provides sensible school-level answers and declines unsupported questions.
- The system clearly labels privacy limitations, non-causal review suggestions and missing dates.

## Requirements and constraints
- **Functional:** load CSV, validate schema, train/compare classifiers, predict record classes, filter by grade/stage/subject/review band, describe semester groups, provide NLP question-answering, export summaries.
- **Non-functional:** usable responsive layout, low cost, reproducible evaluation, pseudonymous labels, sensible input checks and secure handling of any future data.
- **Constraints:** small cross-sectional benchmark, no identities/dates/marks/actual school integration, uncalibrated probability thresholds, unknown transfer performance, restricted resource and development time.
- **Risks:** false predictions, unfair group differences, mistaken causal conclusions, overreliance by teachers or parents, and privacy risks if real learner data is later introduced.
- **Techniques:** pandas/scikit-learn data preparation, Logistic Regression / Random Forest / multilayer perceptron classification, held-out macro-F1 and confusion matrix, TF-IDF retrieval chatbot; optional real-date time-series module.

## What is not yet delivered
No production-level security, school account integration, guardian messaging, automatic risk alerts or verified impact on grades/attendance. The official BA 3.2 project rubric requires time-series evidence, but xAPI alone lacks timestamps. A separate legitimate dated dataset is needed to satisfy that criterion fully.
