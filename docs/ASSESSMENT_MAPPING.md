# VUT Business Analysis 3.2 - Evidence and remaining submissions

This map is based on the 2026 assignment brief (pages 2-7), **not** a guarantee of awarded marks. The application is a proof of concept using the unchanged xAPI-Edu-Data CSV.

| Rubric area | Practical evidence | Submission note |
|---|---|---|
| AI solution for an industry (5) | Python school learner performance prediction and study support | School education sector; not a university LMS |
| Business objectives, constraints, tools (25) | Project report business-analysis sections; README | Team must check written objectives and explain business benefits as proposed, not achieved |
| Problem definition (10) | Project report, concise <=250-word statement | Supports early academic review; no proven failure/dropout reduction |
| Poster (10) | Separate editable and PDF project poster in submission materials | Review and approve before Blackboard upload |
| Machine learning approach (5) | `train_models.py` logistic regression, random forest, MLP comparisons | Macro-F1 selection on validation set |
| Data / model (5 combined rubric block) | `data/xAPI-Edu-Data.csv`; 480 rows, 17 columns; `/reports` evaluation | No marks, unique learner identities, exact dates, or observed weekly series |
| Time-series analysis (10) | `xapi_simulation.py`, `/forecast` + real F/S cohort summary | **W1-W8 generated, W9-W11 simulated forecast**; not real history; lecturer acceptance not assured |
| Solution techniques (5) | Pipelines, stratification, imputation, scaling/one-hot encoding, held-out testing | Risk thresholds are illustrative and probabilities uncalibrated |
| NLP/speech (5) | `chatbot.py` TF-IDF retrieval-based school tutor | Offline FAQ retrieval, not a generative LLM or speech recognizer |
| Deep learning (5) | Multilayer perceptron comparison in `train_models.py` | Model not necessarily superior to selected random forest |
| Chatbot (5) | `/tutor` and NLP module | Study support; not human teacher replacement |
| Practical (10) | FastAPI responsive website, filterable analytics, predictions, reports | Test `python -m pytest -q`; not production ready |
| Administrative rules | README, GitHub Desktop guide, project-board task plan | Upload by 2 Nov 2026 23:59; **signed declaration**, Grammarly evidence, GitHub Project tool, correct PDF and slides required |

**IMPORTANT:** Only the team can sign the declaration, provide real Grammarly results and confirm originality. Every submitted component must be reviewed against the project brief. Use the accompanying school-focused report/poster/presentation as editable drafts, not as a final certified academic submission.
