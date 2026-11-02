# GitHub Desktop — School Learner xAPI Project

**This archive is a standalone project, not a surgical patch of your group's existing repository.** Your GitHub screenshot shows files and commits from teammates; do not overwrite them blindly.

1. Download this ZIP and **Extract All**. The folder contains `app.py`, `data/xAPI-Edu-Data.csv`, Python modules, HTML/CSS/JS, and tests.
2. Open **GitHub Desktop**. If you already cloned the team repo, use **Repository → Show in Explorer** and compare the project files. Create a branch such as `feature/school-learner-xapi` before making any changes.
3. Copy the selected new files into your GitHub working directory. If the team already has `app.py`, `predict.py` or `train_models.py`, **review/merge rather than replace them**. Do not remove your teammates' work.
4. Verify `data/xAPI-Edu-Data.csv` is included, and none of `.env`, `instance/`, `*.joblib` or real private learner data are staged.
5. In GitHub Desktop, review changes, add a commit message such as **"Build school learner predictor using xAPI benchmark"**, click **Commit**, then **Push origin**.
6. Open a pull request and request review before merging into `main`; log the work in **GitHub Projects** (required by your assignment).
7. To run locally: `pip install -r requirements.txt`, `python train_models.py`, `uvicorn app:app --reload`.
8. To make it accessible online, connect your GitHub repo to a **Python web-service host** (such as Render). Build: `pip install -r requirements.txt && python data_preprocessing.py && python train_models.py`. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`. Set `DEMO_MODE=true` and an `APP_SECRET_KEY` environment secret.

GitHub Pages serves static files; this project requires a **running Python server**. Public mode demonstrates an open academic benchmark and disables importing private data. A production school system requires more security and governance.
