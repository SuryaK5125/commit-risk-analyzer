# Commit Risk Analyzer

Predicts whether a Git commit is likely to introduce bugs using only metadata and message text — no diff analysis, no manual labels.

**Live demo:** https://surya-commit-risk-analyzer.streamlit.app/

---

## The approach

The hard part was labelling. There's no ground truth dataset of "this commit introduced a bug," so I approximated it: a commit is marked risky if a bug-fix commit appears within the next 3 commits in the repo timeline. Grounded in real activity, but still a proxy.

From there I pulled 3,000+ commits across Flask, Django, and FastAPI via the GitHub REST API, built 220 features (20 structured + 200 TF-IDF from the commit message), and benchmarked four models. Random Forest came out on top — XGBoost had slightly better AUC but worse F1 on the minority class, which is the metric that matters when positive examples are rare.

Final numbers: **0.67 F1, 0.83 ROC-AUC**.

---

## Run locally

```bash
git clone https://github.com/SuryaK5125/commit-risk-analyzer.git
cd commit-risk-analyzer
pip install -r requirements.txt
streamlit run app.py
```

---

## What it can't do

- Doesn't read diffs; everything is metadata and message text
- Keywords like `fix` or `crash` inflate scores somewhat mechanically
- Author experience and time-of-day features are hardcoded at inference since the UI doesn't collect them
- Labels are proxies, not verified bug reports

## Demo Screenshots

### Medium Risk Example
![Medium Risk Example](Screenshot%20risk%20assessment%201.png)

### High Risk Example
![High Risk Example](Screenshot%20risk%20assessment%202.png)
add explanation to why it chose a specific risk score

Built by Surya Kalimuthu
