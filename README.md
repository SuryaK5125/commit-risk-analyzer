# Commit Risk Analyzer
Predicts bug-prone Git commits using machine learning and real GitHub data

---

## Overview
- Predicts whether a commit is likely to introduce bugs  
- Uses:
  - commit message (NLP)
  - code changes (additions, deletions)
  - commit patterns  

---

## Why this project
- Built using real GitHub API data (not static datasets)  
- Combines NLP with structured features  
- Simulates a pre-merge risk analysis tool for developers  

---

## Tech Stack
- Python, Scikit-learn, XGBoost  
- TF-IDF (NLP)  
- Streamlit  

---

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The model predicts high risk because:
- The commit message indicates bug fixing activity
- The number of changes is relatively high
- Changes affect multiple files, increasing complexity
---

Demo
<img width="1106" height="1140" alt="image" src="https://github.com/user-attachments/assets/e9a5480e-efd9-4799-8434-11b2ba456fc4" /> <img width="1114" height="1070" alt="image" src="https://github.com/user-attachments/assets/30ca43bb-6610-4b04-860e-8306e5b06de1" />
---

Why this matters

Predicting risky commits can help:
- prioritize code reviews
- reduce bugs in production
- improve software quality

---

## Limitations
- Labels are heuristic-based (not actual bug data)
- Model may over-rely on keywords like "fix" or "bug"
- Does not analyze actual code diffs
