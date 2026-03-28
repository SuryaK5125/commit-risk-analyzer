\# 🚨 Commit Risk Analyzer



Predict whether a Git commit is likely to introduce bugs using Machine Learning.



\---



\## 🔍 Overview



This project analyzes GitHub commits and predicts whether a commit is "risky" based on:



\- Commit message (NLP using TF-IDF)

\- Code changes (additions, deletions)

\- Commit behavior (time, frequency, patterns)



\---



\## 🧠 Approach



1\. Collected commit data using GitHub API  

2\. Engineered features like:

&#x20;  - commit size

&#x20;  - message keywords

&#x20;  - developer behavior

3\. Created labels using future bug-fix heuristics  

4\. Trained ML models (Logistic Regression, Random Forest, etc.)

5\. Built a Streamlit app for real-time prediction  



\---



\## 📸 Demo



!\[App Screenshot](screenshot.png)



\---



\## ▶️ Run Locally



```bash

pip install -r requirements.txt

python -m streamlit run app.py

