Commit Risk Analyzer
Predicts bug-prone Git commits using ML + real GitHub data

What it does
Predicts if a commit is high risk (likely to introduce bugs)
Uses:
  commit message (NLP)
  code changes (additions, deletions)
  commit patterns
  
Why it’s interesting
  Built using real GitHub API data (not static datasets)
  Combines NLP + structured features
  Simulates a pre-merge risk checker for developers

Tech Stack
  Python, Scikit-learn, XGBoost
  TF-IDF (NLP)
  Streamlit (UI)
  
Run locally
  pip install -r requirements.txt
  streamlit run app.py
  
Demo
<img width="1106" height="1140" alt="image" src="https://github.com/user-attachments/assets/e9a5480e-efd9-4799-8434-11b2ba456fc4" />
<img width="1114" height="1070" alt="image" src="https://github.com/user-attachments/assets/30ca43bb-6610-4b04-860e-8306e5b06de1" />

Input:

“fix login bug”, 50 additions, 20 deletions, 4 files

Output:
High Risk Commit (0.67)

Reason:
  bug-related keywords
  large change size
  multiple files
  
Future Work
CodeBERT / LLM-based embeddings
GitHub bot integration
Developer behavior modeling
🎯 Why this works
Clean
Fast to read
Still shows:
real data
ML
product thinking
