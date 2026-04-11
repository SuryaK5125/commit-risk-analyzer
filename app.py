import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

# Load model + vectorizer
model = joblib.load("best_commit_risk_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Commit Risk Analyzer", layout="centered")
st.title("Commit Risk Analyzer")
st.caption("Trained on 3,000 commits from Flask, Django, FastAPI.")
st.markdown("By Surya Kalimuthu")
st.markdown("Predict whether a Git commit is likely to introduce bugs.")
st.divider()

# Helpers
RISK_WORDS   = ["fix", "bug", "error", "issue", "crash", "hotfix", "rollback"]
URGENT_WORDS = ["urgent", "hotfix", "quick", "workaround"]

def references_issue(msg: str) -> int:
    return int(bool(re.search(r"(fix|close|resolve)[sd]?\s+#\d+", msg.lower())))

def build_features(message: str, additions: int, deletions: int, files_changed: int):
    total_changes=additions + deletions
    message_lower= message.lower()

    message_length= len(message)
    has_fix_word= int(any(w in message_lower for w in ["fix", "bug", "error", "issue", "patch", "hotfix"]))
    has_urgent_word= int(any(w in message_lower for w in URGENT_WORDS))
    ref_issue = references_issue(message)
    risk_word_count= sum(w in message_lower for w in RISK_WORDS)

    large_commit= int(total_changes > 50)
    multi_file_commit = int(files_changed > 1)
    change_density= total_changes / (files_changed + 1)
    delete_ratio= deletions / (total_changes + 1)
    change_spike= int(total_changes > 100)

    hour= 12
    day_of_week= 2
    author_experience= 5
    is_new_contributor= 0
    time_since_last_commit = 1000
    is_late_night= 0
    core_code_ratio= 0.5

    numeric=np.array([[
        additions, deletions,total_changes,files_changed,
        hour, day_of_week,message_length,
        has_fix_word,has_urgent_word, risk_word_count,
        change_density,delete_ratio, large_commit,multi_file_commit,
        author_experience, is_new_contributor,time_since_last_commit,
        is_late_night, change_spike,core_code_ratio
    ]])

    return hstack([csr_matrix(numeric), tfidf.transform([message])])

def heuristic_boost(message: str, total: int, files_changed: int) -> float:
    lower=message.lower()
    risk_word_count=sum(w in lower for w in RISK_WORDS)
    boost=0.0
    if any(w in lower for w in ["fix", "bug", "error"]):   boost += 0.15
    if any(w in lower for w in ["urgent", "hotfix", "rollback"]): boost += 0.15
    if risk_word_count >= 3: boost += 0.20
    if total > 100: boost += 0.20
    if files_changed > 3: boost += 0.15
    return boost

def risk_signals(message: str, additions: int, deletions: int, files_changed: int) -> list[str]:
    total=additions + deletions
    lower=message.lower()
    signals=[]

    matched=[w for w in RISK_WORDS if w in lower]
    if matched:
        signals.append(f"Commit message contains risk keywords: `{', '.join(matched)}`")
    if references_issue(message):
        signals.append("References a GitHub issue (`fix #NNN` pattern) — strong bug-fix signal")
    if total > 100:
        signals.append(f"Very large change ({total} lines) — higher chance of introducing errors")
    elif total > 50:
        signals.append(f"Above-average change size ({total} lines)")
    if files_changed > 3:
        signals.append(f"Touches {files_changed} files — broad changes carry more risk")
    elif files_changed > 1:
        signals.append("Spans multiple files")

    density=total / (files_changed + 1)
    if density > 80:
        signals.append(f"High change density ({density:.0f} lines/file) — concentrated large edits")

    delete_r=deletions / (total + 1)
    if delete_r < 0.1 and total > 30:
        signals.append("Mostly additions with few deletions — could be untested new code")

    return signals

# Inputs
message=st.text_input("Commit Message", placeholder='e.g. fix crash in auth handler')

col1, col2, col3 = st.columns(3)
with col1:
    additions= st.number_input("Lines Added",    min_value=0, value=10)
with col2:
    deletions= st.number_input("Lines Deleted",  min_value=0, value=5)
with col3:
    files_changed = st.number_input("Files Changed",  min_value=1, value=1)

# Prediction
if st.button("Analyze Commit", type="primary"):
    if not message.strip():
        st.warning("Please enter a commit message.")
    else:
        X= build_features(message, additions, deletions, files_changed)
        ml_prob = model.predict_proba(X)[0][1]
        total= additions + deletions
        prob= min(ml_prob + heuristic_boost(message, total, files_changed), 1.0)
        score = round(prob * 10, 1)

        st.divider()
        st.subheader("Risk Assessment")

        if score < 3.5:
            level, detail = "Low Risk",    "No significant signals detected."
        elif score < 6.5:
            level, detail = "Medium Risk", "Some signals worth reviewing."
        else:
            level, detail = "High Risk",   "Multiple risk signals present."

        col_score, col_bar = st.columns([1, 3])
        with col_score:
            st.metric("Risk Score", f"{score} / 10")
            st.markdown(f"**{level}**")
        with col_bar:
            st.markdown(f"_{detail}_")
            st.progress(prob)

        st.divider()
        st.markdown("#### Why this score?")

        signals=risk_signals(message, additions, deletions, files_changed)
        if signals:
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.markdown("No strong risk signals detected.")

        if 0.4 <= prob <= 0.6:
            st.info("Model confidence is low for this input. Consider reviewing manually.")