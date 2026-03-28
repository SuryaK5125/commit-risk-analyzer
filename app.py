import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix


model = joblib.load("best_commit_risk_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("Commit Risk Analyzer")
st.caption("Built by Surya K")
st.markdown("Predict whether a Git commit is likely to introduce bugs using machine learning")


message = st.text_input("Commit Message")
col1, col2 = st.columns(2)
with col1:
    additions = st.number_input("Additions", min_value=0, value=10)
    files_changed = st.number_input("Files Changed", min_value=1, value=1)

with col2:
    deletions = st.number_input("Deletions", min_value=0, value=5)


def build_features(message, additions, deletions, files_changed):
    total_changes = additions + deletions
    message_lower = message.lower()

    message_length = len(message)
    has_fix_word = int(any(word in message_lower for word in ["fix", "bug", "error"]))
    has_urgent_word = int(any(word in message_lower for word in ["urgent", "hotfix"]))

    large_commit = int(total_changes > 50)
    multi_file_commit = int(files_changed > 1)

    hour = 12
    day_of_week = 2
    author_experience = 5
    is_new_contributor = 0
    time_since_last_commit = 1000
    is_late_night = 0
    change_spike = int(total_changes > 100)
    core_code_ratio = 0.5

    numeric = np.array([[ 
        additions,
        deletions,
        total_changes,
        files_changed,
        hour,
        day_of_week,
        message_length,
        has_fix_word,
        has_urgent_word,
        large_commit,
        multi_file_commit,
        author_experience,
        is_new_contributor,
        time_since_last_commit,
        is_late_night,
        change_spike,
        core_code_ratio
    ]])

    numeric_sparse = csr_matrix(numeric)
    text_features = tfidf.transform([message])
    final_features = hstack([numeric_sparse, text_features])

    return final_features

if st.button("Predict Risk"):

    if message.strip()=="":
        st.warning("Enter a commit message")
    else:
        X = build_features(message, additions, deletions, files_changed)

        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"High Risk Commit (Score: {prob:.2f})")
        else:
            st.success(f"Low Risk Commit (Score: {prob:.2f})")

        st.markdown("Why this prediction?")

        reasons=[]

        if "fix" in message.lower() or "bug" in message.lower():
            reasons.append("Commit message indicates bug fixing activity")

        if additions + deletions > 50:
            reasons.append("Large number of code changes")

        if files_changed > 1:
            reasons.append("Changes span multiple files")

        for r in reasons:
            st.write(f"- {r}")
    