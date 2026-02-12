pip install spacy scikit-learn pandas matplotlib
py -m spacy download en_core_web_sm
python resume_screening.py  
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Load NLP Model
# ------------------------------
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# Sample Resume Data
# ------------------------------
resumes = {
    "Aarav_Shah": """
    Skills: Python, Machine Learning, Data Analysis, SQL, Pandas, NumPy
    Experience: 1 year internship in Data Science
    Projects: Built ML models for prediction and customer segmentation.
    """,

    "Meera_Reddy": """
    Skills: Java, HTML, CSS, JavaScript
    Experience: 2 years Web Development
    Projects: Built responsive websites and e-commerce systems.
    """,

    "Rohan_Verma": """
    Skills: Python, NLP, Deep Learning, Data Analysis, Machine Learning
    Experience: 6 months ML research assistant
    Projects: Built sentiment analysis system using NLP.
    """
}

# ------------------------------
# Job Description
# ------------------------------
job_description = """
Looking for a Machine Learning Intern with strong skills in
Python, Machine Learning, Data Analysis, SQL, NLP, and Deep Learning.
"""

# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)

clean_job_desc = clean_text(job_description)
clean_resumes = {name: clean_text(text) for name, text in resumes.items()}

# ------------------------------
# Skill Dictionary with Weights
# ------------------------------
skill_weights = {
    "python": 3,
    "machine learning": 3,
    "data analysis": 2,
    "sql": 2,
    "nlp": 2,
    "deep learning": 2,
    "java": 1,
    "html": 1,
    "css": 1,
    "javascript": 1
}

skills = list(skill_weights.keys())

def extract_skills(text):
    return [skill for skill in skills if skill in text]

# ------------------------------
# TF-IDF + Cosine Similarity
# ------------------------------
documents = [clean_job_desc] + list(clean_resumes.values())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
).flatten()

# ------------------------------
# Hybrid Scoring
# ------------------------------
job_skills = extract_skills(clean_job_desc)

final_scores = []
skill_match_percentage = {}
skill_gaps = {}

for i, candidate in enumerate(resumes.keys()):
    candidate_skills = extract_skills(clean_resumes[candidate])

    matched = sum(skill_weights[s] for s in candidate_skills if s in job_skills)
    total_weight = sum(skill_weights[s] for s in job_skills)

    skill_percent = (matched / total_weight) if total_weight > 0 else 0
    skill_match_percentage[candidate] = round(skill_percent * 100, 2)

    skill_gaps[candidate] = list(set(job_skills) - set(candidate_skills))

    final_score = 0.6 * similarity_scores[i] + 0.4 * skill_percent
    final_scores.append(final_score)

results = pd.DataFrame({
    "Candidate": resumes.keys(),
    "Similarity Score": similarity_scores,
    "Skill Match %": list(skill_match_percentage.values()),
    "Final Hybrid Score": final_scores
}).sort_values(by="Final Hybrid Score", ascending=False)

# ------------------------------
# Print Output
# ------------------------------
print("\n=== Candidate Ranking ===")
print(results)

print("\n=== Skill Gap Analysis ===")
for candidate, gaps in skill_gaps.items():
    print(f"{candidate} missing skills: {gaps}")

# ------------------------------
# Save Results to CSV
# ------------------------------
results.to_csv("ranking_results.csv", index=False)
print("\nResults saved to ranking_results.csv")

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(8,5))
plt.bar(results["Candidate"], results["Final Hybrid Score"])
plt.title("Candidate Ranking - Hybrid Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ranking_chart.png")
plt.show()

print("Chart saved as ranking_chart.png")
