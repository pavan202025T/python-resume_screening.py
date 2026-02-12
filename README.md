# 🚀 Intelligent Resume Screening System

## 📌 Overview

This project implements a Machine Learning-based Resume Screening System that automatically ranks candidates based on a given job description.

The system combines NLP text processing with hybrid scoring logic to deliver intelligent candidate evaluation.

---

## 🧠 How the System Works

### 1️⃣ Text Preprocessing
- Lowercasing
- Stopword removal
- Lemmatization (spaCy)

### 2️⃣ Feature Extraction
- TF-IDF Vectorization
- Converts resume text into numerical vectors

### 3️⃣ Similarity Scoring
- Cosine similarity compares resume to job description

### 4️⃣ Skill-Based Weighting
Each skill has a priority weight:

| Skill | Weight |
|-------|--------|
| Python | 3 |
| Machine Learning | 3 |
| SQL | 2 |
| NLP | 2 |
| Deep Learning | 2 |

### 5️⃣ Hybrid Scoring Formula

Final Score =  
**(0.6 × Cosine Similarity) + (0.4 × Weighted Skill Score)**

---

## 📊 Sample Output

### 🖥 Terminal Output

![Terminal Output](assets/output_terminal.png)

---

### 📈 Ranking Visualization

![Ranking Chart](assets/ranking_chart.png)

---

## 📂 Project Structure

