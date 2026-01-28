# 🛡️ FakeGuard Pro — Fake News Detection System

A modern, lightweight Fake News Detection system that classifies news articles as **Real** or **Fake** using classic NLP + ML techniques.  
FakeGuard Pro includes reproducible data preprocessing, model training, and a Streamlit web interface for real-time predictions — ideal for demos, coursework, and interviews.

---

## ✨ Highlights

- ✅ Fast, explainable baseline using TF‑IDF + Logistic Regression  
- 🌐 Polished Streamlit UI for real-time classification and confidence scores  
- 📁 Reproducible pipelines: data prep → train → serve  
- 🧪 Evaluation scripts and basic metrics (accuracy, confusion matrix)  
- 🔁 Easy to extend: swap vectorizer, test other classifiers, or add deployment

---

## 📌 Why this project

Misinformation spreads quickly. FakeGuard Pro demonstrates an end-to-end Machine Learning workflow that:
- Ingests raw news datasets
- Cleans and vectorizes textual content
- Trains a robust baseline classifier
- Exposes predictions via a simple, interactive web app

This repository is suitable for coursework, portfolio presentation, and interview walk-throughs.

---

## 🚀 Quick Demo (what you get)

Paste a news article into the web app and receive:
- 🔍 Label: Real or Fake
- 📊 Confidence score (probability)
- ⚠️ Helpful error messages for invalid input or missing model files

---

## 📁 Project structure

```
Fake_news_detect/
│
├── app/
│   └── streamlit.py          # Streamlit web application (UI + inference)
│
├── data/
│   ├── raw/                  # original downloaded datasets (fake.csv, true.csv)
│   └── processed/            # cleaned & split CSVs (train.csv, test.csv)
│
├── models/
│   ├── tfidf_vect.joblib     # saved TF-IDF vectorizer
│   └── tfidf_clf.joblib      # saved classifier
│
├── src/
│   ├── data_prep.py          # data cleaning, merging, train/test split
│   ├── features.py           # tokenization / TF-IDF helper functions
│   ├── train.py              # training, evaluation, and model export
│   └── eval.py               # evaluation utilities and plots
│
├── notebooks/                # optional EDA and training experiments
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech stack

- Language: Python 3.8+  
- Data: pandas, numpy  
- ML: scikit-learn (TF‑IDF, LogisticRegression)  
- Serialization: joblib  
- Web UI: Streamlit  
- Optional: matplotlib / seaborn for plots

---

## ⚙️ Setup & Run

1. Clone the repo
```bash
git clone https://github.com/Manjunath-G-K/Fake_news_detect.git
cd Fake_news_detect
```

2. Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Prepare the dataset (downloads / cleaning)
```bash
python src/data_prep.py
# Produces: data/processed/train.csv and data/processed/test.csv
```

5. Train the model
```bash
python src/train.py --input data/processed/train.csv --output models/
# Trains TF-IDF + LogisticRegression and saves vectorizer + model to models/
```

6. Run the web app
```bash
streamlit run app/streamlit.py
# Open http://localhost:8501
```

Tip: Use `--help` on each script for available flags (e.g., --n_features, --test-size).

---

## 📊 Model summary 

- Vectorizer: TF‑IDF (unigrams + bigrams)  
- Classifier: Logistic Regression (L2, default regularization)  
- Features: top N TF‑IDF features (configurable in train.py)  
- Evaluation: accuracy, precision/recall/F1, confusion matrix — reported after training

Example (illustrative) metrics:
- Accuracy: ~0.94  
- Precision / Recall / F1: reported per class in eval output

(Exact numbers will depend on preprocessing and dataset split.)

---

## ✅ Best practices included

- Reproducible train/eval pipeline with fixed random seed  
- Minimal but readable preprocessing (lowercasing, punctuation removal, basic stopwords)  
- Model & vectorizer versioned via joblib files in `models/`  
- Streamlit app checks for model presence and shows helpful instructions if missing

---



## 👥 Team

This project was developed as a group college mini project by:

- 👩‍💻 M Anitha  
- 👩‍💻 Madiha Naz  
- 👨‍💻 Mallikarjun M B  
- 👨‍💻 Manjunath G K
