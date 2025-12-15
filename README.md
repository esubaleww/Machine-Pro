# 📩 SMS Spam Detection (Streamlit)

Simple end‑to‑end SMS spam detector built with **Python**, **scikit‑learn**, and **Streamlit**.  
You can type or paste any message and see how likely it is to be **spam** or **not spam**.

---

## Files

All files are in the `Machine-Pro` folder:

- `spam.csv` – SMS Spam Collection dataset (ham/spam messages)
- `spam_model_training.ipynb` – Jupyter notebook for training and experiments
- `spam_model.pkl` – trained Logistic Regression model
- `vectorizer.pkl` – fitted TfidfVectorizer
- `app.py` – Streamlit web app
- `requirements.txt` – Python dependencies

---

## Model

- Dataset: SMS Spam Collection (+ some custom examples for modern scams and tricky ham)
- Features: `TfidfVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2)`
- Model: `LogisticRegression(max_iter=2000, class_weight="balanced")`
- Output: spam probability from `predict_proba`, mapped to:
  - Likely SPAM
  - Likely NOT spam
  - Borderline (uncertain)

---

## Setup

- cd Machine-Pro
- pip install -r requirements.txt

If you want to retrain the model, open `spam_model_training.ipynb` in Jupyter and run all cells.

---

## Run the app

- cd Machine-Pro
- streamlit run app.py

Then open the URL shown in the terminal (usually `http://localhost:8501`), paste a message, and check the prediction.
