**🧠 Fake Job Post Detector**
A Machine Learning project that detects whether a job posting is real or fake using Natural Language Processing (NLP).
This project helps identify fraudulent job advertisements that can scam job seekers.

**🚀 Project Overview**
With the rise of online job portals, many fake job listings are created to collect user data or money.
This model analyzes job descriptions and predicts if a job posting is genuine or fraudulent.
**Goal:** Classify job posts as Real or Fake
**Model Used:** Logistic Regression
**Tech Stack:** Python, Pandas, Scikit-learn, TF-IDF, Streamlit

**🧩 Workflow**
**Data Preprocessing**
Cleaned text by removing stopwords, numbers, and punctuation
Lemmatized words for better understanding
Selected important features like title, description, and requirements

**Feature Extraction**
Used TF-IDF Vectorizer to convert job descriptions into numerical format

**Model Training**
Trained a Logistic Regression classifier
Achieved high accuracy on validation data

**Deployment**
Created an interactive Streamlit app (app.py)
User can paste a job description to check if it’s real or fake

**🧪 How to Run Locally**

1. Clone this repository
git clone https://github.com/yourusername/fake-job-post-detector.git
cd fake-job-post-detector

2.Install the dependencies
pip install -r requirements.txt

3.Run the Streamlit app
streamlit run app.py
Paste any job description and check the prediction result!

**📂 Project Files**
fake_job_postings.csv → Dataset used for training
model.pkl → Trained Logistic Regression model
app.py → Streamlit web application
requirements.txt → Dependencies list

**📈 Results**

✅ Successfully detects fake job descriptions
✅ Lightweight and fast NLP model
✅ Can be extended with deep learning models in the future
