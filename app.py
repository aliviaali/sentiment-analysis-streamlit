import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    nb_baseline = pickle.load(open("nb_baseline.pkl", "rb"))
    nb_opt = pickle.load(open("nb_optimized.pkl", "rb"))
    svm_baseline = pickle.load(open("svm_baseline.pkl", "rb"))
    svm_opt = pickle.load(open("svm_optimized.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return nb_baseline, nb_opt, svm_baseline, svm_opt, tfidf

nb_baseline, nb_opt, svm_baseline, svm_opt, tfidf = load_models()

# ================= PREPROCESSING =================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = stemmer.stem(text)
    return text

# ================= SIDEBAR =================
st.sidebar.title("Pengaturan Model")
model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ["Naive Bayes Baseline", 
     "Naive Bayes Optimized", 
     "SVM Baseline", 
     "SVM Optimized"]
)

# ================= INPUT USER =================
st.title("📊 Sentiment Analysis Web App")

user_input = st.text_area("Masukkan Kalimat:")

if st.button("Analisis"):

    clean_text = preprocess(user_input)
    vector = tfidf.transform([clean_text])

    if model_choice == "Naive Bayes Baseline":
        model = nb_baseline
    elif model_choice == "Naive Bayes Optimized":
        model = nb_opt
    elif model_choice == "SVM Baseline":
        model = svm_baseline
    else:
        model = svm_opt

    prediction = model.predict(vector)[0]

    st.success(f"Hasil Prediksi: {prediction}")

    # ================= PROBABILITAS =================
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        prob_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": proba
        })
        st.subheader("Probabilitas Kelas")
        st.dataframe(prob_df)

    # ================= PERHITUNGAN MANUAL (SIMULASI EXCEL) =================
    st.subheader("Perhitungan Manual (Simulasi Excel - TF-IDF)")

    feature_names = tfidf.get_feature_names_out()
    tfidf_vector = vector.toarray()[0]

    manual_df = pd.DataFrame({
        "Term": feature_names,
        "TF-IDF Value": tfidf_vector
    })

    manual_df = manual_df[manual_df["TF-IDF Value"] > 0]
    st.dataframe(manual_df.head(20))

# ================= UPLOAD DATASET UNTUK EVALUASI =================
st.header("Evaluasi Model + Uji Hipotesis")

uploaded_file = st.file_uploader("Upload Dataset CSV (text, label)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df["clean"] = df["text"].apply(preprocess)
    X = tfidf.transform(df["clean"])

    nb_acc = nb_opt.score(X, df["label"])
    svm_acc = svm_opt.score(X, df["label"])

    st.write("Accuracy Naive Bayes:", nb_acc)
    st.write("Accuracy SVM:", svm_acc)

    # ================= UJI HIPOTESIS (T-Test) =================
    st.subheader("Uji Hipotesis")

    nb_pred = nb_opt.predict(X)
    svm_pred = svm_opt.predict(X)

    t_stat, p_value = stats.ttest_ind(
        nb_pred == df["label"],
        svm_pred == df["label"]
    )

    st.write("T-Statistic:", t_stat)
    st.write("P-Value:", p_value)

    if p_value < 0.05:
        st.success("Terdapat perbedaan signifikan antara model NB dan SVM")
    else:
        st.warning("Tidak terdapat perbedaan signifikan antara model NB dan SVM")
