import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Demo Sentiment Analysis", layout="wide")
st.title("📊 Demo Skripsi Sentiment Analysis")
st.markdown("TF-IDF + Naive Bayes + SVM (Baseline & Optimized)")

# ===============================
# DOWNLOAD STOPWORDS (SAFE)
# ===============================
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# ===============================
# LOAD ALL MODELS
# ===============================
@st.cache_resource
def load_models():
    models = {}

    file_list = {
        "Naive Bayes Baseline": "nb_baseline.pkl",
        "Naive Bayes Optimized": "nb_optimized.pkl",
        "SVM Baseline": "svm_baseline.pkl",
        "SVM Optimized": "svm_optimized.pkl"
    }

    for name, file in file_list.items():
        try:
            models[name] = pickle.load(open(file, "rb"))
        except:
            st.warning(f"{file} tidak ditemukan")

    try:
        tfidf = pickle.load(open("tfidf.pkl", "rb"))
    except:
        st.error("tfidf.pkl tidak ditemukan")
        tfidf = None

    return models, tfidf

models, tfidf = load_models()

# ===============================
# PREPROCESSING
# ===============================
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocessing(text):
    original = text

    # Case Folding
    case_fold = text.lower()

    # Cleaning
    cleaning = re.sub(r'[^a-zA-Z\s]', '', case_fold)

    # Tokenization
    tokens = cleaning.split()

    # Stopword Removal
    stop_removed = [word for word in tokens if word not in stop_words]

    # Stemming
    stemming = [stemmer.stem(word) for word in stop_removed]

    final_text = " ".join(stemming)

    return {
        "original": original,
        "case_folding": case_fold,
        "cleaning": cleaning,
        "tokenization": tokens,
        "stopword_removal": stop_removed,
        "stemming": stemming,
        "final": final_text
    }

# ===============================
# INPUT
# ===============================
text_input = st.text_area("Masukkan Teks Berita Ekonomi:")

model_choice = st.selectbox(
    "Pilih Model:",
    ["Semua Model"] + list(models.keys())
)

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔍 Analisis Sentimen"):

    if text_input == "":
        st.warning("Masukkan teks terlebih dahulu")
    elif tfidf is None:
        st.error("TF-IDF tidak tersedia")
    else:

        result = preprocessing(text_input)

        st.subheader("📌 Tahapan Preprocessing")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Sebelum Preprocessing")
            st.write(result["original"])

        with col2:
            st.markdown("### Setelah Preprocessing")
            st.write(result["final"])

        st.markdown("### Detail Proses")
        st.write("Case Folding:", result["case_folding"])
        st.write("Cleaning:", result["cleaning"])
        st.write("Tokenization:", result["tokenization"])
        st.write("Stopword Removal:", result["stopword_removal"])
        st.write("Stemming:", result["stemming"])

        # ============================
        # TF-IDF
        # ============================
        st.subheader("📐 Hasil TF-IDF")

        vector = tfidf.transform([result["final"]])
        feature_names = tfidf.get_feature_names_out()
        dense = vector.todense().tolist()[0]

        tfidf_df = pd.DataFrame({
            "Term": feature_names,
            "TF-IDF Score": dense
        })

        tfidf_df = tfidf_df[tfidf_df["TF-IDF Score"] > 0]
        tfidf_df = tfidf_df.sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(tfidf_df)

        # ============================
        # PREDICTION
        # ============================
        st.subheader("🤖 Hasil Prediksi Model")

        if model_choice == "Semua Model":

            for name, model in models.items():
                try:
                    prediction = model.predict(vector)[0]
                    st.success(f"{name} ➜ {prediction}")
                except:
                    st.error(f"{name} gagal melakukan prediksi")

        else:
            try:
                prediction = models[model_choice].predict(vector)[0]
                st.success(f"{model_choice} ➜ {prediction}")
            except:
                st.error("Model gagal melakukan prediksi")
