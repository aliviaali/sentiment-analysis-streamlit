import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Demo Skripsi Sentiment Analysis", layout="wide")
st.title("📊 Sentiment Analysis Berita Ekonomi Indonesia")
st.markdown("TF-IDF + Naive Bayes + SVM (Baseline & Optimized)")

# ===============================
# NLTK SAFE DOWNLOAD
# ===============================
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    models = {}

    files = {
        "Naive Bayes Baseline": "nb_baseline.pkl",
        "Naive Bayes Optimized": "nb_optimized.pkl",
        "SVM Baseline": "svm_baseline.pkl",
        "SVM Optimized": "svm_optimized.pkl"
    }

    for name, file in files.items():
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
    case_fold = text.lower()
    cleaning = re.sub(r'[^a-zA-Z\s]', '', case_fold)
    tokens = cleaning.split()
    stop_removed = [w for w in tokens if w not in stop_words]
    stemming = [stemmer.stem(w) for w in stop_removed]
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
# INPUT TEXT
# ===============================
text_input = st.text_area("Masukkan Teks Berita Ekonomi:")

model_choice = st.selectbox(
    "Pilih Model:",
    ["Semua Model"] + list(models.keys())
)

# ===============================
# ANALYSIS BUTTON
# ===============================
if st.button("🔍 Analisis Lengkap"):

    if text_input == "":
        st.warning("Masukkan teks terlebih dahulu")
    elif tfidf is None:
        st.error("TF-IDF tidak tersedia")
    else:

        result = preprocessing(text_input)

        # ===============================
        # PREPROCESSING DISPLAY
        # ===============================
        st.subheader("📌 Tahapan Preprocessing")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Sebelum")
            st.write(result["original"])
        with col2:
            st.markdown("### Sesudah")
            st.write(result["final"])

        st.write("Case Folding:", result["case_folding"])
        st.write("Cleaning:", result["cleaning"])
        st.write("Tokenization:", result["tokenization"])
        st.write("Stopword Removal:", result["stopword_removal"])
        st.write("Stemming:", result["stemming"])

        # ===============================
        # TF-IDF
        # ===============================
        st.subheader("📐 Perhitungan TF-IDF")

        st.latex(r"TF(t,d) = \frac{f(t,d)}{\sum f(t,d)}")
        st.latex(r"IDF(t) = \log\left(\frac{N}{df(t)}\right)")
        st.latex(r"TFIDF(t,d) = TF(t,d) \times IDF(t)")

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

        # ===============================
        # PREDICTION
        # ===============================
        st.subheader("🤖 Hasil Prediksi")

        results = {}

        if model_choice == "Semua Model":
            for name, model in models.items():
                pred = model.predict(vector)[0]
                results[name] = pred
                st.success(f"{name} ➜ {pred}")
        else:
            pred = models[model_choice].predict(vector)[0]
            results[model_choice] = pred
            st.success(f"{model_choice} ➜ {pred}")

        # ===============================
        # EVALUATION SIMULASI
        # ===============================
        st.subheader("📊 Evaluasi Model (Simulasi)")

        true_label = st.selectbox("Masukkan Label Sebenarnya:", ["positif", "negatif", "netral"])

        if st.button("Hitung Evaluasi"):

            for name, pred in results.items():

                y_true = [true_label]
                y_pred = [pred]

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

                st.write(f"### {name}")
                st.write("Accuracy:", acc)
                st.write("Precision:", prec)
                st.write("Recall:", rec)
                st.write("F1-Score:", f1)

                cm = confusion_matrix(y_true, y_pred, labels=["positif","negatif","netral"])

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d',
                            xticklabels=["positif","negatif","netral"],
                            yticklabels=["positif","negatif","netral"])
                plt.ylabel("Actual")
                plt.xlabel("Predicted")
                st.pyplot(fig)

        # ===============================
        # UJI HIPOTESIS
        # ===============================
        st.subheader("📑 Uji Hipotesis")

        st.markdown("""
        **H0:** Tidak terdapat perbedaan performa signifikan antara model baseline dan optimized.  
        **H1:** Terdapat perbedaan performa signifikan antara model baseline dan optimized.
        """)

        st.info("""
        Jika nilai Accuracy / F1-Score model optimized lebih tinggi,
        maka H0 ditolak dan H1 diterima.
        """)
