import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from statsmodels.stats.contingency_tables import mcnemar
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Sentiment Analysis Research App", layout="wide")

# ==========================================================
# ================= LOAD MODEL =============================
# ==========================================================
@st.cache_resource
def load_models():
    nb_baseline = pickle.load(open("nb_baseline.pkl", "rb"))
    nb_opt = pickle.load(open("nb_optimized.pkl", "rb"))
    svm_baseline = pickle.load(open("svm_baseline.pkl", "rb"))
    svm_opt = pickle.load(open("svm_optimized.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return nb_baseline, nb_opt, svm_baseline, svm_opt, tfidf

nb_baseline, nb_opt, svm_baseline, svm_opt, tfidf = load_models()

# ==========================================================
# ================= PREPROCESSING ==========================
# ==========================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = stemmer.stem(text)
    return text

# ==========================================================
# ================= SIDEBAR MODEL ==========================
# ==========================================================
st.sidebar.title("Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    [
        "Naive Bayes Baseline",
        "Naive Bayes Optimized",
        "SVM Baseline",
        "SVM Optimized"
    ]
)

if model_choice == "Naive Bayes Baseline":
    selected_model = nb_baseline
elif model_choice == "Naive Bayes Optimized":
    selected_model = nb_opt
elif model_choice == "SVM Baseline":
    selected_model = svm_baseline
else:
    selected_model = svm_opt

# ==========================================================
# ================= JUDUL ==========================
# ==========================================================
st.title("📊 Website Sentiment Analysis Skripsi")
st.markdown("Implementasi Naive Bayes & SVM menggunakan TF-IDF + Uji Hipotesis")

# ==========================================================
# ================= ANALISIS MANUAL ========================
# ==========================================================
st.header("Analisis Kalimat Manual")

user_input = st.text_area("Masukkan Kalimat:")

if st.button("Analisis Sentimen"):

    if user_input.strip() != "":
        cleaned = preprocess(user_input)
        vector = tfidf.transform([cleaned])
        prediction = selected_model.predict(vector)[0]

        st.subheader("Hasil Preprocessing")
        st.write("Kalimat Asli:", user_input)
        st.write("Setelah Preprocessing:", cleaned)

        st.success(f"Hasil Prediksi: {prediction}")

        # ================= TF-IDF EXPLANATION =================
        st.subheader("Perhitungan TF-IDF (Simulasi Excel)")

        st.markdown("""
        ### Rumus TF-IDF:

        **TF (Term Frequency)**  
        TF = (Jumlah kemunculan term dalam dokumen) / (Total kata dalam dokumen)

        **IDF (Inverse Document Frequency)**  
        IDF = log (Jumlah Dokumen / Jumlah Dokumen yang mengandung term)

        **TF-IDF = TF × IDF**
        """)

        feature_names = tfidf.get_feature_names_out()
        tfidf_values = vector.toarray()[0]

        tfidf_df = pd.DataFrame({
            "Term": feature_names,
            "TF-IDF Value": tfidf_values
        })

        tfidf_df = tfidf_df[tfidf_df["TF-IDF Value"] > 0]
        st.dataframe(tfidf_df.sort_values(by="TF-IDF Value", ascending=False).head(20))

# ==========================================================
# ================= EVALUASI MODEL =========================
# ==========================================================
st.header("Evaluasi Model + Uji Hipotesis")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")
    st.dataframe(df.head())
    st.write("Kolom tersedia:", list(df.columns))

    with st.form("evaluasi_form"):

        text_column = st.selectbox("Pilih Kolom Text:", df.columns)
        label_column = st.selectbox("Pilih Kolom Label:", df.columns)

        submit_eval = st.form_submit_button("Proses Evaluasi")

    if submit_eval:

        if text_column == label_column:
            st.error("Kolom Text dan Label tidak boleh sama!")
        else:
            st.info("Sedang memproses evaluasi...")

            df["clean"] = df[text_column].astype(str).apply(preprocess)

            st.subheader("Sebelum & Sesudah Preprocessing")
            preview = pd.DataFrame({
                "Text Asli": df[text_column].head(10),
                "Setelah Preprocessing": df["clean"].head(10)
            })
            st.dataframe(preview)

            X = tfidf.transform(df["clean"])
            y = df[label_column]

            nb_pred = nb_opt.predict(X)
            svm_pred = svm_opt.predict(X)

            # ================= METRICS =================
            nb_acc = accuracy_score(y, nb_pred)
            svm_acc = accuracy_score(y, svm_pred)

            st.subheader("Accuracy")
            st.write("Naive Bayes:", round(nb_acc, 4))
            st.write("SVM:", round(svm_acc, 4))

            nb_f1 = precision_recall_fscore_support(y, nb_pred, average="macro")[2]
            svm_f1 = precision_recall_fscore_support(y, svm_pred, average="macro")[2]

            st.subheader("F1 Macro")
            st.write("Naive Bayes:", round(nb_f1, 4))
            st.write("SVM:", round(svm_f1, 4))

            # ================= CONFUSION MATRIX =================
            st.subheader("Confusion Matrix (SVM)")
            cm = confusion_matrix(y, svm_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # ================= UJI HIPOTESIS =================
            st.subheader("Uji Hipotesis")

            # T-Test
            nb_correct = (nb_pred == y).astype(int)
            svm_correct = (svm_pred == y).astype(int)

            t_stat, p_value = stats.ttest_ind(nb_correct, svm_correct)

            st.write("T-Test p-value:", round(p_value, 5))

            if p_value < 0.05:
                st.success("Terdapat perbedaan signifikan (T-Test)")
            else:
                st.warning("Tidak terdapat perbedaan signifikan (T-Test)")

            # McNemar
            table = [[0, 0], [0, 0]]

            for i in range(len(y)):
                if nb_pred[i] == y.iloc[i] and svm_pred[i] == y.iloc[i]:
                    table[0][0] += 1
                elif nb_pred[i] == y.iloc[i] and svm_pred[i] != y.iloc[i]:
                    table[0][1] += 1
                elif nb_pred[i] != y.iloc[i] and svm_pred[i] == y.iloc[i]:
                    table[1][0] += 1
                else:
                    table[1][1] += 1

            result = mcnemar(table, exact=True)

            st.write("McNemar p-value:", round(result.pvalue, 5))

            if result.pvalue < 0.05:
                st.success("Terdapat perbedaan signifikan (McNemar)")
            else:
                st.warning("Tidak terdapat perbedaan signifikan (McNemar)")
