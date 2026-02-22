import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import stats
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analysis Skripsi", layout="wide")

st.title("📊 Website Sentiment Analysis Skripsi")
st.write("Implementasi SVM + TF-IDF + Uji Hipotesis + Stratified Random Sampling")

# ==========================================================
# LOAD MODEL (RINGAN & AMAN)
# ==========================================================
@st.cache_resource
def load_model():
    model = pickle.load(open("svm_optimized.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return model, tfidf

model, tfidf = load_model()

# ==========================================================
# PREPROCESSING
# ==========================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = stemmer.stem(text)
    return text

# ==========================================================
# ANALISIS MANUAL
# ==========================================================
st.header("1️⃣ Analisis Kalimat Manual")

user_input = st.text_area("Masukkan Kalimat")

if st.button("Analisis Sentimen"):
    cleaned = preprocess(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    st.subheader("Hasil Preprocessing")
    st.write("Kalimat Asli:", user_input)
    st.write("Setelah Preprocessing:", cleaned)

    st.success(f"Hasil Prediksi: {prediction}")

    # ================= TF-IDF =================
    st.subheader("Perhitungan TF-IDF (Simulasi Excel)")

    st.markdown("""
    ### Rumus TF-IDF

    **TF = (Jumlah term dalam dokumen) / (Total kata dalam dokumen)**  
    **IDF = log(N / df)**  
    **TF-IDF = TF × IDF**
    """)

    feature_names = tfidf.get_feature_names_out()
    values = vector.toarray()[0]

    tfidf_df = pd.DataFrame({
        "Term": feature_names,
        "TF-IDF Value": values
    })

    tfidf_df = tfidf_df[tfidf_df["TF-IDF Value"] > 0]
    st.dataframe(tfidf_df.sort_values(by="TF-IDF Value", ascending=False).head(15))


# ==========================================================
# EVALUASI DATASET
# ==========================================================
st.header("2️⃣ Evaluasi Model + Uji Hipotesis")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    text_col = st.selectbox("Pilih Kolom Text", df.columns)
    label_col = st.selectbox("Pilih Kolom Label", df.columns)

    if st.button("Proses Evaluasi"):

        df["clean"] = df[text_col].astype(str).apply(preprocess)

        X = tfidf.transform(df["clean"])
        y = df[label_col]

        pred = model.predict(X)

        acc = accuracy_score(y, pred)
        st.write("Accuracy:", round(acc, 4))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i][j], ha="center", va="center")

        st.pyplot(fig)

        # Uji T-Test sederhana
        correct = (pred == y).astype(int)
        t_stat, p_val = stats.ttest_1samp(correct, 0.5)

        st.subheader("Uji Hipotesis")
        st.write("H0: Model tidak lebih baik dari tebakan acak (50%)")
        st.write("p-value:", round(p_val, 5))

        if p_val < 0.05:
            st.success("Model signifikan lebih baik dari random")
        else:
            st.warning("Model tidak signifikan")

# ==========================================================
# STRATIFIED RANDOM SAMPLING
# ==========================================================
st.header("3️⃣ Stratified Random Sampling (Perhitungan Excel)")

st.markdown("""
### Rumus Stratified Random Sampling

Jika:

N = Total populasi  
Nh = Jumlah populasi tiap kelas  
n = Jumlah sampel yang diambil  

Maka jumlah sampel tiap kelas:

**nh = (Nh / N) × n**
""")

if uploaded_file is not None:

    total_data = len(df)
    sample_size = st.number_input("Masukkan jumlah sampel (n)", min_value=1, max_value=total_data, value=int(total_data*0.3))

    if st.button("Hitung Stratified Sampling"):

        distribusi = df[label_col].value_counts().reset_index()
        distribusi.columns = ["Kelas", "Jumlah (Nh)"]

        distribusi["Total (N)"] = total_data
        distribusi["Sampel (nh)"] = (distribusi["Jumlah (Nh)"] / total_data * sample_size).round().astype(int)

        st.subheader("Perhitungan Manual (Seperti Excel)")
        st.dataframe(distribusi)

        st.markdown("""
        ### Rumus Excel

        Jika:
        - Nh di sel B2
        - N di sel C2
        - n di sel D1

        Maka rumus Excel:

        `= (B2 / C2) * $D$1`
        """)
