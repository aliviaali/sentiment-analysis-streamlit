import streamlit as st
import pickle
import re
import pandas as pd
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis Berita Ekonomi")
st.write("TF-IDF + Naive Bayes + SVM")

# =============================
# DOWNLOAD STOPWORDS
# =============================
nltk.download('stopwords', quiet=True)

# =============================
# LOAD MODELS
# =============================
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

# =============================
# PREPROCESSING
# =============================
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocessing(text):
    case_folding = text.lower()
    cleaning = re.sub(r'[^a-zA-Z\s]', '', case_folding)
    tokens = cleaning.split()
    stop_removed = [word for word in tokens if word not in stop_words]
    stemming = [stemmer.stem(word) for word in stop_removed]
    final_text = " ".join(stemming)

    return case_folding, cleaning, tokens, stop_removed, stemming, final_text

# =============================
# INPUT
# =============================
text_input = st.text_area("Masukkan Teks:")

model_choice = st.selectbox(
    "Pilih Model:",
    ["Semua Model"] + list(models.keys())
)

# =============================
# BUTTON
# =============================
if st.button("Analisis"):

    if text_input == "":
        st.warning("Masukkan teks terlebih dahulu.")
    elif tfidf is None:
        st.error("TF-IDF tidak tersedia.")
    else:

        case_folding, cleaning, tokens, stop_removed, stemming, final_text = preprocessing(text_input)

        # =============================
        # PREPROCESSING OUTPUT
        # =============================
        st.subheader("Tahapan Preprocessing")

        st.write("Teks Asli:")
        st.write(text_input)

        st.write("Case Folding:")
        st.write(case_folding)

        st.write("Cleaning:")
        st.write(cleaning)

        st.write("Tokenization:")
        st.write(tokens)

        st.write("Stopword Removal:")
        st.write(stop_removed)

        st.write("Stemming:")
        st.write(stemming)

        st.write("Hasil Akhir Preprocessing:")
        st.write(final_text)

        # =============================
        # TF-IDF
        # =============================
        st.subheader("Hasil TF-IDF")

        vector = tfidf.transform([final_text])
        feature_names = tfidf.get_feature_names_out()
        dense = vector.todense().tolist()[0]

        tfidf_df = pd.DataFrame({
            "Term": feature_names,
            "Score": dense
        })

        tfidf_df = tfidf_df[tfidf_df["Score"] > 0]
        tfidf_df = tfidf_df.sort_values(by="Score", ascending=False)

        st.dataframe(tfidf_df)

        # =============================
        # PREDICTION
        # =============================
        st.subheader("Hasil Prediksi")

        if model_choice == "Semua Model":
            for name, model in models.items():
                try:
                    prediction = model.predict(vector)[0]
                    st.write(f"{name}: {prediction}")
                except:
                    st.write(f"{name}: Gagal prediksi")
        else:
            prediction = models[model_choice].predict(vector)[0]
            st.write(f"{model_choice}: {prediction}")
