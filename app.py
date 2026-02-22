# ================= EVALUASI MODEL + UJI HIPOTESIS =================
st.header("Evaluasi Model + Uji Hipotesis")

uploaded_file = st.file_uploader("Upload Dataset CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    st.write("Kolom dalam dataset:", list(df.columns))

    # ================= AUTO DETECT KOLOM =================
    text_column = st.selectbox("Pilih Kolom Text:", df.columns)
    label_column = st.selectbox("Pilih Kolom Label:", df.columns)

    if st.button("Proses Evaluasi"):

        # ====== TAMPILKAN SEBELUM & SESUDAH PREPROCESS ======
        st.subheader("Hasil Preprocessing")

        df["clean"] = df[text_column].astype(str).apply(preprocess)

        preview_df = pd.DataFrame({
            "Text Asli": df[text_column].head(10),
            "Setelah Preprocessing": df["clean"].head(10)
        })

        st.dataframe(preview_df)

        # ====== TRANSFORM TF-IDF ======
        X = tfidf.transform(df["clean"])

        y = df[label_column]

        nb_acc = nb_opt.score(X, y)
        svm_acc = svm_opt.score(X, y)

        st.subheader("Accuracy Model")
        st.write("Accuracy Naive Bayes:", round(nb_acc,4))
        st.write("Accuracy SVM:", round(svm_acc,4))

        # ================= UJI HIPOTESIS =================
        st.subheader("Uji Hipotesis (Independent T-Test)")

        nb_pred = nb_opt.predict(X)
        svm_pred = svm_opt.predict(X)

        nb_correct = (nb_pred == y).astype(int)
        svm_correct = (svm_pred == y).astype(int)

        t_stat, p_value = stats.ttest_ind(nb_correct, svm_correct)

        st.write("T-Statistic:", round(t_stat,4))
        st.write("P-Value:", round(p_value,4))

        if p_value < 0.05:
            st.success("Terdapat perbedaan signifikan antara NB dan SVM (Tolak H0)")
        else:
            st.warning("Tidak terdapat perbedaan signifikan antara NB dan SVM (Gagal Tolak H0)")
