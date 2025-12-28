# ===============================
# IMPORT LIBRARY
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen Ulasan UMKM",
    page_icon="📊",
    layout="wide"
)

# ===============================
# CUSTOM CSS (FIXED)
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#e5e7eb 0%,#dbeafe 40%,#e9d5ff 100%);
}

/* TEXT */
h1,h2,h3,h4,h5,h6 { color:#0f172a!important; }
p,label,span,div { color:#1e293b; }

/* BUTTON (ALL) */
.stButton > button,
.stFormSubmitButton > button,
button[kind="primary"],
button[kind="secondary"]{
    background: linear-gradient(90deg,#4f46e5,#7c3aed)!important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.55rem 1.3rem !important;
}
button:hover { opacity:0.9; }

/* INPUT */
textarea, input{
    background:#f8fafc!important;
    color:#0f172a!important;
    border:1px solid #a5b4fc;
    border-radius:8px;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"]{
    background:#4f46e5!important;
    border:2px dashed #e0e7ff!important;
    border-radius:14px;
    padding:1rem;
}
div[data-testid="stFileUploader"] *{
    color:white!important;
}

/* CARD */
div[data-testid="stVerticalBlock"] > div{
    background:rgba(248,250,252,0.95);
    padding:1rem;
    border-radius:14px;
    box-shadow:0 10px 25px rgba(0,0,0,0.08);
}

/* TABLE */
thead th{
    background:#c7d2fe!important;
    color:#0f172a!important;
}
tbody td{
    color:#0f172a!important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("📊 Analisis Sentimen Ulasan Produk UMKM")
st.write("Analisis cepat, akurat, dan menghasilkan insight bisnis.")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("models/logistic_regression/best_model_logistic_regression.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/logistic_regression/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ===============================
# NLP PREPROCESSING
# ===============================
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in text.split() if t not in stopwords]
    return stemmer.stem(" ".join(tokens))

# ===============================
# KONSTANTA
# ===============================
CLASS_LABEL_MAP = {-1: "Negatif", 0: "Netral", 1: "Positif"}
CONF_THRESHOLD = 0.6

ISSUE_RULES = {
    "Pengiriman": ["kirim","pengiriman","lama","kurir"],
    "Kualitas Produk": ["rusak","cacat","pecah","jelek","kurang"],
    "Pelayanan": ["cs","admin","respon","pelayanan"],
    "Harga": ["mahal","murah","harga"]
}

RECOMMENDATION_MAP = {
    "Pengiriman": "Evaluasi kurir dan estimasi pengiriman.",
    "Kualitas Produk": "Perbaiki quality control sebelum pengiriman.",
    "Pelayanan": "Tingkatkan respon CS (<24 jam).",
    "Harga": "Sesuaikan harga dengan kualitas produk.",
    "Umum": "Pantau ulasan pelanggan secara berkala."
}

# ===============================
# HELPER
# ===============================
def detect_issue(clean_text):
    for issue, keywords in ISSUE_RULES.items():
        if any(k in clean_text for k in keywords):
            return issue
    return "Umum"

def detect_urgency(sentiment, confidence):
    if sentiment == "Negatif" and confidence > 0.75:
        return "Tinggi"
    elif sentiment == "Netral":
        return "Sedang"
    else:
        return "Rendah"

# ===============================
# MODE INPUT
# ===============================
mode = st.radio(
    "🧭 Pilih metode analisis:",
    ["✍️ Teks Tunggal", "📂 Upload CSV"],
    horizontal=True
)

# ===============================
# MODE 1 : TEKS TUNGGAL
# ===============================
if mode == "✍️ Teks Tunggal":
    text_input = st.text_area("Masukkan teks ulasan")

    if st.button("🔍 Analisis"):
        clean = preprocess_text(text_input)
        X = tfidf.transform([clean])

        proba = model.predict_proba(X)[0]
        confidence = proba.max()
        pred_label = model.classes_[np.argmax(proba)]

        sentiment = "Netral" if confidence < CONF_THRESHOLD else CLASS_LABEL_MAP[pred_label]

        st.subheader("📌 Hasil Analisis")
        st.success(f"**Sentimen:** {sentiment}\n\n**Confidence:** {confidence:.2f}")

        urgency = detect_urgency(sentiment, confidence)
        issue = detect_issue(clean)

        st.write(f"**Urgensi:** {urgency}")
        st.write(f"**Isu Utama:** {issue}")

        st.subheader("🧠 Rekomendasi")
        st.info(RECOMMENDATION_MAP.get(issue))

# ===============================
# MODE 2 : CSV
# ===============================
else:
    st.subheader("📂 Analisis CSV")

    with st.form("csv_form"):
        col1, col2 = st.columns([3,1])
        with col1:
            file = st.file_uploader("Upload CSV (kolom wajib: text)", type=["csv"])
        with col2:
            submit = st.form_submit_button("🚀 Jalankan Analisis")

    if submit:
        if file is None:
            st.error("⚠️ Silakan upload file CSV terlebih dahulu.")
        else:
            df = pd.read_csv(file)

            if "text" not in df.columns:
                st.error("❌ CSV harus memiliki kolom 'text'")
            else:
                with st.spinner("Menganalisis data..."):
                    df["clean_text"] = df["text"].astype(str).apply(preprocess_text)

                    X = tfidf.transform(df["clean_text"])
                    proba = model.predict_proba(X)

                    df["confidence"] = proba.max(axis=1)
                    pred_label = model.classes_[np.argmax(proba, axis=1)]

                    df["sentiment"] = [
                        "Netral" if df["confidence"][i] < CONF_THRESHOLD
                        else CLASS_LABEL_MAP[pred_label[i]]
                        for i in range(len(df))
                    ]

                    df["issue"] = df["clean_text"].apply(detect_issue)
                    df["urgency"] = [
                        detect_urgency(df["sentiment"][i], df["confidence"][i])
                        for i in range(len(df))
                    ]

                # ===============================
                # INSIGHT
                # ===============================
                st.subheader("📊 Ringkasan Sentimen")
                st.bar_chart(df["sentiment"].value_counts())

                st.subheader("🧠 Insight Bisnis")

                top_issue = df["issue"].value_counts().idxmax()
                st.write(f"**Isu paling sering muncul:** {top_issue}")

                neg = df[df["sentiment"] == "Negatif"]
                if not neg.empty:
                    neg_issue = neg["issue"].value_counts().idxmax()
                    st.error(f"⚠️ **Isu negatif utama:** {neg_issue}")
                    st.info(RECOMMENDATION_MAP.get(neg_issue))
                else:
                    st.success("✅ Tidak ada isu negatif dominan.")

                high_urgency = df[df["urgency"] == "Tinggi"]
                if not high_urgency.empty:
                    st.warning(
                        f"🚨 **{len(high_urgency)} ulasan** memiliki urgensi tinggi "
                        "dan perlu ditindaklanjuti segera."
                    )

                st.subheader("📋 Hasil Lengkap")
                st.dataframe(df)

                st.download_button(
                    "⬇️ Download hasil_sentimen.csv",
                    df.to_csv(index=False),
                    file_name="hasil_sentimen.csv",
                    mime="text/csv"
                )
