import streamlit as st
import pandas as pd
import pickle
import re
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="WhatsApp House Classifier",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# ADVANCED COLORFUL CSS
# ============================================================
st.markdown("""
<style>

/* ---------- Background ---------- */
.stApp {
    background: radial-gradient(circle at top left, #1a1a40, #0f0f1a);
    color: white;
}

/* ---------- Container ---------- */
.block-container {
    padding-top: 2rem;
}

/* ---------- Title ---------- */
h1 {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ff4ecd, #6a5cff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(255, 78, 205, 0.4);
}

/* ---------- Subtitles ---------- */
h2, h3 {
    color: #ffffff;
    text-shadow: 0 0 15px rgba(255,255,255,0.25);
}

/* ---------- Glass Card ---------- */
.glass {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 1.6rem;
    box-shadow: 0 15px 40px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.15);
}

/* ---------- Metrics ---------- */
.metric-value {
    font-size: 2.6rem;
    font-weight: 800;
}

.metric-label {
    letter-spacing: 1.4px;
    opacity: 0.85;
}

/* ---------- File uploader ---------- */
section[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #ff4ecd, #6a5cff);
    padding: 1.5rem;
    border-radius: 22px;
    color: white;
    box-shadow: 0 0 45px rgba(255, 78, 205, 0.5);
}

/* ---------- Table ---------- */
table {
    background: rgba(255,255,255,0.95);
    border-radius: 18px;
    overflow: hidden;
}

/* ---------- House Pills ---------- */
.house {
    padding: 7px 16px;
    border-radius: 999px;
    font-weight: 800;
    color: white;
    display: inline-block;
    text-shadow: 0 0 8px rgba(0,0,0,0.35);
}

.gryffindor { background: linear-gradient(90deg, #ff512f, #dd2476); }
.slytherin  { background: linear-gradient(90deg, #11998e, #38ef7d); }
.ravenclaw  { background: linear-gradient(90deg, #396afc, #2948ff); }
.hufflepuff { background: linear-gradient(90deg, #f7971e, #ffd200); }

/* ---------- Divider ---------- */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #6a5cff, transparent);
    margin: 2.5rem 0;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================
MODEL_PATH = Path(__file__).parent / "house_classifier.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model, tfidf = load_model()

# ============================================================
# NLP PREPROCESSING
# ============================================================
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ============================================================
# WHATSAPP CHAT PARSER
# ============================================================
def parse_whatsapp_chat(uploaded_file):
    uploaded_file.seek(0)
    text = uploaded_file.read().decode("utf-8", errors="ignore")

    pattern = re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[AP]M)?)\s-\s([^:]+):\s(.*)"
    )

    data = []
    current_speaker = None
    current_message = ""

    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            if current_speaker and current_message:
                data.append((current_speaker, current_message.strip()))
            current_speaker = match.group(3).strip()
            current_message = match.group(4).strip()
        else:
            if current_speaker:
                current_message += " " + line.strip()

    if current_speaker and current_message:
        data.append((current_speaker, current_message.strip()))

    df = pd.DataFrame(data, columns=["speaker_first_name", "message"])

    df = df[
        (~df["message"].str.contains("<Media omitted>", na=False)) &
        (~df["speaker_first_name"].str.contains("WhatsApp", na=False))
    ]

    return df

# ============================================================
# HEADER
# ============================================================
st.title("🏠 WhatsApp House Classifier")
st.markdown("""
<p style="text-align:center; font-size:1.15rem; opacity:0.85;">
Upload a WhatsApp chat and reveal everyone’s true house ✨
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass" style="text-align:center;">
📤 <b>How to use</b><br><br>
Export WhatsApp chat → <b>WITHOUT media</b> → Upload the <code>.txt</code> file
</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_file = st.file_uploader(
    "Upload WhatsApp chat (.txt)",
    type="txt"
)

# ============================================================
# MAIN LOGIC
# ============================================================
if uploaded_file:
    chat_df = parse_whatsapp_chat(uploaded_file)

    if chat_df.empty:
        st.error("No messages could be parsed.")
        st.stop()

    # Group messages by user
    chat_user = (
        chat_df
        .groupby("speaker_first_name")["message"]
        .apply(" ".join)
        .reset_index()
    )

    # Preprocess
    chat_user["clean_text"] = chat_user["message"].apply(preprocess)
    chat_user = chat_user[chat_user["clean_text"].str.strip() != ""]

    if chat_user.empty:
        st.error("No usable text after preprocessing.")
        st.stop()

    # Predict
    X_new = tfidf.transform(chat_user["clean_text"])
    chat_user["predicted_house"] = model.predict(X_new)

    dominant_house = chat_user["predicted_house"].value_counts().idxmax()

    # ========================================================
    # METRICS
    # ========================================================
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="glass">
            <div class="metric-value">👥 {len(chat_user)}</div>
            <div class="metric-label">USERS</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass">
            <div class="metric-value">💬 {len(chat_df)}</div>
            <div class="metric-label">MESSAGES</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass">
            <div class="metric-value">🏆 {dominant_house}</div>
            <div class="metric-label">DOMINANT HOUSE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ========================================================
    # HOUSE STYLING
    # ========================================================
    def style_house(house):
        h = house.lower()
        return f'<span class="house {h}">🪄 {house}</span>'

    chat_user["pretty_house"] = chat_user["predicted_house"].apply(style_house)

    # ========================================================
    # RESULTS TABLE
    # ========================================================
    st.subheader("🧙 User House Predictions")

    pretty_df = chat_user[["speaker_first_name", "pretty_house"]].rename(
        columns={
            "speaker_first_name": "User",
            "pretty_house": "House"
        }
    )

    st.markdown(
        pretty_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ========================================================
    # HOUSE DISTRIBUTION
    # ========================================================
    st.subheader("📊 House Distribution")
    st.bar_chart(chat_user["predicted_house"].value_counts())
