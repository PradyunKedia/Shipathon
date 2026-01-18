import streamlit as st
import pickle
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🎵 Genre Predictor",
    page_icon="🎧",
    layout="centered"
)

# ============================================================
# FUN + DYNAMIC CSS
# ============================================================
st.markdown("""
<style>

/* ===== ANIMATED BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        -45deg,
        #ff4ecd,
        #6a5cff,
        #00e5ff,
        #7f00ff
    );
    background-size: 400% 400%;
    animation: bgMove 18s ease infinite;
    color: white;
}

@keyframes bgMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== PAGE ENTER ===== */
.block-container {
    animation: slideUp 1s ease forwards;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(25px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== TITLE ===== */
h1 {
    text-align: center;
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #fff, #ffe259, #ffa751);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 2.5s infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 10px rgba(255,255,255,0.5); }
    to   { text-shadow: 0 0 35px rgba(255,255,255,0.9); }
}

/* ===== GLASS CARD ===== */
.glass {
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 1.6rem;
    box-shadow: 0 25px 60px rgba(0,0,0,0.4);
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}

.glass:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 40px 90px rgba(0,0,0,0.7);
}

/* ===== TEXT AREA ===== */
textarea {
    border-radius: 18px !important;
    font-size: 1.05rem !important;
}

/* ===== BUTTON ===== */
button[kind="primary"] {
    background: linear-gradient(90deg, #ffe259, #ffa751) !important;
    color: #1a1a1a !important;
    border-radius: 999px !important;
    font-weight: 900 !important;
    padding: 0.65rem 1.8rem !important;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,226,89,0.7); }
    70% { box-shadow: 0 0 0 26px rgba(255,226,89,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,226,89,0); }
}

/* ===== RESULT FLOAT ===== */
.result {
    font-size: 2.4rem;
    font-weight: 900;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-12px); }
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL SAFELY
# ============================================================
MODEL_PATH = Path(__file__).parent / "genre_classifier.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model, tfidf = load_model()

# ============================================================
# UI
# ============================================================
st.title("🎵 Feel the Lyrics. Name the Genre.")

if model is None:
    st.error("❌ Model not found. Run `python ML.py` first.")
    st.stop()

st.markdown("""
<div class="glass" style="text-align:center;">
🎤 <b>Drop the lyrics.</b><br>
🎧 <span style="opacity:0.9;">We’ll decode the vibe and call the genre.</span>
</div>
""", unsafe_allow_html=True)

lyrics = st.text_area(
    "🎶 Lyrics go here",
    height=240,
    placeholder=(
        "The bass hits low, the crowd goes wild,\n"
        "Lost in rhythm, heart untamed...\n\n"
        "Paste any lyrics — chorus, verse, or chaos."
    )
)

st.write("")

# ============================================================
# PREDICTION
# ============================================================
if st.button("🔮 Reveal the Genre"):
    if not lyrics.strip():
        st.info("🎶 Give me some lyrics first.")
    else:
        try:
            vec = tfidf.transform([lyrics])
            prediction = model.predict(vec)[0]

            st.markdown(f"""
            <div class="glass" style="text-align:center; margin-top:1.8rem;">
                <p style="letter-spacing:2px; opacity:0.8;">THE VIBE FEELS LIKE</p>
                <div class="result">🎧 {prediction.upper()}</div>
                <p style="opacity:0.85;">Turn the volume up.</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception:
            st.error("⚠️ Could not classify these lyrics. Try a longer sample.")
