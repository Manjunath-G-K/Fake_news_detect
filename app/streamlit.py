

import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="FakeGuard Pro - AI News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional dark mode UI
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero p {
        font-size: 1.3rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Info cards */
    .info-card {
        background: #1e293b;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .info-card h3 {
        color: #60a5fa;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Result styling */
    .result-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
        margin: 2rem 0;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
        margin: 2rem 0;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
    }
    
    .confidence {
        font-size: 1.5rem;
        color: white;
        opacity: 0.9;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 2px solid #334155 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('models/tfidf_clf.joblib')
        vectorizer = joblib.load('models/tfidf_vect.joblib')
        return classifier, vectorizer
    except FileNotFoundError:
        return None, None

# Prediction function
def predict_news(text, classifier, vectorizer):
    text_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(text_vectorized)[0]
    confidence = classifier.predict_proba(text_vectorized)[0]
    return prediction, max(confidence) * 100

# Hero Section
st.markdown("""
<div class="hero">
    <h1>🛡️ FakeGuard Pro</h1>
    <p>AI-Powered Fake News Detection System</p>
    <p style="font-size: 1rem; color: #94a3b8;">Analyze news articles instantly with advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# How It Works Section
st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📝 Step 1: Input</h3>
        <p>Paste or type the news article text you want to verify</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🤖 Step 2: Analysis</h3>
        <p>Our AI model analyzes the text using advanced NLP techniques</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>✅ Step 3: Results</h3>
        <p>Get instant classification with confidence score</p>
    </div>
    """, unsafe_allow_html=True)

# Analyzer Section
st.markdown('<h2 class="section-header" id="analyzer">News Analyzer</h2>', unsafe_allow_html=True)

# Load models
classifier, vectorizer = load_models()

if classifier is None or vectorizer is None:
    st.error("""
    ⚠️ **Models not found!** 
    
    Please follow these steps:
    1. Run `python src/data_prep.py` to prepare the data
    2. Run `python src/train.py` to train the model
    3. Restart this Streamlit app
    """)
else:
    st.success("✅ Models loaded successfully!")
    
    # Text input
    news_text = st.text_area(
        "Enter news article text:",
        height=250,
        placeholder="Paste the news article text here...",
        help="Enter the complete text of the news article you want to analyze"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze News", use_container_width=True)
    
    # Prediction
    if analyze_button:
        if news_text.strip():
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_news(news_text, classifier, vectorizer)
                
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-real">
                        <div class="result-title">✅ REAL NEWS</div>
                        <div class="confidence">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fake">
                        <div class="result-title">⚠️ FAKE NEWS</div>
                        <div class="confidence">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p style="font-size: 0.9rem;">FakeGuard Pro - Powered by Machine Learning</p>
    <p style="font-size: 0.8rem;">Using TF-IDF Vectorization & Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

