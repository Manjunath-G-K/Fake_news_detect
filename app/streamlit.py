

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


# import streamlit as st
# import joblib
# import time
# import os

# # Page configuration
# st.set_page_config(
#     page_title="TruthGuard AI | Fake News Detection",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS for professional dark mode UI with all sections
# st.markdown("""
# <style>
#     /* Global Dark Mode Styling */
#     .stApp {
#         background-color: #0f1419;
#         color: #e8eaed;
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* Sticky Navigation Bar */
#     .nav-bar {
#         position: sticky;
#         top: 0;
#         z-index: 1000;
#         background: rgba(21, 32, 43, 0.95);
#         backdrop-filter: blur(10px);
#         padding: 1rem 2rem;
#         box-shadow: 0 2px 20px rgba(0, 0, 0, 0.5);
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         margin-bottom: 0;
#     }
    
#     .logo {
#         font-size: 1.8rem;
#         font-weight: 800;
#         background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#     }
    
#     .nav-links {
#         display: flex;
#         gap: 2rem;
#     }
    
#     .nav-links a {
#         color: #94a3b8;
#         text-decoration: none;
#         font-weight: 600;
#         transition: color 0.3s ease;
#     }
    
#     .nav-links a:hover {
#         color: #3b82f6;
#     }
    
#     /* Hero Section */
#     .hero-section {
#         padding: 8rem 2rem 6rem 2rem;
#         text-align: center;
#         background: linear-gradient(135deg, #1a1f2e 0%, #15202b 100%);
#         border-radius: 20px;
#         margin: 2rem 0 4rem 0;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .hero-section::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         bottom: 0;
#         background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
#     }
    
#     .hero-title {
#         font-size: 4.5rem;
#         font-weight: 900;
#         background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 1.5rem;
#         line-height: 1.2;
#     }
    
#     .hero-subtitle {
#         font-size: 1.5rem;
#         color: #94a3b8;
#         margin-bottom: 3rem;
#         line-height: 1.6;
#     }
    
#     .cta-button {
#         background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
#         color: white;
#         padding: 1rem 3rem;
#         border-radius: 50px;
#         font-size: 1.2rem;
#         font-weight: 700;
#         border: none;
#         cursor: pointer;
#         box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);
#         transition: all 0.3s ease;
#         text-decoration: none;
#         display: inline-block;
#     }
    
#     .cta-button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 15px 40px rgba(59, 130, 246, 0.6);
#     }
    
#     .hero-features {
#         margin-top: 3rem;
#         color: #64748b;
#         font-size: 1rem;
#     }
    
#     /* Stats Section */
#     .stats-section {
#         background: #15202b;
#         padding: 4rem 2rem;
#         border-radius: 20px;
#         margin: 4rem 0;
#     }
    
#     .stat-card {
#         text-align: center;
#         padding: 2rem;
#     }
    
#     .stat-number {
#         font-size: 3rem;
#         font-weight: 900;
#         background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
    
#     .stat-label {
#         color: #94a3b8;
#         font-size: 1.1rem;
#     }
    
#     /* Section Header */
#     .section-header {
#         font-size: 2.8rem;
#         font-weight: 800;
#         color: #f1f5f9;
#         margin: 4rem 0 2rem 0;
#         text-align: center;
#     }
    
#     .section-subtitle {
#         text-align: center;
#         color: #94a3b8;
#         font-size: 1.2rem;
#         margin-bottom: 3rem;
#     }
    
#     /* Feature Cards */
#     .feature-card {
#         background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
#         padding: 2.5rem;
#         border-radius: 15px;
#         margin: 1rem 0;
#         border: 1px solid #334155;
#         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
#         transition: all 0.3s ease;
#         height: 100%;
#     }
    
#     .feature-card:hover {
#         transform: translateY(-5px);
#         border-color: #3b82f6;
#         box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
#     }
    
#     .feature-icon {
#         font-size: 3rem;
#         margin-bottom: 1rem;
#     }
    
#     .feature-title {
#         color: #60a5fa;
#         font-size: 1.5rem;
#         font-weight: 700;
#         margin-bottom: 1rem;
#     }
    
#     .feature-description {
#         color: #cbd5e1;
#         font-size: 1rem;
#         line-height: 1.6;
#     }
    
#     /* How It Works Cards */
#     .step-card {
#         background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
#         padding: 3rem 2rem;
#         border-radius: 15px;
#         text-align: center;
#         border: 1px solid #334155;
#         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
#         transition: all 0.3s ease;
#         height: 100%;
#     }
    
#     .step-card:hover {
#         transform: translateY(-5px);
#         border-color: #3b82f6;
#     }
    
#     .step-number {
#         font-size: 4rem;
#         font-weight: 900;
#         background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 1rem;
#     }
    
#     .step-title {
#         font-size: 1.5rem;
#         font-weight: 700;
#         color: #f1f5f9;
#         margin-bottom: 1rem;
#     }
    
#     .step-description {
#         color: #94a3b8;
#         font-size: 1rem;
#         line-height: 1.6;
#     }
    
#     /* Technology Section */
#     .tech-card {
#         background: #15202b;
#         padding: 2rem;
#         border-radius: 15px;
#         border-left: 4px solid #3b82f6;
#         margin: 1.5rem 0;
#     }
    
#     .tech-title {
#         color: #60a5fa;
#         font-size: 1.3rem;
#         font-weight: 700;
#         margin-bottom: 0.5rem;
#     }
    
#     .tech-description {
#         color: #cbd5e1;
#         line-height: 1.6;
#     }
    
#     /* Analyzer Section */
#     .analyzer-container {
#         background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
#         padding: 3rem;
#         border-radius: 20px;
#         border: 1px solid #334155;
#         margin: 2rem 0;
#     }
    
#     /* Result Cards */
#     .result-real {
#         background: linear-gradient(135deg, #059669 0%, #10b981 100%);
#         padding: 3rem;
#         border-radius: 20px;
#         text-align: center;
#         box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4);
#         margin: 2rem 0;
#     }
    
#     .result-fake {
#         background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
#         padding: 3rem;
#         border-radius: 20px;
#         text-align: center;
#         box-shadow: 0 10px 40px rgba(239, 68, 68, 0.4);
#         margin: 2rem 0;
#     }
    
#     .result-title {
#         font-size: 3rem;
#         font-weight: 900;
#         color: white;
#         margin-bottom: 1rem;
#     }
    
#     .confidence {
#         font-size: 1.8rem;
#         color: white;
#         opacity: 0.95;
#         font-weight: 600;
#     }
    
#     /* FAQ Section */
#     .faq-item {
#         background: #15202b;
#         padding: 2rem;
#         border-radius: 15px;
#         margin: 1rem 0;
#         border: 1px solid #334155;
#     }
    
#     .faq-question {
#         color: #60a5fa;
#         font-size: 1.2rem;
#         font-weight: 700;
#         margin-bottom: 1rem;
#     }
    
#     .faq-answer {
#         color: #cbd5e1;
#         line-height: 1.6;
#     }
    
#     /* Footer */
#     .footer {
#         background: #0a0f14;
#         padding: 4rem 2rem 2rem 2rem;
#         margin-top: 6rem;
#         border-top: 1px solid #1e293b;
#     }
    
#     .footer-content {
#         max-width: 1200px;
#         margin: 0 auto;
#     }
    
#     .footer-section {
#         margin-bottom: 2rem;
#     }
    
#     .footer-title {
#         color: #60a5fa;
#         font-size: 1.3rem;
#         font-weight: 700;
#         margin-bottom: 1rem;
#     }
    
#     .footer-text {
#         color: #94a3b8;
#         line-height: 1.8;
#     }
    
#     .footer-bottom {
#         text-align: center;
#         padding-top: 2rem;
#         margin-top: 2rem;
#         border-top: 1px solid #1e293b;
#         color: #64748b;
#     }
    
#     /* Streamlit Widget Overrides */
#     .stTextArea textarea {
#         background-color: #15202b !important;
#         color: #e8eaed !important;
#         border: 2px solid #334155 !important;
#         border-radius: 10px !important;
#         font-size: 1rem !important;
#     }
    
#     .stButton button {
#         background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
#         color: white !important;
#         font-size: 1.1rem !important;
#         font-weight: 700 !important;
#         padding: 0.8rem 2.5rem !important;
#         border: none !important;
#         border-radius: 50px !important;
#         box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
#         transition: all 0.3s ease !important;
#     }
    
#     .stButton button:hover {
#         transform: translateY(-3px) !important;
#         box-shadow: 0 10px 30px rgba(59, 130, 246, 0.6) !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load models
# @st.cache_resource
# def load_models():
#     try:
#         classifier = joblib.load('models/tfidf_clf.joblib')
#         vectorizer = joblib.load('models/tfidf_vect.joblib')
#         return classifier, vectorizer
#     except FileNotFoundError:
#         return None, None

# # Prediction function
# def predict_news(text, classifier, vectorizer):
#     text_vectorized = vectorizer.transform([text])
#     prediction = classifier.predict(text_vectorized)[0]
#     confidence = classifier.predict_proba(text_vectorized)[0]
#     return prediction, max(confidence) * 100

# # ============================================
# # NAVIGATION BAR
# # ============================================
# st.markdown("""
# <div class="nav-bar">
#     <div class="logo">🛡️ TruthGuard AI</div>
#     <div class="nav-links">
#         <a href="#home">Home</a>
#         <a href="#features">Features</a>
#         <a href="#howitworks">How It Works</a>
#         <a href="#analyzer">Analyzer</a>
#         <a href="#faq">FAQ</a>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # ============================================
# # HERO SECTION
# # ============================================
# st.markdown('<div id="home"></div>', unsafe_allow_html=True)

# st.markdown("""
# <div class="hero-section">
#     <h1 class="hero-title">Stop Fake News with AI-Powered Detection</h1>
#     <p class="hero-subtitle">Verify news authenticity in seconds. Our advanced AI analyzes articles<br>and statements with precision using Machine Learning.</p>
#     <a href="#analyzer"><button class="cta-button">Start Analyzing Now →</button></a>
#     <div class="hero-features">
#         ✓ Instant Analysis  |  ✓ Machine Learning Core  |  ✓ High Accuracy  |  ✓ Easy to Use
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # ============================================
# # STATS SECTION
# # ============================================
# st.markdown("""
# <div class="stats-section">
# """, unsafe_allow_html=True)

# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.markdown("""
#     <div class="stat-card">
#         <div class="stat-number">99%</div>
#         <div class="stat-label">Accuracy</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="stat-card">
#         <div class="stat-number">&lt;2s</div>
#         <div class="stat-label">Analysis Time</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="stat-card">
#         <div class="stat-number">5000+</div>
#         <div class="stat-label">Features Analyzed</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col4:
#     st.markdown("""
#     <div class="stat-card">
#         <div class="stat-number">24/7</div>
#         <div class="stat-label">Availability</div>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("</div>", unsafe_allow_html=True)

# # ============================================
# # FEATURES SECTION
# # ============================================
# st.markdown('<div id="features"></div>', unsafe_allow_html=True)
# st.markdown('<h2 class="section-header">Powerful Features</h2>', unsafe_allow_html=True)
# st.markdown('<p class="section-subtitle">Everything you need to detect fake news effectively</p>', unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">⚡</div>
#         <div class="feature-title">Lightning Fast</div>
#         <div class="feature-description">Get instant results in under 2 seconds. Our optimized ML pipeline ensures rapid analysis without compromising accuracy.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">🎯</div>
#         <div class="feature-title">High Accuracy</div>
#         <div class="feature-description">Trained on thousands of articles, our model achieves exceptional accuracy in distinguishing real from fake news.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">🔒</div>
#         <div class="feature-title">Secure & Private</div>
#         <div class="feature-description">Your data is processed securely. We don't store or share any articles you analyze.</div>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# col4, col5, col6 = st.columns(3)

# with col4:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">📊</div>
#         <div class="feature-title">Confidence Scores</div>
#         <div class="feature-description">Not just a verdict - get detailed confidence percentages to understand how certain the model is.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col5:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">🤖</div>
#         <div class="feature-title">Advanced NLP</div>
#         <div class="feature-description">Powered by TF-IDF vectorization and sophisticated linguistic pattern recognition.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col6:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">💡</div>
#         <div class="feature-title">Easy to Use</div>
#         <div class="feature-description">Simple interface - just paste your text and click analyze. No technical knowledge required.</div>
#     </div>
#     """, unsafe_allow_html=True)

# # ============================================
# # HOW IT WORKS SECTION
# # ============================================
# st.markdown('<div id="howitworks"></div>', unsafe_allow_html=True)
# st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)
# st.markdown('<p class="section-subtitle">Simple three-step process to verify any news article</p>', unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-number">1</div>
#         <div class="step-title">Paste Content</div>
#         <div class="step-description">Copy and paste any news article, social media post, or statement into the text analyzer below.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-number">2</div>
#         <div class="step-title">AI Analysis</div>
#         <div class="step-description">Our advanced AI examines the content using NLP techniques, checking credibility patterns and linguistic markers.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-number">3</div>
#         <div class="step-title">Get Results</div>
#         <div class="step-description">Receive instant verdict (Real/Fake) with a confidence percentage showing the model's certainty level.</div>
#     </div>
#     """, unsafe_allow_html=True)

# # ============================================
# # TECHNOLOGY SECTION
# # ============================================
# st.markdown('<h2 class="section-header">Technology Behind TruthGuard</h2>', unsafe_allow_html=True)
# st.markdown('<p class="section-subtitle">Built with cutting-edge Machine Learning technologies</p>', unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("""
#     <div class="tech-card">
#         <div class="tech-title">🧮 TF-IDF Vectorization</div>
#         <div class="tech-description">Transforms text into numerical features by analyzing word importance and frequency, enabling the model to understand content patterns.</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="tech-card">
#         <div class="tech-title">🎓 Logistic Regression</div>
#         <div class="tech-description">A powerful classification algorithm that learns patterns from thousands of real and fake news articles to make accurate predictions.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="tech-card">
#         <div class="tech-title">📚 Trained on Thousands of Articles</div>
#         <div class="tech-description">Our model has been trained on diverse datasets covering multiple topics, ensuring robust performance across different news categories.</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="tech-card">
#         <div class="tech-title">⚙️ Optimized Pipeline</div>
#         <div class="tech-description">Efficient data preprocessing and model optimization ensure fast analysis without sacrificing accuracy.</div>
#     </div>
#     """, unsafe_allow_html=True)

# # ============================================
# # ANALYZER SECTION
# # ============================================
# st.markdown('<div id="analyzer"></div>', unsafe_allow_html=True)
# st.markdown('<h2 class="section-header">News Analyzer</h2>', unsafe_allow_html=True)
# st.markdown('<p class="section-subtitle">Paste any news article and get instant verification</p>', unsafe_allow_html=True)

# # Load models
# classifier, vectorizer = load_models()

# if classifier is None or vectorizer is None:
#     st.error("""
#     ⚠️ **Models not found!** 
    
#     Please follow these steps:
#     1. Run `python src/data_prep.py` to prepare the data
#     2. Run `python src/train.py` to train the model
#     3. Restart this Streamlit app
#     """)
# else:
#     st.success("✅ Models loaded successfully!")
    
#     st.markdown('<div class="analyzer-container">', unsafe_allow_html=True)
    
#     # Text input
#     news_text = st.text_area(
#         "Enter news article text:",
#         height=300,
#         placeholder="Paste the complete news article text here... (minimum 20 characters for accurate analysis)",
#         help="For best results, paste the complete article including headline and body text",
#         key="news_input"
#     )
    
#     # Analyze button
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         analyze_button = st.button("🔍 Analyze Article", use_container_width=True, type="primary")
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Prediction
#     if analyze_button:
#         if news_text.strip() and len(news_text) > 20:
#             with st.spinner("🤖 AI is analyzing the article..."):
#                 time.sleep(1.5)  # Simulate processing
#                 prediction, confidence = predict_news(news_text, classifier, vectorizer)
                
#                 st.markdown('<h3 style="color: #f1f5f9; text-align: center; margin-top: 3rem;">Analysis Results</h3>', unsafe_allow_html=True)
                
#                 if prediction == 0:
#                     st.markdown(f"""
#                     <div class="result-real">
#                         <div class="result-title">✅ REAL NEWS</div>
#                         <div class="confidence">Confidence: {confidence:.2f}%</div>
#                         <p style="color: rgba(255,255,255,0.8); margin-top: 1.5rem; font-size: 1.1rem;">
#                             This article appears to be genuine based on our AI analysis.
#                         </p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="result-fake">
#                         <div class="result-title">⚠️ FAKE NEWS</div>
#                         <div class="confidence">Confidence: {confidence:.2f}%</div>
#                         <p style="color: rgba(255,255,255,0.8); margin-top: 1.5rem; font-size: 1.1rem;">
#                             This article shows characteristics of misinformation. Verify with trusted sources.
#                         </p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 # Additional info
#                 st.info("💡 **Tip:** Always cross-reference important news with multiple trusted sources for complete verification.")
                
#         elif len(news_text.strip()) <= 20:
#             st.warning("⚠️ Please enter at least 20 characters for accurate analysis.")
#         else:
#             st.warning("⚠️ Please enter some text to analyze.")

# # ============================================
# # FAQ SECTION
# # ============================================
# st.markdown('<div id="faq"></div>', unsafe_allow_html=True)
# st.markdown('<h2 class="section-header">Frequently Asked Questions</h2>', unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("""
#     <div class="faq-item">
#         <div class="faq-question">How accurate is the detector?</div>
#         <div class="faq-answer">Our model achieves high accuracy rates on test datasets. However, always use this as one tool among many for verifying news authenticity.</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="faq-item">
#         <div class="faq-question">What types of content can I analyze?</div>
#         <div class="faq-answer">You can analyze news articles, social media posts, press releases, and any text-based content. For best results, use complete articles.</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="faq-item">
#         <div class="faq-question">Is my data stored or shared?</div>
#         <div class="faq-answer">No. We process your text in real-time and don't store or share any content you analyze. Your privacy is our priority.</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="faq-item">
#         <div class="faq-question">How does the AI work?</div>
#         <div class="faq-answer">The system uses TF-IDF vectorization to convert text into numerical features, then applies a Logistic Regression classifier trained on thousands of articles.</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="faq-item">
#         <div class="faq-question">Can I use this for fact-checking?</div>
#         <div class="faq-answer">Yes, but use it as a supplementary tool. Always verify critical information through multiple trusted sources and official channels.</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="faq-item">
#         <div class="faq-question">What languages are supported?</div>
#         <div class="faq-answer">Currently, the model is optimized for English text. Support for additional languages may be added in future versions.</div>
#     </div>
#     """, unsafe_allow_html=True)

# # ============================================
# # FOOTER
# # ============================================
# st.markdown("""
# <div class="footer">
#     <div class="footer-content">
# """, unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     <div class="footer-section">
#         <div class="footer-title">🛡️ TruthGuard AI</div>
#         <div class="footer-text">
#             An advanced AI-powered system for detecting fake news and misinformation. 
#             Built with Machine Learning and designed for accuracy.
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="footer-section">
#         <div class="footer-title">Technology</div>
#         <div class="footer-text">
#             • TF-IDF Vectorization<br>
#             • Logistic Regression<br>
#             • Natural Language Processing<br>
#             • Python & scikit-learn<br>
#             • Streamlit Framework
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="footer-section">
#         <div class="footer-title">Quick Links</div>
#         <div class="footer-text">
#             • <a href="#home" style="color: #3b82f6; text-decoration: none;">Home</a><br>
#             • <a href="#features" style="color: #3b82f6; text-decoration: none;">Features</a><br>
#             • <a href="#analyzer" style="color: #3b82f6; text-decoration: none;">Try Analyzer</a><br>
#             • <a href="#faq" style="color: #3b82f6; text-decoration: none;">FAQ</a>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("""
#         <div class="footer-bottom">
#             <p>© 2024 TruthGuard AI - Fake News Detection System</p>
#             <p style="margin-top: 0.5rem; font-size: 0.9rem;">
#                 MLOps Demonstration Project | Built with Python, scikit-learn & Streamlit
#             </p>
#         </div>
#     </div>
# </div>
# """, unsafe_allow_html=True)