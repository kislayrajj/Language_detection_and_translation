# app.py (COMPLETE & WORKING)

import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version inconsistency warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

try:
    from googletrans import Translator
    TRANSLATE_AVAILABLE = True
except:
    TRANSLATE_AVAILABLE = False

st.set_page_config(page_title="🌐 Language Detection & Translation", layout="wide")

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load("language_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Run: python language_detection.py")
        st.stop()

model, vectorizer = load_models()

# Load translator
@st.cache_resource
def load_translator():
    if TRANSLATE_AVAILABLE:
        return Translator()
    return None

translator = load_translator()

# Language codes
lang_codes = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Italian': 'it',
    'Hindi': 'hi'
}

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.title("🌐 Language Detection & Translation")
st.markdown("Powered by TF-IDF + Naive Bayes")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Supported Languages:**
    - 🇬🇧 English
    - 🇫🇷 French
    - 🇪🇸 Spanish
    - 🇩🇪 German
    - 🇮🇹 Italian
    - 🇮🇳 Hindi
    """)
    st.header("📊 Model Info")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Features:** TF-IDF (2-4 char n-grams)")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detect", "📊 Batch", "🌍 Translate", "📜 History"])

# TAB 1: Single Detection
with tab1:
    st.subheader("Detect Language")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area("Enter text:", height=150, placeholder="Type or paste text...")
    with col2:
        st.write("")
        st.write("")
        detect_btn = st.button("🔍 Detect", use_container_width=True, type="primary")
    
    if detect_btn:
        if text_input.strip():
            features = vectorizer.transform([text_input])
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs) * 100
            
            st.success(f"✅ Detected: **{pred}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            st.subheader("All Languages")
            lang_dict = {lang: prob*100 for lang, prob in zip(model.classes_, probs)}
            lang_sorted = dict(sorted(lang_dict.items(), key=lambda x: x[1], reverse=True))
            
            col1, col2 = st.columns(2)
            with col1:
                for lang, prob in lang_sorted.items():
                    st.write(f"{lang}: **{prob:.2f}%**")
            with col2:
                st.bar_chart(lang_sorted)
            
            st.session_state.history.append({
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Type': 'Detection',
                'Text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
                'Result': f"{pred} ({confidence:.1f}%)"
            })
        else:
            st.warning("⚠️ Enter text first!")

# TAB 2: Batch Detection
with tab2:
    st.subheader("Batch Detection")
    st.write("Detect multiple texts at once (one per line)")
    
    batch_text = st.text_area(
        "Enter texts:",
        height=200,
        placeholder="Hello\nBonjour\nHola",
        key="batch_input"
    )
    
    if st.button("🚀 Detect All", type="primary", use_container_width=True):
        if batch_text.strip():
            texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
            
            if texts:
                features = vectorizer.transform(texts)
                predictions = model.predict(features)
                probs = model.predict_proba(features)
                
                results = []
                for i, (txt, pred, prob) in enumerate(zip(texts, predictions, probs)):
                    conf = np.max(prob) * 100
                    results.append({
                        'No': i + 1,
                        'Text': txt[:40] + "..." if len(txt) > 40 else txt,
                        'Language': pred,
                        'Confidence': f"{conf:.1f}%"
                    })
                    st.session_state.history.append({
                        'Time': datetime.now().strftime("%H:%M:%S"),
                        'Type': 'Batch Detection',
                        'Text': txt[:50] + "..." if len(txt) > 50 else txt,
                        'Result': f"{pred} ({conf:.1f}%)"
                    })
                
                st.dataframe(results, use_container_width=True, hide_index=True)
                
                # CSV Download
                csv_str = "No,Text,Language,Confidence\n"
                for r in results:
                    csv_str += f"{r['No']},{r['Text']},{r['Language']},{r['Confidence']}\n"
                
                st.download_button(
                    "📥 Download CSV",
                    csv_str,
                    "batch_results.csv",
                    "text/csv",
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Enter at least one text!")

# TAB 3: Translate
with tab3:
    if not TRANSLATE_AVAILABLE:
        st.error("❌ Translation not available. Install: pip install googletrans==4.0.0rc1")
    else:
        st.subheader("Translate Text")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Source**")
            trans_input = st.text_area(
                "Text to translate:",
                height=120,
                placeholder="Enter text...",
                key="trans_input"
            )
            
            auto_detect = st.checkbox("🔍 Auto-detect", value=True)
            
            if not auto_detect:
                src_lang = st.selectbox("Source language:", list(lang_codes.keys()), key="src_lang")
            
            tgt_lang = st.selectbox("Target language:", list(lang_codes.keys()), key="tgt_lang")
        
        with col2:
            st.write("**Translation**")
            
            if st.button("🌐 Translate", type="primary", use_container_width=True):
                if trans_input.strip():
                    try:
                        if auto_detect:
                            feat = vectorizer.transform([trans_input])
                            detected = model.predict(feat)[0]
                            src_code = lang_codes[detected]
                            st.info(f"📍 Detected: {detected}")
                        else:
                            src_code = lang_codes[src_lang]
                            detected = src_lang
                        
                        tgt_code = lang_codes[tgt_lang]
                        result = translator.translate(trans_input, src=src_code, dest=tgt_code)
                        trans_text = result.text
                        
                        st.text_area("Result:", value=trans_text, height=120, disabled=True)
                        
                        st.session_state.history.append({
                            'Time': datetime.now().strftime("%H:%M:%S"),
                            'Type': 'Translation',
                            'Text': trans_input[:50] + "..." if len(trans_input) > 50 else trans_input,
                            'Result': f"{detected if auto_detect else src_lang} → {tgt_lang}"
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Enter text!")

# TAB 4: History
with tab4:
    st.subheader("History")
    
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No history yet!")

st.markdown("---")
st.markdown("Built with ❤️ | NLP Project")