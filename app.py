import streamlit as st
import time
import os
from src.fact_extraction import AtomicFactExtractor
from src.verification import FactVerifier

st.set_page_config(page_title="XAI Bias & Hallucination Detector", layout="wide", page_icon="🛡️")

# --- Initialize AI Models (Cached so they only load once) ---
@st.cache_resource
def load_models():
    v = FactVerifier(model_name='all-MiniLM-L6-v2')
    # Use dummy for zero-shot testing if API key is missing
    e = AtomicFactExtractor() 
    return v, e

# Attempt to load, but don't block the UI
try:
    verifier, extractor = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure your .env file is set up with an API Key.")
    st.stop()


# Inject custom CSS for premium look
st.markdown("""
<style>
    .reportview-container { background: #fafafa; }
    .stButton>button {
        background-color: #4F46E5; color: white; border-radius: 8px;
        padding: 0.5rem 1rem; font-weight: 600; border: none; transition: all 0.2s;
    }
    .stButton>button:hover { background-color: #4338CA; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Explainable AI (XAI) Hallucination Detector")
st.markdown("Audit your LLM outputs by dissecting them into **atomic facts** and verifying them against a trusted **knowledge base**.")

# 1. Inputs
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📚 Trusted Knowledge Base")
    context = st.text_area(
        "Ground Truth Documents", 
        height=200, 
        value="Albert Einstein was a German-born theoretical physicist. Einstein is best known for developing the theory of relativity. He won the Nobel Prize in Physics in 1921."
    )

with col2:
    st.markdown("### 🤖 LLM Response to Audit")
    llm_out = st.text_area(
        "Generated Content", 
        height=200, 
        value="Albert Einstein was a Swiss physicist who invented the atomic bomb and won the Nobel Prize in 1925."
    )

# 2. Controls
if st.button("🔍 Run Explainable Audit"):
    st.markdown("---")
    st.markdown("### 🔍 Audit Results")
    
    with st.status("Analyzing LLM Response...", expanded=True) as status:
        st.write("Extracting Atomic Facts from the LLM response...")
        
        # --- 1. Fact Extraction (LLM-based or rule-based fallback) ---
        if "OPENAI_API_KEY" not in os.environ or os.environ.get("OPENAI_API_KEY") == "your_openai_api_key_here":
            st.info("No OpenAI API key found. Using rule-based fact extraction from your input.")
        extracted_facts = extractor.extract(llm_out)
        
        time.sleep(1) # Small UX pause
        
        st.write("Calculating Semantic Similarity against Knowledge Base using all-MiniLM-L6-v2...")
        
        # --- 2. Real Semantic Verification ---
        # verify_response returns a dict with 'facts' as a list of results
        verification_data = verifier.verify_response(extracted_facts, context)
        
        status.update(label="Audit Complete!", state="complete", expanded=False)
        
    st.markdown("#### Phase 1: Semantic Verification (Explainability Layer)")
    
    # Render dynamic results
    for res in verification_data['facts']:
        fact_text = res['fact']
        details = res['verification']
        
        score_pct = details['confidence'] * 100
        
        # Grading threshold logic
        if details['is_supported']:
            status_label = "✅ Supported"
            theme_color = "green"
            evidence_str = details['evidence']
        elif score_pct > 30 and score_pct < 50:
            status_label = "⚠️ Contradiction / Low Confidence"
            theme_color = "orange"
            evidence_str = f"Best matching sentence (confidence too low): {details['evidence']}"
        else:
            status_label = "🛑 Hallucination"
            theme_color = "red"
            evidence_str = "No semantic evidence found in the trusted Knowledge Base."
            
        with st.expander(f"{status_label}: {fact_text}"):
            st.markdown(f"**Similarity Score:** `{score_pct:.1f}%`")
            st.markdown(f"**Explanation (Chain-of-Thought):**")
            st.info(evidence_str)

st.sidebar.markdown("### XAI Pipeline Metrics")
st.sidebar.metric(label="Embedding Model", value="all-MiniLM-L6-v2")
st.sidebar.metric(label="Extraction Engine", value="LangChain (Zero-Shot)")
