
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import torch
import os

# --- 1. Page Configuration & CSS (The Polish) ---
st.set_page_config(
    page_title="Chatgaiya AI | Precision Translation",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Premium Startup UI Styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-container {
        padding-top: 2rem;
    }
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Result Box Styling */
    .result-box-bangla {
        background-color: #F0FDF4;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .result-box-english {
        background-color: #EFF6FF;
        border-left: 6px solid #3B82F6;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    .result-label {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        color: #6B7280;
    }
    .result-text {
        font-size: 1.4rem;
        color: #111827;
        line-height: 1.4;
    }

    /* Custom Button Overrides */
    div.stButton > button[kind="primary"] {
        background-color: #1E3A8A !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #152C69 !important;
    }

    /* QNFL Badge Styling */
    .qnfl-badge {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 8px 12px;
        border-radius: 6px;
        text-align: center;
        font-size: 0.85rem;
        color: #4b5563;
        font-weight: 500;
        margin-top: 20px;
    }
    .qnfl-badge span {
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

device = "cpu"

# --- 2. Dual Model Loading (Fixed Path Logic) ---
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Safety check: Did the extraction script run?
        if not os.path.exists("/content/model_path.txt"):
            return None, None, None, None, "Missing '/content/model_path.txt'. Please run the 'Extraction and Reunite' Cell 1 again."

        # Automatically gets the path from the reuniting script
        with open("/content/model_path.txt", "r") as f:
             path_mt5 = "your-username/Chatgaiya-mT5"

        tokenizer_mt5 = AutoTokenizer.from_pretrained(path_mt5)
        model_mt5 = AutoModelForSeq2SeqLM.from_pretrained(path_mt5).to(device)

        # Step 2 Model: Pre-trained Helsinki
        path_en = "Helsinki-NLP/opus-mt-bn-en"
        tokenizer_en = MarianTokenizer.from_pretrained(path_en)
        model_en = MarianMTModel.from_pretrained(path_en).to(device)

        return tokenizer_mt5, model_mt5, tokenizer_en, model_en, None
    except Exception as e:
        return None, None, None, None, str(e)

with st.spinner("🚀 Booting Neural Translation Pipeline..."):
    tokenizer_mt5, model_mt5, tokenizer_en, model_en, error_msg = load_models()

if error_msg:
    st.error(f"⚠️ Model Loading Failed! Exact Error: {error_msg}")

# --- 3. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🌊</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Chatgaiya AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### **Market Impact**")
    st.write("Unlocking digital access for **13M+ speakers** in the Chittagong division.")

    st.markdown("### **Enterprise Ready**")
    st.markdown("✅ Fast Inference\n\n✅ Natively Fine-tuned Core\n\n✅ Optional Global Export")

    st.markdown("---")
    st.markdown("#### **Example Queries**")
    st.code("অ্যাঁই ভাত ন হাইয়্যুম")
    st.code("তোঁয়ার নাম কি?")
    st.code("অ্যাঁই তোঁয়ারে ভালোবাসি")

    st.markdown("<br>", unsafe_allow_html=True)

    # Built by QNFL Badge
    st.markdown("""
        <div class="qnfl-badge">
            Built by <span>QNFL</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# --- 4. Main Interface ---
st.markdown('<p class="app-title">Chatgaiya AI</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Seamless Chittagonian to Standard Bangla Translation Engine.</p>', unsafe_allow_html=True)

input_text = st.text_area("Chittagonian Input:", placeholder="এখানে চাটগাঁইয়া ভাষা লিখুন...", height=120)

# The English Checkbox
st.markdown("#### Options")
include_english = st.checkbox("🌍 Enable Global Export (Translate to English)")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2.5, 1, 1])
with col1:
    btn_text = "🌐 Generate Translation" if not include_english else "🌐 Run 2-Stage Pipeline"
    translate_btn = st.button(btn_text, use_container_width=True, type="primary")
with col3:
    if st.button("🗑️ Reset", use_container_width=True):
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun() # Fallback for older Streamlit versions

# --- 5. Translation Logic ---
if translate_btn:
    if not input_text.strip():
        st.error("Action Required: Please provide input text.")
    elif model_mt5 is None or (include_english and model_en is None):
        st.error("System Error: Neural Engine Offline. Please check model paths.")
    else:
        # ALWAYS DO STEP 1: Chittagonian to Bangla
        with st.spinner("Executing Stage 1: Dialect Normalization..."):
            inputs_mt5 = tokenizer_mt5(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
            with torch.no_grad():
                outputs_mt5 = model_mt5.generate(
                    inputs_mt5.input_ids,
                    max_length=128, num_beams=5, repetition_penalty=2.5, early_stopping=True
                )
            bangla_translation = tokenizer_mt5.decode(outputs_mt5[0], skip_special_tokens=True)

        # Display Step 1 Output immediately
        st.markdown(f"""
            <div class="result-box-bangla">
                <div class="result-label">Standard Bangla Output</div>
                <div class="result-text">{bangla_translation}</div>
            </div>
        """, unsafe_allow_html=True)

        # CONDITIONAL STEP 2: Bangla to English
        if include_english:
            with st.spinner("Executing Stage 2: Global Synthesis..."):
                inputs_en = tokenizer_en(bangla_translation, return_tensors="pt", max_length=128, truncation=True).to(device)
                with torch.no_grad():
                    outputs_en = model_en.generate(
                        inputs_en.input_ids,
                        max_length=128, num_beams=4, early_stopping=True
                    )
                english_translation = tokenizer_en.decode(outputs_en[0], skip_special_tokens=True)

            # Display Step 2 Output
            st.markdown(f"""
                <div class="result-box-english">
                    <div class="result-label">English Output</div>
                    <div class="result-text">{english_translation}</div>
                </div>
            """, unsafe_allow_html=True)

        st.balloons()
