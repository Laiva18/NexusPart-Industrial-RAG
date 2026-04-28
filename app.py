import streamlit as st
import pandas as pd
import faiss
import numpy as np
import time
import ollama
from sentence_transformers import SentenceTransformer

## --- 1. NEXUSPART PREMIUM UI CONFIGURATION ---
st.set_page_config(page_title="NexusPart | Industrial RAG", layout="wide", page_icon="⚙️")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { 
        background-color: #1E293B; 
    }
    
    /* Premium Gradient Heading */
    h1 {
        background: linear-gradient(90deg, #FFFFFF, #60A5FA, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem !important;
        padding-bottom: 10px;
        letter-spacing: -1px;
    }       

    /* Global Text Visibility */
    h2, h3, p, span, label { 
        color: #FFFFFF !important; 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background-color: #0F172A; 
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { 
        color: #E2E8F0 !important; 
    }

    /* Main Card / Verdict Container Fix */
    .main-card {
        background-color: #1E293B !important;
        padding: 24px !important;
        border-radius: 14px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        color: #FFFFFF !important; /* Forces root text to white */
        display: block;
    }

    /* Target all nested elements in AI response for visibility */
    .main-card * {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
    }

    .main-card b, .main-card strong {
        color: #60A5FA !important;
    }

    .main-card:hover {
        transform: translateY(-5px);
        border-color: #60A5FA !important;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }

    /* Search Input Fix */
    .stTextInput > div > div > input {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #FFFFFF !important;
        opacity: 0.7;
    }

    /* Table Styling */
    .stDataFrame {
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. NEXUS ENGINE (Data & MLOps) ---
@st.cache_resource
def init_nexus_engine():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("parts_vector_db.index")
    df = pd.read_csv("processed_parts_for_rag.csv")
    
    if 'PRODUCT_ID' not in df.columns:
        df['PRODUCT_ID'] = [f"NP-{i+1001}" for i in range(len(df))]
    
    return embed_model, index, df

try:
    model, index, df = init_nexus_engine()
except Exception as e:
    st.error(f"Initialization Failed: {e}")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("NexusPart")
    st.caption("Industrial RAG Intelligence")
    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Select Search Depth (K)", 1, 5, 3)

    st.subheader("🛠️ MLOps Dashboard")
    st.info(f"Database Size: {len(df)} Units")
    st.info("LLM: Llama 3 (Quantized)")
    st.info("Engine: CUDA/GPU Accelerated")
    

# --- 4. MAIN INTERFACE ---
st.title("Industrial Supply Chain Intelligence")

query = st.text_input(
    "🔍 Input Part Specifications (e.g., '2A 250V Ceramic Time-Lag')", 
    placeholder="Type technical requirements here..."
)

if query:
    start_time = time.time()
    
    with st.spinner("Executing Semantic Retrieval & Reasoning..."):
        query_vec = model.encode([query])
        distances, indices = index.search(np.array(query_vec).astype('float32'), k=top_k)
        results = df.iloc[indices[0]].copy()
        
        results['Match Score'] = (1 / (1 + distances[0])) * 100 
        
        context_str = ""
        for _, row in results.iterrows():
            context_str += f"ID: {row['ID']}, Current: {row['Rated Current (A)']}A, Voltage: {row['Rated Voltage (V)']}V, Material: {row['Material']}\n"
        
        nexus_expert_prompt = f"""
### IDENTITY & ROLE
You are 'NexusPart AI', a friendly and expert Senior Industrial Procurement Consultant. 
Your goal is to help a colleague find the safest and most efficient part substitute from our warehouse.

### CORE CONTEXT
- User's Specific Need: "{query}"
- Available Warehouse Inventory: 
{context_str}

### OPERATIONAL GUIDELINES
1. **Be Human, Not a Robot**: Use a warm, professional, and encouraging tone. Talk like a mentor, not a datasheet.
2. **Strict Verification**: Compare the 'Rated Current (A)' and 'Rated Voltage (V)' of the found parts against the user's requirements.
3. **Safety First**: If a part is a "Close Match" but has a safety risk (e.g., higher amperage than needed), explain the risk in simple words (e.g., "it might not protect the circuit in time").
4. **Use Original IDs**: Always refer to the parts using the exact IDs provided in the data.

### RESPONSE STRUCTURE (Keep it concise & simple)
- **Hello & Quick Find**: A warm 1-sentence greeting and what you found.
- **Reason**: Explain *why* these parts are good matches in easy-to-understand English.
- **Safety Check**: Clearly point out any risks or precautions in simple terms.
- **My Recommendation**: A final, polite verdict on which part to pick or if they should keep searching.

### CONSTRAINTS
- Word limit: Maximum 150-200 words for a fast response.
- Language: Plain, simple English (No heavy engineering jargon).
"""
        try:
            # --- AI GENERATION ---
            response = ollama.generate(
                model='llama3', 
                prompt=nexus_expert_prompt,
                options={"num_predict": 180, "temperature": 0.2}
            )
            
            st.markdown("### 🤖 NexusPart Engineering Verdict")
            
            # Formatting and wrapping in a safe div
            res_text = response['response'].replace("\n", "<br>")
            st.markdown(f"""
                <div class='main-card' style='border-left: 5px solid #60A5FA !important;'>
                    {res_text}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- 2. PREMIUM TECHNICAL TABLE ---
            st.markdown("### 📦 Verified Technical Matches")
            
            # 1. ID Fix: Mapping strictly to your actual data ID
            # 2. Description Removed as requested
            display_df = results[['ID', 'Rated Current (A)', 'Rated Voltage (V)', 'Material', 'Match Score']]
            
            # Styling the Dataframe to make it pop against the background
            styled_df = display_df.style.format({
                'Match Score': '{:.1f}%', 
                'Rated Current (A)': '{:.2f} A', 
                'Rated Voltage (V)': '{:.0f} V'
            }).set_table_styles([
                # Header: Bright Blue with Glow effect
                {'selector': 'th', 'props': [
                    ('background-color', '#1E293B'), 
                    ('color', '#60A5FA'), 
                    ('font-weight', 'bold'), 
                    ('text-align', 'center'),
                    ('border-bottom', '2px solid #60A5FA'),
                    ('padding', '15px')
                ]},
                # Rows: High contrast and readable
                {'selector': 'td', 'props': [
                    ('background-color', '#0F172A'), 
                    ('color', '#FFFFFF'), 
                    ('border-bottom', '1px solid #334155'),
                    ('text-align', 'center'),
                    ('padding', '12px')
                ]},
                # Row Hover: Subtle highlight
                {'selector': 'tr:hover', 'props': [('background-color', '#1E293B')]}
            ])

            # Displaying with full width
            st.dataframe(
                styled_df,
                hide_index=True, 
                use_container_width=True
            )

            st.markdown("""
                <p style='color: #94A3B8; font-size: 0.85rem; font-style: italic;'>
                    * Safety Note: Verify all substitutes against live circuit conditions before procurement.
                </p>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Nexus Engine Error: {str(e)}")
            st.info("Check if Ollama is running and the database is correctly linked.")

else:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
            <div class='main-card'>
                <div style='font-size: 24px; margin-bottom: 10px;'>🔍</div>
                <b style='color: #60A5FA; font-size: 20px;'>Semantic Search</b>
                <p style='color: #E2E8F0; font-size: 14px; margin-top: 8px;'>
                    Engineer Your Search.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div class='main-card'>
                <div style='font-size: 24px; margin-bottom: 10px;'>🛡️</div>
                <b style='color: #60A5FA; font-size: 20px;'>Fort Knox Privacy</b>
                <p style='color: #E2E8F0; font-size: 14px; margin-top: 8px;'>
                    Your Data Your Rules.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
            <div class='main-card'>
                <div style='font-size: 24px; margin-bottom: 10px;'>🧠</div>
                <b style='color: #60A5FA; font-size: 20px;'>Expert Safety Logic</b>
                <p style='color: #E2E8F0; font-size: 14px; margin-top: 8px;'>
                    Precision without the guesswork.
                </p>
            </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("NexusPart | Powered by Llama 3 & FAISS ")