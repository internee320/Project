import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# --- CONFIGURATION ---
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- 1. PROFESSIONAL CSS & ANIMATIONS ---
st.set_page_config(page_title="Smart Email Search", page_icon="📧", layout="wide")

st.markdown("""
<style>
    /* --- FONTS --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- MAIN CONTAINER --- */
    .stApp {
        animation: fadeIn 0.6s ease-out;
    }

    /* --- CARDS --- */
    .email-card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .email-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        border-color: #cbd5e1;
    }

    /* --- SEARCH BOX --- */
    .search-container {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: background 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
    }

    /* --- HEADINGS --- */
    h1, h2, h3 {
        font-weight: 700;
        color: #0f172a;
    }

    /* --- TEXT AREA FIX (DARK BLACK TEXT) --- */
    /* This specifically targets the textarea element inside the Streamlit component */
    div[data-testid="stTextArea"] textarea {
        color: #000000 !important;      /* Force Pure Black Text */
        background-color: #ffffff !important; /* Ensure White Background */
        opacity: 1 !important;           /* Remove the 'grayed out' disabled effect */
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource
def load_summarizer():
    # Load a smaller, faster model for CPU
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

def summarize_text(text, summarizer):
    try:
        if len(text) < 50: return "Email is too short to summarize."
        # Truncate to keep it fast
        result = summarizer(text[:1000], max_length=130, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

# --- 3. MAIN APP ---

st.title("📧 Smart Email Search")
st.markdown("Find emails by specific details and get instant AI summaries.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV", type=['csv'], label_visibility="collapsed")

if uploaded_file:
    # --- FILE SIZE CHECK ---
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"🚫 File is too large! Maximum size is {MAX_FILE_SIZE_MB}MB. Your file is {uploaded_file.size / (1024*1024):.2f}MB.")
        st.stop()

    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Normalize columns to lowercase to match inputs easily
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure text exists for summarization
    if 'body' not in df.columns:
        st.error("CSV must have a 'body' column.")
        st.stop()

    # --- SEARCH UI (THE FILTERS) ---
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    st.subheader("🔎 Search Filters")
    
    # Create a grid for search inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Input: Username
        search_username = st.text_input("Search by Username", placeholder="e.g., john.doe")
        # Input: Department
        search_dept = st.text_input("Search by Department", placeholder="e.g., Sales")
        
    with col2:
        # Input: Email
        search_email = st.text_input("Search by Email", placeholder="e.g., john@company.com")
        # Input: Body Keyword
        search_body = st.text_input("Keyword in Body", placeholder="e.g., invoice")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # --- FILTER LOGIC ---
    # Start with showing everything
    mask = pd.Series([True] * len(df), index=df.index)

    # Apply filters ONLY if the user typed something in that box
    if search_username:
        if 'username' in df.columns:
            mask &= df['username'].str.contains(search_username, case=False, na=False)
        else:
            st.warning("Column 'username' not found in CSV.")

    if search_email:
        if 'email' in df.columns:
            mask &= df['email'].str.contains(search_email, case=False, na=False)
        else:
            st.warning("Column 'email' not found in CSV.")

    if search_dept:
        if 'department' in df.columns:
            mask &= df['department'].str.contains(search_dept, case=False, na=False)
        else:
            st.warning("Column 'department' not found in CSV.")

    if search_body:
        mask &= df['body'].str.contains(search_body, case=False, na=False)

    # Get Filtered Results
    results_df = df[mask]

    # Load AI Model (Load once after data is ready)
    summarizer = load_summarizer()

    # --- DISPLAY RESULTS ---
    if not results_df.empty:
        st.write(f"Found **{len(results_df)}** emails matching your criteria.")
        st.divider()

        for index, row in results_df.iterrows():
            # Extract data safely
            subject = row.get('subject', 'No Subject')
            sender = row.get('username', row.get('email', 'Unknown Sender'))
            date = row.get('date', '')
            body = str(row.get('body', ''))
            
            # Card Container
            with st.container():
                st.markdown(f"""
                <div class='email-card'>
                    <h3 style="margin-top:0; color: #2563eb;">{subject}</h3>
                    <div style="color: #64748b; font-size: 0.9em; margin-bottom: 10px;">
                        From: <b>{sender}</b> &nbsp;|&nbsp; {date}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # The Action Button
                # We use the body preview inside the expander
                with st.expander("View Email & Summarize"):
                    st.text_area("Email Content", body, height=120, disabled=True, label_visibility="collapsed")
                    
                    # Summarize Button
                    if st.button(f"✨ Summarize this Email", key=f"sum_{index}", use_container_width=True):
                        with st.spinner("AI is analyzing..."):
                            summary = summarize_text(body, summarizer)
                            st.success(summary)
                
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No emails found matching those specific criteria. Try clearing some filters.")

else:
    # Empty State

    st.info("Please upload a CSV file to start searching.")
