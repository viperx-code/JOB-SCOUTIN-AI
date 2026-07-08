import os
import re
import io
import fitz  # PyMuPDF
import docx2txt
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ==========================================
# 0. INITIALIZATION & SESSION MEMORY STATE
# ==========================================
load_dotenv()

# Memory Engine: Track seen/applied job URLs across agent runs
if "seen_job_urls" not in st.session_state:
    st.session_state.seen_job_urls = set()

# Feed Engine: Hold the evaluated jobs in memory so they don't vanish on refresh
if "current_feed" not in st.session_state:
    st.session_state.current_feed = []

# Streamlit Core Config
st.set_page_config(page_title="Job ScoutIN AI", page_icon="🕵️‍♂️", layout="wide")

# Sophisticated Cyberpunk Design System Injection
st.markdown("""
    <style>
    /* Premium Deep Black Background */
    .stApp { background-color: #050505; color: #E0E0E0; }
    
    /* Neon Cyberpunk Main Elements */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    .neon-text { color: #00FF66 !important; font-family: monospace; }
    
    /* High-Fidelity Sophisticated Cards */
    .job-card { 
        background-color: #0F1115; 
        padding: 24px; 
        border-radius: 8px; 
        margin-bottom: 20px; 
        border: 1px solid #1A1F26;
        border-left: 4px solid #00FF66;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .job-card-rejected { 
        border-left: 4px solid #FF3333;
        background-color: #0D090A;
    }
    
    /* Sleek metric fonts */
    .metric-score { font-size: 26px; font-weight: 800; font-family: monospace; }
    
    /* Clean Divider lines */
    hr { border-color: #1A1F26 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MULTI-FORMAT FILE PARSING ENGINE
# ==========================================
def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name
    try:
        if file_name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "".join([page.get_text() for page in doc])
        elif file_name.endswith('.docx'):
            file_bytes = uploaded_file.read()
            return docx2txt.process(io.BytesIO(file_bytes))
        elif file_name.endswith('.txt'):
            return str(uploaded_file.read(), 'utf-8', errors='ignore')
    except Exception as e:
        st.error(f"Failed to read file structure for {file_name}: {e}")
    return ""

@st.cache_resource
def build_vector_db(file_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(file_text)
    
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectors = embeddings_model.embed_documents(chunks)
    
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="resume",
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
    )
    
    points = [
        PointStruct(id=idx, vector=vec, payload={"text": chunk})
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name="resume", points=points)
    return client

# ==========================================
# 2. POWERFUL DATA SCRAPING PIPELINE
# ==========================================
@st.cache_data(ttl=300)
def fetch_live_jobs(role, location, max_jobs=10):
    """Bulletproof API ingestion handling deeply nested JSearch v2 structures."""
    try:
        url = "https://jsearch.p.rapidapi.com/search-v2"
        
        querystring = {
            "query": f"{role} in {location}",
            "page": "1",
            "num_pages": "1"
        }
        
        headers = {
            "X-RapidAPI-Key": st.secrets["RAPIDAPI_KEY"].strip(),
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code != 200:
            st.error(f"API Failed (Status {response.status_code}): {response.text}")
            return []
            
        json_response = response.json()
        
        # --- THE TRUE FIX: Unpacking the double-layer dictionary ---
        # 1. Grab the "data" payload (which we now know is a dictionary)
        raw_data = json_response.get("data", {})
        
        # 2. Extract the actual list of jobs from inside that dictionary
        if isinstance(raw_data, dict):
            jobs_list = raw_data.get("jobs", [])
        elif isinstance(raw_data, list):
            jobs_list = raw_data
        else:
            jobs_list = []
            
        # 3. Final safety check
        if not jobs_list:
            st.sidebar.warning("Agent returned no results. Try broadening the search.")
            return []
            
        formatted_jobs = []
        # Now we can safely slice because we know for an absolute fact jobs_list is a List!
        for job in jobs_list[:max_jobs]:
            formatted_jobs.append({
                "title": job.get("job_title", "Unknown Role"),
                "company": job.get("employer_name", "Unknown Company"),
                "job_url": job.get("job_apply_link", "#"),
                "description": job.get("job_description", "No description provided.")
            })
            
        return formatted_jobs

    except Exception as e:
        st.error(f"🛑 CRITICAL API FAILURE: {str(e)}")
        return []

def evaluate_job(db_client, job_description):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    job_vector = embeddings_model.embed_query(job_description)
    
    search_results = db_client.query_points(collection_name="resume", query=job_vector, limit=4)
    retrieved_context = "\n\n".join([hit.payload["text"] for hit in search_results.points])
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)
    prompt = f"""
    You are an elite, highly analytical Technical Recruiter and Career Strategist. 
    Your expertise lies in evaluating candidates against job descriptions with brutal honesty.
    CONTEXT (Candidate Resume Excerpts): {retrieved_context}
    TARGET JOB DESCRIPTON: {job_description}

    Provide a highly technical evaluation of the match.
    
    OUTPUT FORMAT (Strictly match labels):
    Match Score: [Integer between 1 and 100 representing compatibility]
    Missing Skills: [List missing hard tools, frameworks, or experience gaps]
    Analysis: [Brutally honest reasoning. If score >= 60, write a short, precise 3-sentence cover letter accentuating specific strengths. If below 60, explain why application is illogical.]
    """
    response = llm.invoke(prompt)
    
    text_output = response.content
    if isinstance(text_output, list):
        text_output = text_output[0].get("text", str(text_output))
        
    return text_output.strip()

# ==========================================
# CALLBACK FUNCTION (Instant UI Updates)
# ==========================================
def mark_applied_callback(url):
    """Fires instantly when 'Mark as Applied' is clicked."""
    # 1. Add to the excluded list so it never scrapes again
    st.session_state.seen_job_urls.add(url)
    
    # 2. Remove it directly from the current feed so the UI updates instantly
    st.session_state.current_feed = [
        job for job in st.session_state.current_feed 
        if job["job_data"]["job_url"] != url
    ]

# ==========================================
# 3. SOPHISTICATED DASHBOARD FRONTEND
# ==========================================
st.title("🕵️‍♂️ Autonomous Job ScoutIN AI")
st.markdown("### <span class='neon-text'>[System Ready] Deploy AI Job agent to hunt, analyze, score, and isolate open positions.</span>", unsafe_allow_html=True)
st.divider()

st.subheader("⚡ Scout Setup")
uploaded_file = st.file_uploader("Upload Resume/Portfolio File (Supported: .pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

col1, col2 = st.columns(2)
with col1:
    target_role = st.text_input("Target Role", placeholder="Python Backend Developer")
with col2:
    target_location = st.text_input("Location", placeholder="Bengaluru, Delhi")

deploy_clicked = st.button("🚀 DEPLOY SCOUT AGENT", use_container_width=True)

with st.sidebar:
    st.markdown("<h2 style='letter-spacing: -1px;'>⚡ Job ScoutIN OS</h2>", unsafe_allow_html=True)
    st.caption("Agent Platform v1.0")
    st.caption("Built By Fxhan")
    st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Control Layer")
    sort_preference = st.selectbox("Sort By", [
        "🌟 Highest Match First", 
        "⚠️ Lowest Match First"
    ])
    
    st.markdown("---")
    st.markdown("### 🧠 Agent Session State")
    st.metric("Excluded (Applied) Jobs", len(st.session_state.seen_job_urls))
    if st.button("Reset Jobs"):
        st.session_state.seen_job_urls.clear()
        st.session_state.current_feed.clear()
        st.success("Session history cleared!")
        st.rerun()
        
    st.markdown("---")
    st.caption("Platform Engine: v1.0")
    st.caption("Architect Codebase: Fxhan")

# ==========================================
# 4. AGENT OPERATIONS & RUNTIME EXECUTION
# ==========================================
if deploy_clicked:
    if not uploaded_file:
        st.warning("🚨 Resume required (Upload the Resume).")
        st.stop()
        
    with st.spinner("🧠 Parsing workspace profile..."):
        file_text = extract_text_from_file(uploaded_file)
        if not file_text.strip():
            st.error("Incompatible file context structure. Ensure file contains clear readable text.")
            st.stop()
        db_client = build_vector_db(file_text)
        
    with st.spinner("🌐 Launching Scout Agent..."):
        raw_harvested_jobs = fetch_live_jobs(target_role, target_location, max_jobs=8)
        
    live_jobs = [j for j in raw_harvested_jobs if j["job_url"] not in st.session_state.seen_job_urls]
    
    if not live_jobs:
        st.info("🛰️ No fresh jobs found matching parameters that aren't already tracked in your applied memory list.")
        st.stop()
        
    st.success(f"Discovered {len(live_jobs)} fresh un-applied positions. Processing...")
    
    evaluated_jobs_list = []
    for job in live_jobs:
        with st.spinner(f"Processing: {job['title']} at {job['company']}..."):
            evaluation = evaluate_job(db_client, job['description'])
            try:
                score_str = evaluation.split("Match Score:")[1].split("\n")[0].strip()
                score = int(re.sub(r'\D', '', score_str))
            except:
                score = 0
                
            evaluated_jobs_list.append({
                "job_data": job,
                "ai_text": evaluation,
                "match_score": score
            })
            
    # Save the evaluations to our persistent memory feed
    st.session_state.current_feed = evaluated_jobs_list


# ==========================================
# 5. HIGH-TECH RENDER PASS (Outside the if block!)
# ==========================================
# Only render if we have jobs saved in the memory feed
if st.session_state.current_feed:
    
    # Sort a copy of the feed based on the sidebar preference
    display_feed = st.session_state.current_feed.copy()
    if sort_preference == "🌟 Highest Match First":
        display_feed.sort(key=lambda x: x["match_score"], reverse=True)
    else:
        display_feed.sort(key=lambda x: x["match_score"], reverse=False)
        
    st.subheader("📊 Active Job Feed")
    
    for index, item in enumerate(display_feed):
        job = item["job_data"]
        score = item["match_score"]
        evaluation = item["ai_text"]
        url = job["job_url"]
        
        card_style = "job-card" if score >= 60 else "job-card job-card-rejected"
        status_tag = "🟢 HIGH RELEVANCE" if score >= 75 else ("🟡 MARGINAL MATCH" if score >= 60 else "🔴 MISMATCH")
        
        st.markdown(f"""
        <div class="{card_style}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: monospace; font-size: 12px; color: #8A99AD;">{status_tag}</span>
                <span style="font-family: monospace; font-size: 12px; color: #00FF66;">ID: SCOUT-{1000 + index}</span>
            </div>
            <h2 style="margin: 8px 0;">{job['title']}</h2>
            <p style="color: #8A99AD; margin-bottom: 12px;">Entity: <b>{job['company']}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        m_col1, m_col2 = st.columns([1, 4])
        with m_col1:
            st.markdown(f"<div class='metric-score'>{score}/100</div>", unsafe_allow_html=True)
        with m_col2:
            st.progress(score / 100)
            
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            # The Callback is attached here!
            st.button("Mark as Applied ✔️", key=f"apply_{url}", on_click=mark_applied_callback, args=(url,))
        with btn_col2:
            st.markdown(f"""<a href="{url}" target="_blank"><button style="background-color: #00FF66; color: black; border: none; padding: 6px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">Secure Portal Application 🔗</button></a>""", unsafe_allow_html=True)
            
        with st.expander("Expand Agent Diagnostics", expanded=(score >= 75)):
            st.markdown(evaluation)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
