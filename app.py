import os
import fitz
import streamlit as st
from jobspy import scrape_jobs
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

st.set_page_config(page_title="Job ScoutIN AI", page_icon="🕵️‍♂️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .job-card { background-color: #1E2530; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #00FF00; }
    .job-card-rejected { border-left: 5px solid #FF0000; }
    h1, h2, h3 { color: #FFFFFF; }
    .metric-text { font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_vector_db(pdf_bytes):
    
    """Reads the uploaded PDF and builds an in-memory vector database."""
    
    raw_text = ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        raw_text += page.get_text()
        
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = textsplitter.split_text(raw_text)
    
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

@st.cache_data(ttl=300)
def fetch_jobs(role, location, max_jobs=5):
    
    """Scrapes live jobs from the internet."""
    try:
        jobs_df = scrape_jobs(
            site_name=["linkedin", "indeed", "naukri"],
            search_term=role,
            location=location,
            results_wanted=max_jobs,
            country_indeed='INDIA'
        )
        
        if jobs_df.empty:
            print("❌ No jobs found. Try broadening your search.")
            return []
        
        clean_jobs = jobs_df[["title", "company", "job_url", "description"]].dropna()
        return clean_jobs.to_dict(orient='records')
    except Exception as e:
        st.error(f"Scouting failed: {e}")
        return []
    

def evaluate_job(db_client, job_description):
    
    """Runs the RAG pipeline and returns the AI's evaluation."""
    # 1. Vectorize the job description
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    job_vector = embeddings_model.embed_query(job_description)
    
    # 2. Search the resume database
    search_results = db_client.query_points(collection_name="resume", query=job_vector, limit=4)
    retrieved_resume_context = "\n\n".join([hit.payload["text"] for hit in search_results.points])
    
    # 3. Gemini 3.5 Flash Evaluation
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    prompt = f"""
    You are an elite, highly analytical Technical Recruiter and Career Strategist. 
    Your expertise lies in evaluating candidates against job descriptions with brutal honesty.
    CONTEXT (Candidate's Resume): {retrieved_resume_context}
    TARGET JOB: {job_description}

    TASK:
    1. Score the match from 1 to 100 based ONLY on the provided context. Do not invent skills.
    2. Identify the top missing skills or weak points the candidate has for this role.
    3. RULE: If the match score is >= 60, write a highly tailored cover letter (under 150 words) highlighting exact strengths. No robotic fluff. No "I hope this email finds you well."
    4. RULE: If the match score is < 60, DO NOT write a cover letter. Instead, explain exactly why they should not apply for this job.

    OUTPUT FORMAT (Strict Markdown):
    Match Score: [Score]/100
    Missing Skills: 
    * [Skill 1]
    * [Skill 2]
    * [Skill 3]

    Cover Letter (OR Rejection Explanation):
    [Content here]
    """
    
    response = llm.invoke(prompt)
    text_output = response.content
    if isinstance(text_output, list):
        text_output = text_output[0].get("text", str(text_output))
        
    return text_output.strip()

st.title("🕵️‍♂️ Autonomous Job ScoutIN/Analyzer Agent")
st.markdown("Upload your resume, search for a role, and let the Agent hunt and evaluate live jobs for you.")

# Sidebar Controls
with st.sidebar:
    st.markdown("<h2 style='letter-spacing: -1px;'>⚡ Job ScoutIN AI OS</h2>", unsafe_allow_html=True)
    st.caption("Agent Platform v1.0")
    st.caption("Built By Fxhan")
    st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type="pdf")
    target_role = st.text_input("2. Target Role", value="Python Backend Developer")
    target_location = st.text_input("3. Location", value="Bengaluru")
    sort_preference = st.selectbox("4. Sort Results By", [
    "🌟 Highest Match First", 
    "⚠️ Lowest Match First"
    ])
    run_agent = st.button("🚀 Deploy Agent", use_container_width=True)

# Main Execution Logic
if run_agent:
    if not uploaded_file:
        st.warning("⚠️ Please upload a resume PDF first.")
        st.stop()
        
    # Step 1: Process Resume
    with st.spinner("🧠 Analyzing your resume..."):
        db_client = build_vector_db(uploaded_file.read())
        
    # Step 2: Scrape Jobs
    with st.spinner(f"🌐 Scouting live job portals for '{target_role}'..."):
        live_jobs = fetch_jobs(target_role, target_location, max_jobs=5)
        
    if not live_jobs:
        st.error("No jobs found. Try changing the role or location.")
        st.stop()
        
    st.success(f"Successfully harvested {len(live_jobs)} jobs. Commencing evaluation...")
    
    # Step 3: Evaluate ALL Jobs (Pass 1)
    # We must collect all scores before we can sort them.
    evaluated_jobs_list = []
    
    st.info("🧠 Agent is evaluating all harvested jobs. Please stand by...")
    
    for job in live_jobs:
        with st.spinner(f"Scoring: {job['title']}..."):
            evaluation = evaluate_job(db_client, job['description'])
            
            # Robust score extraction
            try:
                score_str = evaluation.split("Match Score:")[1].split("/100")[0].strip()
                import re
                score = int(re.sub(r'\D', '', score_str)) 
            except:
                score = 0
                
            # Store the job, its evaluation, and its integer score as a package
            evaluated_jobs_list.append({
                "job_data": job,
                "ai_text": evaluation,
                "match_score": score
            })
            
    st.success("✅ All evaluations complete. Rendering dashboard...")
            
    # Step 4: Sort the Data (Pass 2)
    if sort_preference == "🌟 Highest Match First":
        # Sort descending (100 down to 0)
        evaluated_jobs_list.sort(key=lambda x: x["match_score"], reverse=True)
    else:
        # Sort ascending (0 up to 100)
        evaluated_jobs_list.sort(key=lambda x: x["match_score"], reverse=False)
        
    # Step 5: Render the UI (Pass 3)
    for item in evaluated_jobs_list:
        job = item["job_data"]
        score = item["match_score"]
        evaluation = item["ai_text"]
        
        # Dynamic styling based on score thresholds
        if score >= 75:
            card_class = "job-card" # High Match
            status_emoji = "🔥"
        elif score >= 50:
            card_class = "job-card" # Average Match
            status_emoji = "✅"
        else:
            card_class = "job-card job-card-rejected" # Rejected
            status_emoji = "❌"
            
        st.markdown(f"""
        <div class="{card_class}">
            <h3>{status_emoji} {job['title']} @ {job['company']}</h3>
            <p><a href="{job['job_url']}" target="_blank" style="color: #4DA8DA;">🔗 Apply on Portal</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(score / 100)
        st.markdown(f"<span class='metric-text'>Agent Score: {score}/100</span>", unsafe_allow_html=True)
        
        with st.expander("View Agent Analysis & Missing Skills", expanded=(score>=60)):
            st.markdown(evaluation)
        st.divider()
    
    