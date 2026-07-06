import os
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

google_gemini_key = os.getenv("GEMINI_API_KEY")

def setup_persistent_database(pdf_path: str, collection_name: str) -> QdrantClient:
    
    db_path = "./local_qdrant_db"
    client = QdrantClient(path=db_path)
    
    if client.collection_exists(collection_name):
        print(f"Database '{collection_name}' already exists on disk. Skipping rebuild.")
        return client
    
    print("Building persistent vector database...")
    
    raw_text = ""
    
    doc = fitz.open(pdf_path)
    for page in doc:
        raw_text += page.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = text_splitter.split_text(raw_text)
    
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectors = embeddings_model.embed_documents(chunks)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    
    points = [
        PointStruct(id=idx, vector=vec, payload={"text": chunk})
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print("Database built and saved to disk successfully.")
    
    return client


def evaluate_candidate(client: QdrantClient, collection_name: str, job_description: str):
    
    print("\n[AGENT] Analyzing Job Description...")
    
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    job_vector = embeddings_model.embed_query(job_description)
    
    search_results = client.query_points(
        collection_name=collection_name,
        query=job_vector,
        limit=4
    )
    
    retrieved_resume_context = "\n\n".join([hit.payload["text"] for hit in search_results.points])
    
    print("[AGENT] Generating evaluation via Job Agent...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2
    )
    
    prompt = f"""
    You are an elite, highly analytical Technical Recruiter and Career Strategist. 
    Your expertise lies in evaluating candidates against job descriptions with brutal honesty.

    CONTEXT (Candidate's Actual Experience Retrieved via Database):
    {retrieved_resume_context}

    TARGET JOB DESCRIPTION:
    {job_description}

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
    
    print("\n" + "="*50)
    print("🤖 JOB SCOUT AGENT OUTPUT")
    print("="*50)
    print(text_output.strip())
    

if __name__ == "__main__":
    
    RESUME_PDF = "test_resume.pdf"
    COLLECTION = "resume_collection"
    
    DUMMY_JOB = """
    Looking for a Junior AI Engineer. 
    Must have 1+ years experience. 
    Required skills: Linux, GenerativeAI, Terraform, and advanced Python backend development.
    """
    
    # Run the pipeline
    db_client = setup_persistent_database(RESUME_PDF, COLLECTION)
    evaluate_candidate(db_client, COLLECTION, DUMMY_JOB)