# import os
# import fitz
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct

# load_dotenv()

# google_gemini_key = os.getenv("GEMINI_API_KEY")

# def extract_from_text(pdf_path):
    
#     print(f"Starting PDF extraction for: {pdf_path}")
#     text = ""
    
#     try:
#         doc = fitz.open(pdf_path)
        
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
            
#             text += page.get_text()
#         print(f" Successfully extracted {len(text)} raw characters.")
#         return text
#     except Exception as e:
#         return f"PDF ERROR: {e}"

# def chunk_text(raw_text):
    
#     print("Chunking raw text into semantic blocks...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len
#     )
    
#     chunks = text_splitter.split_text(raw_text)
#     return chunks

# def generative_embeddings(raw_chunks):
    
#     print("Generating vector embeddings via Google API...")
#     try:
        
#         embeddings_model = GoogleGenerativeAIEmbeddings(model= "gemini-embedding-2-preview")
#         vector_embeddings = embeddings_model.embed_documents(raw_chunks)
        
#         print(f" Successfully generated {len(vector_embeddings)} dense vectors.")
#         return vector_embeddings
#     except Exception as e:
#         print(f"❌ Error during embedding generation: {e}")
#         return []
    

# def store_in_qdrant(chunks, vectors, collection_name):
    
#     print(f"Initializing Vector DB and storing in collection: '{collection_name}'...")
#     client = QdrantClient(location=":memory:")
    
#     vector_size = len(vectors[0])
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
#     )
    
#     points = []
    
#     for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
#         points.append(
#             PointStruct(
#                 id=idx,
#                 vector=vector,
#                 payload={"page_content": chunk}
#             )
#         )
    
#     client.upsert(collection_name=collection_name, points=points)
#     print(f" Successfully indexed {len(points)} vectors into Qdrant.")
    
#     # Verify database contents by running a test retrieval query
#     print("\n=== RUNNING DB VERIFICATION VERDICT ===")
#     collection_info = client.get_collection(collection_name=collection_name)
#     print(f"Collection Status: {collection_info.status}")
#     print(f"Total Vectors Successfully Stored: {collection_info.points_count}")
#     return client



# if __name__ == "__main__":
    
#     resume_path = "test_resume.pdf"
#     collection_name = "resume_collection"
    
#     extracted_resume = extract_from_text(resume_path)
    
#     if extracted_resume.strip():
#         chunks = chunk_text(extracted_resume)
#         vectors = generative_embeddings(chunks)
        
#         print("\n=== PIPELINE EXECUTION SUCCESSFUL ===")
#         print(f"Verification: Processed {len(chunks)} text chunks into {len(vectors)} vectors.")
#         # The new gemini-embedding-2-preview model creates highly dense multidimensional vectors
#         print(f"Dimension check: Each vector contains {len(vectors[0])} dimensions.")
        
#         if vectors:
#             qdrant_db_instance = store_in_qdrant(chunks, vectors, collection_name)
#             print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
#         else:
#             print("❌ Pipeline broken at step 3 (Embeddings generation failed).")
#     else:
#         print("\n❌ Pipeline failed. Aborting execution.")
    