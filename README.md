# 🕵️‍♂️ Autonomous Job ScoutIN AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-orange)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**Autonomous Job ScoutIN AI** is a full-stack, cloud-deployed Retrieval-Augmented Generation (RAG) agent. It autonomously scrapes live job boards, mathematically scores candidate resumes against open roles, and generates brutally honest gap analyses and custom cover letters.

[**Launch Live Agent**](https://job-scoutin-ai-jnph4wsaf53spjfyxgptrn.streamlit.app/) • [**Report a Bug**](https://github.com/viperx-code/job-scoutin-ai/issues)

---

## 🚀 The Architecture (3-Tier Engine)

This application is built with a decoupled architecture, ensuring fault tolerance and zero-latency UI interactions.

### 1. The Ingestion & Memory Layer (Vector Database)
* **Multi-Format Parsing:** Extracts raw text from `.pdf`, `.docx`, and `.txt` utilizing `PyMuPDF` and `docx2txt`.
* **Semantic Chunking:** Prevents LLM context-bloat by splitting documents into 500-character overlapping semantic blocks via LangChain's `RecursiveCharacterTextSplitter`.
* **Vector Engine:** Leverages `gemini-embedding-2-preview` to transform chunks into 3072-dimensional vectors, stored in an ephemeral, in-memory **Qdrant** database.

### 2. The Network Layer (API Ingestion)
* **Bulletproof Sourcing:** Bypasses datacenter bot-blockers and CAPTCHAs by integrating directly with the authenticated **JSearch (RapidAPI)** network.
* **Defensive Parsing:** Employs aggressive type-checking to safely unpack dynamic, deeply nested JSON payloads and handle 404/403 routing drops gracefully.

### 3. The Orchestration Layer (LLM & State Management)
* **RAG Execution:** Executes a Cosine Similarity search against the Qdrant DB to extract only the top 4 most relevant resume chunks for a specific job.
* **Deterministic Output:** Feeds highly targeted context to `gemini-3.5-flash-lite` (Temperature: 0.1) enforced by a strict recruiter persona prompt to generate actionable, non-hallucinated scoring.
* **Session Memory:** Utilizes Streamlit `st.session_state` and callback functions to track applied jobs and instantly update the UI without triggering heavy backend reruns.

---

## ⚡ UI/UX Design System
Moving away from standard flat data apps, this project features a custom-injected CSS matrix utilizing a deep black and neon green Cyberpunk aesthetic. The UI was engineered with a focus on high-fidelity metric rendering, visual feedback loops, and clean expandable diagnostic cards.

<img width="1905" height="923" alt="Screenshot 2026-07-08 160201" src="https://github.com/user-attachments/assets/07fde412-9e26-41a4-a217-07b7f39011ff" />
<img width="1892" height="876" alt="Screenshot 2026-07-08 160307" src="https://github.com/user-attachments/assets/623673ba-c37d-4a42-a1eb-e68fb849ba28" />


---

## 🛠️ Local Installation & Deployment

### Prerequisites
* Python 3.9+
* A Google Gemini API Key
* A RapidAPI (JSearch) Key

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/viperx-code/job-scoutin-ai.git](https://github.com/viperx-code/job-scoutin-ai.git)
   cd job-scoutin-ai
   
2. **Create a virtual environment:**

python -m venv venv

source venv/bin/activate  # On Powershell use `.\venv\Scripts\activate.ps1`

3. **Install dependencies:**

pip install -r requirements.txt

4. **Configure Environment Variables:**
Create a .env file in the root directory and add your keys securely:

Code snippet

GEMINI_API_KEY="your_google_key_here"

RAPIDAPI_KEY="your_rapidapi_key_here"

5. **Initialize the Agent:**

streamlit run app.py

6. **☁️ Cloud Deployment (Streamlit Community Cloud)**

This application is optimized for headless Linux container deployment.

Connect this repository to Streamlit Community Cloud.

Ensure the Main File Path is set to app_V2.py.

Under Advanced Settings, inject your secrets using TOML format:

Ini, TOML

GEMINI_API_KEY = "your_google_key_here"

RAPIDAPI_KEY = "your_rapidapi_key_here"

---

**👨‍💻 Author**


Built by Fxhan

Technical Engineer | Agentic AI Developer

📍 Bengaluru, India
