import os
import pickle
import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from openai import OpenAI
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Antibiotic Resistance Expert",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .profile-section {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 2rem;
    }
    
    .social-links {
        margin-top: 1rem;
    }
    
    .social-links a {
        margin-right: 1rem;
        text-decoration: none;
        color: #0066cc;
    }
    
    .main-content {
        margin-top: 2rem;
    }
    
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    
    .publication {
        padding: 1rem;
        border-left: 3px solid #0066cc;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize AIMLAPI client
AIMLAPI_KEY = "8c827310ef554c7fbfc5fe0df77b6d1c"
openai_client = OpenAI(api_key=AIMLAPI_KEY, base_url="https://api.aimlapi.com/v1")

# ——— 1) Load FAISS metadata & index from disk ———
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    with open("data/faiss_metadata.pkl","rb") as f:
        raw = pickle.load(f)
    docs = [Document(page_content=d["text"], metadata=d) for d in raw]

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device":"cpu"}  # CPU is OK for embeddings on small app
    )
    vs = FAISS.load_local(
        "data/faiss_index",
        emb,
        allow_dangerous_deserialization=True
    )
    return vs

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error("Error loading vectorstore. Make sure data/faiss_metadata.pkl and data/faiss_index exist.")
    st.stop()

# ——— 2) RAG "ask" function ———
def ask(question: str, k: int = 4):
    # a) retrieve
    retr = vectorstore.as_retriever(search_kwargs={"k": k})
    top = retr.get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in top)

    # b) build messages for AIMLAPI
    messages = [
        {"role": "system", "content": "You are a multilingual research assistant on antibiotic resistance. Answer using ONLY the context provided, in the language of the question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    # c) call AIMLAPI
    try:
        response = openai_client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=messages,
            temperature=0.2,
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()
        return answer, top
    except Exception as e:
        st.error(f"Error calling AIMLAPI: {str(e)}")
        return "Sorry, there was an error generating the response.", []

# Profile Section
col1, col2 = st.columns([1, 2])

with col1:
    # Check if profile image exists
    profile_path = Path("profile.jpg")
    if profile_path.exists():
        st.image("profile.jpg", width=200, caption="Dr. Amine Aiddi")
    else:
        st.info("Please add profile.jpg to the project directory")

with col2:
    st.markdown("""
    <div class='profile-section'>
        <h1>Dr. Amine Aiddi</h1>
        <p>PhD Student in Molecular Bacteriology</p>
        <p>Institut Pasteur du Maroc, Faculté des Sciences Ben M'Sik Casablanca</p>
        <div class='social-links'>
            <a href='https://www.linkedin.com/in/amine-aiddi/' target='_blank'>🔗 LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Publications Section
st.markdown("### 📚 Recent Publications")
st.markdown("""
<div class='publication'>
    <h4>Antibiotic Resistance in Surface Waters: A Comprehensive Review</h4>
    <p><em>Journal of Public Health in Africa, 2024</em></p>
    <p>A systematic analysis of antibiotic resistance mechanisms in aquatic environments...</p>

<a href='https://publichealthinafrica.org/index.php/jphia/article/view/598' target='_blank'>Read more</a>
</div>
""", unsafe_allow_html=True)

# Chatbot Section
st.title("📖 Antibiotic-Resistance RAG Chatbot")
st.markdown("""
This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about antibiotic resistance 
using a local knowledge base. Ask your questions in French or English!
""")

question = st.text_input("Posez votre question :", placeholder="Ex : Quels mécanismes de résistance aux antibiotiques...")
if st.button("Demander") and question:
    with st.spinner("🧠 Recherche et génération..."):
        answer, _ = ask(question)  # Ignore sources
    st.subheader("Réponse")
    st.write(answer)
st.markdown("</div>", unsafe_allow_html=True)
