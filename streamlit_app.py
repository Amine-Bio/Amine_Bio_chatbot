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

# Custom CSS and Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    .profile-section {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #f0f2f6;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .social-links {
        margin-top: 1rem;
        display: flex;
        gap: 1rem;
    }
    
    .social-links a {
        text-decoration: none;
        color: #0066cc;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background-color: rgba(0,102,204,0.1);
        transition: all 0.2s ease;
    }
    
    .social-links a:hover {
        background-color: rgba(0,102,204,0.2);
        transform: translateY(-2px);
    }
    
    .main-content {
        margin-top: 2rem;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #f8f9fa;
        color: #1f2937;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        text-align: left;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .publication {
        padding: 1.5rem;
        border-left: 4px solid #0066cc;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    
    .response-section {
        margin-top: 1.5rem;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #1f2937;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize AIMLAPI client
# set up in an .env file 
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
        <p>Laboratoire d'Ecologie et Environnement, Faculté des sciences Ben M\'sick Université Hassan II de Casablanca</p>
        <div class='social-links'>
            <a href='https://www.linkedin.com/in/amine-aiddi/' target='_blank'><i class='fas fa-link'></i> LinkedIn</a>
            <a href='https://publichealthinafrica.org/index.php/jphia/article/view/598' target='_blank'><i class='fas fa-book'></i> Publications</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Publications Section
st.markdown("### 📚 Recent Publications")
st.markdown("""
<div class='publication'>
    <h4>Letters to the Editors</h4>
    <p><em>Journal of Public Health in Africa, 2024</em></p>
    <p>First report of blaOXA-48 producing Klebsiella pneumoniae isolates from wastewater in Morocco</p>

<a href='https://publichealthinafrica.org/index.php/jphia/article/view/598' target='_blank'>Read more</a>
</div>
""", unsafe_allow_html=True)

# Chatbot Section
st.title("📖 Antibiotic-Resistance RAG Chatbot")
st.markdown("""
Ce chatbot utilise l'Intelligence Artificielle pour répondre à vos questions sur la résistance aux antibiotiques 
dans les milieux aquatiques. Posez vos questions en français!
""")

# Suggested Questions Section
st.markdown("### 💡 Questions suggérées")
suggested_questions = [
    "Quel est le rôle des stations d'épuration dans la dissémination vs. réduction de la charge en bactéries multirésistantes ?",
    "Comment comparer les effluents hospitaliers, industriels et agricoles en termes de charge en gènes de résistance ?",
    "Quelles technologies de traitement de l'eau (ozonation, UV, nanofiltration) sont efficaces contre les bactéries multirésistantes ?",
    "Quelles politiques publiques pourraient limiter la résistance aux antibiotiques dans les milieux aquatiques ?"
]

# Create columns for suggested questions (2 columns)
col1, col2 = st.columns(2)
for i, q in enumerate(suggested_questions):
    with col1 if i % 2 == 0 else col2:
        if st.button(q, key=f"suggested_{i}", help="Cliquez pour poser cette question"):
            question = q
            with st.spinner("🧠 Recherche et génération..."):
                answer, _ = ask(question)
            st.subheader("Réponse")
            st.write(answer)
            st.markdown("</div>", unsafe_allow_html=True)

# Custom question input
st.markdown("### 🔍 Posez votre propre question")
question = st.text_input("", placeholder="Ex: Quels mécanismes de résistance aux antibiotiques...", key="custom_question")
if st.button("Demander", key="ask_button"):
    with st.spinner("🧠 Recherche et génération..."):
        answer, _ = ask(question)
    st.subheader("Réponse")
    st.write(answer)
    st.markdown("</div>", unsafe_allow_html=True)
