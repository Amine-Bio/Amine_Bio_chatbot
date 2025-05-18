# Antibiotic-Resistance RAG Chatbot

A Streamlit-based chatbot that uses RAG (Retrieval-Augmented Generation) with AIMLAPI to answer questions about antibiotic resistance using a local knowledge base.

## Setup

1. Create the data directory and copy your FAISS files:
   ```bash
   mkdir data
   # Copy your faiss_metadata.pkl and faiss_index/ into the data/ directory
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

The AIMLAPI key is already configured in the application.

## Running the App

Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will be available at http://localhost:8501

## Usage

1. Enter your question in French or English in the text input field
2. Click "Demander" to get an answer
3. The response will include both the answer and the source documents used

## Features

- Multilingual support (French/English)
- RAG-powered responses using local FAISS index
- AIMLAPI integration with Llama model
- Source document references for transparency
