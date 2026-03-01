"""
In-class Activity 11 - Production Ready RAG Chatbot
Babson Student Handbook Q&A using LlamaIndex + Gemini + Streamlit

Improvements over Activity 10 (MVP):
  [1] validate_config()     - checks for API key presence, fails fast with clear message
  [2] validate_data_dir()   - checks data directory exists and contains files
  [3] @st.cache_resource    - caches the query engine to reduce latency on reruns
  [4] try/except on index   - catches errors during document loading and indexing
  [5] try/except on query   - catches runtime errors during LLM query
"""

import os
import streamlit as st
from dotenv import load_dotenv

# load .env FIRST before any API clients are instantiated
load_dotenv()

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

"""
STUDENT CHANGE LOG & AI DISCLOSURE:
----------------------------------
1. Did you use an LLM (ChatGPT/Claude/etc.)? No
2. If yes, what was your primary prompt? N/A
----------------------------------
"""

# --- CONFIGURATION ---
DATA_DIR = "data/handbook"
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Babson Handbook Chatbot", layout="centered")

Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# --- [1] CONFIGURATION VALIDATION ---
def validate_config():
    """Check that required API keys are present. Fails fast with a clear message."""
    if not os.getenv("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY is missing. Add it to your .env file and restart.")
        st.stop()


# --- [2] DATA DIRECTORY VALIDATION ---
def validate_data_dir(path):
    """Check that the data directory exists and contains at least one file."""
    if not os.path.exists(path):
        st.error(f"❌ Data directory '{path}' not found. Create it and add your documents.")
        st.stop()
    files = os.listdir(path)
    if not files:
        st.error(f"❌ Data directory '{path}' is empty. Add at least one document.")
        st.stop()


# --- [3] CACHED QUERY ENGINE ---
@st.cache_resource
def get_query_engine():
    """Load documents, build index, and return query engine.
    Cached with st.cache_resource to avoid reloading on every rerun."""
    # [4] try/except around document loading and indexing
    try:
        with st.spinner("📚 Loading handbook and building index..."):
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(documents)
        st.success("✅ Index ready!")
        return index.as_query_engine()
    except Exception as e:
        st.error(f"❌ Failed to load documents or build index: {e}")
        st.stop()


# --- STREAMLIT UI ---
st.title("📖 Babson Student Handbook Chatbot")
st.caption("Ask me anything about the Babson undergraduate student handbook.")

# run validations before doing anything else
validate_config()
validate_data_dir(DATA_DIR)

query_engine = get_query_engine()

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
prompt = st.chat_input("Ask a question about the handbook...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # [5] try/except around the RAG query call
        try:
            with st.spinner("Searching handbook..."):
                response = query_engine.query(prompt)
                bot_response = response.response
            st.markdown(bot_response)
        except Exception as e:
            bot_response = f"⚠️ Sorry, something went wrong while searching: {e}"
            st.error(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
