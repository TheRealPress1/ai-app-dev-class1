"""
In-class Activity 10 - Context Aware RAG Chatbot
Babson Student Handbook Q&A using LlamaIndex + Gemini + Streamlit
"""

import os
import streamlit as st
from dotenv import load_dotenv
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
load_dotenv()
DATA_DIR = "data/handbook"

st.set_page_config(page_title="Babson Handbook Chatbot", layout="centered")

api_key = os.getenv("GEMINI_API_KEY")
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# --- CORE LOGIC ---
@st.cache_resource
def get_query_engine():
    if not os.getenv("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY not found in .env file.")
        st.stop()
    with st.spinner("📚 Loading handbook and building index..."):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
    st.success("✅ Index ready!")
    return index.as_query_engine()


# --- STREAMLIT UI ---
st.title("📖 Babson Student Handbook Chatbot")
st.caption("Ask me anything about the Babson undergraduate student handbook.")

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
        with st.spinner("Searching handbook..."):
            response = query_engine.query(prompt)
            bot_response = response.response
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
