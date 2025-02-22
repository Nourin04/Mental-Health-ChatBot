import os
import zipfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Extract ChromaDB if not already extracted
db_path = "./chroma_db"
if not os.path.exists(db_path):
    with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Initialize the LLM
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Setup the QA Chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Load Vector DB
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
llm = initialize_llm()
qa_chain = setup_qa_chain(vector_db, llm)

# Streamlit UI
st.title("🧠 Mental Health Chatbot 🤖")
st.markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")

user_input = st.text_input("You: ", "")

if user_input:
    response = qa_chain.run(user_input)
    st.text_area("Chatbot:", value=response, height=200, max_chars=None)

st.markdown("⚠️ This chatbot provides general support. For urgent issues, seek help from licensed professionals.")
