from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st
loader = TextLoader('carboniferous.txt')
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device' : 'cpu'},
    encode_kwargs = {'normalize_embeddings' : True}
)

vector = FAISS.from_documents(docs, embeddings)

st.title("RAG Demo")
question = st.text_input(label="mention your query")
if question:
    result = vector.similarity_search(question)[0].page_content
    st.write(result)
