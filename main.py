import io
import random
import string
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import numpy as np

# Hardcoded configuration
PINECONE_METRIC = "cosine"
PINECONE_DIMENSIONS = 384
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
MODEL_NAME = "all-MiniLM-L6-v2"  # This model outputs 384-dimensional embeddings

def process_pdf(pdf_file, chunk_size, chunk_overlap):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    texts = text_splitter.split_text(text)

    return texts

def get_embeddings(texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts)
    return embeddings

def generate_random_id():
    return ''.join(random.choices(string.digits, k=6))

def upsert_to_pinecone(texts, embeddings, pdf_name, namespace, api_key, index_name):
    pc = Pinecone(api_key=api_key)
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric=PINECONE_METRIC,
            dimension=PINECONE_DIMENSIONS,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )

    index = pc.Index(index_name)

    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        chunk_number = str(i).zfill(4)  # パディングして4桁にする
        vector = {
            "id": f"PDF_{generate_random_id()}",
            "values": embedding.tolist(),
            "metadata": {
                "filename": f"{pdf_name}_{chunk_number}",
                "chunk": text
            }
        }
        
        index.upsert(vectors=[vector], namespace=namespace)
        st.write(f"Uploaded vector {i+1} of {len(texts)}")

def main():
    st.title("Pinecone登録ツール スタンドアロン版")

    st.write(f"Pinecone Configuration:")
    st.write(f"- Metric: {PINECONE_METRIC}")
    st.write(f"- Dimensions: {PINECONE_DIMENSIONS}")
    st.write(f"- Cloud: {PINECONE_CLOUD}")
    st.write(f"- Region: {PINECONE_REGION}")
    st.write(f"- Model: {MODEL_NAME}")

    pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
    pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    chunk_size = st.number_input("Enter chunk size:", value=1000)
    chunk_overlap = st.number_input("Enter chunk overlap:", value=200)
    namespace = st.text_input("Enter Pinecone namespace:")

    if st.button("Process PDF and Upload to Pinecone"):
        if uploaded_file is not None and pinecone_api_key and pinecone_index_name:
            pdf_name = uploaded_file.name
            with st.spinner("Processing PDF..."):
                texts = process_pdf(uploaded_file, chunk_size, chunk_overlap)
            with st.spinner("Generating embeddings..."):
                embeddings = get_embeddings(texts)
            with st.spinner("Uploading to Pinecone..."):
                upsert_to_pinecone(texts, embeddings, pdf_name, namespace, pinecone_api_key, pinecone_index_name)
            st.success(f"Processed and uploaded: {pdf_name}")
        else:
            st.error("Please upload a PDF file and enter Pinecone API Key and Index Name.")

if __name__ == "__main__":
    main()
