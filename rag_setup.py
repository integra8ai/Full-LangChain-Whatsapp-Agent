# rag_setup.py
import os
from pathlib import Path
from langchain.document_loaders import TextLoader, DirectoryLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "vector_store")

def load_documents(folder="docs_to_index"):
    loader = DirectoryLoader(folder, glob="**/*.*",
                             loader_cls_map={
                                 ".pdf": UnstructuredPDFLoader,
                                 ".docx": UnstructuredWordDocumentLoader,
                                 ".txt": TextLoader
                             })
    docs = loader.load()
    return docs

def build_vector_store():
    docs = load_documents()
    # split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, emb)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("Saved vectorstore to", VECTOR_STORE_PATH)

if __name__ == "__main__":
    build_vector_store()
