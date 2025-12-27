from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBED_MODEL_NAME, RETRIEVAL_K

def build_vector_store(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return retriever
