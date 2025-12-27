from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

def load_pdf(pdf_path: str):
    print(f"[INFO] Loading PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text = "".join([d.page_content for d in docs]).strip()
        if len(text) == 0:
            raise Exception("PDF has no text layer")
        
        print("[INFO] Loaded using PyPDFLoader")
        return docs

    except Exception as e:
        print(f"[WARN] Standard load failed ({e}). Switching to OCR mode...")
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        docs = loader.load()
        print("[INFO] Loaded using OCR fallback")
        return docs
