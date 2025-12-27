import streamlit as st
import tempfile
import os

from pdf_loader import load_pdf
from splitter import split_docs
from embeddings_store import build_vector_store
from stage1_coverage import run_stage1
from stage2_pseudocode import run_stage2


st.set_page_config(
    page_title="Control Narrative → Pseudocode AI",
    layout="centered"
)

st.title("📘 Control Narrative → Pseudocode Generator")

st.write("Upload a document and generate structured system pseudocode.")


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

# ---------- TASK INPUT ----------
task_prompt = st.text_input(
    "Enter Task Instruction",
    value="Generate complete system pseudocode"
)

# ---------- OPTIONAL QUERY ----------
default_query = (
    "extract complete control / system logic requirements including startup, "
    "operating control logic, alarms, interlocks, shutdown, abort, transitions, "
    "and subsystem behavior"
)

retriever_query = st.text_area(
    "Retrieval Guidance (optional)",
    value=default_query
)

run_button = st.button("🚀 Run AI System")


# ---------- PIPELINE ----------
if run_button:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_pdf_path = tmp.name

            try:
                st.write("📥 Loading PDF...")
                docs = load_pdf(temp_pdf_path)

                st.write("✂️ Splitting text...")
                chunks = split_docs(docs)

                st.write("🧠 Building vector store...")
                retriever = build_vector_store(chunks)

                st.write("🔍 Extracting logic (Stage 1)...")
                coverage_logic = run_stage1(retriever, retriever_query)

                st.write("🧾 Generating pseudocode (Stage 2)...")
                pseudocode = run_stage2(coverage_logic, task_prompt)

                st.success("Done!")

                st.subheader("🧠 Final Pseudocode Output")
                st.code(pseudocode, language="text")

            finally:
                os.remove(temp_pdf_path)
