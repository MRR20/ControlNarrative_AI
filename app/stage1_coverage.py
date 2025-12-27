from api_key import load_mistral_key
load_mistral_key()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from config import MISTRAL_MODEL, TEMPERATURE, MAX_TOKENS

coverage_prompt = ChatPromptTemplate.from_template("""
You are a CONTROL LOGIC REQUIREMENT EXTRACTOR.

Extract ONLY actionable system logic. Do NOT write pseudocode.
Extract structured engineering logic for:

1. Startup requirements
2. Operating / feed / control logic
3. Alarm strategy and compliance expectations
4. Shutdown behavior logic
5. Abort behavior logic
6. Subsystem behavior (such as vessels, auxiliary units)
7. Interlocks, permissives, safety logic
8. State model transition rules
9. Timing / overrides / dependencies

Write as structured technical logic statements.

<context>
{context}
</context>
""")

GENERIC_RAG_QUERY = (
    "extract complete control / system logic requirements including startup, "
    "operating control logic, alarms, interlocks, shutdown, abort, transitions, "
    "and subsystem behavior"
)

def join_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def run_stage1(retriever, query):
    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    chain = (
        {
            "context": retriever | join_docs
        }
        | coverage_prompt
        | llm
    )

    result = chain.invoke(query)
    return result.content
