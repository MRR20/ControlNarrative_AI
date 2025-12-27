from api_key import load_mistral_key
load_mistral_key()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from config import MISTRAL_MODEL, TEMPERATURE, MAX_TOKENS

RULE_BLOCK = """
You are a STRICT PSEUDOCODE GENERATOR.

GENERAL
1. Output ONLY executable logic. No narration or titles.
2. No quotes, no printed document text.
3. Ignore non-executable text.

STRUCTURE
4. Control keywords MUST BE CAPITAL:
   BEGIN, END, FUNCTION, RETURN, IF, ELSE, ELSE IF, FOR, WHILE
5. Use meaningful variable names aligned to terminology.
6. Proper indentation and hierarchy.
7. Only pseudocode, no commentary.

MANDATORY COVERAGE
8. MUST include:
   - Startup logic
   - Operating / feed logic
   - Alarm strategy
   - Shutdown logic
   - Abort logic
   - Subsystem behavior
9. Merge logic from scattered sections.
10. Do not skip relevant behavior.

QUALITY
11. Deterministic and logically runnable.
12. Valid state machine integrity.
13. Avoid vague placeholders.
"""

pseudocode_prompt = ChatPromptTemplate.from_template("""
{rules}

<context>
{context}
</context>

TASK: {input}
""")


def run_stage2(extracted_logic, task):
    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    chain = (
        {
            "rules": lambda _: RULE_BLOCK,
            "context": lambda _: extracted_logic,
            "input": lambda _: task
        }
        | pseudocode_prompt
        | llm
    )

    result = chain.invoke({})
    return result.content
