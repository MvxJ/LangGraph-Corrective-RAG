from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of facts
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is granted in / supported by the facts
"""


llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")


class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generated answer."""

    binary_score: bool = Field(
        description="Answer is granted in the facts 'yes' or 'no'."
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination)
prompt = ChatPromptTemplate(
    [
        ('system', _SYSTEM_PROMPT),
        ('human', "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader: RunnableSequence = prompt | structured_llm_grader