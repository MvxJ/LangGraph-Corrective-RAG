from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

_SYSTEM_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
"""


class GradeAnswer(BaseModel):
    """Binary score for the answer. Show if answer address the question."""

    binary_score: bool = Field(
        description='Answer address the question \'yes\' or \'no\''
    )


llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
structured_llm_grader = llm.with_structured_output(GradeAnswer)
prompt = ChatPromptTemplate(
    [
        ('system', _SYSTEM_PROMPT),
        ('user', """User question: \n\n {question} \n\n LLM generation: {generation}""")
    ]
)

grade_answer: RunnableSequence = prompt | structured_llm_grader