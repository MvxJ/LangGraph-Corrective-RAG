from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

_SYSTEM_PROMPT = ('You are a grader assessing relevance of a retrieved document to a user question. \n'
    'If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant \n'
    'Give a binary score \'yes\' or  \'no\' score to indicate whether the document ios relevant to question.'
)

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description='Documents are relevant to the question, \'yes\' or \'no\'.'
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', _SYSTEM_PROMPT),
        ('human', 'Retrieve document: \n\n {document} \n\n User question: \n\n {question}'),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader