from typing import Literal
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

_SYSTEM_PROMPT = """You are an expert at routing a user question to a vectorstore or web search. \n
The vector store contains documents related toa gents, prompt engineering and adversarial attacks.\n
Use the vector store for questions on these topics. for all other questions use web-search."""


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal['vectorstore', 'websearch'] = Field(
        ...,
        description='Given a user question to choose route, whether websearch or vectorstore',
    )


llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
structured_llm_router = llm.with_structured_output(RouteQuery)
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', _SYSTEM_PROMPT),
        ('user', "{question}")
    ]
)

question_router = prompt | structured_llm_router