from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("human",
     "You are an assistant for question-answering tasks. Use the following "
     "retrieved context to answer the question. If you don't know the answer, "
     "say you don't know. Keep the answer to three sentences maximum and be concise.\n"
     "Question: {question}\n"
     "Context: {context}\n"
     "Answer:"),
])

generation_chain = prompt | llm | StrOutputParser()