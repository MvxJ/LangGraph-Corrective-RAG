from pprint import pprint

from dotenv import load_dotenv

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

def test_retrieval_grade_related_to_question() -> None:
    question = 'memory management'
    docs = retriever.invoke(question)
    doc_text = docs[3].page_content

    response: GradeDocuments = retrieval_grader.invoke(
        {
            'question': question,
            'document': doc_text,
        }
    )

    assert response.binary_score == 'yes'


def test_retrieval_grade_not_related_to_question() -> None:
    question = 'memory management'
    docs = retriever.invoke(question)
    doc_text = docs[3].page_content

    response: GradeDocuments = retrieval_grader.invoke(
        {
            'question': 'how to make pizza',
            'document': doc_text,
        }
    )

    assert response.binary_score == 'no'


def test_generation_chain() -> None:
    topic = 'memory management'
    docs = retriever.invoke(topic)
    generation = generation_chain.invoke({
        'context': docs,
        'question': topic,
    })
    assert generation.lower().__contains__(topic)