from pprint import pprint

from dotenv import load_dotenv

from graph.chains.answe_grader import GradeAnswer, grade_answer
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)

def test_retrieval_grade_related_to_question() -> None:
    question = 'Short-term memory'
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


def test_hallucination_grader_response_on_facts() -> None:
    topic = 'memory management'
    docs = retriever.invoke(topic)
    generation = generation_chain.invoke(
        {
            'question': topic,
            'context': docs,
        }
    )

    hallucination_grade = hallucination_grader.invoke(
        {
            'documents': docs,
            'generation': generation,
        }
    )

    assert hallucination_grade.binary_score


def test_hallucination_grader_response_hallucinate() -> None:
    topic = 'memory management'
    docs = retriever.invoke(topic)
    generation = 'Lorem ipsum lorem ipsum'

    hallucination_grade = hallucination_grader.invoke(
        {
            'documents': docs,
            'generation': generation,
        }
    )

    assert not hallucination_grade.binary_score


def test_answer_grader_response_address_question() -> None:
    question = 'What is agent memory?'
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            'question': question,
            'context': docs,
        }
    )

    question_address_answer_grade: GradeAnswer = grade_answer.invoke(
        {
            'question': question,
            'generation': generation,
        }
    )

    assert question_address_answer_grade.binary_score


def test_answer_grader_response_not_address_question() -> None:
    question = 'What is agent memory?'
    docs = retriever.invoke(question)
    generation = 'Sample text not about agent memory at all. Lorem ipsum lorem ipsum'

    question_address_answer_grade: GradeAnswer = grade_answer.invoke(
        {
            'question': question,
            'generation': generation,
        }
    )

    assert not question_address_answer_grade.binary_score

def test_router_route_to_vector_store() -> None:
    question = 'What is agent short-term memory?'
    router_query: RouteQuery = question_router.invoke({
        'question': question,
    })

    assert router_query.datasource == 'vectorstore'

def test_router_route_to_web_search() -> None:
    question = 'How to make a pizza?'
    router_query: RouteQuery = question_router.invoke({
        'question': question,
    })

    assert router_query.datasource == 'websearch'
