from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answe_grader import grade_answer
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.state import GraphState

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print('### CHECK HALLUCINATION ###')
    question = state.get('question', '')
    documents = state.get('documents', [])
    generation = state.get('generation', '')

    hallucination_score = hallucination_grader.invoke(
        {
            'generation': generation,
            'documents': documents,
        }
    )

    if hallucination_grade := hallucination_score.binary_score:
        print('### DECISION: GENERATION IS GROUNDED IN DOCS ###')
        print('### GRADE THE GENERATION ACROSS QUESTION ###')

        generation_score = grade_answer.invoke({
            'question': question,
            'generation': generation,
        })

        if answer_grade := generation_score.binary_score:
            print('### DECISION: GENERATION ADDRESS THE QUESTION ###')
            return 'useful'
        else:
            print('### Answer is not useful ###')
            return 'not_useful'
    else:
        return 'not_supported'


def decide_to_generate(state: GraphState):
    print('### ASSESS GRADED DOCUMENTS ###')

    if state.get('web_search', False):
        print('### DECISION: RUN SEARCH WEB ###')

        return WEB_SEARCH

    return GENERATE

graph = StateGraph(GraphState)
graph.add_node(RETRIEVE, retrieve)
graph.add_node(GRADE_DOCUMENTS, grade_documents)
graph.add_node(GENERATE, generate)
graph.add_node(WEB_SEARCH, web_search)

graph.set_entry_point(RETRIEVE)
graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    }
)
graph.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        'not_supported': GENERATE,
        'useful': END,
        'not_useful': WEB_SEARCH,
    }
)
graph.add_edge(WEB_SEARCH, GENERATE)
graph.add_edge(GENERATE, END)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')