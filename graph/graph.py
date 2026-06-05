from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.state import GraphState

load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)


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
graph.add_edge(WEB_SEARCH, GENERATE)
graph.add_edge(GENERATE, END)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')