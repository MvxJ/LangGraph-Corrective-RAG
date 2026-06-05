from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState


load_dotenv(verbose=True)
load_dotenv(".env.local", override=True)


web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print('### WEB SEARCH ###')
    question = state.get('question', '')
    documents = state.get('documents', [])

    tavily_results = web_search_tool.invoke({
        'query': question,
    })
    results = tavily_results.get('results', [])
    tavily_results_string = "\n".join(
        [result.get('content', '') for result in results]
    )
    lang_chain_document = Document(
        page_content=tavily_results_string,
    )

    if documents is not None:
        documents.append(lang_chain_document)
    else:
        documents = [lang_chain_document]

    return {
        'documents': documents,
        'question': question,
    }
