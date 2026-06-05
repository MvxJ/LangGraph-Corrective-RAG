from typing import Dict, Any
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any is not relevant, we will set a flag to run web search

    Args:
         state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and update web_search state
    """

    print('### GRADE DOCUMENTS ###')

    filtered_docs = []
    web_search = state.get('web_search', False)
    for document in state.get("documents", []):
        document_grade: GradeDocuments = retrieval_grader.invoke(
            {
                'question': state.get('question', ''),
                'document': document.page_content,
            }
        )

        if document_grade.binary_score == 'yes':
            filtered_docs.append(document)

    if len(state.get('documents', [])) != len(filtered_docs):
        web_search = True

    return {
        'documents': filtered_docs,
        'question': state.get('question', ''),
        'web_search': web_search,
    }