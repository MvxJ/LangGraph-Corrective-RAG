from graph.graph import app


def main():
    print('### Advanced RAG ###')
    question = str(input('What would you like to know about agents?'))
    llm_response = app.invoke(
        input={
            'question': question
        }
    )
    print('### FINAL ANSWER ###')
    print(llm_response.get('generation', ''))


if __name__ == "__main__":
    main()
