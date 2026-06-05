from graph.graph import app


def main():
    print('### Advanced RAG ###')
    question = str(input('What would you like to know about agents?'))
    print(app.invoke(
        input={
            'question': question
        }
    ))


if __name__ == "__main__":
    main()
