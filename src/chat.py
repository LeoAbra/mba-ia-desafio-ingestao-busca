from search import search_prompt


def main():
    chain = search_prompt()
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Digite sua pergunta (ou 'sair' para encerrar).")

    while True:
        question = input("PERGUNTA: ").strip()
        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            break

        try:
            answer = chain(question)
        except Exception as exc:
            print(f"Erro ao responder: {exc}")
            continue

        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
