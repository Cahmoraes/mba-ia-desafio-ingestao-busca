import os
from search import search_prompt
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()


def embedding_generator():
    return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", ""))


def main():
    question = input("Realize sua pergunta:\n")
    # print(question)
    embedding = embedding_generator().embed_query(question)
    print(embedding)
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    pass


if __name__ == "__main__":
    main()
