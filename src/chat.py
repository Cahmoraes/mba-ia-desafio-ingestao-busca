import os
from search import search_prompt
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache


load_dotenv()
set_llm_cache(InMemoryCache())


def embedding_generator():
    return OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )


def main():
    question = input("Realize sua pergunta:\n")
    # print(question)
    # embedding = embedding_generator().embed_query(question)
    # print(embedding)
    # chain = search_prompt()
    # if not chain:
    #     print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
    #     return
    model = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    )
    chain = search_prompt(question)
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    result = model.invoke(chain)
    print(result)


if __name__ == "__main__":
    main()
