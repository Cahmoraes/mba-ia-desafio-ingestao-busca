import os
from search import search_prompt

from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_postgres import PGVector


load_dotenv()
set_llm_cache(InMemoryCache())


def embedding_generator() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )


def model() -> ChatOllama:
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    )


def store() -> PGVector:
    return PGVector(
        embeddings=embedding_generator(),
        collection_name=collection_name(),
        connection=connection_url(),
        use_jsonb=True,
    )


def collection_name() -> str:
    return os.getenv("PG_VECTOR_COLLECTION_NAME", "")


def connection_url() -> str:
    return os.getenv("DATABASE_URL", "")


def main():
    question = input("Realize sua pergunta:\n")
    # Busca top-k documentos (pega só os Documents)
    search_results = store().similarity_search_with_score(question, k=10)
    documents = [d[0] for d in search_results]
    # Cria o PromptTemplate esperado pela chain (context + pergunta)
    chain = search_prompt(question)
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    stuff = create_stuff_documents_chain(
        llm=model(), prompt=chain, document_variable_name="context"
    )
    result = stuff.invoke({"context": documents, "pergunta": question})
    print(result)


if __name__ == "__main__":
    main()
