from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector


load_dotenv()


def file_path() -> str:
    return os.getenv("PDF_PATH", "")


def splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def embedding() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", ""))


def collection_name() -> str:
    return os.getenv("PG_VECTOR_COLLECTION_NAME", "")


def connection_url() -> str:
    return os.getenv("DATABASE_URL", "")


def create_store() -> PGVector:
    return PGVector(
        embeddings=embedding(),
        collection_name=collection_name(),
        connection=connection_url(),
        use_jsonb=True,
    )


def main():
    # Carrega o PDF para ler seu conteúdo
    pdfLoader = PyPDFLoader(file_path())
    documents = pdfLoader.load()
    # Utiliza o recursive text splitter para quebrar o documento em chunks
    chunks = splitter().split_documents(documents)
    # Aplica uma limpeza de metadados vazios ou nulos
    enrich_document = [
        Document(
            page_content=document.page_content,
            metadata={
                k: v for k, v in document.metadata.items() if v not in ("", None)
            },
        )
        for document in chunks
    ]
    # Cria um ID incremental para cada documento "chunk".
    ids = [f"doc-{i}" for i in range(len(enrich_document))]
    # Realizar ingestão no PGVector dos dados
    store = create_store()
    store.add_documents(documents=enrich_document, ids=ids)


if __name__ == "__main__":
    main()
