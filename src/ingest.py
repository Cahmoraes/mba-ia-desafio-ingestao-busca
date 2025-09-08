from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


load_dotenv()


def file_path() -> str:
    return os.getenv("PDF_PATH", "")


def splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def embedding() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", ""))


def main():
    pdfLoader = PyPDFLoader(file_path())
    documents = pdfLoader.load()
    chunks = splitter().split_documents(documents)
    chunk = chunks[0]
    enrich_document = [
        Document(
            # v for v in chunks if v not in ("", None)
            page_content=document.page_content,
            metadata={
                k: v for k, v in document.metadata.items() if v not in ("", None)
            },
        )
        for document in chunks
    ]
    ids = [f"doc-{i}" for i in range(len(enrich_document))]
    print(ids)
    print(enrich_document[0])
    print("-" * 50)
    print(chunk)
    # print(vars(chunk))

    # vector = embedding().embed_query(chunks[0].page_content[:2])
    # print(vector)


if __name__ == "__main__":
    main()
