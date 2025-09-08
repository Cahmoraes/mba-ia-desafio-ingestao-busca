from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()
PDF_PATH = os.getenv("PDF_PATH", "")

loader = PyPDFLoader(PDF_PATH)
doc = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

if not splitter:
    raise SystemExit(0)


def main():
    chunks = splitter.split_documents(doc)
    print(chunks[0])
    embedding = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
    )
    preview_text = chunks[0].page_content[:20]
    vector = embedding.embed_query(chunks[0].page_content[:20])
    print("Preview:")
    print(preview_text, "...")
    print("\nEmbedding gerador:")
    print(vector)
    # print(metadata)


if __name__ == "__main__":
    main()
