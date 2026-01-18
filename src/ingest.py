import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = PROJECT_ROOT / "document.pdf"

PDF_PATH = Path(os.getenv("PDF_PATH", str(DEFAULT_PDF_PATH)))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
)
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")


def normalize_provider(provider):
    provider = (provider or "").strip().lower()
    if provider == "gemini":
        return "google"
    return provider


def provider_priority(value, default_provider):
    raw = normalize_provider(value or default_provider)
    if raw in ("auto", "fallback"):
        return ["google", "openai"]
    if "," in raw:
        return [normalize_provider(part) for part in raw.split(",") if part.strip()]
    return [raw]


def is_quota_error(error):
    message = str(error).lower()
    return "quota" in message or "insufficient_quota" in message or "429" in message


def build_embeddings(provider):
    provider = normalize_provider(provider)

    if provider == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    if provider == "google":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def ingest_pdf():
    pdf_path = PDF_PATH
    if not pdf_path.is_absolute():
        pdf_path = (PROJECT_ROOT / pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    provider_value = os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER", "openai")
    providers = provider_priority(provider_value, "openai")

    last_error = None
    for provider in providers:
        try:
            embeddings = build_embeddings(provider)
            PGVector.from_documents(
                documents=chunks,
                embedding=embeddings,
                connection=DATABASE_URL,
                collection_name=COLLECTION_NAME,
                pre_delete_collection=True,
            )
            print(f"Embeddings provider: {provider}")
            print(f"Ingested {len(chunks)} chunks into '{COLLECTION_NAME}'.")
            return
        except Exception as exc:
            last_error = exc
            if not is_quota_error(exc) or provider == providers[-1]:
                break
            print(f"Quota no provedor '{provider}', tentando fallback...")

    raise last_error


if __name__ == "__main__":
    ingest_pdf()
