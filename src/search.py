import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.
- Se a pergunta citar uma empresa, só responda se o nome aparecer exatamente no CONTEXTO.
- Antes de responder, cite a linha do CONTEXTO que contém a informação usada.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

DEFAULT_NO_INFO = "Não tenho informações necessárias para responder sua pergunta."

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


def build_llm(provider):
    provider = normalize_provider(provider)
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        return ChatOpenAI(model=model, temperature=0)
    if provider == "google":
        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def build_vector_store(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def search_prompt():
    provider_value = os.getenv("LLM_PROVIDER", "openai")
    embedding_value = os.getenv("EMBEDDING_PROVIDER", provider_value)

    llm_providers = provider_priority(provider_value, "openai")
    embedding_providers = provider_priority(embedding_value, llm_providers[0])

    try:
        embeddings = build_embeddings(embedding_providers[0])
        vector_store = build_vector_store(embeddings)
        llm = build_llm(llm_providers[0])
    except Exception as exc:
        print(f"Erro ao inicializar a busca: {exc}")
        return None

    state = {
        "embedding_index": 0,
        "llm_index": 0,
        "vector_store": vector_store,
        "llm": llm,
    }

    def switch_embeddings():
        if state["embedding_index"] + 1 >= len(embedding_providers):
            return False
        state["embedding_index"] += 1
        provider = embedding_providers[state["embedding_index"]]
        state["vector_store"] = build_vector_store(build_embeddings(provider))
        print(f"Fallback de embeddings para {provider}.")
        return True

    def switch_llm():
        if state["llm_index"] + 1 >= len(llm_providers):
            return False
        state["llm_index"] += 1
        provider = llm_providers[state["llm_index"]]
        state["llm"] = build_llm(provider)
        print(f"Fallback de LLM para {provider}.")
        return True

    def search_with_fallback(query):
        try:
            return state["vector_store"].similarity_search_with_score(query, k=10)
        except Exception as exc:
            if is_quota_error(exc) and switch_embeddings():
                return state["vector_store"].similarity_search_with_score(query, k=10)
            raise

    def extract_company_name(text):
        match = re.search(r"\bempresa\s+([A-Za-z0-9_-]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def run(question):
        question = (question or "").strip()
        if not question:
            return DEFAULT_NO_INFO

        results = search_with_fallback(question)
        company = extract_company_name(question)
        if company and not any(
            company.lower() in (doc.page_content or "").lower() for doc, _ in results
        ):
            results = search_with_fallback(company)

        if os.getenv("DEBUG_RAG", "").lower() in ("1", "true", "yes", "on"):
            print("DEBUG_RAG: resultados top-10")
            for idx, (doc, score) in enumerate(results, start=1):
                preview = (doc.page_content or "").replace("\n", " ")[:200]
                print(f"{idx:02d} score={score:.6f} text={preview}")

        context = "\n\n".join(
            doc.page_content for doc, _score in results if doc.page_content
        )
        if not context.strip():
            return DEFAULT_NO_INFO

        prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=question)
        try:
            response = state["llm"].invoke(prompt)
        except Exception as exc:
            if is_quota_error(exc) and switch_llm():
                response = state["llm"].invoke(prompt)
            else:
                raise

        return response.content.strip()

    return run
