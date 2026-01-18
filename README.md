# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto realiza ingestão de um PDF em um banco PostgreSQL com pgVector e permite busca semântica via CLI usando LangChain.

## Requisitos
- Python 3.10+
- Docker e Docker Compose
- Chave de API OpenAI e/ou Google Gemini

## Configuração
1) Crie e ative o ambiente virtual:
```bash
python -m venv venv
source venv/Scripts/activate
```

2) Instale as dependências:
```bash
pip install -r requirements.txt
```

3) Copie o `.env.example` para `.env` e preencha:
```ini
LLM_PROVIDER=auto
EMBEDDING_PROVIDER=auto
OPENAI_API_KEY=
OPENAI_MODEL='gpt-5-nano'
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
GOOGLE_API_KEY=
GOOGLE_MODEL='gemini-2.5-flash-lite'
GOOGLE_EMBEDDING_MODEL='models/embedding-001'
DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='documents'
PDF_PATH='document.pdf'
```

Observações:
- `LLM_PROVIDER` e `EMBEDDING_PROVIDER` aceitam `openai`, `google` ou `auto` (tenta `google` e faz fallback para `openai`).
- `EMBEDDING_PROVIDER` é opcional; se vazio, usa o mesmo valor de `LLM_PROVIDER`.

## Subir o banco
```bash
docker compose up -d
```

## Ingestão do PDF
```bash
python src/ingest.py
```

## Chat no terminal
```bash
python src/chat.py
```

## Exemplo de uso
```
Digite sua pergunta (ou 'sair' para encerrar).
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Notas de implementação
- O PDF é dividido em chunks de 1000 caracteres com overlap de 150.
- A busca usa `similarity_search_with_score(query, k=10)` no pgVector.
- O prompt restritivo está definido em `src/search.py`.
- Se a pergunta tiver "Empresa X" e o chunk não aparecer no top-10, uma segunda busca vetorial é feita usando apenas "X".
