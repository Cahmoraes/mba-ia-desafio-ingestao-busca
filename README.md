# Desafio MBA Engenharia de Software com IA - Full Cycle

## Observações:
Adicionei o Ollama para ser capaz de executar modelos OpenSource e não consumir tokens. Aprendi sobre o Ollama, novos modelos, técnicas e dialeto do Python durante o desafio. Utilizei técnicas de Cache em Memória, Stuff... Foram coisas que aprendi enquanto me questionava como poderia melhorar ou dar um passo a mais. Deixei comentários  pelo código, fiz isso intencionalmente enquanto eu revisava a solução e comentava comigo mesmo, conscientemente sobre o que estou fazendo passo a passo.


### Descreva abaixo como executar a sua solução.

### 1 - Executar Docker Compose
```sh
 docker compose -d
```

### 2 - Instalar modelos no Ollama:
```sh
 docker exec -it ollama bash
 ollama pull nomic-embed-text:latest
 ollama pull llama3.2:1b
```

### 3 - Instalar dependências do Python
```sh
  # Executar no modo de ambiente virtual:
  source ./.venv/bin/activate
  # Instalar dependências:
  pip install -r requirements.md
```

### 4 - Executar o ingest para carregar o PDF e realizar a ingestão no Postgres
```sh
  python3 ./src/ingest.py
```

### 5 - Executar o chat para fornecer uma entrada para consulta

```sh
  python3 ./src/chat.py
```
