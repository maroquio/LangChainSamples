############################################
#
# Exemplo de Invocation Config (RunnableConfig)
#
############################################


############################################
# PASSO 1 - Básico: Passando Config para invoke()
############################################

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 70)
print("PASSANDO CONFIG PARA invoke()")
print("=" * 70)

# Criar um config
config = RunnableConfig(
    tags=["exemplo", "teste"],
    metadata={"user_id": "123", "session": "abc"},
    run_name="Primeira Invocação",
)

# Passar config para invoke()
response = model.invoke(
    "Qual é a capital da França?",
    config=config,
)

print(f"Resposta: {response.content}\n")
print("Config aplicado com:")
print(f"  Tags: {config.get('tags')}")
print(f"  Metadata: {config.get('metadata')}")
print(f"  Run name: {config.get('run_name')}")
print("\nℹ️ Estes metadados aparecem em ferramentas de tracing (LangSmith, etc.)")
print()


############################################
# PASSO 2 - Usando tags para Organização
############################################

print("=" * 70)
print("USANDO tags PARA ORGANIZAÇÃO")
print("=" * 70)

# Tags ajudam a categorizar execuções
config_qa = RunnableConfig(tags=["qa", "production"])
config_translation = RunnableConfig(tags=["translation", "batch"])

response_qa = model.invoke(
    "O que é Python?",
    config=config_qa,
)

response_translation = model.invoke(
    "Traduza para inglês: Bom dia",
    config=config_translation,
)

print("Resposta 1 (tags: qa, production):")
print(f"  {response_qa.content}\n")

print("Resposta 2 (tags: translation, batch):")
print(f"  {response_translation.content}\n")

print(
    """
Tags são úteis para:
- Filtrar logs por tipo de operação
- Análise de custos por categoria
- Debugging específico
- Dashboards separados por tag
"""
)
print()


############################################
# PASSO 3 - Usando metadata para Contexto Adicional
############################################

print("=" * 70)
print("USANDO metadata PARA CONTEXTO ADICIONAL")
print("=" * 70)

# Metadata pode conter qualquer informação relevante
config_with_metadata = RunnableConfig(
    metadata={
        "user_id": "user_456",
        "request_id": "req_789",
        "feature": "chatbot",
        "version": "v2.1",
        "environment": "staging",
    },
    run_name="Chatbot Interaction",
)

response = model.invoke(
    "Explique machine learning em uma frase.",
    config=config_with_metadata,
)

print(f"Resposta: {response.content}\n")
print("Metadata anexado:")
for key, value in config_with_metadata.get("metadata", {}).items():
    print(f"  {key}: {value}")

print(
    """
Metadata é útil para:
- Rastreamento por usuário
- Análise de features específicas
- Debugging de problemas
- Auditoria e compliance
- A/B testing
"""
)
print()


############################################
# PASSO 4 - run_name para Tracing Legível
############################################

print("=" * 70)
print("run_name PARA TRACING LEGÍVEL")
print("=" * 70)

# run_name dá um nome descritivo para a execução
questions = [
    ("Qual é 2+2?", "Simple Math"),
    ("Traduza: Hello", "Translation Task"),
    ("O que é AI?", "Q&A About AI"),
]

print("Executando múltiplas tarefas com run_name:\n")

for question, run_name in questions:
    config = RunnableConfig(run_name=run_name)
    response = model.invoke(question, config=config)
    print(f"[{run_name}] {question}")
    print(f"  → {response.content}\n")

print("run_name torna traces mais fáceis de identificar no LangSmith.")
print()


############################################
# PASSO 5 - Callbacks em Config
############################################

print("=" * 70)
print("CALLBACKS EM CONFIG")
print("=" * 70)

from langchain_core.callbacks import StdOutCallbackHandler

# Callback que imprime eventos no stdout
callback = StdOutCallbackHandler()

config_with_callback = RunnableConfig(
    callbacks=[callback],
    run_name="Execution with Callback",
)

print("Executando com StdOutCallbackHandler:\n")
response = model.invoke(
    "Liste 2 cores.",
    config=config_with_callback,
)

print(f"\nResposta final: {response.content}")
print()


############################################
# PASSO 6 - Config com stream()
############################################

print("=" * 70)
print("CONFIG COM stream()")
print("=" * 70)

config_stream = RunnableConfig(
    tags=["streaming", "chat"],
    metadata={"mode": "interactive"},
    run_name="Streaming Example",
)

print("Streaming com config:\n")

for chunk in model.stream("Conte até 3.", config=config_stream):
    print(chunk.content, end="", flush=True)

print("\n\nConfig aplicado durante o streaming.")
print()


############################################
# PASSO 7 - Config com batch()
############################################

print("=" * 70)
print("CONFIG COM batch()")
print("=" * 70)

inputs = ["Um", "Dois", "Três"]

# Um único config para todo o batch
config_batch = RunnableConfig(
    tags=["batch-processing"],
    metadata={"batch_size": len(inputs)},
    run_name="Batch of 3",
)

responses = model.batch(inputs, config=config_batch)

print("Batch executado com config:\n")
for i, resp in enumerate(responses, 1):
    print(f"  {i}. {resp.content}")

print(f"\nTags: {config_batch.get('tags')}")
print(f"Metadata: {config_batch.get('metadata')}")
print()


############################################
# PASSO 8 - configurable em Config (runtime params)
############################################

print("=" * 70)
print("PARÂMETRO configurable (runtime configuration)")
print("=" * 70)

# configurable permite passar parâmetros específicos do runnable
config_with_configurable = RunnableConfig(
    configurable={
        "llm_temperature": 0.9,
        "max_retries": 3,
    },
)

print(
    """
O campo 'configurable' permite passar parâmetros
que podem ser usados por Runnables configuráveis.

Por exemplo, com configurable_fields() ou configurable_alternatives()
você pode mudar comportamento em runtime.

Veja sample030.py para exemplos de configurable models.
"""
)
print()


############################################
# PASSO 9 - Combinando Todos os Elementos
############################################

print("=" * 70)
print("COMBINANDO TODOS OS ELEMENTOS DO CONFIG")
print("=" * 70)

from langchain_core.callbacks import get_usage_metadata_callback

# Config completo com todos os campos
complete_config = RunnableConfig(
    tags=["production", "user-query", "v2"],
    metadata={
        "user_id": "user_999",
        "session_id": "session_abc123",
        "feature": "qa_system",
        "version": "2.0.1",
        "timestamp": "2024-01-15T10:30:00Z",
    },
    run_name="Complete Config Example",
    configurable={
        "model_name": "gpt-4o-mini",
    },
)

with get_usage_metadata_callback() as cb:
    response = model.invoke(
        "Qual é a velocidade da luz?",
        config=complete_config,
    )

    # cb.usage_metadata retorna dict aninhado por model
    usage_data = cb.usage_metadata

print("Resposta:", response.content)
print("\nConfig aplicado:")
print(f"  Tags: {complete_config.get('tags')}")
print(f"  Metadata: {complete_config.get('metadata')}")
print(f"  Run name: {complete_config.get('run_name')}")

# Agregar tokens de todos os models
if usage_data:
    total_tokens = 0
    for model_name, usage in usage_data.items():
        total_tokens += usage.get('total_tokens', 0)
    print(f"\nUsage:")
    print(f"  Tokens: {total_tokens}")
else:
    print(f"\nUsage: não disponível")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print(
    """
1. RunnableConfig:
   - Dicionário de configuração para Runnables
   - Passado via parâmetro config= em invoke/stream/batch
   - Afeta tracing, logging e comportamento
   - Não altera parâmetros do model (use configurable para isso)

2. CAMPOS DO RunnableConfig:

   tags (list[str]):
   - Marcadores/labels para a execução
   - Útil para filtrar e categorizar
   - Aparece em LangSmith e outros tracers
   - Exemplo: ["production", "qa", "v2"]

   metadata (dict):
   - Informações contextuais arbitrárias
   - Chave-valor com dados relevantes
   - Útil para debugging e auditoria
   - Exemplo: {"user_id": "123", "feature": "chat"}

   run_name (str):
   - Nome descritivo da execução
   - Torna traces mais legíveis
   - Aparece em dashboards de tracing
   - Exemplo: "User Query - Product Search"

   callbacks (list[BaseCallbackHandler]):
   - Handlers para eventos durante execução
   - Logging, metrics, tracing customizado
   - Múltiplos callbacks podem ser combinados
   - Exemplo: [StdOutCallbackHandler(), CustomLogger()]

   configurable (dict):
   - Parâmetros para Runnables configuráveis
   - Permite mudar comportamento em runtime
   - Usado com configurable_fields() e configurable_alternatives()
   - Veja sample030.py para detalhes

3. QUANDO USAR tags:
   - Categorizar por tipo de operação (qa, translation, summarization)
   - Separar ambientes (dev, staging, production)
   - Versões de features (v1, v2, experimental)
   - Filtrar logs e traces
   - Análise de custos por categoria

4. QUANDO USAR metadata:
   - Rastreamento de usuários (user_id)
   - Request IDs para debugging
   - Informações de sessão
   - Feature flags
   - Timestamps
   - Qualquer contexto relevante

5. QUANDO USAR run_name:
   - Identificar execuções específicas
   - Debugging de problemas
   - Dashboards legíveis
   - Documentação de fluxos

6. CALLBACKS:
   - StdOutCallbackHandler: imprime eventos
   - LangChainTracer: envia para LangSmith
   - CustomCallbackHandler: seu próprio logging
   - get_usage_metadata_callback(): rastrear tokens

7. LANGSMITH INTEGRATION:
   - Tags e metadata aparecem no LangSmith dashboard
   - Permite filtrar e buscar execuções
   - Útil para análise de performance
   - Debugging de problemas em produção

8. CONFIG COM CHAINS E AGENTS:
   - Config funciona com qualquer Runnable
   - Chains, agents, retrievers, etc.
   - Config é propagado através da cadeia
   - Cada step pode adicionar seus próprios tags/metadata

9. BOAS PRÁTICAS:
   - Use tags consistentes (convenções de nomenclatura)
   - Inclua user_id em metadata quando relevante
   - run_name descritivo (não genérico)
   - Não coloque dados sensíveis em metadata
   - Use callbacks para metrics customizados

10. PRÓXIMOS PASSOS:
    - Para configurable models (runtime params), veja sample030.py
    - Para log probabilities, veja sample031.py
    - Para tool choice control, veja sample032.py
"""
)
