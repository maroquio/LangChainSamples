############################################

# Exemplo de Agente com memória e múltiplos
# contextos de conversa. Demonstra gerenciamento
# de diferentes threads (conversas separadas e
# independentes) usando thread_id distintos.
# Mostra como manter múltiplas conversas
# simultâneas sem interferência entre elas.

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um especialista em previsão do tempo, que fala em trocadilhos.

Você tem acesso a duas ferramentas:

- get_weather_for_location: use isso para obter o clima de um local específico
- get_user_location: use isso para obter a localização do usuário

Se um usuário perguntar sobre o clima, certifique-se de saber a localização. Se você puder perceber pela pergunta que eles querem saber o clima de onde estão, use a ferramenta get_user_location para encontrar a localização deles."""

############################################
# PASSO 2 - Configurar o esquema do contexto
############################################

from dataclasses import dataclass


@dataclass
class Context:
    """Esquema personalizado para o contexto de runtime."""

    user_id: str


############################################
# PASSO 3 - Definir as ferramentas
############################################

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime


@tool
def get_weather_for_location(city: str) -> str:
    """Obter o clima para uma determinada cidade."""
    return f"Sempre está ensolarado em {city}!"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Recuperar informações do usuário com base no ID do usuário."""
    user_id = runtime.context.user_id
    match user_id:
        case "1":
            return "Cachoeiro de Itapemirim"
        case "2":
            return "Vitória"
        case _:
            return "São Paulo"


############################################
# PASSO 4 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.5,
    timeout=10,
    max_tokens=1000,
)

############################################
# PASSO 5 - Definir um formato de resposta
############################################

from dataclasses import dataclass


@dataclass
class ResponseFormat:
    """Esquema de resposta para o agente."""

    punny_response: str
    weather_conditions: str | None = None


############################################
# PASSO 6 - Criar a memória do agente
############################################

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()


############################################
# PASSO 7 - Inicializar o agente com memória
############################################

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        get_user_location,
        get_weather_for_location,
    ],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

############################################
# PASSO 8 - Definir uma configuração com 
# identificador de contexto de conversa
############################################

from langchain_core.runnables.config import RunnableConfig

# `thread_id` é um identificador único para uma determinada conversa.
run_config1: RunnableConfig = {"configurable": {"thread_id": "1"}}

############################################
# PASSO 9 - Usar o agente para conversar
############################################

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Como está o clima lá fora?"}]},
    config=run_config1,
    context=Context(user_id="1"),
)

print(response["structured_response"])

# Continuar a conversa usando o mesmo `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "O meu nome é João da Silva."}]},
    config=run_config1,
    context=Context(user_id="1"),
)

print(response["structured_response"])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Você se lembra do meu nome?"}]},
    config=run_config1,
    context=Context(user_id="1"),
)

print(response["structured_response"])

############################################
# PASSO 10 - Definir uma segunda configuração
# de runtime personalizada
############################################

run_config2: RunnableConfig = {"configurable": {"thread_id": "2"}}

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Você se lembra do meu nome?",
            }
        ]
    },
    config=run_config2,  # Se passar run_config1, ele se lembra do "João da Silva"
    context=Context(user_id="1"),
)

print(response["structured_response"])

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print()
print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. MÚLTIPLAS CONVERSAS SIMULTÂNEAS:
   - Cada thread_id representa uma conversa independente
   - Conversas não interferem umas nas outras
   - Mesmo agente pode gerenciar múltiplas conversas

2. GERENCIAMENTO DE THREADS:
   - thread_id="1": Primeira conversa (João da Silva)
   - thread_id="2": Segunda conversa (nova, sem contexto)
   - MemorySaver mantém cada thread separadamente

3. O QUE ACONTECE:
   - Thread 1: Agente pergunta clima → diz nome → lembra nome
   - Thread 2: Nova conversa, NÃO lembra de João (thread diferente)
   - Se usar thread_id="1" no final, lembraria de João

4. CASOS DE USO:
   - Chatbots multiusuário (cada usuário = um thread_id)
   - Múltiplas sessões de atendimento paralelas
   - Diferentes contextos de conversa do mesmo usuário

5. ISOLAMENTO DE DADOS:
   - Threads são completamente isoladas
   - Privacidade entre usuários garantida
   - Não há vazamento de contexto entre threads

6. GESTÃO DE thread_id EM PRODUÇÃO:
   - Use UUIDs para garantir unicidade
   - Associe thread_id ao user_id + session_id
   - Exemplo: f"{user_id}_{session_id}"

7. LIMPEZA DE MEMÓRIA:
   - MemorySaver() cresce ilimitadamente
   - Em produção, implemente limpeza periódica
   - Ou use persistência com TTL (Time To Live)

8. PRÓXIMOS PASSOS:
   - Para seleção dinâmica de modelo, veja sample010.py
   - Para tratamento de erros, veja sample011.py
   - Para prompts dinâmicos, veja sample012.py
""")
