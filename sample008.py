############################################

# Exemplo de Agente COM memória usando
# MemorySaver e checkpointer. Demonstra como
# o agente mantém contexto de conversas usando
# thread_id, permitindo continuidade entre
# múltiplas invocações.

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

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

model = init_chat_model(
    "gpt-4o-mini",  # substitua pelo modelo desejado (este é o modo estático de definir o modelo)
    temperature=0.5,  # controle de aleatoriedade, criativo vs. preciso
    timeout=10,  # tempo máximo de resposta em segundos
    max_tokens=1000,  # limite de tokens na resposta
)

############################################
# PASSO 5 - Definir um formato de resposta
############################################

from dataclasses import dataclass


# Pode-se usar dataclass aqui, mas modelos Pydantic também são suportados.
@dataclass
class ResponseFormat:
    """Esquema de resposta para o agente."""

    # Uma resposta engraçada com trocadilhos sobre o clima
    punny_response: str
    # Qualquer informação interessante sobre o clima, se disponível
    weather_conditions: str | None = None


############################################
# PASSO 6 - Criar a memória do agente
############################################

from langgraph.checkpoint.memory import MemorySaver

# Instancia a memória do agente (em RAM, neste caso)
checkpointer = MemorySaver()


############################################
# PASSO 7 - Inicializar o agente COM MEMÓRIA
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

# `thread_id` é um identificador único para um contexto de
# conversa e é de uso OBRIGATÓRIO para agentes com memória.
run_config: RunnableConfig = {"configurable": {"thread_id": "1"}}


############################################
# PASSO 9 - Usar o agente para conversar
############################################

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Como está o clima lá fora?"}]},
    context=Context(user_id="1"),
    config=run_config,
)

print(response["structured_response"])

# Continuar a conversa usando o mesmo `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "O meu nome é João da Silva."}]},
    context=Context(user_id="1"),
    config=run_config,
)

print(response["structured_response"])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Você se lembra do meu nome?"}]},
    context=Context(user_id="1"),
    config=run_config,
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
1. AGENTE COM MEMÓRIA:
   - MemorySaver() armazena histórico de conversas
   - Cada thread_id representa uma conversa única
   - O agente LEMBRA de mensagens anteriores

2. COMPONENTES NECESSÁRIOS:
   - checkpointer: MemorySaver() para armazenar em memória
   - thread_id: Identificador único da conversa
   - RunnableConfig: Configuração com thread_id

3. THREAD_ID:
   - Obrigatório para agentes com memória
   - Identifica uma conversa específica
   - Mesmo thread_id = mesma conversa

4. O QUE ACONTECE NESTE EXEMPLO:
   - 1ª invocação: Pergunta clima (thread_id="1")
   - 2ª invocação: Diz o nome (thread_id="1", lembra contexto)
   - 3ª invocação: Pergunta nome (thread_id="1", LEMBRA!)

5. PERSISTÊNCIA:
   - MemorySaver() armazena em RAM (memória volátil)
   - Dados são perdidos quando o programa termina
   - Para persistência permanente, use SqliteSaver ou PostgresSaver

6. DIFERENÇA DO sample007.py:
   - sample007.py: SEM memória (esquece tudo)
   - sample008.py: COM memória (lembra tudo do thread)

7. CUSTO E TOKENS:
   - Com memória, o histórico completo é enviado a cada invoke()
   - Mais tokens = maior custo
   - Considere limitar tamanho do histórico em produção

8. PRÓXIMOS PASSOS:
   - Para múltiplas conversas paralelas, veja sample009.py
   - Para controle manual de mensagens, veja sample013.py
   - Para memória persistente (banco de dados), consulte exemplos posteriores
""")
