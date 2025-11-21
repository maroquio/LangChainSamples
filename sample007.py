############################################

# Exemplo de Agente SEM memória. Demonstra
# que o agente não mantém contexto entre
# invocações, esquecendo informações da
# conversa anterior. Este é um exemplo
# importante para contrastar com os exemplos
# seguintes que incluem memória.

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
# PASSO 6 - Inicializar o agente com formato
# de resposta personalizado, SEM MEMÓRIA
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
)

############################################
# PASSO 7 - Usar o agente para conversar
############################################

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Como está o clima lá fora?"}]},
    context=Context(user_id="1"),
)

print(response["structured_response"].punny_response)

# Continuar a conversa usando o mesmo `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "O meu nome é João da Silva."}]},
    context=Context(user_id="1"),
)

print(response["structured_response"].punny_response)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Você se lembra do meu nome?"}]},
    context=Context(user_id="1"),
)

print(response["structured_response"].punny_response)

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print()
print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. AGENTE SEM MEMÓRIA:
   - Cada invocação é INDEPENDENTE
   - O agente NÃO lembra de conversas anteriores
   - Cada chamada agent.invoke() é uma nova conversa

2. O QUE ACONTECE NESTE EXEMPLO:
   - 1ª invocação: Pergunta sobre clima (responde corretamente)
   - 2ª invocação: Usuário diz seu nome (agente responde)
   - 3ª invocação: Pergunta "lembra do meu nome?" (NÃO lembra!)

3. POR QUE NÃO LEMBRA:
   - Não há checkpointer configurado
   - Cada invoke() recebe apenas as mensagens passadas
   - As mensagens anteriores são descartadas

4. QUANDO USAR SEM MEMÓRIA:
   - Perguntas isoladas e independentes
   - Quando não precisa de contexto histórico
   - Para reduzir custos (menos tokens enviados)
   - APIs stateless

5. COMO ADICIONAR MEMÓRIA:
   - Use checkpointer=MemorySaver() no create_agent()
   - Passe thread_id no config ao invocar
   - Veja sample008.py para implementação completa

6. DIFERENÇA ENTRE STATE E MEMÓRIA:
   - State: mensagens passadas manualmente em cada invoke()
   - Memória: checkpointer persiste mensagens automaticamente
   - sample013.py mostra como passar state manualmente

7. PRÓXIMOS PASSOS:
   - Para agente COM memória, veja sample008.py
   - Para múltiplas conversas paralelas, veja sample009.py
   - Para passar mensagens manualmente, veja sample013.py
""")
