############################################

# Exemplo de Agente com Estado Customizado
# definido via Middleware (ABORDAGEM PREFERIDA).
# Demonstra como estender AgentState para
# rastrear informações adicionais além das
# mensagens, mantendo escopo organizado.

############################################


############################################
# PASSO 1 - Definir o estado customizado
# estendendo AgentState
############################################

from langchain.agents import AgentState
from typing import Any


class CustomState(AgentState):
    """Estado customizado que estende AgentState com campos adicionais."""

    # Preferências do usuário (estilo de resposta, verbosidade)
    user_preferences: dict

    # Contador de interações
    interaction_count: int


############################################
# PASSO 2 - Criar middleware com state_schema
############################################

from langchain.agents.middleware import AgentMiddleware


class PreferencesMiddleware(AgentMiddleware):
    """Middleware que gerencia preferências do usuário e estado customizado."""

    # Define o schema de estado que este middleware usa
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """Hook executado antes de cada chamada ao modelo."""
        # Incrementar contador de interações
        current_count = state.get("interaction_count", 0)

        # Adaptar prompt baseado nas preferências
        preferences = state.get("user_preferences", {})
        style = preferences.get("style", "balanced")
        verbosity = preferences.get("verbosity", "normal")

        print(f"[Middleware] Interação #{current_count + 1}")
        print(f"[Middleware] Estilo preferido: {style}, Verbosidade: {verbosity}")

        # Retornar updates para o estado
        return {
            "interaction_count": current_count + 1
        }


############################################
# PASSO 3 - Definir ferramentas que LEEM
# o estado customizado
############################################

from langchain.tools import tool, ToolRuntime


@tool
def obter_preferencias(runtime: ToolRuntime[CustomState]) -> str:
    """Obter as preferências atuais do usuário."""
    preferences = runtime.state.get("user_preferences", {})

    if not preferences:
        return "Nenhuma preferência definida."

    style = preferences.get("style", "não definido")
    verbosity = preferences.get("verbosity", "não definido")

    return f"Preferências do usuário:\n- Estilo: {style}\n- Verbosidade: {verbosity}"


@tool
def obter_contador_interacoes(runtime: ToolRuntime[CustomState]) -> str:
    """Obter o contador de interações."""
    count = runtime.state.get("interaction_count", 0)
    return f"Número de interações até agora: {count}"


############################################
# PASSO 4 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.5,
    timeout=15,
    max_tokens=1500,
)

############################################
# PASSO 5 - Criar agente com middleware
############################################

from langchain.agents import create_agent

SYSTEM_PROMPT = """
Você é um assistente inteligente que adapta seu estilo de resposta
às preferências do usuário.

Você tem acesso às seguintes ferramentas:
- obter_preferencias: Para verificar as preferências do usuário
- obter_contador_interacoes: Para ver quantas interações já aconteceram

Adapte suas respostas de acordo com as preferências:
- Estilo "técnico": Use termos técnicos e seja preciso
- Estilo "casual": Use linguagem simples e exemplos do dia a dia
- Verbosidade "conciso": Respostas breves e diretas
- Verbosidade "detalhado": Respostas completas com exemplos
"""

# Criar o middleware
preferences_middleware = PreferencesMiddleware()

# Vincular as ferramentas ao middleware
preferences_middleware.tools = [
    obter_preferencias,
    obter_contador_interacoes,
]

# Criar o agente
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    middleware=[preferences_middleware],  # Middleware define o estado customizado
)

############################################
# PASSO 6 - Demonstrar uso com estado
# customizado
############################################

print("=" * 70)
print("EXEMPLO 1 - Iniciar conversa com preferências técnicas")
print("=" * 70)

# Invocar o agente passando estado inicial customizado
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Explique o que é uma API REST"}
    ],
    # Estado customizado inicial
    "user_preferences": {
        "style": "técnico",
        "verbosity": "detalhado"
    },
    "interaction_count": 0,
})

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print(f"Contador de interações: {result.get('interaction_count', 0)}")
print()

print("=" * 70)
print("EXEMPLO 2 - Mudar para estilo casual e conciso")
print("=" * 70)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Agora me explique o que é Docker"}
    ],
    # Mudando as preferências
    "user_preferences": {
        "style": "casual",
        "verbosity": "conciso"
    },
    "interaction_count": result.get("interaction_count", 0),
})

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print(f"Contador de interações: {result.get('interaction_count', 0)}")
print()

print("=" * 70)
print("EXEMPLO 3 - Verificar preferências usando ferramenta")
print("=" * 70)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Quais são minhas preferências configuradas?"}
    ],
    "user_preferences": {
        "style": "casual",
        "verbosity": "conciso"
    },
    "interaction_count": result.get("interaction_count", 0),
})

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print(f"Contador de interações: {result.get('interaction_count', 0)}")
print()

print("=" * 70)
print("EXEMPLO 4 - Verificar contador de interações")
print("=" * 70)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Quantas vezes já conversamos?"}
    ],
    "user_preferences": {
        "style": "casual",
        "verbosity": "conciso"
    },
    "interaction_count": result.get("interaction_count", 0),
})

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print(f"Contador final: {result.get('interaction_count', 0)}")
print()

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. ESTADO CUSTOMIZADO VIA MIDDLEWARE (ABORDAGEM PREFERIDA):
   - Estende AgentState com campos adicionais
   - Mantém escopo organizado: estado + ferramentas juntas
   - Middleware pode acessar e modificar o estado em hooks
   - Melhor separação de responsabilidades

2. VANTAGENS DO MIDDLEWARE:
   - before_model(), after_model() podem acessar o estado
   - Ferramentas e estado relacionados ficam agrupados
   - Facilita reutilização do middleware em outros agentes
   - Escopo conceitual claro (estado é do middleware, não global)

3. ACESSO AO ESTADO:
   - Em ferramentas: runtime.state.get("campo") - SOMENTE LEITURA
   - Em hooks: state.get("campo") - para ler
   - Atualizar estado: retornar dict do hook before_model()
   - O middleware modifica o estado, ferramentas apenas leem

4. CAMPOS DO ESTADO CUSTOMIZADO:
   - user_preferences: Preferências do usuário
   - interaction_count: Contador de interações
   - Você pode adicionar qualquer campo que precisar

5. INICIALIZAÇÃO DO ESTADO:
   - Passe campos customizados em agent.invoke()
   - Se não passar, campos terão valores None/default
   - Estado é passado junto com "messages"

6. DIFERENÇA DO sample017.py:
   - Este usa middleware (mais organizado)
   - sample017.py usa state_schema (mais simples)
   - Middleware é recomendado para projetos complexos

7. PERSISTÊNCIA DO ESTADO:
   - Estado não persiste automaticamente entre invocações
   - Você precisa passar manualmente: result.get("campo")
   - Para persistência automática, use checkpointer (sample008/009)

8. PRÓXIMOS PASSOS:
   - Para abordagem mais simples, veja sample017.py
   - Para combinar com memória, veja sample008.py e sample009.py
""")
