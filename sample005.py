############################################

# Exemplo de Agente com contexto de runtime
# personalizado (ToolRuntime). Demonstra como
# injetar contexto customizado nas ferramentas
# para acessar informações específicas durante
# a execução (ex: user_role, user_id, etc.).

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um especialista em investimentos muito inteligente e prestativo.
Se um usuário perguntar sobre o seu percentual de retorno de um investimento,
use a ferramenta get_investment_return para calcular o resultado.
"""

############################################
# PASSO 2 - Configurar o esquema do contexto
############################################

from dataclasses import dataclass


@dataclass
class Context:
    user_role: str

############################################
# PASSO 3 - Definir as ferramentas
############################################

from langchain.tools import tool, ToolRuntime


@tool
# Injeta o contexto de runtime personalizado para acessar informações do usuário
def get_investment_return(investment_value: float, runtime: ToolRuntime[Context]) -> str:
    """Calcular o percentual de retorno de um investimento com base no perfil do usuário."""
    user_role = runtime.context.user_role
    # Lógica personalizada com base no user_role
    match user_role:
        case "conservative":
            multiplier = 1.05
        case "balanced":
            multiplier = 1.10
        case "aggressive":
            multiplier = 1.20
        case _:
            multiplier = 1.0
    return f"O retorno do investimento será de {multiplier * investment_value}."


############################################
# PASSO 4 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

# init_chat_model identifica e inicializa o modelo de chat da classe apropriada
model = init_chat_model(
    "gpt-4o-mini",  # substitua pelo modelo desejado
    temperature=0.5,  # controle de aleatoriedade, criativo vs. preciso
    timeout=10,  # tempo máximo de resposta em segundos
    max_tokens=1000,  # limite de tokens na resposta
)


############################################
# PASSO 5 - Inicializar o agente
############################################

from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

agent = create_agent(
    model=model,  # o modelo de linguagem a ser usado
    system_prompt=SYSTEM_PROMPT,  # o prompt do sistema definido no Passo 1
    tools=[get_investment_return],  # a ferramenta definida no Passo 2
)

############################################
# PASSO 6 - Usar o agente com contexto personalizado
############################################

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Vou investir R$ 1.000,00. Qual será meu retorno?"}]},
    context=Context(user_role="aggressive"),
    # O perfil do usuário poderia ser obtido de um banco de dados ou sistema de autenticação
)

print(response["messages"][-1].content)

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print()
print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. CONTEXTO PERSONALIZADO (ToolRuntime):
   - Permite injetar informações específicas nas ferramentas
   - Útil para: user_id, permissões, preferências, configurações
   - Acessível via runtime.context dentro das ferramentas

2. DEFININDO O CONTEXTO:
   - Crie uma classe (dataclass ou TypedDict) com os dados necessários
   - Passe o contexto em agent.invoke(context=...)
   - Ferramentas acessam via: runtime: ToolRuntime[Context]

3. CASOS DE USO:
   - Personalização por usuário (como neste exemplo)
   - Controle de acesso e permissões
   - Configurações específicas da sessão
   - Informações de autenticação

4. TIPAGEM GENÉRICA:
   - ToolRuntime[Context] fornece type hints corretos
   - IDE pode autocompletar runtime.context.user_role
   - Ajuda a evitar erros em tempo de desenvolvimento

5. SEPARAÇÃO DE RESPONSABILIDADES:
   - System prompt: comportamento geral do agente
   - Context: dados específicos da execução
   - Ferramentas: lógica de negócio que usa o contexto

6. PRÓXIMOS PASSOS:
   - Para contexto + resposta estruturada, veja sample006.py
   - Para memória entre conversas, veja sample008.py
""")
