############################################

# Exemplo de Agente com contexto personalizado
# e formato de resposta estruturado usando
# ResponseFormat. Demonstra como combinar
# contexto de runtime com output estruturado
# em formato customizado (dataclass).

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
def get_investment_return(
    investment_value: float, runtime: ToolRuntime[Context]
) -> str:
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
# PASSO 5 - Definir um formato de resposta
############################################

from dataclasses import dataclass


# Pode-se usar dataclass aqui, mas modelos Pydantic também são suportados.
@dataclass
class ResponseFormat:
    """Esquema de resposta para o agente."""

    # O valor total do investimento mais o lucro
    investiment_return: float
    # O valor do lucro obtido
    investiment_profit: float | None = None
    # O percentual de retorno aplicado
    return_percentage: float | None = None


############################################
# PASSO 6 - Inicializar o agente com formato
# de resposta personalizado
############################################

from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_investment_return],
    response_format=ResponseFormat,  # o formato de resposta definido no Passo 5
)

############################################
# PASSO 7 - Usar o agente com contexto 
# personalizado
############################################

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Vou investir R$ 1.000,00. Qual será meu retorno?",
            }
        ]
    },
    context=Context(user_role="aggressive"),
    # O perfil do usuário poderia ser obtido de um banco de dados ou sistema de autenticação
)

formatted_response = response["structured_response"]

print(f"Valor total: {formatted_response.investiment_return}")
print(f"Lucro: {formatted_response.investiment_profit}")
print(f"Percentual de retorno: {formatted_response.return_percentage}%")

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print()
print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. RESPOSTA ESTRUTURADA (response_format):
   - Define um schema para a resposta do agente
   - Garante que a saída tenha um formato previsível
   - Pode usar dataclass ou Pydantic BaseModel

2. ACESSO À RESPOSTA ESTRUTURADA:
   - Use result["structured_response"] ao invés de result["messages"]
   - Retorna uma instância do schema definido (ResponseFormat)
   - Permite acesso tipado: formatted_response.investiment_return

3. COMBINAÇÃO PODEROSA:
   - Context (ToolRuntime): Entrada personalizada para ferramentas
   - response_format: Saída estruturada do agente
   - Permite controle completo de entrada e saída

4. DATACLASS VS PYDANTIC:
   - Ambos funcionam para response_format
   - Pydantic: mais validação, mais recursos
   - Dataclass: mais simples, nativo do Python

5. ATENÇÃO - LANÇAMENTO 1.0:
   - A partir do LangChain 1.0, deve-se usar ToolStrategy ou ProviderStrategy
   - Em códigos antigos, passa-se o schema diretamente (deprecated)
   - Para a abordagem correta, veja sample014.py e sample015.py

6. CAMPOS OPCIONAIS:
   - Use | None e valor default para campos opcionais
   - Exemplo: investiment_profit: float | None = None
   - O modelo decide se preenche ou não

7. PRÓXIMOS PASSOS:
   - Para exemplos SEM memória, veja sample007.py
   - Para memória com checkpointer, veja sample008.py
   - Para structured output correto (LangChain 1.0+), veja sample014.py e sample015.py
""")