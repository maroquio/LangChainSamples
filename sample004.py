############################################

# Exemplo de Agente com modelo multi-provedor
# usando init_chat_model e configuração
# personalizada. Demonstra inicialização de
# modelo compatível com qualquer fornecedor
# (OpenAI, Anthropic, etc.).

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um especialista em matemática.
Se um usuário perguntar sobre o resultado de uma equação, use a ferramenta get_equation_result para calcular o resultado.
"""

############################################
# PASSO 2 - Definir as ferramentas
############################################

from langchain.tools import tool


@tool
def get_equation_result(equation: str) -> str:
    """Calcular o resultado de uma equação matemática."""
    try:
        result = eval(equation)
        return f"O resultado de {equation} é {result}."
    except Exception as e:
        return f"Erro ao calcular a equação: {e}"


############################################
# PASSO 3 - Configurar o modelo
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
# PASSO 4 - Inicializar o agente
############################################

from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

agent = create_agent(
    model=model,  # o modelo de linguagem a ser usado
    system_prompt=SYSTEM_PROMPT,  # o prompt do sistema definido no Passo 1
    tools=[get_equation_result],  # a ferramenta definida no Passo 2
)

############################################
# PASSO 5 - Usar o agente
############################################

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Qual é o resultado de 12 * 8 + 5?"}]},
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
1. INIT_CHAT_MODEL - MULTI-PROVIDER:
   - init_chat_model() é a abordagem recomendada no LangChain 1.0+
   - Identifica automaticamente o provider pelo nome do modelo
   - Permite trocar de provider facilmente (OpenAI, Anthropic, etc.)

2. PORTABILIDADE:
   - Código funciona com qualquer provider suportado
   - Basta mudar o nome do modelo (ex: "gpt-4o-mini" → "claude-3-sonnet")
   - Configurações padrão (temperature, timeout, etc.) funcionam em todos

3. PARÂMETROS COMUNS:
   - temperature: Controla criatividade (0.0 = preciso, 1.0 = criativo)
   - timeout: Tempo máximo de espera em segundos
   - max_tokens: Limite de tokens (diferente de max_completion_tokens)

4. DIFERENÇA DO sample003.py:
   - Este usa init_chat_model (recomendado, multi-provider)
   - sample003.py usa ChatOpenAI (específico OpenAI, mais controle)

5. QUANDO USAR init_chat_model:
   - Quando quiser portabilidade entre providers
   - Para projetos que podem mudar de modelo no futuro
   - Abordagem padrão recomendada pelo LangChain

6. QUANDO USAR CLASSE ESPECÍFICA (ChatOpenAI):
   - Quando precisar de parâmetros exclusivos do provider
   - Para funcionalidades avançadas específicas
   - Veja sample003.py para exemplo

7. PRÓXIMOS PASSOS:
   - Para contexto personalizado, veja sample005.py
   - Para formato de resposta estruturado, veja sample006.py
""")
