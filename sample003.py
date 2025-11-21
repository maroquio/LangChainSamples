############################################

# Exemplo de Agente com modelo ChatOpenAI
# configurado com parâmetros personalizados
# (temperature, timeout, max_completion_tokens).
# Demonstra o uso de classe específica da
# OpenAI para acessar configurações avançadas.

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
# PASSO 3 - Configurar o modelo específico do ChatGPT
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

# Para outros modelos, veja https://docs.langchain.com/oss/python/integrations/chat
model = ChatOpenAI(
    model="gpt-4o-mini",  # substitua pelo modelo desejado (este é o modo estático de definir o modelo)
    temperature=0.1,  # controle de aleatoriedade, criativo vs. preciso
    timeout=30,  # tempo máximo de resposta em segundos
    max_completion_tokens=1000,  # limite de tokens na resposta
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
1. CHATGPT COM CONFIGURAÇÕES ESPECÍFICAS:
   - ChatOpenAI permite configurações avançadas específicas da OpenAI
   - Use quando precisar de parâmetros específicos do provider
   - Mais controle, mas menos portabilidade entre providers

2. PARÂMETROS IMPORTANTES:
   - temperature: Controla aleatoriedade (0.0 = determinístico, 1.0 = criativo)
   - timeout: Tempo máximo de espera pela resposta em segundos
   - max_completion_tokens: Limite de tokens na resposta (controla custo)

3. TEMPERATURA - QUANDO USAR:
   - 0.0-0.3: Tarefas que exigem precisão (matemática, extração de dados)
   - 0.4-0.7: Uso geral balanceado
   - 0.8-1.0: Tarefas criativas (escrita, brainstorming)

4. DIFERENÇA DO sample004.py:
   - Este usa ChatOpenAI (específico da OpenAI)
   - sample004.py usa init_chat_model (multi-provider)
   - Use ChatOpenAI quando precisar de recursos exclusivos da OpenAI

5. OUTROS PROVIDERS:
   - Para outros modelos, consulte: https://docs.langchain.com/oss/python/integrations/chat
   - Exemplos: ChatAnthropic, ChatCohere, ChatGoogle, etc.

6. PRÓXIMOS PASSOS:
   - Para abordagem multi-provider, veja sample004.py
   - Para contexto personalizado, veja sample005.py
""")
