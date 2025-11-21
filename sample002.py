############################################

# Exemplo de Agente que usa uma ferramenta

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
# PASSO 3 - Inicializar o agente
############################################

from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

agent = create_agent(
    "gpt-4o-mini",  # o modelo de linguagem a ser usado
    system_prompt=SYSTEM_PROMPT,  # o prompt do sistema definido no Passo 1
    tools=[get_equation_result],  # a ferramenta definida no Passo 2
)

############################################
# PASSO 4 - Usar o agente
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
1. FERRAMENTAS (@tool):
   - Use o decorador @tool para criar ferramentas customizadas
   - A docstring é usada pelo modelo para entender quando usar a ferramenta
   - Os parâmetros tipados ajudam o modelo a fornecer os argumentos corretos

2. INTEGRAÇÃO AUTOMÁTICA:
   - O agente decide automaticamente quando usar a ferramenta
   - Baseado no system_prompt e na pergunta do usuário
   - Não é necessário chamar a ferramenta manualmente

3. SEGURANÇA COM eval():
   - ATENÇÃO: Este exemplo usa eval() apenas para demonstração
   - eval() é PERIGOSO em produção (pode executar código malicioso)
   - Em aplicações reais, use bibliotecas seguras como ast ou sympy

4. COMO O AGENTE DECIDE:
   - O model lê o system_prompt que menciona a ferramenta
   - Quando detecta uma pergunta matemática, chama get_equation_result
   - O resultado da ferramenta é usado para formular a resposta final

5. PRÓXIMOS PASSOS:
   - Para configurar o modelo com mais opções, veja sample003.py e sample004.py
   - Para adicionar contexto às ferramentas, veja sample005.py
   - Para múltiplas ferramentas, veja exemplos posteriores
""")
