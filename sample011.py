############################################

# Exemplo de Agente com Tratamento de Erros
# em Ferramentas usando Middleware. Este
# exemplo demonstra como capturar e tratar
# erros que ocorrem durante a execução de
# ferramentas, retornando mensagens amigáveis
# ao modelo para que ele possa responder
# adequadamente ao usuário.

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um especialista em matemática e análise de dados muito inteligente e prestativo.

Você tem acesso às seguintes ferramentas:
- divide_numbers: para fazer divisões
- calculate_square_root: para calcular raiz quadrada
- get_user_age: para obter a idade de um usuário

Se uma ferramenta retornar um erro, explique ao usuário de forma amigável
o que aconteceu e sugira uma alternativa se possível.
"""

############################################
# PASSO 2 - Definir ferramentas que podem
# gerar erros
############################################

from langchain.tools import tool


@tool
def divide_numbers(dividend: float, divisor: float) -> float:
    """Dividir dois números. Pode gerar erro se o divisor for zero."""
    # Esta operação pode gerar ZeroDivisionError
    result = dividend / divisor
    return result


@tool
def calculate_square_root(number: float) -> float:
    """Calcular a raiz quadrada de um número. Pode gerar erro se o número for negativo."""
    # Esta operação pode gerar ValueError para números negativos
    if number < 0:
        raise ValueError("Não é possível calcular raiz quadrada de número negativo em números reais")
    return number ** 0.5


@tool
def get_user_age(user_id: str) -> int:
    """Obter a idade de um usuário pelo ID. Pode gerar erro se o usuário não existir."""
    # Simulando um banco de dados de usuários
    users_database = {
        "user_001": 25,
        "user_002": 34,
        "user_003": 42,
    }
    # Esta operação pode gerar KeyError se o usuário não existir
    return users_database[user_id]


############################################
# PASSO 3 - Criar middleware de tratamento
# de erros
############################################

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Intercepta erros na execução de ferramentas e retorna mensagens amigáveis."""
    try:
        # Tenta executar a ferramenta normalmente
        return handler(request)
    except ZeroDivisionError:
        # Tratamento específico para divisão por zero
        return ToolMessage(
            content="Erro: Não é possível dividir por zero. Por favor, forneça um divisor diferente de zero.",
            tool_call_id=request.tool_call["id"]
        )
    except ValueError as e:
        # Tratamento para erros de valor (ex: raiz quadrada de negativo)
        return ToolMessage(
            content=f"Erro de valor: {str(e)}. Não é possível calcular a raiz quadrada de um número negativo em números reais.",
            tool_call_id=request.tool_call["id"]
        )
    except KeyError:
        # Tratamento para chaves não encontradas (ex: usuário inexistente)
        return ToolMessage(
            content="Erro: Usuário não encontrado no sistema. Verifique o ID fornecido.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        # Tratamento genérico para outros erros
        return ToolMessage(
            content=f"Erro inesperado: {str(e)}. Por favor, tente novamente ou reformule sua pergunta.",
            tool_call_id=request.tool_call["id"]
        )


############################################
# PASSO 4 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.3,
    timeout=10,
    max_tokens=1000,
)

############################################
# PASSO 5 - Inicializar agente com middleware
############################################

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        divide_numbers,
        calculate_square_root,
        get_user_age,
    ],
    middleware=[handle_tool_errors],  # Middleware que trata erros de ferramentas
)

############################################
# PASSO 6 - Testar casos de sucesso e erro
############################################

print("=" * 60)
print("TESTE 1 - Divisão normal (sucesso esperado)")
print("=" * 60)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Quanto é 10 dividido por 2?"}]},
)

print(response["messages"][-1].content)
print()

print("=" * 60)
print("TESTE 2 - Divisão por zero (erro tratado)")
print("=" * 60)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Quanto é 10 dividido por 0?"}]},
)

print(response["messages"][-1].content)
print()

print("=" * 60)
print("TESTE 3 - Raiz quadrada de número negativo (erro tratado)")
print("=" * 60)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Qual é a raiz quadrada de -16?"}]},
)

print(response["messages"][-1].content)
print()

print("=" * 60)
print("TESTE 4 - Buscar usuário inexistente (erro tratado)")
print("=" * 60)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Qual é a idade do usuário user_999?"}]},
)

print(response["messages"][-1].content)
print()

print("=" * 60)
print("TESTE 5 - Buscar usuário existente (sucesso esperado)")
print("=" * 60)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Qual é a idade do usuário user_002?"}]},
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
1. MIDDLEWARE - TRATAMENTO DE ERROS:
   - @wrap_tool_call intercepta execução de ferramentas
   - Captura exceções e retorna mensagens amigáveis
   - Permite que o agente responda adequadamente aos erros

2. COMO FUNCIONA:
   - Try/except dentro do middleware
   - Captura erros específicos (ZeroDivisionError, ValueError, KeyError)
   - Retorna ToolMessage com mensagem de erro ao invés de crash

3. TOOLMESSAGE:
   - Formato específico para respostas de ferramentas
   - Contém content (mensagem) e tool_call_id (identificador)
   - O modelo recebe o erro como resultado da ferramenta

4. VANTAGENS:
   - Aplicação não quebra com erros em ferramentas
   - Usuário recebe explicação clara do problema
   - Modelo pode sugerir alternativas ou correções

5. TIPOS DE ERROS TRATADOS:
   - ZeroDivisionError: Divisão por zero
   - ValueError: Valores inválidos (ex: raiz de negativo)
   - KeyError: Chaves não encontradas (ex: ID inexistente)
   - Exception: Fallback para erros inesperados

6. TRATAMENTO GRANULAR:
   - Cada tipo de erro tem mensagem específica
   - Ajuda o modelo a entender exatamente o que deu errado
   - Permite respostas mais úteis ao usuário

7. ALTERNATIVAS:
   - Tratamento dentro da própria ferramenta
   - Validação de input antes de executar
   - Middleware é mais centralizado e reutilizável

8. EM PRODUÇÃO:
   - Adicione logging dos erros
   - Considere retry para erros transientes
   - Monitore taxa de erro por ferramenta

9. PRÓXIMOS PASSOS:
   - Para prompts dinâmicos, veja sample012.py
   - Para passar mensagens manualmente, veja sample013.py
   - Para structured output correto, veja sample014.py e sample015.py
""")
