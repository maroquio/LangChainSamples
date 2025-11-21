############################################
#
# Exemplo de model.bind_tools() - Binding 
# Manual de Tools
#
############################################


############################################
# PASSO 1 - Definir as ferramentas
############################################

from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Retorna a previsão do tempo para uma cidade."""
    # Simulação (em produção, consultaria uma API real)
    weather_data = {
        "São Paulo": "Ensolarado, 28°C",
        "Rio de Janeiro": "Parcialmente nublado, 32°C",
        "Curitiba": "Chuvoso, 18°C",
    }
    return weather_data.get(city, f"Dados não disponíveis para {city}")


@tool
def calculate(expression: str) -> str:
    """Calcula uma expressão matemática."""
    try:
        # Usando eval apenas para demonstração (use ast.literal_eval ou sympy em produção)
        result = eval(expression)
        return f"Resultado: {result}"
    except Exception as e:
        return f"Erro ao calcular: {e}"


############################################
# PASSO 2 - Fazer bind das tools ao model
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Criar o model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Fazer BIND das tools ao model
# Isso instrui o model sobre quais tools estão disponíveis
model_with_tools = model.bind_tools([get_weather, calculate])

# Invocar o model com uma pergunta que requer uma tool
response = model_with_tools.invoke("Qual é o tempo em São Paulo?")

print("=" * 70)
print("RESPOSTA DO MODEL COM BIND_TOOLS:")
print("=" * 70)
print(f"Conteúdo: {response.content}")
print(f"\nTool calls: {response.tool_calls}")
print()


############################################
# PASSO 3 - Loop Manual de Execução de Tools
############################################

from langchain_core.messages import HumanMessage, ToolMessage

print("=" * 70)
print("EXECUTANDO O LOOP MANUAL DE TOOLS:")
print("=" * 70)

# Lista de mensagens para manter o contexto
messages = [HumanMessage(content="Qual é o tempo em São Paulo?")]

# Primeira invocação
response = model_with_tools.invoke(messages)
messages.append(response)

print(f"\n1. Model decidiu chamar tool: {response.tool_calls[0]['name']}")
print(f"   Argumentos: {response.tool_calls[0]['args']}")

# Verificar se há tool_calls para executar
if response.tool_calls:
    for tool_call in response.tool_calls:
        # Identificar qual tool foi chamada
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # EXECUTAR a tool manualmente
        if tool_name == "get_weather":
            tool_result = get_weather.invoke(tool_args)
        elif tool_name == "calculate":
            tool_result = calculate.invoke(tool_args)
        else:
            tool_result = "Tool desconhecida"

        print(f"\n2. Tool executada: {tool_name}")
        print(f"   Resultado: {tool_result}")

        # Adicionar o resultado da tool às mensagens
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
            )
        )

# Segunda invocação com o resultado da tool
final_response = model_with_tools.invoke(messages)
messages.append(final_response)

print(f"\n3. Resposta final do model:")
print(f"   {final_response.content}")
print()


############################################
# PASSO 4 - Exemplo com Múltiplas Tools
############################################

print("=" * 70)
print("EXEMPLO COM DECISÃO ENTRE MÚLTIPLAS TOOLS:")
print("=" * 70)

# Perguntar algo que requer cálculo
messages_calc = [HumanMessage(content="Quanto é 15 * 8 + 42?")]
response_calc = model_with_tools.invoke(messages_calc)

print(f"\nPergunta: 'Quanto é 15 * 8 + 42?'")
print(f"Tool escolhida: {response_calc.tool_calls[0]['name']}")
print(f"Argumentos: {response_calc.tool_calls[0]['args']}")

# Executar a tool
tool_result_calc = calculate.invoke(response_calc.tool_calls[0]["args"])
print(f"Resultado da tool: {tool_result_calc}")

# Completar o loop
messages_calc.append(response_calc)
messages_calc.append(
    ToolMessage(
        content=tool_result_calc,
        tool_call_id=response_calc.tool_calls[0]["id"],
    )
)
final_calc = model_with_tools.invoke(messages_calc)
print(f"Resposta final: {final_calc.content}")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. model.bind_tools():
   - Vincula tools ao model sem usar agent
   - O model DECIDE quando chamar tools (através de tool_calls)
   - VOCÊ é responsável por EXECUTAR as tools manualmente
   - Retorna response.tool_calls com lista de tools a executar

2. LOOP MANUAL DE EXECUÇÃO:
   - Passo 1: Invocar model com a pergunta
   - Passo 2: Verificar se response.tool_calls existe
   - Passo 3: EXECUTAR cada tool manualmente (tool.invoke())
   - Passo 4: Criar ToolMessage com o resultado
   - Passo 5: Invocar model novamente com o resultado
   - Passo 6: Model gera resposta final usando o resultado da tool

3. DIFERENÇA DE AGENT:
   - Agent: executa o loop automaticamente (ReAct pattern)
   - bind_tools: VOCÊ controla cada passo do loop
   - Mais controle, mais código, mais flexibilidade

4. QUANDO USAR bind_tools():
   - Quando precisa controlar a execução de tools
   - Logging customizado de cada chamada
   - Validação de argumentos antes de executar
   - Rate limiting por tool
   - Implementação de retry logic customizado
   - Quando quer entender como agents funcionam "por baixo dos panos"

5. ESTRUTURA DO tool_call:
   {
     "name": "nome_da_tool",
     "args": {"parametro": "valor"},
     "id": "call_abc123",  # ID único da chamada
   }

6. ToolMessage:
   - Informa ao model o resultado da execução
   - tool_call_id: deve corresponder ao ID do tool_call
   - content: string com o resultado da tool

7. PARALLEL TOOL CALLS:
   - O model pode retornar múltiplos tool_calls de uma vez
   - Você pode executá-los em paralelo para melhor performance
   - Útil quando as tools são independentes

8. PRÓXIMOS PASSOS:
   - Para structured output direto, veja sample021.py
   - Para multimodal (imagens), veja sample022.py
   - Para tool_choice control, veja sample032.py
""")
