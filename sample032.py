############################################
#
# Exemplo de Tool Choice Control (Controle de Seleção de Ferramentas)
#
############################################


############################################
# PASSO 1 - Definindo Múltiplas Tools
############################################

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Retorna o tempo atual de uma cidade."""
    weather_data = {
        "São Paulo": "Ensolarado, 28°C",
        "Rio de Janeiro": "Nublado, 32°C",
        "Curitiba": "Chuvoso, 18°C",
    }
    return weather_data.get(city, f"Dados não disponíveis para {city}")


@tool
def calculate(expression: str) -> str:
    """Calcula uma expressão matemática."""
    try:
        result = eval(expression)
        return f"Resultado: {result}"
    except Exception as e:
        return f"Erro: {e}"


@tool
def search_web(query: str) -> str:
    """Busca informações na web."""
    return f"Resultados da busca para '{query}': [simulado]"


tools = [get_weather, calculate, search_web]


############################################
# PASSO 2 - tool_choice="auto" (Padrão)
############################################

print("=" * 70)
print("tool_choice='auto' (PADRÃO - MODEL DECIDE)")
print("=" * 70)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# tool_choice="auto": model decide se e qual tool usar
model_auto = model.bind_tools(tools, tool_choice="auto")

prompts_auto = [
    "Qual é o tempo em São Paulo?",
    "Quanto é 15 * 8?",
    "Qual é a capital da França?",  # Não precisa de tool
]

print("\nCom tool_choice='auto' (model decide):\n")

for prompt in prompts_auto:
    response = model_auto.invoke(prompt)

    print(f"Prompt: '{prompt}'")

    if response.tool_calls:
        tool_used = response.tool_calls[0]["name"]
        print(f"  → Model decidiu usar: {tool_used}")
    else:
        print(f"  → Model decidiu NÃO usar tools")
        print(f"  → Resposta direta: {response.content}")

    print()


############################################
# PASSO 3 - tool_choice="any" (Força Usar Alguma Tool)
############################################

print("=" * 70)
print("tool_choice='any' (FORÇA USAR ALGUMA TOOL)")
print("=" * 70)

# tool_choice="any": força o model a usar pelo menos uma tool
# (OpenAI usa "required" para este comportamento)
model_any = model.bind_tools(
    tools,
    tool_choice="any",  # ou "required" dependendo do provider
)

prompts_any = [
    "Olá!",  # Normalmente não precisaria de tool
    "Qual é a capital da França?",  # Também não precisaria
]

print("\nCom tool_choice='any' (forçar uso de tool):\n")

for prompt in prompts_any:
    response = model_any.invoke(prompt)

    print(f"Prompt: '{prompt}'")

    if response.tool_calls:
        tool_used = response.tool_calls[0]["name"]
        args = response.tool_calls[0]["args"]
        print(f"  → Model foi FORÇADO a usar tool: {tool_used}")
        print(f"  → Argumentos: {args}")
    else:
        print(f"  → Nenhuma tool usada (inesperado com 'any')")

    print()


############################################
# PASSO 4 - tool_choice com Nome Específico (Força Tool Específica)
############################################

print("=" * 70)
print("tool_choice='{nome}' (FORÇA TOOL ESPECÍFICA)")
print("=" * 70)

# Forçar uso de uma tool específica
model_specific = model.bind_tools(
    tools,
    tool_choice={"type": "function", "function": {"name": "calculate"}},
)

prompts_specific = [
    "Olá!",
    "Quanto é 10 + 5?",
    "Qual é o tempo em São Paulo?",  # Pediria get_weather, mas será forçado calculate
]

print("\nCom tool_choice forçando 'calculate':\n")

for prompt in prompts_specific:
    response = model_specific.invoke(prompt)

    print(f"Prompt: '{prompt}'")

    if response.tool_calls:
        tool_used = response.tool_calls[0]["name"]
        args = response.tool_calls[0]["args"]
        print(f"  → Tool usada: {tool_used}")
        print(f"  → Argumentos: {args}")

        if tool_used != "calculate":
            print(f"  ⚠️ Esperava 'calculate', mas model usou '{tool_used}'")
    else:
        print(f"  → Nenhuma tool usada")

    print()


############################################
# PASSO 5 - tool_choice="none" (Nunca Usar Tools)
############################################

print("=" * 70)
print("tool_choice='none' (NUNCA USAR TOOLS)")
print("=" * 70)

# tool_choice="none": model nunca usa tools (mesmo com tools disponíveis)
model_none = model.bind_tools(tools, tool_choice="none")

prompts_none = [
    "Qual é o tempo em São Paulo?",  # Tem tool, mas não vai usar
    "Quanto é 5 + 5?",  # Tem tool, mas não vai usar
]

print("\nCom tool_choice='none' (tools disponíveis mas nunca usadas):\n")

for prompt in prompts_none:
    response = model_none.invoke(prompt)

    print(f"Prompt: '{prompt}'")

    if response.tool_calls:
        print(f"  → Tool usada: {response.tool_calls[0]['name']} (inesperado!)")
    else:
        print(f"  → Nenhuma tool usada (esperado)")
        print(f"  → Resposta: {response.content}")

    print()


############################################
# PASSO 6 - Parallel Tool Calls
############################################

print("=" * 70)
print("PARALLEL TOOL CALLS (CHAMADAS PARALELAS)")
print("=" * 70)

model_parallel = model.bind_tools(tools, tool_choice="auto")

# Prompt que pode se beneficiar de múltiplas tools
prompt_parallel = "Qual é o tempo em São Paulo e quanto é 10 * 5?"

response = model_parallel.invoke(prompt_parallel)

print(f"Prompt: '{prompt_parallel}'\n")

if response.tool_calls:
    print(f"Model decidiu usar {len(response.tool_calls)} tool(s):\n")

    for i, tool_call in enumerate(response.tool_calls, 1):
        print(f"  {i}. Tool: {tool_call['name']}")
        print(f"     Args: {tool_call['args']}")
        print(f"     ID: {tool_call['id']}\n")

    print("✓ Parallel tool calls permitem múltiplas tools em uma única resposta!")
else:
    print("Nenhuma tool usada.")

print()


############################################
# PASSO 7 - Casos de Uso por Opção
############################################

print("=" * 70)
print("CASOS DE USO POR OPÇÃO DE tool_choice")
print("=" * 70)

print("""
1. tool_choice="auto" (PADRÃO):
   USE QUANDO:
   - Quer que o model decida naturalmente
   - Caso de uso geral (conversação, Q&A)
   - Tools são opcionais, não obrigatórias
   - Melhor UX (model só usa quando necessário)

   EXEMPLO:
   - Chatbot com ferramentas auxiliares
   - Assistant que pode ou não precisar de dados externos

2. tool_choice="any" ou "required":
   USE QUANDO:
   - SEMPRE precisa de uma tool para responder
   - Forçar model a usar recurso externo
   - Garantir que não responda sem consultar tool
   - Workflow onde tool é obrigatória

   EXEMPLO:
   - Sistema que SEMPRE deve buscar dados atualizados
   - RAG obrigatório (sempre retrieve antes de responder)
   - API wrapper (toda resposta precisa de API call)

3. tool_choice={"type": "function", "function": {"name": "..."}":
   USE QUANDO:
   - Quer FORÇAR uma tool específica
   - Fluxo determinístico (sem decisão do model)
   - Testing/debugging de uma tool específica
   - Garantir que apenas uma tool seja usada

   EXEMPLO:
   - Forçar busca no banco de dados
   - Garantir uso de calculadora para math
   - Pipeline com steps fixos

4. tool_choice="none":
   USE QUANDO:
   - Quer DESABILITAR tools temporariamente
   - Resposta deve ser baseada apenas em conhecimento
   - Testing sem side effects
   - Economizar custos (tool calls são mais caros)

   EXEMPLO:
   - Modo "knowledge only"
   - Fallback quando tools estão indisponíveis
   - Testing de prompts sem executar tools
""")
print()


############################################
# PASSO 8 - Combinando tool_choice com Lógica Customizada
############################################

print("=" * 70)
print("LÓGICA CUSTOMIZADA COM tool_choice")
print("=" * 70)

def smart_tool_choice(user_input: str):
    """Decide tool_choice baseado no input do usuário."""
    if "tempo" in user_input.lower() or "weather" in user_input.lower():
        # Forçar get_weather
        return {"type": "function", "function": {"name": "get_weather"}}
    elif any(op in user_input for op in ["+", "-", "*", "/", "="]):
        # Forçar calculate
        return {"type": "function", "function": {"name": "calculate"}}
    else:
        # Deixar model decidir
        return "auto"


test_inputs = [
    "Como está o tempo hoje?",
    "Quanto é 25 * 4?",
    "Qual é a capital do Brasil?",
]

print("\nLógica customizada de tool_choice:\n")

for user_input in test_inputs:
    choice = smart_tool_choice(user_input)
    model_smart = model.bind_tools(tools, tool_choice=choice)

    response = model_smart.invoke(user_input)

    print(f"Input: '{user_input}'")
    print(f"  Choice strategy: {choice}")

    if response.tool_calls:
        print(f"  Tool usada: {response.tool_calls[0]['name']}")
    else:
        print(f"  Sem tools (resposta direta)")

    print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. OPÇÕES DE tool_choice:

   "auto" (padrão):
   - Model decide se e qual tool usar
   - Comportamento natural e flexível
   - Pode não usar nenhuma tool

   "any" ou "required" (varia por provider):
   - Força uso de pelo menos UMA tool
   - Model escolhe qual
   - Garante que sempre use recurso externo

   {"type": "function", "function": {"name": "..."}}:
   - Força uso de tool ESPECÍFICA
   - Determinístico, sem decisão do model
   - Útil para workflows fixos

   "none":
   - Desabilita tools completamente
   - Model nunca usa, mesmo se disponível
   - Útil para fallback ou testing

2. PARALLEL TOOL CALLS:
   - Model pode chamar múltiplas tools de uma vez
   - Mais eficiente que sequential
   - Útil quando tools são independentes
   - Nem todos os providers suportam

3. DIFERENÇAS POR PROVIDER:

   OpenAI:
   - "auto", "none", "required"
   - Suporta parallel tool calls
   - Permite especificar tool por nome

   Anthropic:
   - "auto", "any", "tool"
   - Sintaxe pode variar
   - Consulte documentação

   Google:
   - Suporte varia por model
   - Verifique docs atualizadas

4. CUSTOS:
   - Tool calls geralmente custam mais tokens
   - tool_choice="none" economiza (sem tool calls)
   - tool_choice="required" pode aumentar custo
   - Considere custo vs necessidade

5. QUANDO FORÇAR TOOLS:
   - Dados devem estar sempre atualizados
   - Compliance (audit trail de tool usage)
   - Evitar hallucinations (força consulta externa)
   - Workflows determinísticos

6. QUANDO NÃO FORÇAR:
   - Conversação natural
   - Quando knowledge interno é suficiente
   - Economizar custos
   - Reduzir latência

7. DEBUGGING:
   - Use tool_choice específico para testar tools individualmente
   - tool_choice="none" para testar sem side effects
   - Verifique response.tool_calls para ver decisão do model

8. BOAS PRÁTICAS:
   - Use "auto" como padrão (melhor UX)
   - Use "required" apenas quando realmente necessário
   - Considere lógica customizada (como no Passo 8)
   - Monitore uso de tools (custos e latência)
   - Teste todas as opções de tool_choice

9. ERROS COMUNS:
   - Forçar tool quando não há dados suficientes
   - Não tratar casos onde tool falha
   - Esquecer que tool_choice="required" sempre chama tool
   - Não considerar custo de tool calls

10. PRÓXIMOS PASSOS:
    - Para bind_tools manual, veja sample020.py
    - Para structured output, veja sample021.py
    - Para agents (tool execution automático), veja sample019.py
""")
