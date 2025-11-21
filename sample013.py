############################################

# Exemplo de Agente com Passagem de Sequência
# de Mensagens. Demonstra diferentes formas de
# invocar um agente passando histórico de
# conversa no State, incluindo mensagens únicas
# e múltiplas mensagens com contexto.

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um assistente muito inteligente e prestativo que ajuda com
cálculos e responde perguntas gerais.

Se o usuário mencionar informações de mensagens anteriores, use esse
contexto para fornecer respostas mais relevantes.
"""

############################################
# PASSO 2 - Definir ferramentas
############################################

from langchain.tools import tool


@tool
def calcular_potencia(base: float, expoente: float) -> float:
    """Calcular a potência de um número (base elevada ao expoente)."""
    return base ** expoente


@tool
def converter_temperatura(celsius: float) -> dict:
    """Converter temperatura de Celsius para Fahrenheit e Kelvin."""
    fahrenheit = (celsius * 9/5) + 32
    kelvin = celsius + 273.15
    return {
        "celsius": celsius,
        "fahrenheit": fahrenheit,
        "kelvin": kelvin
    }


############################################
# PASSO 3 - Configurar o modelo
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
# PASSO 4 - Inicializar o agente
############################################

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[calcular_potencia, converter_temperatura],
)

############################################
# PASSO 5 - Demonstrar diferentes formas
# de invocar o agente com mensagens
############################################

print("=" * 70)
print("EXEMPLO 1 - Invocação com mensagem única (formato dict)")
print("=" * 70)

# Forma mais simples: passar uma única mensagem no formato dict
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Olá! Qual é a potência de 2 elevado a 8?"}]}
)

print(result["messages"][-1].content)
print()

print("=" * 70)
print("EXEMPLO 2 - Invocação com sequência de mensagens (histórico)")
print("=" * 70)

# Passar múltiplas mensagens simulando um histórico de conversa
# O agente considera TODAS as mensagens ao gerar a resposta
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Olá! Meu nome é João."},
            {"role": "assistant", "content": "Olá João! Prazer em conhecê-lo. Como posso ajudá-lo hoje?"},
            {"role": "user", "content": "Qual é a temperatura de 25°C em Fahrenheit?"},
            {"role": "assistant", "content": "25°C equivale a 77°F."},
            {"role": "user", "content": "Você se lembra do meu nome?"},
        ]
    }
)

print(result["messages"][-1].content)
print()

print("=" * 70)
print("EXEMPLO 3 - Construindo histórico progressivamente")
print("=" * 70)

# Começar com mensagens iniciais
messages = [
    {"role": "user", "content": "Olá! Estou estudando matemática."},
]

result = agent.invoke({"messages": messages})
print(f"Agente: {result['messages'][-1].content}\n")

# Adicionar a resposta do agente ao histórico
messages.extend([
    {"role": "assistant", "content": result["messages"][-1].content},
    {"role": "user", "content": "Pode calcular 3 elevado a 4 para mim?"},
])

result = agent.invoke({"messages": messages})
print(f"Agente: {result['messages'][-1].content}\n")

# Continuar a conversa
messages.extend([
    {"role": "assistant", "content": result["messages"][-1].content},
    {"role": "user", "content": "Sobre o que eu disse que estava estudando?"},
])

result = agent.invoke({"messages": messages})
print(f"Agente: {result['messages'][-1].content}\n")

print("=" * 70)
print("EXEMPLO 4 - Usando objetos de mensagem do LangChain")
print("=" * 70)

# Além de dicts, você pode usar objetos de mensagem do LangChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Converta 0°C para Fahrenheit e Kelvin."),
        ]
    }
)

print(result["messages"][-1].content)
print()

print("=" * 70)
print("EXEMPLO 5 - Mensagens com contexto rico")
print("=" * 70)

# Passar contexto mais rico para o agente usar
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Estou planejando uma viagem para o Canadá no inverno."},
            {"role": "assistant", "content": "Que legal! O Canadá no inverno é lindo. Posso ajudar com alguma informação?"},
            {"role": "user", "content": "Sim! A temperatura média lá é -10°C. Isso é quanto em Fahrenheit?"},
        ]
    }
)

print(result["messages"][-1].content)
print()

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. DIFERENÇA ENTRE STATE E MEMÓRIA:
   - Este exemplo passa mensagens manualmente no State
   - O agente NÃO tem memória persistente entre invocações
   - Cada invocação é independente - você deve passar todo o histórico

2. QUANDO USAR ESTE PADRÃO:
   - Quando você tem controle total sobre o histórico de mensagens
   - Para testes ou cenários específicos
   - Quando não quer persistência automática

3. ALTERNATIVA COM MEMÓRIA:
   - Para memória automática, veja sample008.py e sample009.py
   - Nesses exemplos, o agente usa MemorySaver/checkpointer
   - Não é necessário passar mensagens antigas manualmente

4. FORMATO DAS MENSAGENS:
   - Dict: {"role": "user", "content": "texto"}
   - role: define se é "user" (usuário), "assistant" (modelo) ou "system" (sistema)
   - content: o texto da mensagem
""")
