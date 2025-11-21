############################################

# Exemplo de Agente com Estado Customizado
# definido via state_schema (ABORDAGEM SIMPLES).
# Demonstra como usar state_schema diretamente
# no create_agent para rastrear informações
# adicionais, sem necessidade de middleware.

############################################


############################################
# PASSO 1 - Definir o estado customizado
# estendendo AgentState
############################################

from langchain.agents import AgentState


class CustomState(AgentState):
    """Estado customizado que estende AgentState com campos adicionais."""

    # Código do pedido
    order_id: str | None

    # Nome do cliente
    customer_name: str | None


############################################
# PASSO 2 - Definir ferramentas que LEEM
# o estado customizado
############################################

from langchain.tools import tool, ToolRuntime


@tool
def obter_info_pedido(runtime: ToolRuntime[CustomState]) -> str:
    """Obter informações do pedido atual."""
    order_id = runtime.state.get("order_id")
    customer_name = runtime.state.get("customer_name")

    if not order_id:
        return "Nenhum pedido ativo no momento."

    result = f"Informações do Pedido:\n"
    result += f"- ID do Pedido: {order_id}\n"
    result += f"- Cliente: {customer_name or 'Não informado'}\n"

    return result


@tool
def verificar_status_pedido(pedido_id: str, runtime: ToolRuntime[CustomState]) -> str:
    """Verificar o status de um pedido."""
    # Simular consulta de status
    statuses = {
        "PED001": "Em preparação",
        "PED002": "Saiu para entrega",
        "PED003": "Entregue",
    }

    status = statuses.get(pedido_id, "Pedido não encontrado")

    # Verificar se é o pedido do cliente atual
    current_order = runtime.state.get("order_id")
    is_current = " (Seu pedido atual)" if pedido_id == current_order else ""

    return f"Status do pedido {pedido_id}{is_current}: {status}"


@tool
def calcular_desconto(valor: float, runtime: ToolRuntime[CustomState]) -> str:
    """Calcular desconto baseado no cliente."""
    customer = runtime.state.get("customer_name")

    # Clientes VIP têm 15% de desconto
    vip_customers = ["João Silva", "Maria Santos"]

    if customer in vip_customers:
        desconto_percentual = 15
        valor_desconto = valor * 0.15
        valor_final = valor - valor_desconto
        return f"Cliente VIP: {desconto_percentual}% de desconto!\nValor original: R$ {valor:.2f}\nDesconto: R$ {valor_desconto:.2f}\nValor final: R$ {valor_final:.2f}"
    else:
        return f"Valor: R$ {valor:.2f}\nSem desconto disponível para este cliente."


############################################
# PASSO 3 - Configurar o modelo
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
# PASSO 4 - Criar agente com state_schema
############################################

from langchain.agents import create_agent

SYSTEM_PROMPT = """
Você é um assistente de atendimento ao cliente para uma loja online.

Você tem acesso às seguintes ferramentas:
- obter_info_pedido: Para mostrar informações do pedido atual
- verificar_status_pedido: Para consultar o status de um pedido
- calcular_desconto: Para calcular descontos baseado no cliente

Ajude os clientes de forma amigável e eficiente.
"""

# Criar o agente com state_schema (abordagem simples)
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        obter_info_pedido,
        verificar_status_pedido,
        calcular_desconto,
    ],
    state_schema=CustomState,  # Define o estado customizado diretamente
)

############################################
# PASSO 5 - Demonstrar uso com estado
# customizado
############################################

print("=" * 70)
print("EXEMPLO 1 - Cliente VIP consultando pedido")
print("=" * 70)

# Invocar o agente passando estado inicial customizado
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Qual o status do meu pedido PED001?"}
        ],
        # Estado customizado inicial
        "order_id": "PED001",
        "customer_name": "João Silva",
    }
)

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print(f"Cliente: {result.get('customer_name')}")
print(f"Pedido: {result.get('order_id')}")
print()

print("=" * 70)
print("EXEMPLO 2 - Calcular desconto para cliente VIP")
print("=" * 70)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Quanto ficaria uma compra de R$ 100,00?"}
        ],
        "order_id": "PED001",
        "customer_name": "João Silva",
    }
)

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print()

print("=" * 70)
print("EXEMPLO 3 - Cliente regular sem desconto")
print("=" * 70)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Quanto ficaria uma compra de R$ 100,00?"}
        ],
        "order_id": "PED002",
        "customer_name": "Pedro Costa",
    }
)

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print(f"Cliente: {result.get('customer_name')}")
print()

print("=" * 70)
print("EXEMPLO 4 - Obter informações do pedido atual")
print("=" * 70)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Mostre as informações do meu pedido"}
        ],
        "order_id": "PED003",
        "customer_name": "Maria Santos",
    }
)

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print()

print("=" * 70)
print("EXEMPLO 5 - Cliente sem pedido ativo")
print("=" * 70)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Qual é o meu pedido?"}],
        "order_id": None,
        "customer_name": "Ana Lima",
    }
)

print(f"\nResposta do agente:\n{result['messages'][-1].content}\n")
print()

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print(
    """
1. ESTADO CUSTOMIZADO VIA STATE_SCHEMA (ABORDAGEM SIMPLES):
   - Passa state_schema diretamente para create_agent()
   - Não requer criação de classe AgentMiddleware
   - Mais simples e direto para casos básicos
   - Sem necessidade de hooks before_model/after_model

2. DIFERENÇA DO MIDDLEWARE (sample016.py):
   - Middleware: Permite hooks e lógica customizada
   - state_schema: Apenas define campos adicionais
   - Use middleware quando precisar de lógica em hooks
   - Use state_schema quando só precisar de campos extras

3. ACESSO AO ESTADO:
   - Em ferramentas: runtime.state.get("campo") - SOMENTE LEITURA
   - Estado é gerenciado pelo framework, não pelas ferramentas
   - Ferramentas apenas leem o estado para personalizar comportamento

4. CAMPOS DO ESTADO CUSTOMIZADO:
   - order_id: Código do pedido
   - customer_name: Nome do cliente
   - Você pode adicionar qualquer campo que precisar

5. INICIALIZAÇÃO DO ESTADO:
   - Passe campos customizados em agent.invoke()
   - Se não passar, campos terão valores None/default
   - Estado é passado junto com "messages"

6. QUANDO USAR state_schema:
   - Projetos simples sem necessidade de hooks
   - Apenas precisa armazenar dados adicionais
   - Quer menos código boilerplate
   - Não precisa de lógica em before_model/after_model

7. QUANDO USAR MIDDLEWARE (sample016.py):
   - Precisa de hooks para processar estado
   - Lógica complexa de transformação
   - Múltiplas ferramentas relacionadas agrupadas
   - Reutilização em múltiplos agentes

8. PERSISTÊNCIA DO ESTADO:
   - Estado não persiste automaticamente entre invocações
   - Você precisa passar manualmente: result.get("campo")
   - Para persistência automática, use checkpointer (sample008/009)

9. CASOS DE USO REAIS:
   - Informações de sessão do usuário
   - Contexto de atendimento ao cliente
   - Preferências temporárias
   - Dados de transação em andamento

10. PRÓXIMOS PASSOS:
    - Para abordagem com middleware, veja sample016.py
    - Para combinar com memória persistente, veja sample008.py
    - Para múltiplas conversas, veja sample009.py
"""
)
