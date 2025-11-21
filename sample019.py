############################################
#
# Exemplo de Uso Direto do Model vs 
# Uso via Agent
#
############################################


############################################
# PASSO 1 - Uso DIRETO do Model (sem agent)
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

# Inicializar o model diretamente
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Invocar o model diretamente (sem agent)
response_direct = model.invoke("Qual é a capital da França?")

print("=" * 70)
print("RESPOSTA DO MODEL DIRETO:")
print("=" * 70)
print(response_direct.content)
print()


############################################
# PASSO 2 - Uso via AGENT (com create_agent)
############################################

from langchain.agents import create_agent

# Criar um agent que usa o mesmo modelo
agent = create_agent(
    "gpt-4o-mini",
    system_prompt="Você é um assistente prestativo.",
)

# Invocar o agent
response_agent = agent.invoke(
    {"messages": [{"role": "user", "content": "Qual é a capital da França?"}]},
)

print("=" * 70)
print("RESPOSTA DO AGENT:")
print("=" * 70)
print(response_agent["messages"][-1].content)
print()


############################################
# PASSO 3 - Comparação: Model com Tools vs Agent com Tools
############################################

from langchain.tools import tool

@tool
def get_current_time() -> str:
    """Retorna a hora atual."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

print("=" * 70)
print("TENTANDO USAR TOOLS COM MODEL DIRETO:")
print("=" * 70)

# Model direto NÃO executa tools automaticamente
# Ele apenas menciona que poderia usar uma ferramenta
response_model_with_tool = model.invoke("Que horas são agora?")
print(response_model_with_tool.content)
print("\n⚠️ Note que o model NÃO executou nenhuma ferramenta,")
print("apenas respondeu com base no conhecimento geral.\n")

print("=" * 70)
print("USANDO TOOLS COM AGENT:")
print("=" * 70)

# Agent EXECUTA tools automaticamente
agent_with_tool = create_agent(
    "gpt-4o-mini",
    system_prompt="Você é um assistente. Use as ferramentas disponíveis quando necessário.",
    tools=[get_current_time],
)

response_agent_with_tool = agent_with_tool.invoke(
    {"messages": [{"role": "user", "content": "Que horas são agora?"}]},
)

print(response_agent_with_tool["messages"][-1].content)
print("\n✓ O agent executou a ferramenta automaticamente!\n")


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. MODEL DIRETO (model.invoke):
   - Acesso direto ao LLM, sem intermediários
   - Retorna uma mensagem simples (AIMessage)
   - NÃO executa tools automaticamente
   - Use quando: respostas simples, sem necessidade de ferramentas
   - Mais controle sobre o comportamento do modelo
   - Acesso direto: response.content

2. AGENT (agent.invoke):
   - Camada de abstração sobre o model
   - Retorna um dicionário com histórico completo de mensagens
   - EXECUTA tools automaticamente em loop (ReAct pattern)
   - Use quando: precisa de ferramentas, raciocínio multi-step
   - Gerencia o loop de execução: think → act → observe → repeat
   - Acesso à resposta: response["messages"][-1].content

3. QUANDO USAR CADA UM:

   USE MODEL DIRETO:
   - Completions simples (tradução, resumo, classificação)
   - Processamento em batch de textos
   - Quando você quer controlar manualmente o loop de tools
   - Estrutura de output customizada (com with_structured_output)
   - Máximo controle sobre o comportamento

   USE AGENT:
   - Quando precisa de ferramentas (APIs, cálculos, buscas)
   - Tarefas que requerem múltiplas etapas de raciocínio
   - Quando o LLM deve decidir quais ferramentas usar
   - Gerenciamento automático de memória e estado
   - Interações conversacionais com contexto

4. DIFERENÇAS NAS ASSINATURAS:
   - model.invoke(input: str | list[dict])
   - agent.invoke(input: dict com chave "messages")

5. TOOLS:
   - Model direto: pode usar bind_tools(), mas VOCÊ controla a execução
   - Agent: executa tools automaticamente quando necessário

6. PRÓXIMOS PASSOS:
   - Para controle manual de tools com model, veja sample020.py
   - Para structured output direto, veja sample021.py
   - Para métodos invoke/stream/batch, veja sample025.py
""")
