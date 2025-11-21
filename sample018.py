############################################

# Exemplo de Agente com Streaming de Respostas.
# Demonstra como usar agent.stream() para
# receber chunks de ESTADOS em tempo real,
# mostrando progresso intermediário durante
# o processamento.
#
# IMPORTANTE: Agents fazem streaming de ESTADOS
# (state chunks), não de tokens individuais.
# Cada chunk representa um passo: pergunta →
# tool_call → ferramenta → resposta final.

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um assistente inteligente que busca informações e faz análises.

Você tem acesso às seguintes ferramentas:
- buscar_informacoes: Para buscar dados sobre um tópico
- calcular_estatisticas: Para calcular média, mínimo e máximo de números
- gerar_relatorio: Para gerar um relatório detalhado sobre um tema

Use as ferramentas quando necessário e forneça respostas completas.
"""

############################################
# PASSO 2 - Definir ferramentas
############################################

from langchain.tools import tool
import time


@tool
def buscar_informacoes(query: str) -> str:
    """Buscar informações sobre um tópico."""
    # Simular processamento com delay
    time.sleep(1)

    # Simular base de conhecimento
    info_db = {
        "python": "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
        "ia": "Inteligência Artificial é um campo da ciência da computação focado em criar sistemas que simulam inteligência humana.",
        "langchain": "LangChain é um framework para desenvolvimento de aplicações com modelos de linguagem.",
    }

    query_lower = query.lower()
    for key, value in info_db.items():
        if key in query_lower:
            return value

    return f"Informação sobre '{query}' não encontrada na base de dados."


@tool
def calcular_estatisticas(numeros: list[float]) -> str:
    """Calcular média, mínimo e máximo de uma lista de números."""
    # Simular processamento
    time.sleep(0.5)

    if not numeros:
        return "Lista vazia fornecida."

    media = sum(numeros) / len(numeros)
    minimo = min(numeros)
    maximo = max(numeros)

    return f"Estatísticas: Média={media:.2f}, Mínimo={minimo}, Máximo={maximo}"


@tool
def gerar_relatorio(topico: str) -> str:
    """Gerar um relatório detalhado sobre um tema."""
    # Simular processamento longo
    time.sleep(1.5)

    return f"""
RELATÓRIO: {topico.upper()}

Resumo: Este é um relatório abrangente sobre {topico}.
Análise: Dados indicam crescimento significativo na área.
Conclusão: Recomenda-se investimento contínuo.
"""


############################################
# PASSO 3 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    timeout=20,
    max_tokens=2000,
)

############################################
# PASSO 4 - Criar o agente
############################################

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        buscar_informacoes,
        calcular_estatisticas,
        gerar_relatorio,
    ],
)

############################################
# PASSO 5 - Demonstrações práticas
############################################

print("=" * 70)
print("EXEMPLO 1 - Comparação: invoke() vs stream()")
print("=" * 70)

print("\nNOTA: Agents fazem streaming de ESTADOS (state chunks), não de tokens.")
print("Cada chunk representa um passo do agente (pergunta → ferramenta → resposta).")
print()

print("[A] Usando invoke() - Aguarda resposta completa:")
print("-" * 70)

import time
start_time = time.time()

result = agent.invoke({
    "messages": [{"role": "user", "content": "Busque informações sobre Python"}]
})

elapsed = time.time() - start_time
print(f"Resposta: {result['messages'][-1].content}")
print(f"Tempo total: {elapsed:.2f}s (tudo processado sem feedback intermediário)")
print()

print("[B] Usando stream() - Mostra progresso em cada passo:")
print("-" * 70)

start_time = time.time()

chunk_count = 0
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Busque informações sobre Python"}]
}, stream_mode="values"):
    chunk_count += 1
    latest_message = chunk["messages"][-1]
    msg_type = getattr(latest_message, 'type', 'unknown')

    # Mostrar cada passo do processo
    if msg_type == 'human':
        print(f"Chunk {chunk_count}: Pergunta do usuário recebida", flush=True)
    elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        print(f"Chunk {chunk_count}: Agente decidiu chamar ferramentas", flush=True)
    elif msg_type == 'tool':
        print(f"Chunk {chunk_count}: Ferramenta executada", flush=True)
    elif latest_message.content and msg_type == 'ai':
        print(f"Chunk {chunk_count}: Resposta final gerada", flush=True)

elapsed = time.time() - start_time
print(f"Tempo total: {elapsed:.2f}s (com {chunk_count} chunks mostrando progresso)")
print()

print("=" * 70)
print("EXEMPLO 2 - Streaming mostrando cada etapa do processo")
print("=" * 70)

print("\nPergunta: Calcule as estatísticas de [15, 25, 35, 45]")
print("Acompanhando cada etapa:")
print("-" * 70)

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Calcule as estatísticas de [15, 25, 35, 45]"}]
}, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    msg_type = getattr(latest_message, 'type', 'unknown')

    # Mostrar cada tipo de mensagem que passa pelo stream
    if msg_type == 'human':
        print(f"📤 Usuário perguntou")
    elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        tools = [tc['name'] for tc in latest_message.tool_calls]
        print(f"🤔 Agente decidiu usar: {', '.join(tools)}")
    elif msg_type == 'tool':
        print(f"⚙️  Ferramenta executada (aguardando...)")
    elif msg_type == 'ai' and latest_message.content:
        print(f"✅ Resposta pronta:")
        print(f"   {latest_message.content[:80]}...")
print()

print("=" * 70)
print("EXEMPLO 3 - Streaming com chamadas de ferramentas")
print("=" * 70)

print("\nPergunta: Busque informações sobre Python e calcule as estatísticas de [10, 20, 30, 40, 50]")
print("Processamento em tempo real:")
print("-" * 70)

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Busque informações sobre Python e calcule as estatísticas de [10, 20, 30, 40, 50]"}]
}, stream_mode="values"):
    latest_message = chunk["messages"][-1]

    # Detectar quando o agente decide chamar ferramentas
    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        tool_names = [tc['name'] for tc in latest_message.tool_calls]
        print(f"🔧 Chamando ferramentas: {', '.join(tool_names)}", flush=True)

    # Mostrar resposta final
    elif latest_message.content and hasattr(latest_message, 'type') and latest_message.type == 'ai':
        print(f"✅ Resposta final:\n{latest_message.content}")

print()

print("=" * 70)
print("EXEMPLO 4 - Análise detalhada dos chunks")
print("=" * 70)

print("\nPergunta: Gere um relatório sobre Inteligência Artificial")
print("Chunks recebidos (estrutura detalhada):")
print("-" * 70)

chunk_count = 0
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Gere um relatório sobre Inteligência Artificial"}]
}, stream_mode="values"):
    chunk_count += 1
    latest_message = chunk["messages"][-1]

    # Mostrar informações sobre cada chunk
    msg_type = getattr(latest_message, 'type', 'unknown')

    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        print(f"Chunk #{chunk_count}: Tipo={msg_type}, Tool Calls={len(latest_message.tool_calls)}")
        for tc in latest_message.tool_calls:
            print(f"  → Ferramenta: {tc['name']}")
    elif latest_message.content:
        content_preview = latest_message.content[:50] + "..." if len(latest_message.content) > 50 else latest_message.content
        print(f"Chunk #{chunk_count}: Tipo={msg_type}, Content='{content_preview}'")

print(f"\nTotal de chunks recebidos: {chunk_count}")
print()

print("=" * 70)
print("EXEMPLO 5 - Stream mode 'values' vs 'updates'")
print("=" * 70)

print("\n[A] Stream mode 'values' (estado completo):")
print("-" * 70)

for i, chunk in enumerate(agent.stream({
    "messages": [{"role": "user", "content": "Calcule as estatísticas de [5, 10, 15]"}]
}, stream_mode="values"), 1):
    total_msgs = len(chunk.get("messages", []))
    print(f"Chunk {i}: Total de mensagens no estado = {total_msgs}")

print()

print("[B] Stream mode 'updates' (apenas mudanças):")
print("-" * 70)

update_count = 0
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Calcule as estatísticas de [5, 10, 15]"}]
}, stream_mode="updates"):
    update_count += 1
    # Updates mostra apenas o que mudou em cada nó do grafo
    # Cada chave é um nó que foi executado
    for node_name, node_data in chunk.items():
        if "messages" in node_data:
            num_msgs = len(node_data["messages"])
            print(f"Update {update_count}: Nó '{node_name}' adicionou {num_msgs} mensagem(s)")
        else:
            print(f"Update {update_count}: Nó '{node_name}' executado")

if update_count == 0:
    print("(Nenhum update recebido - pode variar conforme versão do LangGraph)")

print()

print("=" * 70)
print("EXEMPLO 6 - Indicador de progresso visual")
print("=" * 70)

print("\nPergunta: Busque sobre LangChain e gere um relatório")
print("Progresso:")
print("-" * 70)

import sys

steps = ["Iniciando", "Processando", "Gerando resposta"]
step_index = 0

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Busque sobre LangChain e gere um relatório"}]
}, stream_mode="values"):
    latest_message = chunk["messages"][-1]

    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        # Mostrar indicador de progresso
        tool_names = [tc['name'] for tc in latest_message.tool_calls]
        print(f"⏳ Executando: {', '.join(tool_names)}...", end='', flush=True)
        print(" ✓")

    elif latest_message.content and hasattr(latest_message, 'type') and latest_message.type == 'ai':
        print(f"✅ Concluído!\n")
        # Mostrar apenas parte da resposta
        content_lines = latest_message.content.split('\n')
        print("Resposta (primeiras linhas):")
        for line in content_lines[:3]:
            if line.strip():
                print(f"  {line}")

print()

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. CONCEITO IMPORTANTE - STREAMING DE ESTADOS:
   - Agents fazem streaming de ESTADOS, não de tokens individuais
   - Cada chunk = um PASSO do agente (pergunta → tool_call → resultado → resposta)
   - Diferente do ChatGPT que mostra palavra por palavra (streaming de tokens)
   - Exemplo: Pergunta com ferramenta gera 4 chunks (human → ai tool_call → tool → ai response)

2. INVOKE() VS STREAM():
   - invoke(): Aguarda processamento completo, retorna tudo de uma vez
   - stream(): Retorna chunks incrementais a cada PASSO do processo
   - stream() permite mostrar progresso e melhorar UX

3. STREAM MODES:
   - "values": Cada chunk contém o ESTADO COMPLETO até aquele ponto
   - "updates": Cada chunk contém apenas o que MUDOU
   - "messages": Stream apenas as mensagens (modo legado)
   - Recomendado: Use "values" para maioria dos casos

4. ESTRUTURA DOS CHUNKS:
   - chunk["messages"]: Lista de todas as mensagens até o momento
   - chunk["messages"][-1]: Última mensagem (mais recente)
   - latest_message.content: Texto da resposta (se houver)
   - latest_message.tool_calls: Ferramentas sendo chamadas (se houver)

5. DETECTAR TIPO DE MENSAGEM:
   - latest_message.type == 'ai': Mensagem do assistente
   - latest_message.type == 'tool': Resultado de ferramenta
   - latest_message.type == 'human': Mensagem do usuário
   - hasattr(msg, 'tool_calls'): Verificar se está chamando ferramentas

6. CASOS DE USO PARA STREAMING DE ESTADOS:
   - UI responsiva: Mostrar "pensando..." ou spinner a cada passo
   - Feedback ao usuário: "Buscando informações..." quando tool é chamada
   - Progresso visual: Barra de progresso baseada em etapas
   - Múltiplas ferramentas: Indicar qual ferramenta está sendo executada
   - Log em tempo real: Registrar cada ação do agente

7. PERFORMANCE E UX:
   - Streaming melhora percepção de velocidade
   - Usuário vê progresso imediato a cada passo
   - Especialmente útil com ferramentas lentas (API calls, DB queries)
   - Use flush=True para output imediato no terminal

8. TRATAMENTO EM PRODUÇÃO:
   - Sempre use try/except ao iterar chunks
   - Trate possíveis timeouts ou erros de rede
   - Considere buffer para evitar muitos updates pequenos
   - Implemente retry logic se necessário

9. LIMITAÇÕES E DIFERENÇAS:
   - Streaming de estados ≠ Streaming de tokens
   - Não mostra texto sendo "digitado" palavra por palavra
   - Streaming não reduz tempo total de processamento
   - Apenas melhora experiência do usuário mostrando progresso
   - Chunks podem chegar em ordens diferentes com async

10. STREAMING DE TOKENS (ALTERNATIVA):
    - Para streaming palavra-por-palavra, use model.stream() diretamente
    - Não funciona bem com agents que usam ferramentas
    - Exemplo: model.stream("pergunta") retorna tokens incrementais
    - Agents (create_agent) fazem streaming de ESTADOS, não tokens

11. PRÓXIMOS PASSOS:
   - Para combinar streaming com memória, veja sample008.py
   - Para tratamento de erros, veja sample011.py
   - Para estado customizado, veja sample016.py e sample017.py
""")
