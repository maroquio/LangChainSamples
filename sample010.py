############################################

# Exemplo de Agente com Seleção Dinâmica de
# Modelo usando Middleware. Este exemplo
# demonstra como alternar entre modelos
# básicos e avançados com base na
# complexidade da conversa (número de
# mensagens).

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um assistente muito inteligente e prestativo que pode
responder perguntas sobre diversos tópicos.
"""

############################################
# PASSO 2 - Configurar os modelos
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

# Modelo básico para conversas simples
basic_model = ChatOpenAI(model="gpt-4o-mini")

# Modelo avançado para conversas complexas
advanced_model = ChatOpenAI(model="gpt-4o")

############################################
# PASSO 3 - Criar o middleware de seleção
# dinâmica de modelo
############################################

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Escolhe o modelo com base na complexidade da conversa."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Usar modelo avançado para conversas longas
        print(f"Usando modelo avançado (gpt-4o) - {message_count} mensagens na conversa")
        model = advanced_model
    else:
        # Usar modelo básico para conversas curtas
        print(f"Usando modelo básico (gpt-4o-mini) - {message_count} mensagens na conversa")
        model = basic_model

    # Fazer a substituição do modelo na requisição
    return handler(request.override(model=model))


############################################
# PASSO 4 - Definir ferramentas (opcional)
############################################

from langchain.tools import tool


@tool
def calculate_square(number: float) -> float:
    """Calcular o quadrado de um número."""
    return number ** 2


############################################
# PASSO 5 - Inicializar o agente com
# middleware
############################################

from langchain.agents import create_agent

agent = create_agent(
    model=basic_model,  # Modelo padrão inicial
    system_prompt=SYSTEM_PROMPT,
    tools=[calculate_square],
    middleware=[dynamic_model_selection],  # Middleware que seleciona dinamicamente o modelo
)

############################################
# PASSO 6 - Usar o agente
############################################

# Primeira interação (usa modelo básico)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Olá! Como você está?"}]},
)

print("--- Resposta 1 (conversa curta) ---")
print(response["messages"][-1].content)

# Simular conversa mais longa para acionar o modelo avançado
# Criando 12 mensagens para ultrapassar o limite de 10
messages = []
for i in range(6):
    messages.append({"role": "user", "content": f"Pergunta {i+1}"})
    messages.append({"role": "assistant", "content": f"Resposta {i+1}"})

messages.append({"role": "user", "content": "Qual é o quadrado de 7?"})

response = agent.invoke(
    {"messages": messages},
)

print("--- Resposta 2 (conversa longa) ---")
print(response["messages"][-1].content)

# Veja o que aconteceu na segunda execução do agent.invoke:

# 1ª chamada (13 mensagens):
# - 6 pares de mensagens simuladas (user + assistant) = 12 mensagens
# -  1 mensagem do usuário ("Qual é o quadrado de 7?") = 13 mensagens
# - O modelo analisa e decide chamar a ferramenta calculate_square

# 2ª chamada (15 mensagens):
# - 13 mensagens anteriores
# -  1 mensagem de chamada da ferramenta (tool_call)
# -  1 mensagem de resposta da ferramenta (tool_result) = 15 mensagens
# - O modelo processa o resultado da ferramenta e gera a resposta final

# Isso é o comportamento esperado do padrão ReAct (Reasoning + Acting) do LangChain:
# 1. Reason (raciocinar): O modelo decide qual ferramenta usar
# 2. Act (agir): Executa a ferramenta
# 3. Reason (raciocinar novamente): O modelo processa o resultado e responde

# Cada etapa de "Reason" invoca o modelo, por isso o middleware é executado duas vezes.

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print()
print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. MIDDLEWARE - SELEÇÃO DINÂMICA DE MODELO:
   - @wrap_model_call intercepta chamadas ao modelo
   - Permite trocar o modelo dinamicamente durante a execução
   - Útil para otimizar custo vs. qualidade

2. COMO FUNCIONA:
   - Middleware analisa request.state["messages"]
   - Decide qual modelo usar baseado em critérios customizados
   - Retorna handler(request.override(model=novo_modelo))

3. ESTRATÉGIA DESTE EXEMPLO:
   - ≤ 10 mensagens: usa gpt-4o-mini (mais barato e rápido)
   - > 10 mensagens: usa gpt-4o (mais inteligente e preciso)
   - Conversas longas = mais contexto = modelo mais avançado

4. OUTROS CRITÉRIOS POSSÍVEIS:
   - Complexidade da pergunta (análise de palavras-chave)
   - Uso de ferramentas (ferramentas complexas = modelo melhor)
   - Horário (modelo mais barato fora do pico)
   - Tipo de usuário (premium = modelo avançado)

5. PADRÃO REACT:
   - O agente pode invocar o modelo MÚLTIPLAS vezes
   - 1ª vez: Decide usar ferramenta
   - 2ª vez: Processa resultado da ferramenta
   - Por isso o middleware é chamado mais de uma vez

6. VANTAGENS:
   - Redução de custos (usa modelo barato quando possível)
   - Melhor performance (modelo rápido para tarefas simples)
   - Flexibilidade (lógica customizada de seleção)

7. request.override():
   - Cria nova requisição com modelo substituído
   - Mantém todos os outros parâmetros inalterados
   - É o jeito correto de mudar o modelo no middleware

8. PRÓXIMOS PASSOS:
   - Para tratamento de erros em ferramentas, veja sample011.py
   - Para prompts dinâmicos, veja sample012.py
""")
