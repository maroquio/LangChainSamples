############################################

# Exemplo de Agente com System Prompt Dinâmico
# usando Middleware. Este exemplo demonstra
# como gerar system prompts diferentes
# baseados no contexto do usuário (perfil,
# nível de expertise, etc.), permitindo
# personalizar a experiência para diferentes
# tipos de usuários.

############################################


############################################
# PASSO 1 - Configurar o esquema de contexto
############################################

from typing import TypedDict


class Context(TypedDict):
    """Esquema de contexto com informação do perfil do usuário."""
    user_role: str  # Valores possíveis: "iniciante", "intermediario", "expert"


############################################
# PASSO 2 - Criar middleware de prompt
# dinâmico
############################################

from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Gera system prompt baseado no nível de expertise do usuário."""
    # Acessar o atributo user_role do contexto (TypedDict)
    user_role = getattr(request.runtime.context, "user_role", "intermediario")
    
    # Prompt base comum a todos os perfis
    base_prompt = "Você é um especialista em tecnologia e programação muito inteligente e prestativo."

    # Personalização baseada no perfil do usuário
    if user_role == "iniciante":
        return f"""{base_prompt}

Ao responder:
- Use linguagem simples e evite jargão técnico
- Forneça exemplos práticos do dia a dia
- Explique conceitos básicos quando necessário
- Seja paciente e didático
- Use analogias para facilitar o entendimento
"""

    elif user_role == "intermediario":
        return f"""{base_prompt}

Ao responder:
- Use equilíbrio entre simplicidade e detalhes técnicos
- Mencione termos técnicos, mas explique-os brevemente
- Forneça exemplos de código quando relevante
- Assuma conhecimento básico de programação
"""

    elif user_role == "expert":
        return f"""{base_prompt}

Ao responder:
- Forneça respostas técnicas detalhadas
- Use terminologia especializada sem hesitação
- Discuta padrões de design, arquitetura e best practices
- Assuma conhecimento avançado de programação
- Foque em nuances técnicas e otimizações
"""

    # Fallback para perfis não reconhecidos
    return base_prompt


############################################
# PASSO 3 - Definir ferramentas (opcional)
############################################

from langchain.tools import tool


@tool
def get_code_example(concept: str) -> str:
    """Obter exemplo de código para um conceito específico."""
    examples = {
        "API REST": """
# Exemplo de endpoint REST com Flask
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = {"id": user_id, "name": "João Silva"}
    return jsonify(user)
""",
        "função": """
# Exemplo de função em Python
def calcular_area_retangulo(largura, altura):
    '''Calcula a área de um retângulo'''
    area = largura * altura
    return area

resultado = calcular_area_retangulo(5, 3)
print(f"A área é: {resultado}")
"""
    }
    return examples.get(concept, "Exemplo não disponível para este conceito.")


############################################
# PASSO 4 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,  # Um pouco mais criativo para explicações variadas
    timeout=15,
    max_tokens=1500,
)

############################################
# PASSO 5 - Inicializar agente com middleware
# de prompt dinâmico
############################################

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[get_code_example],
    middleware=[user_role_prompt],  # Middleware que define o prompt dinamicamente
    context_schema=Context,  # Esquema de contexto necessário para o middleware
)

############################################
# PASSO 6 - Demonstrar com diferentes perfis
# de usuário
############################################

# Mesma pergunta será feita para 3 perfis diferentes
question = "Explique o que é uma API REST"

print("=" * 70)
print("TESTE 1 - Usuário INICIANTE")
print("=" * 70)
print(f"Pergunta: {question}\n")

response = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    context=Context(user_role="iniciante"),
)

print(response["messages"][-1].content)
print("\n")

print("=" * 70)
print("TESTE 2 - Usuário INTERMEDIÁRIO")
print("=" * 70)
print(f"Pergunta: {question}\n")

response = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    context=Context(user_role="intermediario"),
)

print(response["messages"][-1].content)
print("\n")

print("=" * 70)
print("TESTE 3 - Usuário EXPERT")
print("=" * 70)
print(f"Pergunta: {question}\n")

response = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    context=Context(user_role="expert"),
)

print(response["messages"][-1].content)
print("\n")

print("=" * 70)
print("TESTE 4 - Usando ferramenta com contexto de INICIANTE")
print("=" * 70)
print("Pergunta: Me mostre um exemplo de API REST\n")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Me mostre um exemplo de código de API REST"}]},
    context=Context(user_role="iniciante"),
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
1. MIDDLEWARE - PROMPT DINÂMICO:
   - @dynamic_prompt permite gerar system prompt em tempo de execução
   - Acessa runtime.context para personalizar o prompt
   - Cada usuário pode ter experiência diferente

2. COMO FUNCIONA:
   - Middleware recebe ModelRequest
   - Acessa request.runtime.context.user_role
   - Retorna system prompt personalizado baseado no perfil

3. VANTAGENS:
   - Personalização por tipo de usuário
   - Adapta tom e complexidade da resposta
   - Mesmo agente, múltiplos comportamentos

4. PERFIS NESTE EXEMPLO:
   - "iniciante": Linguagem simples, didático, usa analogias
   - "intermediario": Balanceado, assume conhecimento básico
   - "expert": Técnico, detalhado, foca em nuances

5. CASOS DE USO:
   - Educação (adaptar ao nível do aluno)
   - Suporte técnico (leigo vs. especialista)
   - Documentação (tutorial vs. referência)
   - Internacionalização (idioma baseado em contexto)

6. COMBINAÇÕES PODEROSAS:
   - Context: Dados do usuário (user_role, etc.)
   - dynamic_prompt: Prompt baseado no contexto
   - Ferramentas: Podem também usar o contexto
   - Memória: Mantém histórico personalizado

7. ESTRUTURA DO PROMPT:
   - Base comum a todos os perfis
   - Seção específica por perfil
   - Fallback para perfis não reconhecidos

8. CONTEXT_SCHEMA:
   - Define estrutura do contexto (TypedDict ou dataclass)
   - Necessário quando usa middleware que acessa context
   - Fornece type hints para IDEs

9. PRÓXIMOS PASSOS:
   - Para passar mensagens manualmente, veja sample013.py
   - Para structured output (LangChain 1.0+), veja sample014.py e sample015.py
""")
