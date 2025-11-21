############################################

# Exemplo de Agente com Saída Estruturada
# usando ToolStrategy. Este exemplo demonstra
# como usar "tool calling artificial" para
# gerar saída estruturada em um formato
# específico, funcionando com qualquer modelo
# que suporte chamada de ferramentas.

############################################


############################################
# PASSO 1 - Definir o schema de saída com
# Pydantic
############################################

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Informações de contato estruturadas."""
    name: str = Field(description="Nome completo da pessoa")
    email: str = Field(description="Endereço de e-mail")
    phone: str = Field(description="Número de telefone")


class ProductReview(BaseModel):
    """Avaliação estruturada de um produto."""
    product_name: str = Field(description="Nome do produto avaliado")
    rating: int = Field(description="Nota de 1 a 5 estrelas", ge=1, le=5)
    pros: list[str] = Field(description="Lista de pontos positivos")
    cons: list[str] = Field(description="Lista de pontos negativos")
    recommendation: str = Field(description="Recomendação final (Sim/Não/Talvez)")


class EventDetails(BaseModel):
    """Detalhes estruturados de um evento."""
    event_name: str = Field(description="Nome do evento")
    date: str = Field(description="Data do evento (formato: DD/MM/YYYY)")
    location: str = Field(description="Local do evento")
    attendees: int = Field(description="Número de participantes esperados")
    topics: list[str] = Field(description="Lista de tópicos que serão abordados")


############################################
# PASSO 2 - Definir ferramentas para o agente
# (opcional, mas útil para demonstrar que
# ToolStrategy funciona junto com ferramentas
# reais)
############################################

from langchain.tools import tool


@tool
def buscar_informacoes(query: str) -> str:
    """Buscar informações adicionais (simulado)."""
    # Simulação de uma base de conhecimento
    database = {
        "telefone empresa": "(11) 98765-4321",
        "email suporte": "suporte@empresa.com",
        "nome CEO": "Maria Silva",
        "email CEO": "maria.silva@empresa.com",
        "telefone CEO": "(11) 91234-5678"
    }
    return database.get(query.lower(), "Informação não encontrada")


############################################
# PASSO 3 - Configurar o modelo
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0,  # Temperatura baixa para saída mais determinística
    timeout=15,
    max_tokens=1000,
)

############################################
# PASSO 4 - Criar agente com ToolStrategy
############################################

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

# Agente configurado para extrair informações de contato
agent_contact = create_agent(
    model=model,
    tools=[buscar_informacoes],  # Ferramentas opcionais
    response_format=ToolStrategy(ContactInfo),  # ToolStrategy com schema
)

# Agente configurado para analisar avaliações de produtos
agent_review = create_agent(
    model=model,
    response_format=ToolStrategy(ProductReview),
)

# Agente configurado para extrair detalhes de eventos
agent_event = create_agent(
    model=model,
    response_format=ToolStrategy(EventDetails),
)

############################################
# PASSO 5 - Demonstrar extração de contatos
############################################

print("=" * 70)
print("EXEMPLO 1 - Extrair Informações de Contato")
print("=" * 70)

texto_contato = """
Entre em contato com nosso gerente de vendas:
João Pedro Santos
E-mail: joao.santos@vendas.com.br
Telefone: (11) 99876-5432
"""

print(f"Texto original:\n{texto_contato}\n")

result = agent_contact.invoke({
    "messages": [{"role": "user", "content": f"Extraia as informações de contato do seguinte texto: {texto_contato}"}]
})

# A resposta estruturada vem no campo 'structured_response'
contact_info = result["structured_response"]
print(f"Estrutura extraída (tipo: {type(contact_info).__name__}):")
print(f"  Nome: {contact_info.name}")
print(f"  Email: {contact_info.email}")
print(f"  Telefone: {contact_info.phone}")
print()

############################################
# PASSO 6 - Demonstrar análise de avaliação
############################################

print("=" * 70)
print("EXEMPLO 2 - Analisar Avaliação de Produto")
print("=" * 70)

texto_avaliacao = """
Comprei o Smartphone XYZ Pro e estou muito impressionado!
A câmera é excelente, a bateria dura o dia todo e o desempenho é muito rápido.
Por outro lado, achei o preço um pouco alto e o aparelho esquenta durante jogos pesados.
No geral, recomendo para quem busca um celular premium.
"""

print(f"Avaliação original:\n{texto_avaliacao}\n")

result = agent_review.invoke({
    "messages": [{"role": "user", "content": f"Analise esta avaliação de produto e estruture as informações: {texto_avaliacao}"}]
})

review = result["structured_response"]
print(f"Análise estruturada (tipo: {type(review).__name__}):")
print(f"  Produto: {review.product_name}")
print(f"  Nota: {review.rating}/5")
print(f"  Pontos positivos:")
for pro in review.pros:
    print(f"    - {pro}")
print(f"  Pontos negativos:")
for con in review.cons:
    print(f"    - {con}")
print(f"  Recomendação: {review.recommendation}")
print()

############################################
# PASSO 7 - Demonstrar extração de evento
############################################

print("=" * 70)
print("EXEMPLO 3 - Extrair Detalhes de Evento")
print("=" * 70)

texto_evento = """
Participe da Conferência Tech Summit 2024!
Data: 15/03/2024
Local: Centro de Convenções São Paulo
Esperamos cerca de 500 participantes.
Tópicos: Inteligência Artificial, Cloud Computing, Segurança Cibernética, DevOps
"""

print(f"Descrição do evento:\n{texto_evento}\n")

result = agent_event.invoke({
    "messages": [{"role": "user", "content": f"Extraia os detalhes estruturados deste evento: {texto_evento}"}]
})

event = result["structured_response"]
print(f"Detalhes estruturados (tipo: {type(event).__name__}):")
print(f"  Evento: {event.event_name}")
print(f"  Data: {event.date}")
print(f"  Local: {event.location}")
print(f"  Participantes esperados: {event.attendees}")
print(f"  Tópicos:")
for topic in event.topics:
    print(f"    - {topic}")
print()

############################################
# PASSO 8 - Demonstrar uso com ferramentas
############################################

print("=" * 70)
print("EXEMPLO 4 - Combinar ToolStrategy com Ferramentas Reais")
print("=" * 70)

# Simplificado: busca apenas o nome, os outros dados são fornecidos
result = agent_contact.invoke({
    "messages": [{"role": "user", "content": "Extraia um contato estruturado: Nome é Maria Silva, email maria.silva@empresa.com, telefone (11) 91234-5678"}]
})

contact_with_tools = result["structured_response"]
print(f"Contato criado (estrutura extraída do texto):")
print(f"  Nome: {contact_with_tools.name}")
print(f"  Email: {contact_with_tools.email}")
print(f"  Telefone: {contact_with_tools.phone}")
print()
print("Nota: ToolStrategy funciona bem com ferramentas, mas para este exemplo")
print("simplificamos a extração direta do texto para evitar múltiplas chamadas.")
print()

############################################
# PASSO 9 - Demonstrar validação do Pydantic
############################################

print("=" * 70)
print("EXEMPLO 5 - Validação Automática do Pydantic")
print("=" * 70)

# O Pydantic valida automaticamente os dados
# Por exemplo, a nota deve estar entre 1 e 5
texto_validacao = """
O produto ABC é horrível! Dou nota 0.
Não funciona direito e o suporte é péssimo.
Definitivamente não recomendo.
"""

print(f"Avaliação para validar:\n{texto_validacao}\n")

result = agent_review.invoke({
    "messages": [{"role": "user", "content": f"Analise esta avaliação: {texto_validacao}"}]
})

validated_review = result["structured_response"]
print(f"Nota extraída (validada para estar entre 1-5): {validated_review.rating}/5")
print(f"  Nota: Note que mesmo o usuário mencionando '0', o schema força o valor mínimo de 1")
print()

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. TOOLSTRATEGY - CARACTERÍSTICAS:
   - Usa "tool calling artificial" para gerar saída estruturada
   - Funciona com QUALQUER modelo que suporte tool calling
   - Mais compatível, mas pode ser menos confiável que ProviderStrategy
   - Ideal para modelos que não têm suporte nativo a structured output

2. ACESSO À RESPOSTA ESTRUTURADA:
   - Use result["structured_response"] para acessar o objeto Pydantic
   - O objeto retornado é uma instância validada do schema fornecido
   - Benefícios de type hints e validação do Pydantic

3. VALIDAÇÃO AUTOMÁTICA:
   - Pydantic valida automaticamente tipos, ranges, e constraints
   - Use Field() com description para guiar o modelo
   - Use ge (>=), le (<=), min_length, max_length para validações

4. COMBINAÇÃO COM FERRAMENTAS:
   - ToolStrategy funciona junto com ferramentas normais
   - O agente pode usar ferramentas E retornar saída estruturada
   - Útil quando o agente precisa buscar dados antes de estruturar

5. CASOS DE USO:
   - Extração de informações (NER - Named Entity Recognition)
   - Classificação e categorização
   - Análise de sentimentos estruturada
   - Parsing de documentos não estruturados
   - Conversão de texto livre para dados estruturados

6. ALTERNATIVA:
   - Para maior confiabilidade, veja ProviderStrategy no sample015.py
   - ProviderStrategy usa structured output nativo do provider (mais confiável)
   - Mas só funciona com providers que suportam (ex: OpenAI)
""")
