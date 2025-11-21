############################################

# Exemplo de Agente com Saída Estruturada
# usando ProviderStrategy. Este exemplo
# demonstra como usar structured output
# NATIVO do provider (ex: OpenAI), que é
# mais confiável que ToolStrategy, mas
# requer suporte nativo do modelo.

############################################


############################################
# PASSO 1 - Definir schemas de saída com
# Pydantic
############################################

from pydantic import BaseModel, Field
from typing import Literal


class Address(BaseModel):
    """Endereço estruturado."""
    street: str = Field(description="Nome da rua com número")
    city: str = Field(description="Nome da cidade")
    state: str = Field(description="Estado (sigla)")
    zip_code: str = Field(description="CEP")


class Person(BaseModel):
    """Informações completas de uma pessoa."""
    full_name: str = Field(description="Nome completo")
    age: int = Field(description="Idade em anos", ge=0, le=120)
    occupation: str = Field(description="Profissão ou ocupação")
    address: Address = Field(description="Endereço completo")
    interests: list[str] = Field(description="Lista de interesses/hobbies")


class SentimentAnalysis(BaseModel):
    """Análise de sentimento estruturada."""
    text: str = Field(description="Texto original analisado")
    sentiment: Literal["positivo", "negativo", "neutro"] = Field(
        description="Sentimento identificado"
    )
    confidence: float = Field(
        description="Nível de confiança (0.0 a 1.0)",
        ge=0.0,
        le=1.0
    )
    key_phrases: list[str] = Field(
        description="Frases-chave que justificam o sentimento"
    )
    emotion: Literal["alegria", "tristeza", "raiva", "medo", "surpresa", "nenhuma"] = Field(
        description="Emoção predominante identificada"
    )


class CodeAnalysis(BaseModel):
    """Análise estruturada de código."""
    language: str = Field(description="Linguagem de programação")
    purpose: str = Field(description="Propósito/objetivo do código")
    complexity: Literal["baixa", "média", "alta"] = Field(
        description="Complexidade do código"
    )
    functions: list[str] = Field(description="Lista de funções/métodos identificados")
    potential_issues: list[str] = Field(
        description="Possíveis problemas ou melhorias"
    )
    best_practices_score: int = Field(
        description="Pontuação de boas práticas (0-10)",
        ge=0,
        le=10
    )


############################################
# PASSO 2 - Configurar o modelo
# IMPORTANTE: ProviderStrategy requer modelo
# com suporte nativo a structured output
############################################

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Usar modelo OpenAI que suporta structured output nativo
# Outros providers podem suportar no futuro
model = init_chat_model(
    "gpt-4o",  # Note: gpt-4o suporta structured output nativo
    temperature=0,
    timeout=20,
    max_tokens=1500,
)

############################################
# PASSO 3 - Criar agentes com ProviderStrategy
############################################

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

# Agente para extrair informações de pessoas
agent_person = create_agent(
    model=model,
    response_format=ProviderStrategy(Person),  # ProviderStrategy com schema
)

# Agente para análise de sentimento
agent_sentiment = create_agent(
    model=model,
    response_format=ProviderStrategy(SentimentAnalysis),
)

# Agente para análise de código
agent_code = create_agent(
    model=model,
    response_format=ProviderStrategy(CodeAnalysis),
)

############################################
# PASSO 4 - Demonstrar extração de pessoa
############################################

print("=" * 70)
print("EXEMPLO 1 - Extrair Informações Completas de Pessoa")
print("=" * 70)

texto_pessoa = """
Conheça Ana Carolina Ferreira, uma desenvolvedora de software de 28 anos.
Ela mora na Rua das Flores, 123, em Campinas, SP, CEP 13010-100.
Ana adora programação em Python, leitura de ficção científica e trilhas na natureza.
"""

print(f"Texto original:\n{texto_pessoa}\n")

result = agent_person.invoke({
    "messages": [{"role": "user", "content": f"Extraia todas as informações estruturadas desta pessoa: {texto_pessoa}"}]
})

person = result["structured_response"]
print(f"Informações estruturadas (tipo: {type(person).__name__}):")
print(f"  Nome: {person.full_name}")
print(f"  Idade: {person.age} anos")
print(f"  Ocupação: {person.occupation}")
print(f"  Endereço:")
print(f"    Rua: {person.address.street}")
print(f"    Cidade: {person.address.city}")
print(f"    Estado: {person.address.state}")
print(f"    CEP: {person.address.zip_code}")
print(f"  Interesses:")
for interest in person.interests:
    print(f"    - {interest}")
print()

############################################
# PASSO 5 - Demonstrar análise de sentimento
############################################

print("=" * 70)
print("EXEMPLO 2 - Análise de Sentimento Estruturada")
print("=" * 70)

# Sentimento positivo
texto_positivo = """
Estou absolutamente encantado com este produto! A qualidade superou todas as
minhas expectativas. A entrega foi rápida e o atendimento ao cliente foi
excepcional. Com certeza vou comprar novamente e recomendar para meus amigos!
"""

print(f"Texto 1 (Positivo):\n{texto_positivo}\n")

result = agent_sentiment.invoke({
    "messages": [{"role": "user", "content": f"Analise o sentimento deste texto: {texto_positivo}"}]
})

sentiment = result["structured_response"]
print(f"Análise de sentimento:")
print(f"  Sentimento: {sentiment.sentiment}")
print(f"  Confiança: {sentiment.confidence:.2%}")
print(f"  Emoção: {sentiment.emotion}")
print(f"  Frases-chave:")
for phrase in sentiment.key_phrases:
    print(f"    - {phrase}")
print()

# Sentimento negativo
print("-" * 70)
texto_negativo = """
Que experiência terrível! O produto chegou com defeito, o suporte não responde
e ainda por cima querem cobrar pelo envio de volta. Estou extremamente frustrado
e arrependido desta compra. Nunca mais compro nesta loja!
"""

print(f"Texto 2 (Negativo):\n{texto_negativo}\n")

result = agent_sentiment.invoke({
    "messages": [{"role": "user", "content": f"Analise o sentimento deste texto: {texto_negativo}"}]
})

sentiment_neg = result["structured_response"]
print(f"Análise de sentimento:")
print(f"  Sentimento: {sentiment_neg.sentiment}")
print(f"  Confiança: {sentiment_neg.confidence:.2%}")
print(f"  Emoção: {sentiment_neg.emotion}")
print(f"  Frases-chave:")
for phrase in sentiment_neg.key_phrases:
    print(f"    - {phrase}")
print()

############################################
# PASSO 6 - Demonstrar análise de código
############################################

print("=" * 70)
print("EXEMPLO 3 - Análise Estruturada de Código")
print("=" * 70)

codigo_exemplo = '''
def calcular_media(numeros):
    total = 0
    for num in numeros:
        total = total + num
    return total / len(numeros)

def processar_dados(dados):
    resultado = []
    for item in dados:
        if item > 0:
            resultado.append(item * 2)
    return resultado

media = calcular_media([10, 20, 30, 40])
print(media)
'''

print(f"Código para análise:\n{codigo_exemplo}\n")

result = agent_code.invoke({
    "messages": [{"role": "user", "content": f"Analise este código:\n{codigo_exemplo}"}]
})

code_analysis = result["structured_response"]
print(f"Análise de código estruturada:")
print(f"  Linguagem: {code_analysis.language}")
print(f"  Propósito: {code_analysis.purpose}")
print(f"  Complexidade: {code_analysis.complexity}")
print(f"  Funções identificadas:")
for func in code_analysis.functions:
    print(f"    - {func}")
print(f"  Possíveis melhorias:")
for issue in code_analysis.potential_issues:
    print(f"    - {issue}")
print(f"  Score de boas práticas: {code_analysis.best_practices_score}/10")
print()

############################################
# PASSO 7 - Demonstrar schemas aninhados
############################################

print("=" * 70)
print("EXEMPLO 4 - Schemas Aninhados (Person com Address)")
print("=" * 70)

texto_complexo = """
Roberto Alves é um professor de 45 anos que vive em São Paulo.
Seu endereço é Avenida Paulista, 1000, São Paulo, SP, 01310-100.
Roberto gosta de xadrez, fotografia e culinária italiana.
"""

print(f"Texto com informações aninhadas:\n{texto_complexo}\n")

result = agent_person.invoke({
    "messages": [{"role": "user", "content": f"Extraia as informações: {texto_complexo}"}]
})

person_complex = result["structured_response"]
print(f"Pessoa extraída:")
print(f"  {person_complex.full_name}, {person_complex.age} anos")
print(f"  Profissão: {person_complex.occupation}")
print(f"  Mora em: {person_complex.address.street}, {person_complex.address.city}-{person_complex.address.state}")
print()

############################################
# PASSO 8 - Comparação ProviderStrategy vs
# ToolStrategy
############################################

print("=" * 70)
print("EXEMPLO 5 - Benefícios do ProviderStrategy")
print("=" * 70)

# ProviderStrategy é mais confiável porque usa structured output nativo
# Vamos demonstrar com um caso que exige precisão

texto_preciso = """
Maria tem 32 anos e trabalha como engenheira. Mora em Belo Horizonte, MG.
Seu endereço é Rua Goiás, 456, Belo Horizonte, MG, 30190-060.
Gosta de yoga e música clássica.
"""

result = agent_person.invoke({
    "messages": [{"role": "user", "content": f"Extraia com máxima precisão: {texto_preciso}"}]
})

precise_person = result["structured_response"]

print("Estrutura extraída com ProviderStrategy (Native Structured Output):")
print(f"  Nome: '{precise_person.full_name}'")
print(f"  Idade: {precise_person.age} (tipo: {type(precise_person.age).__name__})")
print(f"  Ocupação: '{precise_person.occupation}'")
print(f"  CEP: '{precise_person.address.zip_code}' (formato preservado)")
print()
print("Vantagens do ProviderStrategy:")
print("  ✓ Maior confiabilidade na estrutura")
print("  ✓ Tipos garantidos pelo provider")
print("  ✓ Menos erros de parsing")
print("  ✓ Melhor performance")
print()

############################################
# PASSO 9 - Demonstrar uso de Literal types
############################################

print("=" * 70)
print("EXEMPLO 6 - Validação com Literal Types")
print("=" * 70)

# Literal garante que apenas valores específicos são aceitos
textos_diversos = [
    "Este filme é incrível! Adorei cada minuto!",
    "Meh, foi ok. Nada de especial.",
    "Que desastre! Perdi meu tempo assistindo isso."
]

print("Analisando múltiplos textos com sentimentos diferentes:\n")

for i, texto in enumerate(textos_diversos, 1):
    result = agent_sentiment.invoke({
        "messages": [{"role": "user", "content": f"Analise: {texto}"}]
    })
    sent = result["structured_response"]
    print(f"{i}. '{texto[:50]}...'")
    print(f"   Sentimento: {sent.sentiment} | Emoção: {sent.emotion} | Confiança: {sent.confidence:.0%}")
    print()

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. PROVIDERSTRATEGY - CARACTERÍSTICAS:
   - Usa structured output NATIVO do provider (ex: OpenAI)
   - MAIS CONFIÁVEL que ToolStrategy
   - Melhor performance e precisão
   - Requer modelo com suporte nativo (ex: gpt-4o, gpt-4o-mini)

2. DIFERENÇA ENTRE TOOLSTRATEGY E PROVIDERSTRATEGY:

   ToolStrategy (sample014.py):
   - Usa "tool calling artificial"
   - Funciona com qualquer modelo que suporte tools
   - Menos confiável, pode ter erros de parsing

   ProviderStrategy (este arquivo):
   - Usa structured output nativo do provider
   - Mais confiável e preciso
   - Só funciona com providers que suportam (OpenAI, etc)

3. QUANDO USAR CADA UM:

   Use ProviderStrategy quando:
   - Precisar de máxima confiabilidade
   - Usar OpenAI ou outro provider com suporte nativo
   - Estruturas complexas e aninhadas
   - Aplicações em produção críticas

   Use ToolStrategy quando:
   - Usar modelo sem suporte nativo a structured output
   - Precisar de compatibilidade máxima
   - Prototipar rapidamente

4. SCHEMAS COMPLEXOS:
   - ProviderStrategy suporta schemas aninhados (ex: Person com Address)
   - Use Literal para valores restritos (ex: "positivo", "negativo", "neutro")
   - Field() com ge/le para validação numérica
   - Tipo list[str] para listas de strings

5. VALIDAÇÃO AUTOMÁTICA:
   - Pydantic + Provider garantem tipos corretos
   - Erros de validação são capturados automaticamente
   - Constraints (ge, le, Literal) são aplicados pelo provider

6. CASOS DE USO IDEAIS:
   - Extração de entidades complexas (reconhecimento de entidades aninhadas)
   - Análise de sentimento com múltiplas dimensões
   - Classificação com múltiplas categorias
   - Parsing de documentos estruturados
   - Sistemas que exigem alta confiabilidade

7. REQUISITOS:
   - Modelo: gpt-4o, gpt-4o-mini, ou outro com suporte nativo
   - LangChain 1.0+: deve usar explicitamente ProviderStrategy
   - Não pode passar schema diretamente (deprecated)

8. MIGRAÇÃO DE CÓDIGO ANTIGO:
   Antes (deprecated):
     response_format=ContactInfo

   Agora (correto):
     response_format=ProviderStrategy(ContactInfo)
""")
