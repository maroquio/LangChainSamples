############################################
#
# Exemplo de model.with_structured_output()
# - Structured Output
#
############################################


############################################
# PASSO 1 - Definir o Schema com Pydantic
############################################

from pydantic import BaseModel, Field
from typing import List

class Person(BaseModel):
    """Informações sobre uma pessoa."""
    name: str = Field(description="Nome completo da pessoa")
    age: int = Field(description="Idade em anos")
    occupation: str = Field(description="Profissão ou ocupação")
    hobbies: List[str] = Field(description="Lista de hobbies")


class MovieReview(BaseModel):
    """Review estruturado de um filme."""
    title: str = Field(description="Título do filme")
    rating: int = Field(description="Nota de 1 a 10", ge=1, le=10)
    pros: List[str] = Field(description="Pontos positivos do filme")
    cons: List[str] = Field(description="Pontos negativos do filme")
    recommendation: bool = Field(description="Se você recomenda o filme ou não")


############################################
# PASSO 2 - Usar with_structured_output()
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Criar o model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Aplicar structured output com schema Pydantic
structured_model = model.with_structured_output(Person)

# Invocar com texto não estruturado
text = """
João Silva tem 35 anos e trabalha como engenheiro de software.
Nas horas vagas, ele gosta de tocar violão, fazer trilhas e ler ficção científica.
"""

response = structured_model.invoke(text)

print("=" * 70)
print("EXEMPLO 1: EXTRAÇÃO DE DADOS ESTRUTURADOS")
print("=" * 70)
print(f"Tipo do response: {type(response)}")
print(f"\nDados estruturados:")
print(f"  Nome: {response.name}")
print(f"  Idade: {response.age}")
print(f"  Ocupação: {response.occupation}")
print(f"  Hobbies: {', '.join(response.hobbies)}")
print()


############################################
# PASSO 3 - Exemplo com Schema Complexo
############################################

structured_review_model = model.with_structured_output(MovieReview)

review_text = """
Assisti Inception ontem. É um filme incrível! A fotografia é lindíssima,
o roteiro é muito inteligente e a atuação é impecável. O único problema
é que o final deixa muitas perguntas sem resposta, o que pode frustrar
algumas pessoas. Mesmo assim, super recomendo! Dou nota 9.
"""

review = structured_review_model.invoke(review_text)

print("=" * 70)
print("EXEMPLO 2: REVIEW ESTRUTURADO DE FILME")
print("=" * 70)
print(f"Filme: {review.title}")
print(f"Nota: {review.rating}/10")
print(f"\nPontos Positivos:")
for pro in review.pros:
    print(f"  ✓ {pro}")
print(f"\nPontos Negativos:")
for con in review.cons:
    print(f"  ✗ {con}")
print(f"\nRecomenda? {'Sim' if review.recommendation else 'Não'}")
print()


############################################
# PASSO 4 - Usando include_raw para ver output bruto
############################################

# include_raw=True retorna tanto o parsed quanto o raw output
structured_model_with_raw = model.with_structured_output(Person, include_raw=True)

response_with_raw = structured_model_with_raw.invoke(text)

print("=" * 70)
print("EXEMPLO 3: USANDO include_raw=True")
print("=" * 70)
print(f"Tipo do response: {type(response_with_raw)}")
print(f"\nParsed (estruturado):")
print(f"  {response_with_raw['parsed']}")
print(f"\nRaw (original):")
print(f"  Conteúdo: {response_with_raw['raw'].content}")
print(f"  Tool calls: {response_with_raw['raw'].tool_calls}")
print()


############################################
# PASSO 5 - Usando TypedDict como Schema
############################################

from typing import TypedDict

class ProductTypedDict(TypedDict):
    """Informações sobre um produto."""
    name: str
    price: float
    category: str
    in_stock: bool


# TypedDict também funciona como schema
structured_product_model = model.with_structured_output(ProductTypedDict)

product_text = "O notebook Dell Inspiron custa R$ 3.500 e está disponível na categoria Eletrônicos."

product = structured_product_model.invoke(product_text)

print("=" * 70)
print("EXEMPLO 4: USANDO TypedDict COMO SCHEMA")
print("=" * 70)
print(f"Tipo do response: {type(product)}")
print(f"Produto: {product}")
print()


############################################
# PASSO 6 - Parâmetro 'method' para controlar a estratégia
############################################

# method pode ser: 'function_calling' (default) ou 'json_mode'
structured_model_json = model.with_structured_output(
    Person,
    method="function_calling"  # ou "json_mode" dependendo do provider
)

print("=" * 70)
print("EXEMPLO 5: PARÂMETRO method")
print("=" * 70)
print("""
O parâmetro 'method' controla como o structured output é implementado:

- 'function_calling' (padrão):
  * Usa function/tool calling nativo do model
  * Mais preciso e confiável
  * Suportado pela maioria dos providers modernos (OpenAI, Anthropic, etc.)

- 'json_mode':
  * Força o model a retornar JSON válido
  * Útil para models sem suporte a function calling
  * Pode ser menos preciso

- 'json_schema' (alguns providers):
  * Usa JSON Schema enforcement nativo
  * Mais rígido e preciso que json_mode

Recomendação: Use 'function_calling' quando disponível.
""")


############################################
# PASSO 7 - Schema Aninhado (Nested)
############################################

class Address(BaseModel):
    """Endereço."""
    street: str
    city: str
    state: str
    zip_code: str


class Company(BaseModel):
    """Informações de uma empresa."""
    name: str = Field(description="Nome da empresa")
    founded_year: int = Field(description="Ano de fundação")
    employees: int = Field(description="Número de funcionários")
    address: Address = Field(description="Endereço da empresa")


structured_company_model = model.with_structured_output(Company)

company_text = """
A TechCorp foi fundada em 2010 e hoje tem 250 funcionários.
Sua sede fica na Av. Paulista, 1000, São Paulo, SP, CEP 01310-100.
"""

company = structured_company_model.invoke(company_text)

print("=" * 70)
print("EXEMPLO 6: SCHEMA ANINHADO (NESTED)")
print("=" * 70)
print(f"Empresa: {company.name}")
print(f"Fundada em: {company.founded_year}")
print(f"Funcionários: {company.employees}")
print(f"\nEndereço:")
print(f"  Rua: {company.address.street}")
print(f"  Cidade: {company.address.city}")
print(f"  Estado: {company.address.state}")
print(f"  CEP: {company.address.zip_code}")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. with_structured_output():
   - Força o model a retornar dados em formato estruturado
   - Sem necessidade de parsers manuais
   - Usa Pydantic, TypedDict ou JSON Schema como schema
   - Retorna instância do tipo especificado (não string)

2. VANTAGENS:
   - Validação automática dos dados (Pydantic valida tipos e constraints)
   - Type safety no código (IDE autocomplete, type checking)
   - Sem necessidade de regex ou parsing manual
   - Menos propenso a erros de formato
   - Ideal para extração de dados, formulários, APIs

3. SCHEMAS SUPORTADOS:
   - Pydantic BaseModel (recomendado) - mais rico em validações
   - TypedDict - mais simples, apenas tipos
   - JSON Schema dict - máxima flexibilidade
   - Qualquer classe com __annotations__

4. PARÂMETRO include_raw:
   - False (padrão): retorna apenas o objeto parsed
   - True: retorna dict com 'parsed' e 'raw'
   - Útil para debugging ou quando precisa do original

5. PARÂMETRO method:
   - 'function_calling': usa tool/function calling (mais preciso)
   - 'json_mode': força JSON válido (fallback)
   - 'json_schema': enforcement nativo (quando disponível)
   - Default: 'function_calling'

6. DIFERENÇA DE OUTPUT PARSERS:
   - with_structured_output: usa capacidades nativas do model
   - Output parsers: fazem parsing do text response
   - with_structured_output é preferível quando disponível

7. LIMITAÇÕES:
   - Nem todos os models suportam structured output
   - Schemas muito complexos podem confundir o model
   - Pode ser mais lento que completions simples
   - Custo pode ser maior (usa function calling)

8. CASOS DE USO:
   - Extração de informações de textos
   - Classificação estruturada
   - Preenchimento de formulários
   - Análise de sentimento com campos específicos
   - Geração de dados para APIs
   - Qualquer cenário onde você precisa de JSON confiável

9. PRÓXIMOS PASSOS:
   - Para multimodal com imagens, veja sample022.py
   - Para métodos invoke/stream/batch, veja sample025.py
   - Para parâmetros de model, veja sample026.py
""")
