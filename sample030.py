############################################
#
# Exemplo de Configurable Models (Runtime Configuration)
#
############################################


############################################
# PASSO 1 - configurable_fields: Parâmetros Configuráveis
############################################

from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField, RunnableConfig
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("configurable_fields: PARÂMETROS CONFIGURÁVEIS")
print("=" * 70)

# Criar model com campo temperature configurável
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # Valor padrão
).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="Temperature do model (0 = determinístico, 1 = criativo)",
    )
)

prompt = "Escreva uma frase sobre gatos."

# Usar com temperature padrão (0.7)
print("\n1. Com temperature padrão (0.7):")
response_default = model.invoke(prompt)
print(f"   {response_default.content}")

# Configurar temperature = 0 em runtime
print("\n2. Com temperature=0 (configurado em runtime):")
config_temp_0 = RunnableConfig(configurable={"llm_temperature": 0})
response_temp_0 = model.invoke(prompt, config=config_temp_0)
print(f"   {response_temp_0.content}")

# Configurar temperature = 1.5 em runtime
print("\n3. Com temperature=1.5 (configurado em runtime):")
config_temp_15 = RunnableConfig(configurable={"llm_temperature": 1.5})
response_temp_15 = model.invoke(prompt, config=config_temp_15)
print(f"   {response_temp_15.content}")

print("\n✓ Mesmo model, parâmetros diferentes em runtime!")
print()


############################################
# PASSO 2 - Múltiplos Campos Configuráveis
############################################

print("=" * 70)
print("MÚLTIPLOS CAMPOS CONFIGURÁVEIS")
print("=" * 70)

# Model com múltiplos campos configuráveis
model_multi = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_completion_tokens=100,
).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="Temperature",
        description="Controle de criatividade",
    ),
    max_tokens=ConfigurableField(
        id="max_tokens",
        name="Max Tokens",
        description="Comprimento máximo da resposta",
    ),
)

prompt_story = "Conte uma história sobre um robô."

print("\n1. Configuração padrão:")
response_1 = model_multi.invoke(prompt_story)
print(f"   {response_1.content[:100]}...")

print("\n2. temperature=0, max_tokens=50 (curto e determinístico):")
config_short = RunnableConfig(configurable={"temperature": 0, "max_tokens": 50})
response_2 = model_multi.invoke(prompt_story, config=config_short)
print(f"   {response_2.content}")

print("\n3. temperature=1.5, max_tokens=200 (criativo e longo):")
config_long = RunnableConfig(configurable={"temperature": 1.5, "max_tokens": 200})
response_3 = model_multi.invoke(prompt_story, config=config_long)
print(f"   {response_3.content[:150]}...")

print()


############################################
# PASSO 3 - configurable_alternatives: Trocar Model em Runtime
############################################

print("=" * 70)
print("configurable_alternatives: TROCAR MODEL EM RUNTIME")
print("=" * 70)

from langchain_anthropic import ChatAnthropic

# Model padrão com alternativas configuráveis
model_switchable = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).configurable_alternatives(
    ConfigurableField(id="llm", name="LLM"),
    # Alternativas disponíveis
    gpt4o=ChatOpenAI(model="gpt-4o", temperature=0),
    claude=ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=0,
        timeout=30,
        stop=["\n\n"],
    ),
    default_key="gpt-4o-mini",  # Chave para o model padrão
)

question = "Qual é a capital do Brasil?"

print("\n1. Model padrão (gpt-4o-mini):")
try:
    response_default = model_switchable.invoke(question)
    print(f"   {response_default.content}")
except Exception as e:
    print(f"   Erro: {e}")

print("\n2. Trocando para gpt-4o em runtime:")
try:
    config_gpt4o = RunnableConfig(configurable={"llm": "gpt4o"})
    response_gpt4o = model_switchable.invoke(question, config=config_gpt4o)
    print(f"   {response_gpt4o.content}")
except Exception as e:
    print(f"   Erro: {e}")

print("\n3. Trocando para claude em runtime:")
try:
    config_claude = RunnableConfig(configurable={"llm": "claude"})
    response_claude = model_switchable.invoke(question, config=config_claude)
    print(f"   {response_claude.content}")
except Exception as e:
    print(f"   Erro: {e}")
    print("   (Certifique-se de ter ANTHROPIC_API_KEY configurado)")

print("\n✓ Um único objeto, múltiplos models!")
print()


############################################
# PASSO 4 - Caso de Uso: A/B Testing
############################################

print("=" * 70)
print("CASO DE USO: A/B TESTING")
print("=" * 70)

# Simular A/B testing com diferentes models
import random

model_ab_test = ChatOpenAI(model="gpt-4o-mini").configurable_alternatives(
    ConfigurableField(id="model_variant", name="Model Variant"),
    variant_a=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    variant_b=ChatOpenAI(model="gpt-4o-mini", temperature=1),
)

print("\nSimulando 5 usuários com A/B testing:")

for user_id in range(1, 6):
    # Atribuir aleatoriamente variante A ou B
    variant = random.choice(["variant_a", "variant_b"])

    config = RunnableConfig(
        configurable={"model_variant": variant},
        metadata={"user_id": user_id, "variant": variant},
    )

    response = model_ab_test.invoke("Diga olá!", config=config)

    print(f"  User {user_id} (Variant: {variant}): {response.content}")

print(
    """
Metadata pode ser usado para rastrear qual variante
foi usada e comparar métricas (satisfação, tempo, custo).
"""
)
print()


############################################
# PASSO 5 - Caso de Uso: Ambientes (Dev/Staging/Prod)
############################################

print("=" * 70)
print("CASO DE USO: AMBIENTES (DEV/STAGING/PROD)")
print("=" * 70)

# Diferentes configs por ambiente
model_env = ChatOpenAI(model="gpt-4o-mini").configurable_alternatives(
    ConfigurableField(id="environment", name="Environment"),
    dev=ChatOpenAI(model="gpt-4o-mini", temperature=1),  # Rápido e barato para dev
    staging=ChatOpenAI(model="gpt-4o", temperature=0.5),  # Intermediário
    prod=ChatOpenAI(model="gpt-4o", temperature=0),  # Melhor qualidade
)

environments = ["dev", "staging", "prod"]

print("\nMesma pergunta em diferentes ambientes:\n")

for env in environments:
    config = RunnableConfig(configurable={"environment": env})
    response = model_env.invoke("O que é LangChain?", config=config)
    print(f"{env.upper()}: {response.content[:80]}...")

print(
    """
Vantagens:
- Mesmo código, diferentes models por ambiente
- Dev usa models baratos, Prod usa melhores
- Fácil trocar configuração via environment variable
"""
)
print()


############################################
# PASSO 6 - Combinando Fields e Alternatives
############################################

print("=" * 70)
print("COMBINANDO configurable_fields E configurable_alternatives")
print("=" * 70)

# Model com AMBOS: fields configuráveis E alternatives
model_combined = (
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    .configurable_fields(
        temperature=ConfigurableField(id="temp", name="Temperature"),
    )
    .configurable_alternatives(
        ConfigurableField(id="provider", name="Provider"),
        anthropic=ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0,
            stop=["\n\n"],
            timeout=30,
        ),
    )
)

print("\n1. OpenAI com temperature=0:")
config_1 = RunnableConfig(configurable={"temp": 0})
response_1 = model_combined.invoke("Diga um número.", config=config_1)
print(f"   {response_1.content}")

print("\n2. OpenAI com temperature=1.5:")
config_2 = RunnableConfig(configurable={"temp": 1.5})
response_2 = model_combined.invoke("Diga um número.", config=config_2)
print(f"   {response_2.content}")

print("\n3. Trocar para Anthropic:")
try:
    config_3 = RunnableConfig(configurable={"provider": "anthropic", "temp": 0})
    response_3 = model_combined.invoke("Diga um número.", config=config_3)
    print(f"   {response_3.content}")
except Exception as e:
    print(f"   Erro: {e}")
    print("   (Configure ANTHROPIC_API_KEY para usar Claude)")

print()


############################################
# PASSO 7 - Configuração via Environment Variables
############################################

print("=" * 70)
print("CONFIGURAÇÃO VIA ENVIRONMENT VARIABLES")
print("=" * 70)

import os

# Ler environment do ambiente (ou usar padrão)
current_env = os.getenv("APP_ENV", "dev")

model_from_env = ChatOpenAI(model="gpt-4o-mini").configurable_alternatives(
    ConfigurableField(id="env", name="Environment"),
    dev=ChatOpenAI(model="gpt-4o-mini", temperature=1),
    prod=ChatOpenAI(model="gpt-4o", temperature=0),
)

# Configurar baseado na env variable
config = RunnableConfig(configurable={"env": current_env})

response = model_from_env.invoke("Olá!", config=config)

print(f"APP_ENV: {current_env}")
print(f"Resposta: {response.content}")
print(
    """
Workflow:
1. Ler environment variable (APP_ENV)
2. Passar para configurable
3. Model é selecionado automaticamente

Deploy:
- Dev: export APP_ENV=dev
- Prod: export APP_ENV=prod
- Mesmo código, diferentes models!
"""
)
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print(
    """
1. configurable_fields():
   - Torna parâmetros do model configuráveis em runtime
   - Parâmetros: temperature, max_tokens, etc.
   - Útil para experimentação sem recriar models
   - ID usado no config: RunnableConfig(configurable={id: value})

2. configurable_alternatives():
   - Permite trocar entre diferentes models em runtime
   - Define um model padrão + alternativas
   - Cada alternativa tem uma chave (string)
   - Trocar via: RunnableConfig(configurable={field_id: key})

3. ConfigurableField:
   - Define um campo configurável
   - Campos: id (obrigatório), name, description
   - id: usado no config
   - name/description: para documentação

4. CASOS DE USO:

   configurable_fields:
   - Experimentar com diferentes temperatures
   - Ajustar max_tokens por contexto
   - Testes de parâmetros
   - Otimização de prompts

   configurable_alternatives:
   - A/B testing (variant_a vs variant_b)
   - Multi-model (OpenAI vs Anthropic vs Google)
   - Ambientes (dev vs staging vs prod)
   - Fallback (model primário vs backup)
   - Custo vs qualidade (gpt-4o vs gpt-4o-mini)

5. A/B TESTING:
   - Defina variantes (variant_a, variant_b)
   - Atribua aleatoriamente aos usuários
   - Rastreie via metadata
   - Compare métricas (custo, latência, satisfação)
   - Escolha o vencedor

6. MULTI-AMBIENTE:
   - dev: models baratos/rápidos
   - staging: intermediário
   - prod: melhor qualidade
   - Controlado via environment variable
   - Mesmo código em todos os ambientes

7. BOAS PRÁTICAS:
   - Use IDs descritivos (temperature, não temp123)
   - Documente cada ConfigurableField
   - Defina default_key em alternatives
   - Combine com metadata para rastreamento
   - Teste todas as alternativas antes de deploy

8. LIMITAÇÕES:
   - Nem todos os parâmetros podem ser configuráveis
   - Depende do runnable específico
   - Alternatives devem ter mesma interface
   - Configuração acontece por invocação (não global)

9. COMBINAÇÕES PODEROSAS:
   - Fields + Alternatives = máxima flexibilidade
   - Environment variables + Configurable = deploy simplificado
   - Metadata + Configurable = rastreamento completo
   - Callbacks + Configurable = observabilidade

10. PRÓXIMOS PASSOS:
    - Para log probabilities, veja sample031.py
    - Para tool choice control, veja sample032.py
"""
)
