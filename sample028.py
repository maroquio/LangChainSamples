############################################
#
# Exemplo de Token Usage Tracking (Rastreamento de Uso de Tokens)
#
############################################


############################################
# PASSO 1 - Acessando usage_metadata da Resposta
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 70)
print("ACESSANDO usage_metadata DA RESPOSTA")
print("=" * 70)

response = model.invoke("Explique o que é Python em 2 frases.")

print(f"Resposta: {response.content}\n")

# Acessar usage metadata
usage = response.usage_metadata

if usage:
    print("Token Usage:")
    print(f"  Input tokens: {usage['input_tokens']}")
    print(f"  Output tokens: {usage['output_tokens']}")
    print(f"  Total tokens: {usage['total_tokens']}")
else:
    print("⚠️ usage_metadata não disponível")
print()


############################################
# PASSO 2 - Rastreando Tokens em Múltiplas Chamadas
############################################

print("=" * 70)
print("RASTREANDO TOKENS EM MÚLTIPLAS CHAMADAS")
print("=" * 70)

questions = [
    "Qual a capital da França?",
    "Qual a capital do Brasil?",
    "Qual a capital do Japão?",
]

total_input = 0
total_output = 0
total_tokens = 0

print("Fazendo múltiplas chamadas:\n")

for i, question in enumerate(questions, 1):
    response = model.invoke(question)
    usage = response.usage_metadata

    if usage:
        total_input += usage['input_tokens']
        total_output += usage['output_tokens']
        total_tokens += usage['total_tokens']

        print(f"{i}. {question}")
        print(f"   Resposta: {response.content}")
        print(
            f"   Tokens: {usage['input_tokens']} in + {usage['output_tokens']} out = {usage['total_tokens']} total\n"
        )
    else:
        print(f"{i}. {question}")
        print(f"   Resposta: {response.content}")
        print(f"   ⚠️ Usage metadata não disponível\n")

print("TOTAIS:")
print(f"  Total input tokens: {total_input}")
print(f"  Total output tokens: {total_output}")
print(f"  Total tokens: {total_tokens}")
print()


############################################
# PASSO 3 - Calculando Custos Baseado em Tokens
############################################

print("=" * 70)
print("CALCULANDO CUSTOS BASEADO EM TOKENS")
print("=" * 70)

# Preços do gpt-4o-mini (exemplo, verifique preços atuais)
PRICE_INPUT_PER_1M = 0.150  # USD por 1 milhão de input tokens
PRICE_OUTPUT_PER_1M = 0.600  # USD por 1 milhão de output tokens


def calculate_cost(input_tokens, output_tokens):
    """Calcula o custo em USD baseado no uso de tokens."""
    cost_input = (input_tokens / 1_000_000) * PRICE_INPUT_PER_1M
    cost_output = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M
    return cost_input + cost_output


prompt = "Escreva um parágrafo sobre inteligência artificial."
response = model.invoke(prompt)
usage = response.usage_metadata

print(f"Prompt: {prompt}")
print(f"Resposta: {response.content[:100]}...\n")

if usage:
    cost = calculate_cost(usage['input_tokens'], usage['output_tokens'])
    print(f"Token Usage:")
    print(f"  Input: {usage['input_tokens']} tokens")
    print(f"  Output: {usage['output_tokens']} tokens")
    print(f"  Total: {usage['total_tokens']} tokens")
    print(f"\nCusto estimado: ${cost:.6f} USD")
    print(f"Custo em centavos: {cost * 100:.4f}¢")
else:
    print("⚠️ Usage metadata não disponível")
print()


############################################
# PASSO 4 - Usando get_usage_metadata_callback()
############################################

print("=" * 70)
print("USANDO get_usage_metadata_callback() CONTEXT MANAGER")
print("=" * 70)

from langchain_core.callbacks import get_usage_metadata_callback

# Context manager para agregar automaticamente
with get_usage_metadata_callback() as cb:
    # Fazer múltiplas chamadas dentro do context
    model.invoke("Traduza para inglês: Olá")
    model.invoke("Traduza para inglês: Bom dia")
    model.invoke("Traduza para inglês: Boa noite")

    # Acessar totais agregados
    total_usage = cb.usage_metadata

if total_usage:
    print("Total agregado de 3 chamadas:")
    # cb.usage_metadata retorna dict aninhado por model
    # Estrutura: {'model-name': {'input_tokens': X, 'output_tokens': Y, ...}}
    # Agregar tokens de todos os models
    input_tok = 0
    output_tok = 0
    total_tok = 0

    for model_name, usage_data in total_usage.items():
        input_tok += usage_data.get('input_tokens', 0)
        output_tok += usage_data.get('output_tokens', 0)
        total_tok += usage_data.get('total_tokens', 0)

    print(f"  Input tokens: {input_tok}")
    print(f"  Output tokens: {output_tok}")
    print(f"  Total tokens: {total_tok}")

    if input_tok and output_tok:
        cost_total = calculate_cost(input_tok, output_tok)
        print(f"\nCusto total: ${cost_total:.6f} USD")
else:
    print("⚠️ Total usage metadata não disponível")
print()


############################################
# PASSO 5 - Rastreando Tokens com Streaming
############################################

print("=" * 70)
print("RASTREANDO TOKENS COM STREAMING")
print("=" * 70)

print("Resposta em streaming:\n")

# Com streaming, usage_metadata vem no último chunk
last_chunk = None
for chunk in model.stream("Liste 3 cores primárias."):
    print(chunk.content, end="", flush=True)
    last_chunk = chunk

print("\n")

# Usage metadata está no último chunk (quando disponível)
usage = None
if last_chunk and hasattr(last_chunk, "usage_metadata"):
    usage = last_chunk.usage_metadata

if usage:
    print(f"Tokens (do último chunk):")
    print(f"  Input: {usage['input_tokens']}")
    print(f"  Output: {usage['output_tokens']}")
    print(f"  Total: {usage['total_tokens']}")
else:
    print("⚠️ usage_metadata pode não estar disponível em streaming")
    print("Depende do provider e configuração.")

print()


############################################
# PASSO 6 - Rastreando Tokens com batch()
############################################

print("=" * 70)
print("RASTREANDO TOKENS COM batch()")
print("=" * 70)

inputs = [
    "Diga: Um",
    "Diga: Dois",
    "Diga: Três",
]

responses = model.batch(inputs)

batch_input = 0
batch_output = 0

print("Respostas do batch:\n")
for i, resp in enumerate(responses, 1):
    usage = resp.usage_metadata

    print(f"{i}. {resp.content}")

    if usage:
        batch_input += usage['input_tokens']
        batch_output += usage['output_tokens']
        print(
            f"   Tokens: {usage['input_tokens']} + {usage['output_tokens']} = {usage['total_tokens']}\n"
        )
    else:
        print(f"   ⚠️ Usage não disponível\n")

print(f"Total do batch:")
print(f"  Input: {batch_input}")
print(f"  Output: {batch_output}")
print(f"  Total: {batch_input + batch_output}")

cost_batch = calculate_cost(batch_input, batch_output)
print(f"  Custo: ${cost_batch:.6f} USD")
print()


############################################
# PASSO 7 - Monitorando Tokens de Múltiplos Models
############################################

print("=" * 70)
print("MONITORANDO TOKENS DE MÚLTIPLOS MODELS")
print("=" * 70)

# Usar callback para rastrear uso de múltiplos models
with get_usage_metadata_callback() as cb:
    model_mini = ChatOpenAI(model="gpt-4o-mini")
    model_4o = ChatOpenAI(model="gpt-4o")

    # Fazer chamadas com modelos diferentes
    model_mini.invoke("Olá!")
    model_mini.invoke("Como vai?")
    model_4o.invoke("Explique brevemente quantum computing.")

    total = cb.usage_metadata

if total:
    print("Total agregado de múltiplos models:")
    # cb.usage_metadata retorna dict aninhado por model
    # Agregar tokens de todos os models
    input_tok = 0
    output_tok = 0
    total_tok = 0

    for model_name, usage_data in total.items():
        input_tok += usage_data.get('input_tokens', 0)
        output_tok += usage_data.get('output_tokens', 0)
        total_tok += usage_data.get('total_tokens', 0)

    print(f"  Input tokens: {input_tok}")
    print(f"  Output tokens: {output_tok}")
    print(f"  Total tokens: {total_tok}")
else:
    print("⚠️ Total usage metadata não disponível")

# NOTA: Para custo preciso, você precisa separar por model
# pois preços variam (gpt-4o é mais caro que gpt-4o-mini)
print("\n⚠️ IMPORTANTE: Custos variam por model!")
print("   Para custo preciso, rastreie cada model separadamente.")
print()


############################################
# PASSO 8 - Tokens vs Caracteres: Estimativa
############################################

print("=" * 70)
print("TOKENS vs CARACTERES - ESTIMATIVA")
print("=" * 70)

texts = [
    "Hello, world!",
    "Olá, mundo!",
    "こんにちは世界",  # Japonês
    "The quick brown fox jumps over the lazy dog.",
]

print("Relação aproximada entre caracteres e tokens:\n")

for text in texts:
    response = model.invoke(f"Repita: {text}")
    usage = response.usage_metadata

    print(f"Texto: '{text}'")
    print(f"  Caracteres: {len(text)}")

    if usage and usage.get('output_tokens'):
        # Input inclui o prompt "Repita: " + text
        char_count = len(text)
        # Estimativa (não exata, pois inclui o prompt)
        ratio = char_count / usage['output_tokens']
        print(f"  Output tokens: {usage['output_tokens']}")
        print(f"  Aprox: {ratio:.2f} caracteres/token\n")
    else:
        print(f"  ⚠️ Usage não disponível\n")

print(
    """
Observações:
- Inglês: ~4 caracteres por token
- Português: ~3-4 caracteres por token
- Idiomas asiáticos: ~1-2 caracteres por token
- Tokens != palavras (podem ser partes de palavras)
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
1. usage_metadata:
   - Disponível em response.usage_metadata
   - Dicionário com: input_tokens, output_tokens, total_tokens
   - Nem todos os providers expõem todos os campos
   - Crucial para monitorar custos

2. CAMPOS DO usage_metadata:
   - input_tokens: tokens do seu prompt
   - output_tokens: tokens da resposta do model
   - total_tokens: input + output
   - Alguns providers incluem campos extras (cache_read_tokens, etc.)

3. CÁLCULO DE CUSTOS:
   - Preços variam por model e provider
   - Geralmente: USD por 1 milhão de tokens
   - Input tokens costumam ser mais baratos que output
   - Verifique preços atuais na documentação do provider

4. get_usage_metadata_callback():
   - Context manager para agregar uso automaticamente
   - Rastreia múltiplas chamadas
   - Acessa totais com cb.usage_metadata
   - Útil para sessões/conversas longas

5. STREAMING E TOKENS:
   - Com stream(), usage_metadata pode vir no último chunk
   - Nem todos os providers expõem usage em streaming
   - Alguns providers só expõem após completar o stream
   - Verifique suporte por provider

6. BATCH E TOKENS:
   - batch() retorna lista de responses
   - Cada response tem seu próprio usage_metadata
   - Some manualmente para obter total do batch

7. PREÇOS POR MODEL (OpenAI - exemplo, verifique preços atuais):

   GPT-4o:
   - Input: $2.50 / 1M tokens
   - Output: $10.00 / 1M tokens

   GPT-4o-mini:
   - Input: $0.150 / 1M tokens
   - Output: $0.600 / 1M tokens

   GPT-3.5-turbo:
   - Input: $0.50 / 1M tokens
   - Output: $1.50 / 1M tokens

8. ESTIMANDO TOKENS:
   - Inglês: ~750 palavras = 1000 tokens
   - 1 token ≈ 4 caracteres em inglês
   - 1 token ≈ 0.75 palavras em inglês
   - Português/espanhol: similar a inglês
   - Idiomas asiáticos: mais tokens por caractere

9. OTIMIZAÇÃO DE CUSTOS:
   - Use models menores quando possível (gpt-4o-mini vs gpt-4o)
   - Limite max_tokens para controlar output
   - Monitore e analise uso regularmente
   - Considere caching de prompts repetidos
   - Use system prompts concisos

10. MONITORAMENTO EM PRODUÇÃO:
    - Log todos os usages para análise
    - Agregue por usuário, sessão, feature
    - Configure alertas para uso anormal
    - Dashboard com métricas de custo
    - Revise custos mensalmente

11. PRÓXIMOS PASSOS:
    - Para invocation config, veja sample029.py
    - Para configurable models, veja sample030.py
    - Para log probabilities, veja sample031.py
"""
)
