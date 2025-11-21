############################################
#
# Exemplo de Parâmetros do Model
#
############################################


############################################
# PASSO 1 - temperature: Controle de Criatividade
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = "Escreva uma frase sobre gatos."

print("=" * 70)
print("PARÂMETRO: temperature")
print("=" * 70)

# temperature = 0 (determinístico, sempre a mesma resposta)
model_temp_0 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("\ntemperature=0 (determinístico):")
for i in range(3):
    response = model_temp_0.invoke(prompt)
    print(f"  Tentativa {i+1}: {response.content}")

# temperature = 0.7 (balanceado)
model_temp_07 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
print("\ntemperature=0.7 (balanceado):")
for i in range(3):
    response = model_temp_07.invoke(prompt)
    print(f"  Tentativa {i+1}: {response.content}")

# temperature = 1.5 (muito criativo)
model_temp_15 = ChatOpenAI(model="gpt-4o-mini", temperature=1.5)
print("\ntemperature=1.5 (muito criativo):")
for i in range(3):
    response = model_temp_15.invoke(prompt)
    print(f"  Tentativa {i+1}: {response.content}")

print("\nObservação: Note como respostas ficam mais variadas com temperature maior.")
print()


############################################
# PASSO 2 - max_tokens: Limite de Comprimento
############################################

print("=" * 70)
print("PARÂMETRO: max_tokens (ou max_completion_tokens)")
print("=" * 70)

# max_tokens limita o tamanho da resposta
model_short = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=20)
model_long = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=200)

prompt_story = "Conte uma história sobre um robô."

print("\nmax_tokens=20 (resposta curta):")
response_short = model_short.invoke(prompt_story)
print(f"  {response_short.content}")
print(f"  Tokens usados: {len(response_short.content.split())}")

print("\nmax_tokens=200 (resposta longa):")
response_long = model_long.invoke(prompt_story)
print(f"  {response_long.content[:200]}...")
print(f"  Tokens usados: ~{len(response_long.content.split())}")

print("\nObservação: max_tokens controla o comprimento máximo da resposta.")
print()


############################################
# PASSO 3 - top_p: Nucleus Sampling
############################################

print("=" * 70)
print("PARÂMETRO: top_p (nucleus sampling)")
print("=" * 70)

# top_p = 1.0 (considera todos os tokens possíveis)
model_top_p_1 = ChatOpenAI(model="gpt-4o-mini", temperature=1, top_p=1.0)

# top_p = 0.1 (considera apenas os top 10% mais prováveis)
model_top_p_01 = ChatOpenAI(model="gpt-4o-mini", temperature=1, top_p=0.1)

prompt_creative = "Complete a frase: O futuro da inteligência artificial será..."

print("\ntop_p=1.0 (mais diversidade):")
for i in range(3):
    response = model_top_p_1.invoke(prompt_creative)
    print(f"  {i+1}. {response.content[:80]}...")

print("\ntop_p=0.1 (mais focado):")
for i in range(3):
    response = model_top_p_01.invoke(prompt_creative)
    print(f"  {i+1}. {response.content[:80]}...")

print("\nObservação: top_p baixo = respostas mais conservadoras.")
print()


############################################
# PASSO 4 - frequency_penalty e presence_penalty
############################################

print("=" * 70)
print("PARÂMETROS: frequency_penalty e presence_penalty")
print("=" * 70)

# Sem penalties
model_no_penalty = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    frequency_penalty=0,
    presence_penalty=0,
)

# Com penalties
model_with_penalty = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    frequency_penalty=1.0,  # Penaliza repetições
    presence_penalty=1.0,   # Encoraja novos tópicos
)

prompt_repeat = "Liste 10 frutas."

print("\nSem penalties:")
response_no = model_no_penalty.invoke(prompt_repeat)
print(f"  {response_no.content}")

print("\nCom frequency_penalty=1.0 e presence_penalty=1.0:")
response_yes = model_with_penalty.invoke(prompt_repeat)
print(f"  {response_yes.content}")

print("""
Observação:
- frequency_penalty: penaliza tokens que já apareceram (evita repetição)
- presence_penalty: encoraja introdução de novos tópicos
- Valores: -2.0 a 2.0 (positivo = penaliza, negativo = encoraja)
""")
print()


############################################
# PASSO 5 - stop: Stop Sequences
############################################

print("=" * 70)
print("PARÂMETRO: stop (stop sequences)")
print("=" * 70)

# stop: lista de strings que param a geração
model_with_stop = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    model_kwargs={"stop": [".", "!", "?"]},  # Para no primeiro ponto final
)

prompt_stop = "Escreva 3 frases sobre o universo"

print("\nSem stop sequences:")
model_no_stop = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response_no_stop = model_no_stop.invoke(prompt_stop)
print(f"  {response_no_stop.content}")

print("\nCom stop=['.', '!', '?'] (para na primeira pontuação):")
response_with_stop = model_with_stop.invoke(prompt_stop)
print(f"  {response_with_stop.content}")

print("\nObservação: stop sequences param a geração quando encontradas.")
print()


############################################
# PASSO 6 - seed: Reproducibilidade
############################################

print("=" * 70)
print("PARÂMETRO: seed (reproducibilidade)")
print("=" * 70)

# seed tenta tornar as respostas reproduzíveis
model_with_seed = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,  # Mesmo com temperature alta
    seed=42,        # seed fixo
)

prompt_seed = "Invente um nome de empresa de tecnologia."

print("\nCom seed=42 (tentativas múltiplas):")
for i in range(5):
    response = model_with_seed.invoke(prompt_seed)
    print(f"  {i+1}. {response.content}")

print("""
Observação:
- seed ajuda com reproducibilidade, mas não garante 100%
- Útil para debugging e testes
- Mesmo seed + mesmos parâmetros = resultados similares
- Pode variar entre versões do modelo
""")
print()


############################################
# PASSO 7 - timeout: Limite de Tempo
############################################

print("=" * 70)
print("PARÂMETRO: timeout")
print("=" * 70)

import time

# timeout em segundos
model_with_timeout = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=5,  # 5 segundos (parâmetro padrão do LangChain)
)

print("\nCom timeout=5 segundos:")
try:
    start = time.time()
    response = model_with_timeout.invoke("Conte-me uma história curta.")
    elapsed = time.time() - start
    print(f"  Sucesso! Tempo: {elapsed:.2f}s")
    print(f"  Resposta: {response.content[:100]}...")
except Exception as e:
    print(f"  Timeout ou erro: {e}")

print("\nObservação: timeout evita requests que demoram demais.")
print()


############################################
# PASSO 8 - Combinando Múltiplos Parâmetros
############################################

print("=" * 70)
print("COMBINANDO MÚLTIPLOS PARÂMETROS")
print("=" * 70)

# Configuração para geração criativa mas controlada
model_creative = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9,           # Criativo
    top_p=0.95,                # Um pouco focado
    max_completion_tokens=150, # Resposta média
    frequency_penalty=0.5,     # Evita repetição moderada
    presence_penalty=0.3,      # Encoraja variedade
    seed=123,                  # Reproducível
)

# Configuração para geração factual e precisa
model_factual = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,             # Determinístico
    top_p=0.1,                 # Muito focado
    max_completion_tokens=100,            # Resposta curta
    frequency_penalty=0,       # Sem penalidades
    presence_penalty=0,
)

prompt_test = "Explique o que é machine learning."

print("\nMODEL CRIATIVO (temp=0.9, top_p=0.95):")
response_creative = model_creative.invoke(prompt_test)
print(f"  {response_creative.content[:150]}...")

print("\nMODEL FACTUAL (temp=0, top_p=0.1):")
response_factual = model_factual.invoke(prompt_test)
print(f"  {response_factual.content[:150]}...")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. temperature (0 a 2, padrão ~0.7):
   - 0: Determinístico, sempre mesma resposta
   - 0.7-1.0: Balanceado (recomendado para a maioria)
   - 1.5-2.0: Muito criativo, pode ser incoerente
   - Use baixo para: código, fatos, classificação
   - Use alto para: escrita criativa, brainstorming

2. max_tokens / max_completion_tokens:
   - Limite de tokens na resposta
   - 1 token ≈ 0.75 palavras em inglês, ~0.5 em português
   - Controla custo e comprimento
   - Se atingir o limite, resposta pode ser cortada

3. top_p (0 a 1, padrão 1):
   - Nucleus sampling (alternativa ao temperature)
   - 1.0: Considera todos os tokens
   - 0.1: Considera apenas top 10% mais prováveis
   - Geralmente use OU temperature OU top_p (não ambos)

4. frequency_penalty (-2 a 2, padrão 0):
   - Penaliza tokens que JÁ apareceram
   - Positivo: evita repetição
   - Negativo: encoraja repetição
   - Útil para listas, variações, evitar loops

5. presence_penalty (-2 a 2, padrão 0):
   - Penaliza tokens do contexto atual
   - Positivo: encoraja novos tópicos
   - Negativo: mantém no mesmo tópico
   - Útil para exploração de ideias

6. stop (lista de strings):
   - Sequências que param a geração
   - Útil para formatos específicos
   - Ex: stop=["\\n\\n"] para um parágrafo
   - Ex: stop=["###"] para seções

7. seed (int):
   - Tenta tornar respostas reproduzíveis
   - Útil para debugging e testes
   - Não garante 100% de reproducibilidade
   - Pode variar entre versões do modelo

8. timeout (segundos):
   - Tempo máximo de espera
   - Evita requests travados
   - Lança TimeoutError se excedido
   - Padrão varia por provider

9. OUTROS PARÂMETROS (varia por provider):
   - n: número de completions a gerar
   - logprobs: retornar log probabilities
   - best_of: gerar N e retornar o melhor
   - logit_bias: aumentar/diminuir probabilidade de tokens
   - user: identificador do usuário (tracking)

10. BOAS PRÁTICAS:
    - Comece com defaults e ajuste conforme necessário
    - temperature=0 para tarefas determinísticas
    - temperature=0.7-1.0 para conversação
    - Use seed para testes e debugging
    - Limite max_tokens para controlar custos
    - Combine parâmetros para efeitos desejados

11. PRÓXIMOS PASSOS:
    - Para rate limiting, veja sample027.py
    - Para token usage tracking, veja sample028.py
    - Para log probabilities, veja sample031.py
""")
