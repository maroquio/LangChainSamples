############################################
#
# Exemplo de Métodos do Model: invoke, stream e batch
#
############################################


############################################
# PASSO 1 - Método invoke() - Resposta Completa
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

print("=" * 70)
print("MÉTODO 1: invoke() - RESPOSTA COMPLETA")
print("=" * 70)

# invoke() retorna a resposta completa de uma vez
start = time.time()
response = model.invoke("Escreva um poema curto sobre o oceano.")
elapsed = time.time() - start

print(f"Resposta completa:\n{response.content}\n")
print(f"Tempo total: {elapsed:.2f}s")
print(f"Tipo do retorno: {type(response)}")
print()


############################################
# PASSO 2 - Método stream() - Token por Token
############################################

print("=" * 70)
print("MÉTODO 2: stream() - TOKEN POR TOKEN")
print("=" * 70)

# stream() retorna chunks conforme são gerados
print("Resposta em streaming (token por token):\n")

start = time.time()
for chunk in model.stream("Escreva um poema curto sobre as estrelas."):
    # chunk.content contém o texto do token
    print(chunk.content, end="", flush=True)
    # end="" evita nova linha, flush=True mostra imediatamente

elapsed = time.time() - start
print(f"\n\nTempo total: {elapsed:.2f}s")
print(f"Tipo de cada chunk: {type(chunk)}")
print()


############################################
# PASSO 3 - stream() com Agregação de Chunks
############################################

print("=" * 70)
print("MÉTODO 3: stream() COM AGREGAÇÃO")
print("=" * 70)

# Coletar todos os chunks e agregar
chunks = []
for chunk in model.stream("Liste 3 fatos interessantes sobre Python."):
    chunks.append(chunk.content)

# Agregar todos os chunks
full_response = "".join(chunks)

print(f"Resposta agregada:\n{full_response}\n")
print(f"Total de chunks recebidos: {len(chunks)}")
print(f"Primeiros 5 chunks: {chunks[:5]}")
print()


############################################
# PASSO 4 - Método batch() - Múltiplas Requests em Paralelo
############################################

print("=" * 70)
print("MÉTODO 4: batch() - MÚLTIPLAS REQUESTS EM PARALELO")
print("=" * 70)

# batch() processa múltiplas inputs de uma vez
inputs = [
    "Traduza para inglês: Bom dia!",
    "Traduza para inglês: Boa noite!",
    "Traduza para inglês: Até logo!",
    "Traduza para inglês: Obrigado!",
]

start = time.time()
responses = model.batch(inputs)
elapsed = time.time() - start

print("Respostas do batch:\n")
for i, response in enumerate(responses, 1):
    print(f"{i}. Input: {inputs[i-1]}")
    print(f"   Output: {response.content}\n")

print(f"Tempo total para {len(inputs)} requests: {elapsed:.2f}s")
print(f"Tempo médio por request: {elapsed/len(inputs):.2f}s")
print()


############################################
# PASSO 5 - Comparação de Performance: Sequencial vs Batch
############################################

print("=" * 70)
print("COMPARAÇÃO: SEQUENCIAL vs BATCH")
print("=" * 70)

questions = [
    "Qual é a capital da França?",
    "Qual é a capital do Brasil?",
    "Qual é a capital do Japão?",
]

# Abordagem SEQUENCIAL (invoke um por vez)
print("1. SEQUENCIAL (invoke para cada):")
start_seq = time.time()
responses_seq = []
for question in questions:
    resp = model.invoke(question)
    responses_seq.append(resp)
time_seq = time.time() - start_seq
print(f"   Tempo total: {time_seq:.2f}s\n")

# Abordagem BATCH (todos de uma vez)
print("2. BATCH (todos juntos):")
start_batch = time.time()
responses_batch = model.batch(questions)
time_batch = time.time() - start_batch
print(f"   Tempo total: {time_batch:.2f}s\n")

speedup = time_seq / time_batch
print(f"SPEEDUP: {speedup:.2f}x mais rápido com batch!")
print()


############################################
# PASSO 6 - stream() vs invoke() - Quando Usar Cada Um
############################################

print("=" * 70)
print("STREAM vs INVOKE - EXPERIÊNCIA DO USUÁRIO")
print("=" * 70)

prompt = "Explique o conceito de recursão em programação em 2 parágrafos."

# invoke: usuário espera até o fim
print("1. COM invoke() - Usuário espera...\n")
start = time.time()
response = model.invoke(prompt)
time_invoke = time.time() - start
print(f"[Aguardando {time_invoke:.2f}s...]\n")
print(response.content)
print()

# stream: usuário vê texto aparecendo
print("2. COM stream() - Usuário vê progresso:\n")
start = time.time()
for chunk in model.stream(prompt):
    print(chunk.content, end="", flush=True)
time_stream = time.time() - start
print(f"\n\n[Streaming completou em {time_stream:.2f}s]")
print()


############################################
# PASSO 7 - batch() com Configurações Diferentes
############################################

print("=" * 70)
print("batch() COM max_concurrency")
print("=" * 70)

# batch() pode ser limitado em concorrência
inputs_many = [f"Conte até {i}" for i in range(1, 6)]

# max_concurrency controla quantos requests rodam simultaneamente
responses = model.batch(inputs_many, config={"max_concurrency": 2})

print(f"Processados {len(inputs_many)} requests com max_concurrency=2")
print("Primeiros 2 resultados:")
for i in range(2):
    print(f"  {i+1}. {responses[i].content[:50]}...")
print()


############################################
# PASSO 8 - Async: ainvoke, astream, abatch
############################################

print("=" * 70)
print("MÉTODOS ASYNC (ainvoke, astream, abatch)")
print("=" * 70)
print("""
LangChain também oferece versões ASYNC dos métodos:

import asyncio

async def async_example():
    model = ChatOpenAI(model="gpt-4o-mini")

    # ainvoke - invoke assíncrono
    response = await model.ainvoke("Hello!")

    # astream - stream assíncrono
    async for chunk in model.astream("Hello!"):
        print(chunk.content, end="")

    # abatch - batch assíncrono
    responses = await model.abatch(["Hi", "Hello", "Hey"])

asyncio.run(async_example())

Vantagens dos métodos async:
- Melhor performance com muitas requests
- Não bloqueia a thread principal
- Ideal para aplicações web (FastAPI, etc.)
- Permite concorrência real

Para exemplos detalhados de async, veja a documentação.
""")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. invoke() - RESPOSTA COMPLETA:
   - Retorna AIMessage completo de uma vez
   - Mais simples de usar
   - Usuário espera até o fim
   - Use quando: batch processing, scripts, não precisa de feedback imediato

2. stream() - TOKEN POR TOKEN:
   - Retorna chunks conforme são gerados
   - Melhor experiência do usuário (vê progresso)
   - Mesmo tempo total, mas parece mais rápido
   - Use quando: chatbots, interfaces interativas, respostas longas

3. batch() - MÚLTIPLAS REQUESTS:
   - Processa múltiplos inputs em paralelo
   - Muito mais rápido que sequencial
   - Economiza tempo de latência
   - Use quando: processar listas, datasets, múltiplas perguntas

4. STREAMING - DETALHES:
   - Chunks são AIMessageChunk objects
   - chunk.content contém o texto parcial
   - Agregue chunks com "".join([chunk.content for chunk in ...])
   - Útil para mostrar progresso em real-time

5. BATCH - DETALHES:
   - Requests são processados em paralelo (até max_concurrency)
   - Retorna lista de respostas na mesma ordem dos inputs
   - Se um falhar, pode lançar exceção (trate individualmente se necessário)
   - max_concurrency evita sobrecarregar a API

6. PERFORMANCE:
   - invoke: ~2-5s por request (depende do tamanho)
   - stream: mesmo tempo total, mas começa mais cedo
   - batch: ~2-5s total para N requests (vs N*2-5s sequencial)
   - Speedup do batch depende de max_concurrency e rate limits

7. ASYNC vs SYNC:
   - Métodos sync: invoke, stream, batch
   - Métodos async: ainvoke, astream, abatch
   - Async permite concorrência sem threads
   - Necessário usar asyncio.run() ou await

8. QUANDO USAR CADA MÉTODO:

   USE invoke():
   - Scripts simples
   - Processamento batch onde UX não importa
   - Quando você precisa da resposta completa antes de continuar

   USE stream():
   - Chatbots e interfaces de chat
   - Respostas longas (artigos, explicações)
   - Quando UX importa (feedback visual)
   - Demonstrações e demos

   USE batch():
   - Processar datasets
   - Múltiplas traduções, classificações
   - Análise de múltiplos textos
   - Quando velocidade é crítica

9. RATE LIMITING:
   - Todos os métodos respeitam rate limits do provider
   - batch() pode bater no rate limit se max_concurrency muito alto
   - Use InMemoryRateLimiter para controlar (veja sample027.py)

10. PRÓXIMOS PASSOS:
    - Para parâmetros de model, veja sample026.py
    - Para rate limiting, veja sample027.py
    - Para token usage tracking, veja sample028.py
""")
