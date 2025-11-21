############################################
#
# Exemplo de Rate Limiting com InMemoryRateLimiter
#
############################################


############################################
# PASSO 1 - Problema: Sem Rate Limiting
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

print("=" * 70)
print("SEM RATE LIMITING - Pode exceder limites da API")
print("=" * 70)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tentar fazer muitas requests rapidamente
print("\nFazendo 5 requests rápidas sem rate limiting:")
start = time.time()

for i in range(5):
    try:
        response = model.invoke(f"Diga o número {i+1}")
        print(f"  {i+1}. {response.content}")
    except Exception as e:
        print(f"  {i+1}. ERRO: {e}")

elapsed = time.time() - start
print(f"\nTempo total: {elapsed:.2f}s")
print("⚠️ Sem rate limiting, você pode exceder os limites da API e receber erros.")
print()


############################################
# PASSO 2 - Usando InMemoryRateLimiter
############################################

from langchain_core.rate_limiters import InMemoryRateLimiter

print("=" * 70)
print("COM RATE LIMITING - InMemoryRateLimiter")
print("=" * 70)

# Criar um rate limiter
# requests_per_second: máximo de requests por segundo
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # Máximo 1 request por segundo
    check_every_n_seconds=0.1,  # Verificar a cada 0.1s
    max_bucket_size=10,  # Tamanho do bucket (burst control)
)

# Aplicar rate limiter ao model
model_with_limiter = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    rate_limiter=rate_limiter,
)

print("\nFazendo 5 requests com rate_limiter (1 req/s):")
start = time.time()

for i in range(5):
    response = model_with_limiter.invoke(f"Diga o número {i+1}")
    print(f"  {i+1}. {response.content} (tempo: {time.time() - start:.1f}s)")

elapsed = time.time() - start
print(f"\nTempo total: {elapsed:.2f}s")
print("✓ Rate limiter garantiu que não excedemos 1 request/segundo.")
print()


############################################
# PASSO 3 - Ajustando requests_per_second
############################################

print("=" * 70)
print("AJUSTANDO requests_per_second")
print("=" * 70)

# Rate limiter mais permissivo
rate_limiter_fast = InMemoryRateLimiter(
    requests_per_second=3,  # 3 requests por segundo
)

model_fast = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    rate_limiter=rate_limiter_fast,
)

print("\nCom rate_limiter (3 req/s):")
start = time.time()

for i in range(6):
    response = model_fast.invoke(f"Número {i+1}")
    timestamp = time.time() - start
    print(f"  {i+1}. Concluído em {timestamp:.1f}s")

elapsed = time.time() - start
print(f"\nTempo total: {elapsed:.2f}s")
print("Observe que os 3 primeiros são rápidos, depois há throttling.")
print()


############################################
# PASSO 4 - max_bucket_size: Controle de Burst
############################################

print("=" * 70)
print("PARÂMETRO: max_bucket_size (burst control)")
print("=" * 70)

print("""
max_bucket_size controla quantos "tokens" podem acumular no bucket.

Pense como um balde que enche a N tokens/segundo:
- requests_per_second=2: balde ganha 2 tokens/segundo
- max_bucket_size=5: balde pode ter no máximo 5 tokens
- Cada request consome 1 token

Com max_bucket_size maior:
- Permite "bursts" (rajadas) de requests
- Útil quando requests vêm em ondas

Exemplo:
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    max_bucket_size=5,
)

Isso permite:
- 5 requests imediatas (esvazia o bucket)
- Depois, 1 request por segundo (conforme recarrega)
""")
print()


############################################
# PASSO 5 - Compartilhando Rate Limiter entre Models
############################################

print("=" * 70)
print("COMPARTILHANDO RATE LIMITER ENTRE MODELS")
print("=" * 70)

# Um único rate limiter para múltiplos models
shared_limiter = InMemoryRateLimiter(
    requests_per_second=2,
)

model_a = ChatOpenAI(model="gpt-4o-mini", rate_limiter=shared_limiter)
model_b = ChatOpenAI(model="gpt-4o-mini", rate_limiter=shared_limiter)

print("\nDois models compartilhando o mesmo rate limiter (2 req/s total):")
start = time.time()

# Alternar entre models
for i in range(4):
    if i % 2 == 0:
        response = model_a.invoke(f"Model A: {i+1}")
        print(f"  Model A: {response.content} ({time.time()-start:.1f}s)")
    else:
        response = model_b.invoke(f"Model B: {i+1}")
        print(f"  Model B: {response.content} ({time.time()-start:.1f}s)")

elapsed = time.time() - start
print(f"\nTempo total: {elapsed:.2f}s")
print("✓ Ambos os models respeitam o mesmo limite de 2 req/s.")
print()


############################################
# PASSO 6 - Rate Limiting com batch()
############################################

print("=" * 70)
print("RATE LIMITING COM batch()")
print("=" * 70)

rate_limiter_batch = InMemoryRateLimiter(requests_per_second=2)
model_batch = ChatOpenAI(model="gpt-4o-mini", rate_limiter=rate_limiter_batch)

inputs = [f"Translate to English: Olá {i}" for i in range(5)]

print("\nbatch() com rate limiter (2 req/s):")
start = time.time()
responses = model_batch.batch(inputs)
elapsed = time.time() - start

for i, resp in enumerate(responses):
    print(f"  {i+1}. {resp.content}")

print(f"\nTempo total: {elapsed:.2f}s")
print("Rate limiter aplica-se a cada request dentro do batch.")
print()


############################################
# PASSO 7 - Thread-Safe: Múltiplas Threads
############################################

print("=" * 70)
print("THREAD-SAFE: Rate Limiter com Múltiplas Threads")
print("=" * 70)

from threading import Thread

def make_request(model, i, results):
    """Faz uma request e armazena o resultado."""
    try:
        response = model.invoke(f"Thread {i}")
        results[i] = f"Thread {i}: {response.content[:50]}"
    except Exception as e:
        results[i] = f"Thread {i}: ERRO - {e}"


rate_limiter_thread = InMemoryRateLimiter(requests_per_second=2)
model_thread = ChatOpenAI(model="gpt-4o-mini", rate_limiter=rate_limiter_thread)

results = {}
threads = []

print("\nCriando 5 threads simultâneas (rate limit 2 req/s):")
start = time.time()

for i in range(5):
    thread = Thread(target=make_request, args=(model_thread, i, results))
    threads.append(thread)
    thread.start()

# Esperar todas as threads
for thread in threads:
    thread.join()

elapsed = time.time() - start

for i in sorted(results.keys()):
    print(f"  {results[i]}")

print(f"\nTempo total: {elapsed:.2f}s")
print("✓ Rate limiter é thread-safe e controla requests de todas as threads.")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. InMemoryRateLimiter:
   - Controla taxa de requests ao model
   - Previne exceder rate limits da API
   - Implementado usando Token Bucket algorithm
   - Thread-safe (pode usar com múltiplas threads)

2. PARÂMETROS:

   requests_per_second (obrigatório):
   - Máximo de requests permitidos por segundo
   - Ex: requests_per_second=5 → máx 5 req/s
   - Ajuste baseado nos limites do seu tier de API

   check_every_n_seconds (padrão: 0.1):
   - Frequência de verificação do bucket
   - Valores menores = mais preciso, mais overhead
   - Valores maiores = menos preciso, menos overhead

   max_bucket_size (padrão: 1):
   - Tamanho máximo do bucket de tokens
   - Controla "burst" (rajadas) de requests
   - max_bucket_size=1: nenhum burst
   - max_bucket_size=10: permite até 10 requests seguidas

3. TOKEN BUCKET ALGORITHM:
   - Bucket acumula tokens a N tokens/segundo
   - Cada request consome 1 token
   - Se bucket vazio, request espera
   - max_bucket_size limita acúmulo

4. QUANDO USAR:
   - APIs com rate limits estritos
   - Evitar erros 429 (Too Many Requests)
   - Controlar custos (limitar requests/segundo)
   - Processar grandes datasets sem sobrecarregar API
   - Ambientes multi-thread

5. RATE LIMITS POR PROVIDER:

   OpenAI (varia por tier):
   - Free: ~3 req/min = 0.05 req/s
   - Tier 1: ~500 req/min = 8.3 req/s
   - Tier 5: ~10,000 req/min = 166 req/s

   Anthropic:
   - Varia por plano (consulte dashboard)

   Google (Gemini):
   - Free: 15 req/min = 0.25 req/s
   - Paid: varia (consulte documentação)

6. COMPARTILHAMENTO:
   - Você pode compartilhar um rate limiter entre múltiplos models
   - Útil quando múltiplos models usam a mesma API key
   - Limita requests TOTAIS de todos os models combinados

7. BATCH E RATE LIMITING:
   - batch() faz múltiplas requests internamente
   - Rate limiter controla cada request individual dentro do batch
   - batch(10) com rate_limiter(2 req/s) leva ~5s

8. ALTERNATIVAS:
   - tenacity: retry com backoff exponencial
   - asyncio.Semaphore: para código assíncrono
   - Redis rate limiter: para aplicações distribuídas
   - Cloud rate limiters: API Gateway, etc.

9. BOAS PRÁTICAS:
   - Defina requests_per_second ABAIXO do limite da API (margem de segurança)
   - Use max_bucket_size para permitir bursts controlados
   - Monitore erros 429 e ajuste o rate limiter
   - Combine com retry logic (tenacity) para robustez
   - Em produção, considere rate limiters distribuídos (Redis)

10. LIMITAÇÕES:
    - InMemoryRateLimiter: apenas processo único
    - Não funciona entre processos/servidores
    - Para sistemas distribuídos, use Redis/external rate limiter
    - Não considera rate limits por token (apenas por request)

11. PRÓXIMOS PASSOS:
    - Para token usage tracking, veja sample028.py
    - Para invocation config, veja sample029.py
    - Para configurable models, veja sample030.py
""")
