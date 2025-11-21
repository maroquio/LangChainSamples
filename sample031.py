############################################
#
# Exemplo de Log Probabilities (Probabilidades de Tokens)
#
############################################


############################################
# PASSO 1 - Habilitando Log Probabilities
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("HABILITANDO LOG PROBABILITIES")
print("=" * 70)

# Habilitar logprobs no model
# logprobs=True retorna probabilidades dos tokens gerados
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"logprobs": True, "top_logprobs": 3},
)

response = model.invoke("A capital da França é")

print(f"Resposta: {response.content}\n")

# Acessar logprobs
if hasattr(response, "response_metadata") and "logprobs" in response.response_metadata:
    logprobs_data = response.response_metadata["logprobs"]
    print("Log probabilities disponíveis!")
    print(f"Estrutura: {list(logprobs_data.keys())}")
else:
    print("⚠️ Log probabilities não disponíveis (pode variar por provider)")

print()


############################################
# PASSO 2 - Entendendo a Estrutura do Logprobs
############################################

print("=" * 70)
print("ESTRUTURA DO LOGPROBS")
print("=" * 70)

response = model.invoke("O céu é")

print(f"Resposta: {response.content}\n")

# Estrutura do logprobs (OpenAI)
if "logprobs" in response.response_metadata:
    logprobs = response.response_metadata["logprobs"]

    # content contém lista de tokens com suas probabilidades
    if "content" in logprobs:
        print("Análise token por token:\n")

        for i, token_data in enumerate(logprobs["content"][:5]):  # Primeiros 5 tokens
            token = token_data["token"]
            logprob = token_data["logprob"]
            prob = 2 ** logprob  # Converter log probability para probability

            print(f"Token {i+1}: '{token}'")
            print(f"  Log probability: {logprob:.4f}")
            print(f"  Probability: {prob:.4f} ({prob*100:.2f}%)")

            # Top alternatives (se top_logprobs foi especificado)
            if "top_logprobs" in token_data and token_data["top_logprobs"]:
                print(f"  Top alternativas:")
                for alt in token_data["top_logprobs"][:3]:
                    alt_token = alt["token"]
                    alt_prob = 2 ** alt["logprob"]
                    print(f"    - '{alt_token}': {alt_prob*100:.2f}%")
            print()

print()


############################################
# PASSO 3 - Medindo Confiança da Resposta
############################################

print("=" * 70)
print("MEDINDO CONFIANÇA DA RESPOSTA")
print("=" * 70)

def calculate_confidence(response):
    """Calcula confiança média baseada em log probabilities."""
    if "logprobs" not in response.response_metadata:
        return None

    logprobs = response.response_metadata["logprobs"]
    if "content" not in logprobs:
        return None

    # Calcular probabilidade média
    probs = []
    for token_data in logprobs["content"]:
        prob = 2 ** token_data["logprob"]
        probs.append(prob)

    avg_prob = sum(probs) / len(probs) if probs else 0
    return avg_prob


questions = [
    "2 + 2 é igual a",
    "A capital do planeta Marte é",
    "Quantas patas tem um cachorro?",
]

print("Comparando confiança em diferentes respostas:\n")

for question in questions:
    response = model.invoke(question)
    confidence = calculate_confidence(response)

    print(f"Pergunta: '{question}'")
    print(f"Resposta: '{response.content}'")

    if confidence:
        print(f"Confiança média: {confidence*100:.2f}%")
        if confidence > 0.8:
            print("  → Alta confiança ✓")
        elif confidence > 0.5:
            print("  → Confiança moderada")
        else:
            print("  → Baixa confiança ⚠️")
    else:
        print("Confiança não disponível")

    print()


############################################
# PASSO 4 - Detectando Incerteza
############################################

print("=" * 70)
print("DETECTANDO INCERTEZA DO MODEL")
print("=" * 70)

def detect_uncertainty(response, threshold=0.5):
    """Detecta tokens com baixa confiança."""
    if "logprobs" not in response.response_metadata:
        return []

    logprobs = response.response_metadata["logprobs"]
    if "content" not in logprobs:
        return []

    uncertain_tokens = []
    for token_data in logprobs["content"]:
        prob = 2 ** token_data["logprob"]
        if prob < threshold:
            uncertain_tokens.append({
                "token": token_data["token"],
                "probability": prob,
            })

    return uncertain_tokens


prompt = "O inventor do telefone foi"
response = model.invoke(prompt)

print(f"Prompt: '{prompt}'")
print(f"Resposta: '{response.content}'\n")

uncertain = detect_uncertainty(response, threshold=0.6)

if uncertain:
    print(f"⚠️ Encontrados {len(uncertain)} tokens com baixa confiança:")
    for item in uncertain[:5]:
        print(f"  - '{item['token']}': {item['probability']*100:.1f}% confiança")
else:
    print("✓ Todos os tokens têm alta confiança")

print()


############################################
# PASSO 5 - Analisando Alternativas (Top Logprobs)
############################################

print("=" * 70)
print("ANALISANDO ALTERNATIVAS (top_logprobs)")
print("=" * 70)

# Configurar para retornar top 5 alternativas por token
model_top5 = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"logprobs": True, "top_logprobs": 5},
)

response = model_top5.invoke("O melhor time de futebol é")

print(f"Resposta do model: '{response.content}'\n")

if "logprobs" in response.response_metadata:
    logprobs = response.response_metadata["logprobs"]

    if "content" in logprobs and logprobs["content"]:
        # Analisar primeiro token da resposta
        first_token = logprobs["content"][0]

        print(f"Primeiro token gerado: '{first_token['token']}'")
        print(f"Probabilidade: {2**first_token['logprob']*100:.2f}%\n")

        if "top_logprobs" in first_token:
            print("Top 5 alternativas consideradas:")
            for i, alt in enumerate(first_token["top_logprobs"], 1):
                alt_token = alt["token"]
                alt_prob = 2 ** alt["logprob"]
                print(f"  {i}. '{alt_token}': {alt_prob*100:.2f}%")

print("""
Top logprobs mostram outras opções que o model considerou.
Útil para entender o "raciocínio" do model.
""")
print()


############################################
# PASSO 6 - Casos de Uso: Validação de Respostas
############################################

print("=" * 70)
print("CASO DE USO: VALIDAÇÃO DE RESPOSTAS")
print("=" * 70)

def should_accept_response(response, min_confidence=0.7):
    """Decide se a resposta deve ser aceita baseado na confiança."""
    confidence = calculate_confidence(response)

    if confidence is None:
        return True, "Sem dados de confiança"

    if confidence >= min_confidence:
        return True, f"Alta confiança ({confidence*100:.1f}%)"
    else:
        return False, f"Baixa confiança ({confidence*100:.1f}%)"


questions_test = [
    "Quanto é 5 + 5?",
    "Qual é a cor do cavalo branco de Napoleão?",
]

print("Sistema de validação automática:\n")

for question in questions_test:
    response = model.invoke(question)
    accept, reason = should_accept_response(response, min_confidence=0.7)

    print(f"Pergunta: {question}")
    print(f"Resposta: {response.content}")
    print(f"Status: {'✓ ACEITA' if accept else '✗ REJEITADA'} - {reason}\n")

print("""
Aplicações:
- Filtrar respostas com baixa confiança
- Solicitar revisão humana para casos incertos
- Routing: alta confiança → automático, baixa → humano
- Evitar hallucinations
""")
print()


############################################
# PASSO 7 - Limitações e Considerações
############################################

print("=" * 70)
print("LIMITAÇÕES E CONSIDERAÇÕES")
print("=" * 70)

print("""
1. DISPONIBILIDADE:
   - Nem todos os providers suportam logprobs
   - OpenAI: ✓ Suportado
   - Anthropic (Claude): ✗ Não suportado nativamente
   - Google (Gemini): Parcialmente suportado
   - Verifique documentação do provider

2. PERFORMANCE:
   - Logprobs aumenta latência (ligeiramente)
   - Aumenta tamanho do response
   - Pode ter custo adicional (verifique pricing)

3. INTERPRETAÇÃO:
   - Alta probabilidade ≠ resposta correta
   - Model pode estar "confiante mas errado"
   - Use como um sinal, não como verdade absoluta
   - Combine com outras validações

4. TEMPERATURE E LOGPROBS:
   - Temperature afeta distribuição de probabilidades
   - temperature=0: mais determinístico, probs mais altas
   - temperature=1: mais variado, probs mais distribuídas
   - Para análise, considere usar temperature=0

5. THRESHOLD RECOMENDADOS:
   - > 80%: Alta confiança
   - 50-80%: Confiança moderada
   - < 50%: Baixa confiança (considere rejeitar)
   - Ajuste baseado no seu caso de uso

6. CUSTOS:
   - top_logprobs aumenta o tamanho do response
   - Mais dados = mais bandwidth
   - Considere usar apenas quando necessário
   - Não use em produção sem medir impacto
""")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. LOG PROBABILITIES:
   - Log probability: logaritmo da probabilidade do token
   - Probability: 2^(log_probability) ou exp(log_probability)
   - Valor entre 0 e 1 (0% a 100%)
   - Quanto maior, mais "confiante" o model

2. HABILITANDO LOGPROBS:
   - OpenAI: model_kwargs={"logprobs": True}
   - top_logprobs: quantas alternativas mostrar (1-20)
   - Nem todos os models/providers suportam

3. ESTRUTURA (OpenAI):
   response.response_metadata["logprobs"]["content"] = [
     {
       "token": str,           # Token gerado
       "logprob": float,       # Log probability
       "top_logprobs": [...]   # Alternativas
     },
     ...
   ]

4. CASOS DE USO:

   VALIDAÇÃO:
   - Rejeitar respostas com baixa confiança
   - Routing baseado em confiança
   - Solicitar revisão humana

   ANÁLISE:
   - Entender "raciocínio" do model
   - Identificar pontos de incerteza
   - Comparar diferentes prompts

   DEBUGGING:
   - Identificar onde o model hesita
   - Testar eficácia de prompts
   - Encontrar edge cases

   SEGURANÇA:
   - Detectar possíveis hallucinations
   - Filtro de qualidade automático
   - Compliance (não responder sem certeza)

5. CONFIANÇA vs CORREÇÃO:
   - Alta confiança ≠ sempre correto
   - Model pode estar confiante mas errado
   - Use como INDICADOR, não como garantia
   - Combine com outras validações (fact-checking, etc.)

6. THRESHOLDS:
   - Defina threshold baseado no caso de uso
   - Aplicações críticas: threshold alto (>80%)
   - Uso geral: threshold médio (>60%)
   - Experimental: threshold baixo (>40%)

7. ALTERNATIVAS SEM LOGPROBS:
   - Multiple samples: gerar N respostas e comparar consistência
   - Self-consistency: perguntar de formas diferentes
   - Prompt engineering: pedir "Quão confiante você está?"
   - Temperature=0: mais determinístico

8. PRÓXIMOS PASSOS:
   - Para tool choice control, veja sample032.py
   - Para structured output, veja sample021.py
   - Para token usage, veja sample028.py
""")
