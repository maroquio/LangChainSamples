############################################
#
# Exemplo de Reasoning Models (Extended Thinking)
#
############################################


############################################
# PASSO 1 - Modelo Padrão (sem reasoning)
############################################

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

# Modelo padrão sem capacidade de reasoning estendido
model_standard = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# Problema que requer raciocínio profundo
problem = """
Três amigos (Ana, Bruno e Carlos) têm idades diferentes.
- Ana é mais velha que Bruno
- Carlos não é o mais velho
- A soma das idades é 90 anos
- A diferença entre a idade mais alta e mais baixa é 20 anos
- Bruno tem 25 anos

Quais são as idades de Ana e Carlos?
"""

print("=" * 70)
print("MODELO PADRÃO (sem reasoning):")
print("=" * 70)

start = time.time()
response_standard = model_standard.invoke(problem)
time_standard = time.time() - start

print(f"Resposta: {response_standard.content}")
print(f"\nTempo: {time_standard:.2f}s")
print()


############################################
# PASSO 2 - Reasoning Model (o1/o3-mini)
############################################

# OpenAI o1 e o3-mini são modelos de reasoning
# Eles realizam "pensamento estendido" antes de responder
try:
    model_reasoning = ChatOpenAI(
        model="o1-mini",  # ou "o3-mini" quando disponível
        temperature=1,  # o1 ignora temperature, sempre usa 1
    )

    print("=" * 70)
    print("REASONING MODEL (o1-mini):")
    print("=" * 70)

    start = time.time()
    response_reasoning = model_reasoning.invoke(problem)
    time_reasoning = time.time() - start

    print(f"Resposta: {response_reasoning.content}")
    print(f"\nTempo: {time_reasoning:.2f}s")
    print(f"⚠️ Note que o reasoning model pode ser mais lento,")
    print(f"pois ele 'pensa' mais antes de responder.")
    print()

except Exception as e:
    print("=" * 70)
    print("REASONING MODEL:")
    print("=" * 70)
    print(f"Erro ao usar o1-mini: {e}")
    print("\nModelos de reasoning (o1, o3-mini) podem não estar")
    print("disponíveis em todas as contas. Verifique seu acesso.")
    print()


############################################
# PASSO 3 - Acessando Reasoning Traces/Metadata
############################################

print("=" * 70)
print("ACESSANDO METADATA DO REASONING:")
print("=" * 70)

# Verificar metadata da resposta
try:
    # Tentar acessar com o1-mini
    model_reasoning = ChatOpenAI(model="o1-mini", temperature=1)
    response = model_reasoning.invoke("Resolva: 2x + 5 = 15")

    print(f"Response metadata: {response.response_metadata}")
    print(f"\nUsage metadata: {response.usage_metadata}")

    # Alguns providers expõem "reasoning tokens" ou "thinking time"
    if hasattr(response, "reasoning_content"):
        print(f"\nReasoning content: {response.reasoning_content}")

except Exception as e:
    print(f"Não foi possível acessar o1-mini.")
    print(f"Usando modelo padrão para demonstração de metadata:\n")

    # Fallback para modelo padrão
    response = model_standard.invoke("Resolva: 2x + 5 = 15")
    print(f"Response metadata: {response.response_metadata}")
    print(f"Usage metadata: {response.usage_metadata}")

print()


############################################
# PASSO 4 - Comparação de Capacidades
############################################

print("=" * 70)
print("COMPARAÇÃO: MODELO PADRÃO vs REASONING MODEL")
print("=" * 70)

# Problema de lógica complexo
logic_problem = """
Você tem 8 bolas idênticas em aparência.
Uma delas é ligeiramente mais pesada que as outras.
Você tem uma balança de dois pratos e pode usá-la apenas 2 vezes.
Como você identifica a bola mais pesada?
"""

print(f"\nProblema: {logic_problem}\n")
print("-" * 70)
print("MODELO PADRÃO:")
print("-" * 70)
response_std = model_standard.invoke(logic_problem)
print(str(response_std.content)[:300] + "...\n")

print("-" * 70)
print("REASONING MODEL (quando disponível):")
print("-" * 70)
print("""
Modelos de reasoning (o1, o3-mini) são especialmente bons em:
- Problemas de lógica e matemática
- Quebra-cabeças complexos
- Programação e debugging
- Raciocínio multi-step
- Análise profunda

Eles 'pensam' internamente antes de responder,
similar a como um humano resolveria um problema difícil.
""")
print()


############################################
# PASSO 5 - Problema de Matemática Avançada
############################################

math_problem = """
Encontre todos os números primos entre 100 e 150 que,
quando divididos por 7, deixam resto 3.
Explique seu raciocínio passo a passo.
"""

print("=" * 70)
print("PROBLEMA DE MATEMÁTICA:")
print("=" * 70)
print(f"Problema: {math_problem}\n")

response_math = model_standard.invoke(math_problem)
print(f"Resposta (modelo padrão):\n{response_math.content}")
print()


############################################
# PASSO 6 - Quando Usar Reasoning Models
############################################

print("=" * 70)
print("QUANDO USAR REASONING MODELS:")
print("=" * 70)
print("""
USE REASONING MODELS (o1, o3-mini) PARA:
✓ Problemas matemáticos complexos
✓ Lógica e quebra-cabeças
✓ Código complexo e debugging
✓ Análise profunda e multi-step
✓ Planejamento estratégico
✓ Provas matemáticas
✓ Otimização de algoritmos
✓ Raciocínio científico

USE MODELOS PADRÕES (gpt-4o, etc.) PARA:
✓ Conversação geral
✓ Geração de texto criativo
✓ Resumos e traduções
✓ Classificação simples
✓ Q&A direto
✓ Tarefas rápidas
✓ Quando velocidade importa mais que profundidade
✓ Quando custo é uma preocupação

TRADE-OFFS:
Reasoning Models:
- PRO: Raciocínio mais profundo e preciso
- PRO: Melhor em problemas complexos
- CON: Mais lento (thinking time)
- CON: Mais caro (reasoning tokens)
- CON: Menos flexível (temperature fixo em 1)

Modelos Padrão:
- PRO: Mais rápido
- PRO: Mais barato
- PRO: Mais controle (temperature, etc.)
- PRO: Melhor para conversação
- CON: Pode errar em problemas complexos
""")
print()


############################################
# PASSO 7 - Extended Thinking com Prompt Engineering
############################################

print("=" * 70)
print("ALTERNATIVA: PROMPT ENGINEERING PARA 'REASONING'")
print("=" * 70)

# Você pode simular reasoning com modelos padrão usando prompt engineering
reasoning_prompt = """
Resolva o problema abaixo passo a passo.
Mostre TODO o seu raciocínio antes de dar a resposta final.

Use este formato:
1. ENTENDIMENTO: [explique o problema]
2. ANÁLISE: [identifique os pontos-chave]
3. RACIOCÍNIO: [mostre cada passo da solução]
4. VERIFICAÇÃO: [confira se a resposta faz sentido]
5. RESPOSTA FINAL: [resposta concisa]

Problema: {problem}
"""

problem_simple = "Se um trem viaja 60 km/h e outro 90 km/h em sentidos opostos, partindo de cidades a 300 km de distância, em quanto tempo eles se encontram?"

response_engineered = model_standard.invoke(
    reasoning_prompt.format(problem=problem_simple)
)

print(f"Com prompt engineering para 'forçar' reasoning:\n")
print(response_engineered.content)
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. REASONING MODELS DA OPENAI:
   - o1-preview: modelo completo de reasoning (mais caro)
   - o1-mini: versão menor e mais barata
   - o3-mini: próxima geração (quando disponível)
   - Estes modelos fazem "thinking" interno antes de responder

2. COMO FUNCIONAM:
   - Model gera "reasoning tokens" internos (pensamento)
   - Depois gera a resposta final baseada no pensamento
   - Você só vê a resposta final (não o pensamento)
   - Alguns providers podem expor reasoning traces

3. LIMITAÇÕES DOS REASONING MODELS:
   - Temperature sempre em 1 (não configurável)
   - Não suportam system messages (o1/o1-mini)
   - Não suportam streaming de tokens
   - Podem não suportar function calling (varia)
   - Mais lentos que modelos padrão
   - Mais caros (cobram reasoning tokens)

4. CUSTOS:
   - Reasoning models cobram por:
     * Input tokens (sua pergunta)
     * Reasoning tokens (pensamento interno)
     * Output tokens (resposta)
   - Pode ser 3-5x mais caro que modelos padrão
   - Use apenas quando a qualidade justifica o custo

5. PERFORMANCE:
   - Latência maior (5-30 segundos para problemas complexos)
   - Não ideal para aplicações real-time
   - Ótimo para batch processing de problemas difíceis

6. ALTERNATIVAS:
   - Prompt engineering (Chain of Thought prompting)
   - ReAct pattern com agents
   - Tree of Thoughts
   - Self-consistency (multiple samples)
   - Todos simulam reasoning com modelos padrão

7. CHAIN OF THOUGHT (CoT) PROMPTING:
   - Técnica de prompt que pede "pense passo a passo"
   - Funciona com modelos padrão
   - Mais barato que reasoning models
   - Você tem controle sobre o formato do raciocínio

8. PRÓXIMOS PASSOS:
   - Para métodos invoke/stream/batch, veja sample025.py
   - Para parâmetros de model, veja sample026.py
   - Para token usage tracking, veja sample028.py
""")
