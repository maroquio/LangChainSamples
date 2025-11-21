############################################

# Exemplo de Agente Básico com LangChain 1.0

############################################


############################################
# PASSO 1 - Definir o prompt do sistema
############################################

SYSTEM_PROMPT = """
Você é um especialista em astronomia muito inteligente e prestativo.
"""

############################################
# PASSO 2 - Inicializar o agente
############################################

from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

agent = create_agent(
    "gpt-4o-mini",  # o modelo de linguagem a ser usado
    system_prompt=SYSTEM_PROMPT,  # o prompt do sistema definido no Passo 1
)

############################################
# PASSO 3 - Usar o agente
############################################

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Qual é a distância da Terra ao Sol?"}]},
)

print(response["messages"][-1].content)

############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print()
print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. AGENTE BÁSICO:
   - Este é o exemplo mais simples de um agente LangChain
   - Usa apenas system_prompt, sem ferramentas ou memória
   - Ideal para começar a entender o funcionamento básico

2. FUNÇÃO create_agent:
   - Pode receber o nome do modelo diretamente ("gpt-4o-mini")
   - O LangChain inicializa o modelo automaticamente
   - Simplifica a criação de agentes básicos

3. FORMATO DA RESPOSTA:
   - agent.invoke() retorna um dicionário com chave "messages"
   - A última mensagem ([-1]) contém a resposta do agente
   - Use .content para acessar o texto da resposta

4. VARIÁVEIS DE AMBIENTE:
   - load_dotenv() carrega as chaves de API do arquivo .env
   - Necessário ter OPENAI_API_KEY configurada
   - O arquivo .env não deve ser commitado no repositório

5. PRÓXIMOS PASSOS:
   - Para adicionar ferramentas, veja sample002.py
   - Para configurações avançadas de modelo, veja sample003.py e sample004.py
   - Para contexto personalizado, veja sample005.py
""")
