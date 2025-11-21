# LangChain Samples - Exemplos Práticos

Coleção de exemplos práticos demonstrando recursos e funcionalidades do LangChain 1.0+, incluindo agentes, ferramentas, middleware, e muito mais, totalmente baseada na documentação oficial.

## 📋 Pré-requisitos

- **Python 3.12+** instalado
- **uv** (gerenciador de pacotes e ambientes virtuais Python)
- **Conta OpenAI** com API key válida

## 🚀 Instalação

### 1. Instalar o uv

O [uv](https://github.com/astral-sh/uv) é um gerenciador de pacotes Python extremamente rápido, escrito em Rust.

**macOS e Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verificar instalação:**
```bash
uv --version
```

### 2. Clonar o Repositório

```bash
git clone https://github.com/maroquio/LangChainSamples.git
cd LangChainSamples
```

### 3. Criar Ambiente Virtual

O uv cria e gerencia automaticamente ambientes virtuais:

```bash
uv venv
```

Isso criará um ambiente virtual na pasta `.venv`.

### 4. Instalar Dependências

**Opção 1 - Usando requirements.txt (Recomendado):**
```bash
uv pip install -r requirements.txt
```

**Opção 2 - Usando pip tradicional:**
```bash
pip install -r requirements.txt
```

**Opção 3 - Sincronizar com pyproject.toml:**
```bash
uv sync
```

**Opção 4 - Instalar pacotes manualmente:**
```bash
uv pip install langchain langchain-openai langchain-core langgraph python-dotenv
```

### Dependências do Projeto

O projeto utiliza as seguintes bibliotecas:

- `langchain>=1.0.8` - Framework principal
- `langchain-openai>=1.0.3` - Integração com OpenAI
- `langchain-core>=1.0.0` - Core do LangChain
- `langgraph>=1.0.0` - Para checkpointer e memória
- `python-dotenv>=1.2.1` - Carregar variáveis de ambiente

## ⚙️ Configuração

### Criar arquivo .env

Crie um arquivo `.env` na raiz do projeto com suas credenciais:

```bash
# Copiar template (se existir)
cp .env.example .env

# Ou criar manualmente
touch .env
```

### Adicionar variáveis de ambiente

Edite o arquivo `.env` e adicione sua API key da OpenAI:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**⚠️ Importante:** Nunca commite o arquivo `.env` no Git! Certifique-se de que está no `.gitignore`.

## 🏃 Como Executar

### Ativar o ambiente virtual

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

### Executar um exemplo

```bash
python sample001.py
```

Ou execute diretamente com uv (sem ativar o ambiente):
```bash
uv run sample001.py
```

## 📁 Estrutura do Projeto

```
LangChainOfficialDocs/
├── .env                 # Variáveis de ambiente (criar manualmente)
├── .gitignore          # Arquivos ignorados pelo Git
├── .python-version     # Versão do Python
├── pyproject.toml      # Configuração do projeto e dependências
├── requirements.txt    # Dependências do projeto (pip/uv)
├── uv.lock            # Lock file de dependências
├── README.md          # Este arquivo
├── sample001.py       # Exemplo 1: Agente básico
├── sample002.py       # Exemplo 2: Agente com ferramenta
├── sample003.py       # Exemplo 3: ChatOpenAI com parâmetros personalizados
├── sample004.py       # Exemplo 4: Modelo multi-provedor
├── sample005.py       # Exemplo 5: Contexto de runtime personalizado
├── sample006.py       # Exemplo 6: Contexto e resposta estruturada
├── sample007.py       # Exemplo 7: Agente SEM memória
├── sample008.py       # Exemplo 8: Agente COM memória
├── sample009.py       # Exemplo 9: Memória e múltiplos contextos
├── sample010.py       # Exemplo 10: Seleção dinâmica de modelo
├── sample011.py       # Exemplo 11: Tratamento de erros em ferramentas
├── sample012.py       # Exemplo 12: System prompt dinâmico
├── sample013.py       # Exemplo 13: Passagem de sequência de mensagens
├── sample014.py       # Exemplo 14: Saída estruturada com ToolStrategy
├── sample015.py       # Exemplo 15: Saída estruturada com ProviderStrategy
├── sample016.py       # Exemplo 16: Estado customizado via middleware
├── sample017.py       # Exemplo 17: Estado customizado via state_schema
└── sample018.py       # Exemplo 18: Streaming de respostas
```

## 📚 Exemplos Disponíveis

| Arquivo | Descrição | Conceitos |
|---------|-----------|-----------|
| **sample001.py** | Agente básico com LangChain | Inicialização básica, system prompt |
| **sample002.py** | Agente com uma ferramenta | Tools, decorador `@tool` |
| **sample003.py** | Modelo ChatOpenAI com parâmetros personalizados | `ChatOpenAI`, temperature, timeout, max_completion_tokens |
| **sample004.py** | Modelo multi-provedor | `init_chat_model`, compatibilidade multi-provedor |
| **sample005.py** | Contexto de runtime personalizado | `Context`, `ToolRuntime`, injeção de contexto |
| **sample006.py** | Contexto e resposta estruturada | `ResponseFormat`, output estruturado, dataclass |
| **sample007.py** | Agente SEM memória | Demonstração de perda de contexto entre invocações |
| **sample008.py** | Agente COM memória | `MemorySaver`, checkpointer, persistência de contexto |
| **sample009.py** | Memória e múltiplos contextos | Múltiplos thread_id, conversas independentes |
| **sample010.py** | Seleção dinâmica de modelo | Middleware, `@wrap_model_call`, troca de modelo |
| **sample011.py** | Tratamento de erros em ferramentas | `@wrap_tool_call`, exception handling |
| **sample012.py** | System prompt dinâmico | `@dynamic_prompt`, personalização por contexto |
| **sample013.py** | Passagem de sequência de mensagens | State, histórico manual, múltiplas mensagens |
| **sample014.py** | Saída estruturada com ToolStrategy | `ToolStrategy`, tool calling artificial, Pydantic models |
| **sample015.py** | Saída estruturada com ProviderStrategy | `ProviderStrategy`, structured output nativo |
| **sample016.py** | Estado customizado via middleware | `AgentMiddleware`, `CustomState`, before_model hooks |
| **sample017.py** | Estado customizado via state_schema | `state_schema`, campos customizados simples |
| **sample018.py** | Streaming de respostas | `agent.stream()`, stream_mode, chunks, progresso em tempo real |

## 🎯 Exemplos de Uso

### Exemplo Rápido - Agente Básico

```python
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    "gpt-4o-mini",
    system_prompt="Você é um assistente prestativo.",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Olá!"}]},
)

print(response["messages"][-1].content)
```

### Exemplo com Ferramenta

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def calcular_quadrado(numero: float) -> float:
    """Calcula o quadrado de um número."""
    return numero ** 2

agent = create_agent(
    "gpt-4o-mini",
    tools=[calcular_quadrado],
)
```

## 🛠️ Comandos Úteis do uv

```bash
# Sincronizar dependências
uv sync

# Adicionar nova dependência
uv add nome-do-pacote

# Remover dependência
uv remove nome-do-pacote

# Listar pacotes instalados
uv pip list

# Atualizar todos os pacotes
uv pip install --upgrade -r requirements.txt

# Executar script sem ativar ambiente
uv run script.py

# Criar novo projeto
uv init nome-do-projeto
```

## 🔧 Configuração do Pylance

O projeto inclui configurações do Pylance no `pyproject.toml` para trabalhar melhor com frameworks dinâmicos como LangChain:

```toml
[tool.pyright]
typeCheckingMode = "basic"
reportAttributeAccessIssue = "none"
reportArgumentType = "none"
reportUnknownMemberType = "none"
reportUnknownArgumentType = "none"
reportMissingTypeStubs = "none"
```

## 📖 Recursos Adicionais

- **[Documentação LangChain](https://python.langchain.com/)** - Documentação oficial
- **[LangChain Agents](https://python.langchain.com/docs/how_to/#agents)** - Guia de agentes
- **[Documentação uv](https://docs.astral.sh/uv/)** - Gerenciador de pacotes
- **[OpenAI Pricing](https://platform.openai.com/docs/pricing)** - Preços de Uso dos Modelos da OpenAI

## 🤝 Contribuindo

Sinta-se à vontade para adicionar novos exemplos ou melhorar os existentes!

## 📝 Licença

Este projeto é apenas para fins educacionais e demonstrativos.

---

**Dica:** Comece pelo `sample001.py` e avance progressivamente para entender os conceitos de forma incremental! 🚀
