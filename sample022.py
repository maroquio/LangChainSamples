############################################
#
# Exemplo de Multimodal: Usando Imagens com 
# Vision Models
#
############################################


############################################
# PASSO 1 - Imagem via URL
############################################

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Usar um model com capacidade de visão
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Criar mensagem com imagem via URL
message_with_image_url = HumanMessage(
    content=[
        {"type": "text", "text": "O que você vê nesta imagem? Descreva em detalhes."},
        {
            "type": "image_url",
            "image_url": {"url": "https://picsum.photos/400/300"},
        },
    ],
)

response_url = model.invoke([message_with_image_url])

print("=" * 70)
print("EXEMPLO 1: IMAGEM VIA URL")
print("=" * 70)
print(f"Resposta: {response_url.content}")
print()


############################################
# PASSO 2 - Imagem via Base64
############################################

import base64
import requests
from io import BytesIO

# Baixar uma imagem de exemplo e converter para base64
image_url = "https://picsum.photos/400/300"
image_data = requests.get(image_url).content
base64_image = base64.b64encode(image_data).decode("utf-8")

# Criar mensagem com imagem em base64
message_with_image_base64 = HumanMessage(
    content=[
        {"type": "text", "text": "Descreva esta imagem."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ],
)

response_base64 = model.invoke([message_with_image_base64])

print("=" * 70)
print("EXEMPLO 2: IMAGEM VIA BASE64")
print("=" * 70)
print(f"Resposta: {response_base64.content}")
print()


############################################
# PASSO 3 - Múltiplas Imagens
############################################

# Criar mensagem com múltiplas imagens
message_multiple_images = HumanMessage(
    content=[
        {"type": "text", "text": "Compare estas duas imagens. Quais são as diferenças?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://picsum.photos/seed/image1/400/300"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://picsum.photos/seed/image2/400/300"},
        },
    ],
)

response_multiple = model.invoke([message_multiple_images])

print("=" * 70)
print("EXEMPLO 3: MÚLTIPLAS IMAGENS")
print("=" * 70)
print(f"Resposta: {response_multiple.content}")
print()


############################################
# PASSO 4 - Imagem Local (via Base64)
############################################

def load_image_as_base64(image_path: str) -> str:
    """Carrega uma imagem local e converte para base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# NOTA: Este exemplo requer um arquivo de imagem local
# Descomente as linhas abaixo se você tiver uma imagem local
"""
local_image_base64 = load_image_as_base64("caminho/para/sua/imagem.jpg")

message_local_image = HumanMessage(
    content=[
        {"type": "text", "text": "O que tem nesta foto?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{local_image_base64}"},
        },
    ],
)

response_local = model.invoke([message_local_image])
print("RESPOSTA DA IMAGEM LOCAL:")
print(response_local.content)
"""

print("=" * 70)
print("EXEMPLO 4: IMAGEM LOCAL")
print("=" * 70)
print("""
Para usar imagens locais:
1. Use a função load_image_as_base64() para carregar a imagem
2. Converta para base64
3. Passe como data:image/jpeg;base64,{base64_string}

Exemplo comentado disponível no código acima.
""")
print()


############################################
# PASSO 5 - Detalhamento da Imagem (detail parameter)
############################################

# OpenAI permite controlar o nível de detalhe da análise
message_high_detail = HumanMessage(
    content=[
        {"type": "text", "text": "Analise esta imagem em alto nível de detalhe."},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://picsum.photos/400/300",
                "detail": "high",  # "low", "high", ou "auto" (default)
            },
        },
    ],
)

response_high_detail = model.invoke([message_high_detail])

print("=" * 70)
print("EXEMPLO 5: CONTROLE DE DETALHE (detail parameter)")
print("=" * 70)
print(f"Resposta (high detail): {response_high_detail.content}")
print()


############################################
# PASSO 6 - Uso Prático: Extração de Texto de Imagem (OCR)
############################################

# Simular uma imagem com texto (usando URL de exemplo)
message_ocr = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Extraia todo o texto visível nesta imagem. Se não houver texto, apenas descreva o que você vê.",
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://picsum.photos/seed/text/400/300"},
        },
    ],
)

response_ocr = model.invoke([message_ocr])

print("=" * 70)
print("EXEMPLO 6: OCR (EXTRAÇÃO DE TEXTO)")
print("=" * 70)
print(f"Texto extraído: {response_ocr.content}")
print()


############################################
# PASSO 7 - Structured Output com Imagens
############################################

from pydantic import BaseModel, Field
from typing import List

class ImageAnalysis(BaseModel):
    """Análise estruturada de uma imagem."""
    main_subjects: List[str] = Field(description="Principais objetos ou pessoas na imagem")
    colors: List[str] = Field(description="Cores predominantes")
    mood: str = Field(description="Mood ou atmosfera da imagem")
    scene_type: str = Field(description="Tipo de cena (interior, exterior, paisagem, etc.)")


# Combinar multimodal com structured output
structured_model = model.with_structured_output(ImageAnalysis)

message_structured = [
    HumanMessage(
        content=[
            {"type": "text", "text": "Analise esta imagem e forneça os dados estruturados."},
            {
                "type": "image_url",
                "image_url": {"url": "https://picsum.photos/seed/analysis/400/300"},
            },
        ],
    )
]

analysis = structured_model.invoke(message_structured)

print("=" * 70)
print("EXEMPLO 7: STRUCTURED OUTPUT COM IMAGENS")
print("=" * 70)
print(f"Principais elementos: {', '.join(analysis.main_subjects)}")
print(f"Cores predominantes: {', '.join(analysis.colors)}")
print(f"Atmosfera: {analysis.mood}")
print(f"Tipo de cena: {analysis.scene_type}")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. VISION MODELS:
   - Use models com capacidade de visão (gpt-4o, gpt-4o-mini, claude-3-sonnet, etc.)
   - Nem todos os models suportam imagens
   - Verifique a documentação do provider para modelos disponíveis

2. FORMATOS DE IMAGEM SUPORTADOS:
   - URL pública (https://)
   - Base64 encoded (data:image/jpeg;base64,...)
   - Formatos: JPEG, PNG, GIF, WebP
   - Tamanho máximo varia por provider (geralmente 20MB)

3. CONTENT BLOCKS:
   - Mensagens multimodais usam lista de content blocks
   - Cada block tem um "type": "text" ou "image_url"
   - Você pode misturar múltiplos textos e imagens
   - Ordem importa: o model processa na sequência

4. PARÂMETRO detail (OpenAI):
   - "low": análise mais rápida e barata, menos tokens
   - "high": análise detalhada, mais tokens, mais precisa
   - "auto": o model decide baseado no tamanho da imagem
   - Afeta custo e qualidade da análise

5. MÚLTIPLAS IMAGENS:
   - Você pode passar várias imagens em uma única mensagem
   - Útil para comparações, análise de sequências, etc.
   - Limite varia por provider (OpenAI permite várias)

6. CASOS DE USO:
   - OCR (extração de texto de imagens)
   - Descrição de imagens para acessibilidade
   - Análise de gráficos e diagramas
   - Moderação de conteúdo visual
   - Identificação de objetos, pessoas, lugares
   - Análise de documentos escaneados
   - Comparação de imagens

7. BOAS PRÁTICAS:
   - Use URLs quando possível (mais eficiente que base64)
   - Comprima imagens grandes antes de enviar
   - Para imagens locais, converta para base64
   - Seja específico nas instruções de texto
   - Use structured output para análises consistentes

8. LIMITAÇÕES:
   - Models podem ter viés ou erros em análise visual
   - Não use para tarefas críticas sem validação
   - Qualidade da imagem afeta a precisão
   - Custo é maior que text-only requests
   - Rate limits podem ser mais restritivos

9. SEGURANÇA:
   - Não envie imagens com informações sensíveis
   - Dados são processados pelo provider
   - Imagens podem ser armazenadas para melhorias
   - Verifique políticas de privacidade do provider

10. PRÓXIMOS PASSOS:
    - Para audio e video, veja sample023.py
    - Para reasoning models, veja sample024.py
    - Para métodos invoke/stream/batch, veja sample025.py
""")
