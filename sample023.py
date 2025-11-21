############################################
#
# Exemplo de Multimodal: Audio, Video e PDFs
#
############################################


############################################
# PASSO 1 - Audio via URL (Transcrição/Análise)
############################################

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import base64

load_dotenv()

# NOTA: Nem todos os models suportam audio/video diretamente
# Alguns providers como Gemini suportam, OpenAI tem APIs separadas

# Para Gemini (suporta audio/video nativamente)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

    # Audio via URL (formato suportado: MP3, WAV, etc.)
    message_audio = HumanMessage(
        content=[
            {"type": "text", "text": "Transcreva e resuma este áudio."},
            {
                "type": "media",  # Gemini usa "media" type
                "mime_type": "audio/mp3",
                "data": "https://example.com/audio.mp3",  # URL do áudio
            },
        ],
    )

    # response_audio = model_gemini.invoke([message_audio])

    print("=" * 70)
    print("EXEMPLO 1: AUDIO VIA URL (Gemini)")
    print("=" * 70)
    print("""
Para usar audio com Gemini:
1. Use ChatGoogleGenerativeAI com model="gemini-2.0-flash-exp" ou similar
2. Content type: "media" com mime_type="audio/mp3" ou "audio/wav"
3. Forneça URL ou base64 do áudio
4. O model pode transcrever, resumir ou analisar o áudio

NOTA: OpenAI não suporta audio direto no chat (use Whisper API separadamente)
""")

except ImportError:
    print("=" * 70)
    print("EXEMPLO 1: AUDIO VIA URL")
    print("=" * 70)
    print("""
Para usar audio:
- Instale: pip install langchain-google-genai
- Configure GOOGLE_API_KEY no .env
- Use Gemini models que suportam multimodal

Exemplo de código disponível no arquivo sample023.py
""")

print()


############################################
# PASSO 2 - Audio via Base64
############################################

def load_audio_as_base64(audio_path: str) -> str:
    """Carrega um arquivo de áudio e converte para base64."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


print("=" * 70)
print("EXEMPLO 2: AUDIO VIA BASE64")
print("=" * 70)
print("""
Para usar audio local:

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

    # Carregar audio local
    audio_base64 = load_audio_as_base64("caminho/para/audio.mp3")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Transcreva este áudio."},
            {
                "type": "media",
                "mime_type": "audio/mp3",
                "data": f"data:audio/mp3;base64,{audio_base64}",
            },
        ],
    )

    response = model.invoke([message])
    print(response.content)

except ImportError:
    print("Instale langchain-google-genai")
""")
print()


############################################
# PASSO 3 - Video via URL
############################################

print("=" * 70)
print("EXEMPLO 3: VIDEO VIA URL")
print("=" * 70)
print("""
Para usar video (Gemini):

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

message_video = HumanMessage(
    content=[
        {"type": "text", "text": "Descreva o que acontece neste vídeo."},
        {
            "type": "media",
            "mime_type": "video/mp4",
            "data": "https://example.com/video.mp4",
        },
    ],
)

response = model.invoke([message_video])
print(response.content)

Formatos suportados: MP4, MOV, AVI, etc.
Limite de tamanho varia (Gemini: até 2GB por vídeo)
""")
print()


############################################
# PASSO 4 - Video Local via Base64
############################################

def load_video_as_base64(video_path: str) -> str:
    """Carrega um arquivo de vídeo e converte para base64."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


print("=" * 70)
print("EXEMPLO 4: VIDEO LOCAL VIA BASE64")
print("=" * 70)
print("""
Para vídeo local:

video_base64 = load_video_as_base64("caminho/para/video.mp4")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Resuma este vídeo em 3 pontos principais."},
        {
            "type": "media",
            "mime_type": "video/mp4",
            "data": f"data:video/mp4;base64,{video_base64}",
        },
    ],
)

response = model.invoke([message])

⚠️ ATENÇÃO: Vídeos grandes em base64 podem exceder limites de payload
Recomendação: Use URLs quando possível para vídeos grandes
""")
print()


############################################
# PASSO 5 - PDF Documents
############################################

print("=" * 70)
print("EXEMPLO 5: PDF DOCUMENTS")
print("=" * 70)
print("""
Para PDFs (Gemini):

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Carregar PDF
pdf_base64 = load_pdf_as_base64("documento.pdf")

message_pdf = HumanMessage(
    content=[
        {"type": "text", "text": "Resuma este documento PDF."},
        {
            "type": "media",
            "mime_type": "application/pdf",
            "data": f"data:application/pdf;base64,{pdf_base64}",
        },
    ],
)

response = model.invoke([message_pdf])

Casos de uso:
- Análise de contratos
- Extração de dados de formulários
- Resumo de documentos longos
- Q&A sobre PDFs
""")
print()


############################################
# PASSO 6 - Múltiplas Modalidades na Mesma Mensagem
############################################

print("=" * 70)
print("EXEMPLO 6: MÚLTIPLAS MODALIDADES")
print("=" * 70)
print("""
Você pode combinar texto, imagem, audio e vídeo:

message_multi = HumanMessage(
    content=[
        {"type": "text", "text": "Analise esta apresentação:"},
        {
            "type": "media",
            "mime_type": "image/jpeg",
            "data": "https://example.com/slide1.jpg",
        },
        {
            "type": "media",
            "mime_type": "audio/mp3",
            "data": "https://example.com/narration.mp3",
        },
        {"type": "text", "text": "Compare o conteúdo visual com a narração."},
    ],
)

response = model.invoke([message_multi])

Útil para:
- Análise de apresentações com narração
- Conferências com slides e audio
- Tutoriais em vídeo com materiais complementares
""")
print()


############################################
# PASSO 7 - OpenAI: Alternativa para Audio (Whisper)
############################################

print("=" * 70)
print("EXEMPLO 7: OPENAI WHISPER PARA AUDIO (Alternativa)")
print("=" * 70)
print("""
OpenAI não suporta audio no ChatOpenAI diretamente.
Use a API Whisper separadamente:

from openai import OpenAI

client = OpenAI()

# Transcrição de audio
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="pt"  # português
    )

print(transcription.text)

# Depois use o texto com ChatOpenAI normalmente
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(f"Resuma este texto: {transcription.text}")

Workflow:
1. Audio → Whisper API → Texto
2. Texto → ChatOpenAI → Análise/Resumo
""")
print()


############################################
# OBSERVAÇÕES IMPORTANTES
############################################

print("=" * 70)
print("OBSERVAÇÕES IMPORTANTES")
print("=" * 70)
print("""
1. SUPORTE POR PROVIDER:

   GEMINI (Google):
   ✓ Imagens (JPEG, PNG, WebP, HEIC, HEIF)
   ✓ Audio (MP3, WAV, AIFF, AAC, OGG, FLAC)
   ✓ Video (MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP)
   ✓ PDF documents

   OPENAI (ChatGPT):
   ✓ Imagens (JPEG, PNG, GIF, WebP)
   ✗ Audio direto no chat (use Whisper API separadamente)
   ✗ Video
   ✗ PDF direto (extraia texto primeiro)

   ANTHROPIC (Claude):
   ✓ Imagens (JPEG, PNG, GIF, WebP)
   ✓ PDF documents (claude-3-5-sonnet e posteriores)
   ✗ Audio direto (processe com Whisper primeiro)
   ✗ Video

2. FORMATOS DE DADOS:
   - URL pública (https://)
   - Base64 encoded (data:{mime_type};base64,...)
   - Sempre especifique o mime_type correto

3. MIME TYPES COMUNS:
   Audio: audio/mp3, audio/wav, audio/aac, audio/ogg
   Video: video/mp4, video/mov, video/avi, video/webm
   PDF: application/pdf
   Imagem: image/jpeg, image/png, image/webp

4. LIMITES DE TAMANHO:
   - Gemini: até 2GB por arquivo de vídeo
   - OpenAI Images: até 20MB
   - Verifique documentação do provider para limites atuais

5. CUSTOS:
   - Audio/video processamento é geralmente mais caro
   - Cobrado por duração (audio/video) ou tamanho (imagens)
   - Vídeos longos podem ter custos significativos
   - Consulte pricing do provider

6. CASOS DE USO POR TIPO:

   AUDIO:
   - Transcrição de reuniões
   - Análise de podcasts
   - Atendimento ao cliente (call center)
   - Acessibilidade (legendas automáticas)

   VIDEO:
   - Resumo de vídeo aulas
   - Análise de segurança (CCTV)
   - Moderação de conteúdo
   - Descrição para acessibilidade

   PDF:
   - Análise de contratos
   - Extração de dados de formulários
   - Resumo de relatórios
   - Q&A sobre documentos

7. BOAS PRÁTICAS:
   - Use URLs para arquivos grandes (melhor que base64)
   - Comprima vídeos antes de enviar
   - Para audio, considere pré-processar com Whisper
   - Seja específico nas instruções
   - Use chunks para vídeos muito longos

8. LIMITAÇÕES:
   - Nem todos os models suportam todas as modalidades
   - Qualidade do audio/video afeta precisão
   - Vídeos longos podem ter delays significativos
   - Rate limits são mais restritivos para multimodal

9. SEGURANÇA E PRIVACIDADE:
   - Não envie conteúdo sensível ou privado
   - Audio/video pode conter informações pessoais
   - Dados são processados nos servidores do provider
   - Verifique compliance (LGPD, GDPR, etc.)

10. PRÓXIMOS PASSOS:
    - Para reasoning models, veja sample024.py
    - Para métodos invoke/stream/batch, veja sample025.py
    - Para parâmetros de model, veja sample026.py
""")
