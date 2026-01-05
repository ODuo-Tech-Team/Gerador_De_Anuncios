"""
AdBlast AI v1.1 - Backend Flask
Gerador de variações de anúncios com imagens usando OpenAI (GPT-4o + DALL-E 3)
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Inicializa o cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Limites de caracteres (Facebook/Instagram Ads) - ATUALIZADOS v1.1
CHAR_LIMITS = {
    "titulo": 40,       # Headline do Facebook Ads
    "descricao": 250,   # Primary text estendido (5-6 linhas)
    "cta": 20           # CTA button text
}

# System prompt para o copywriter AI - ATUALIZADO v1.1
SYSTEM_PROMPT_TEXT = """Você é um Copywriter Sênior e Estrategista de Tráfego Pago especialista em Direct Response para o mercado brasileiro. Sua especialidade é criar anúncios para Meta Ads (Facebook/Instagram) que param o scroll e geram cliques qualificados.

CONTEXTO DE EXECUÇÃO:
O usuário fornecerá: Cliente, Oferta, Função/Nicho e opcionalmente um Estilo Visual.
Use os frameworks AIDA (Atenção, Interesse, Desejo, Ação) e PAS (Problema, Agitação, Solução).

REGRAS RÍGIDAS DE CONTEÚDO E FORMATO:
1. QUANTIDADE: Gere exatamente 5 variações distintas.
2. LIMITES TÉCNICOS (NÃO ULTRAPASSE):
   - TÍTULO: Máximo 40 caracteres (Direto e impactante).
   - DESCRIÇÃO: Máximo 250 caracteres (Texto mais detalhado, 5-6 linhas, com storytelling).
   - CTA: Máximo 20 caracteres (Curto e imperativo).
   - IMAGE_PROMPT: Crie um prompt em INGLÊS para gerar uma imagem impactante para o anúncio (máximo 200 caracteres).
3. IDIOMA: Português do Brasil (PT-BR) para titulo, descricao e cta. INGLÊS para image_prompt.
4. Tom natural, humano e persuasivo. Evite "IA-speak".

ESTRUTURA DAS VARIAÇÕES:
- Variação 1 (PAS): Foco na dor latente do público e na solução rápida.
- Variação 2 (Benefício): Foco na transformação clara após usar o produto/serviço.
- Variação 3 (Autoridade): Foco em prova social ou tempo de mercado do cliente.
- Variação 4 (Escassez): Foco em tempo limitado ou poucas vagas (Urgência Real).
- Variação 5 (Direct/Hook): Um gancho de curiosidade forte ou pergunta provocativa.

REQUISITO TÉCNICO DE SAÍDA:
Retorne EXCLUSIVAMENTE um array JSON puro, sem blocos de código markdown (sem ```json), sem explicações.
Formato: [{"titulo": "...", "descricao": "...", "cta": "...", "image_prompt": "..."}]

O image_prompt deve descrever uma imagem profissional, moderna e relevante para o anúncio. Exemplo:
"Professional smiling person in modern office with growth charts, vibrant colors, flat design style"
"""


def validate_and_truncate_ads(ads: list) -> list:
    """Valida e trunca os textos dos anúncios para garantir limites de caracteres."""
    validated_ads = []

    for ad in ads:
        validated_ad = {
            "titulo": ad.get("titulo", "")[:CHAR_LIMITS["titulo"]],
            "descricao": ad.get("descricao", "")[:CHAR_LIMITS["descricao"]],
            "cta": ad.get("cta", "")[:CHAR_LIMITS["cta"]],
            "image_prompt": ad.get("image_prompt", "")[:200]
        }
        validated_ads.append(validated_ad)

    return validated_ads


def generate_image_with_dalle(prompt: str, style: str = "") -> str:
    """
    Gera uma imagem usando DALL-E 3.

    Args:
        prompt: Descrição da imagem em inglês
        style: Estilo visual opcional

    Returns:
        URL da imagem gerada
    """
    try:
        # Adiciona estilo ao prompt se fornecido
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style"

        # Adiciona instruções para anúncio
        full_prompt = f"Create a professional advertising image: {full_prompt}. High quality, suitable for social media ads, no text overlay."

        response = client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        return response.data[0].url

    except Exception as e:
        print(f"Erro ao gerar imagem: {str(e)}")
        return None


def generate_ads_with_openai(oferta: str, cliente: str, nicho: str, estilo_visual: str = "") -> list:
    """
    Gera variações de anúncios com texto usando GPT-4o.

    Args:
        oferta: A oferta principal do anúncio
        cliente: Nome do cliente/empresa
        nicho: Função ou nicho de mercado
        estilo_visual: Estilo visual para as imagens (opcional)

    Returns:
        Lista de dicionários com as variações de anúncios
    """

    estilo_info = f"\nESTILO VISUAL DESEJADO: {estilo_visual}" if estilo_visual else ""

    user_prompt = f"""Gere 5 variações de anúncios para:

OFERTA PRINCIPAL: {oferta}
CLIENTE/EMPRESA: {cliente}
NICHO/PÚBLICO-ALVO: {nicho}{estilo_info}

Lembre-se:
- Retorne APENAS o array JSON
- Inclua o campo "image_prompt" em INGLÊS para cada variação
- A descrição agora pode ter até 250 caracteres (mais detalhada)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()

        # Remove marcadores markdown
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()
        ads = json.loads(response_text)
        validated_ads = validate_and_truncate_ads(ads)

        return validated_ads

    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao processar resposta da IA: {str(e)}")
    except Exception as e:
        raise Exception(f"Erro na comunicação com a API: {str(e)}")


@app.route("/generate_ads", methods=["POST"])
def generate_ads():
    """
    Endpoint para gerar variações de anúncios com imagens.

    Espera um JSON com:
    - oferta: string (obrigatório)
    - cliente: string (obrigatório)
    - nicho: string (obrigatório)
    - estilo_visual: string (opcional) - Ex: "Realista", "Minimalista", "Ilustração"
    - generate_images: boolean (opcional, default: true)

    Retorna:
    - success: boolean
    - data: array de objetos {titulo, descricao, cta, image_url}
    """

    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({
            "success": False,
            "error": "API Key da OpenAI não configurada. Verifique o arquivo .env"
        }), 500

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "Nenhum dado enviado na requisição"
        }), 400

    # Campos obrigatórios
    oferta = data.get("oferta", "").strip()
    cliente = data.get("cliente", "").strip()
    nicho = data.get("nicho", "").strip()

    # Campos opcionais
    estilo_visual = data.get("estilo_visual", "").strip()
    generate_images = data.get("generate_images", True)

    if not oferta:
        return jsonify({"success": False, "error": "O campo 'oferta' é obrigatório"}), 400
    if not cliente:
        return jsonify({"success": False, "error": "O campo 'cliente' é obrigatório"}), 400
    if not nicho:
        return jsonify({"success": False, "error": "O campo 'nicho' é obrigatório"}), 400

    try:
        # 1. Gera os textos dos anúncios
        ads = generate_ads_with_openai(oferta, cliente, nicho, estilo_visual)

        # 2. Gera as imagens para cada anúncio (se habilitado)
        if generate_images:
            for ad in ads:
                image_prompt = ad.get("image_prompt", "")
                if image_prompt:
                    image_url = generate_image_with_dalle(image_prompt, estilo_visual)
                    ad["image_url"] = image_url
                else:
                    ad["image_url"] = None

                # Remove o prompt da imagem do retorno (não precisa ir pro frontend)
                del ad["image_prompt"]
        else:
            # Se não gerar imagens, remove o prompt e define url como null
            for ad in ads:
                if "image_prompt" in ad:
                    del ad["image_prompt"]
                ad["image_url"] = None

        return jsonify({
            "success": True,
            "data": ads
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint de health check."""
    return jsonify({
        "status": "healthy",
        "service": "AdBlast AI",
        "version": "1.1.0"
    })


@app.route("/")
def serve_frontend():
    """Serve o frontend index.html na rota raiz."""
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AdBlast AI v1.1 - Backend iniciado!")
    print("="*50)
    print("📍 Servidor: http://localhost:5000")
    print("📡 Endpoints:")
    print("   POST /generate_ads - Gera anúncios + imagens")
    print("   GET  /health       - Health check")
    print("="*50)
    print("🆕 Novidades v1.1:")
    print("   • Descrições estendidas (250 chars)")
    print("   • Geração de imagens com DALL-E 3")
    print("   • Campo opcional 'estilo_visual'")
    print("="*50 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
