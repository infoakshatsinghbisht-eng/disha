"""
Interactive Web Application for the Multimodal LLM Image Generation Studio.
Provides a modern, reactive interface to interact with the LLM from scratch.
"""

import os
import io
import base64
import torch
from flask import Flask, render_template_string, request, jsonify
from PIL import Image

from config import LLMConfig, VQVAEConfig, GenerationConfig
from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE
from model.transformer import MultimodalTransformer
from pipeline.sampler import MultimodalGeneratorPipeline

app = Flask(__name__)

# Global Pipeline Singleton
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        ckpt_path = "checkpoints/multimodal_llm.pt"
        if os.path.exists(ckpt_path):
            print(f"[+] Loading trained checkpoint from {ckpt_path}")
            PIPELINE = MultimodalGeneratorPipeline.from_pretrained(ckpt_path, device=DEVICE)
        else:
            print("[!] Initializing baseline model on device:", DEVICE)
            llm_cfg = LLMConfig()
            vq_cfg = VQVAEConfig()
            tokenizer = ByteTokenizer()
            total_vocab = llm_cfg.text_vocab_size + llm_cfg.image_vocab_size + len(tokenizer.SPECIAL_TOKENS)
            
            llm = MultimodalTransformer(
                vocab_size=total_vocab,
                dim=llm_cfg.dim,
                num_layers=llm_cfg.num_layers,
                num_heads=llm_cfg.num_heads,
                num_kv_heads=llm_cfg.num_kv_heads,
            )
            vqvae = VQVAE(
                in_channels=vq_cfg.in_channels,
                hidden_dim=vq_cfg.hidden_dim,
                embedding_dim=vq_cfg.embedding_dim,
                codebook_size=vq_cfg.codebook_size,
            )
            PIPELINE = MultimodalGeneratorPipeline(
                llm=llm,
                vqvae=vqvae,
                tokenizer=tokenizer,
                text_vocab_size=llm_cfg.text_vocab_size,
                device=DEVICE,
            )
    return PIPELINE


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal LLM Studio | Image Generation from Scratch</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0b10;
            --bg-card: rgba(18, 21, 33, 0.75);
            --bg-card-border: rgba(255, 255, 255, 0.08);
            --accent-glow: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 18, 28, 0.9);
            --border-glow: rgba(99, 102, 241, 0.35);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            max-width: 1200px;
            width: 100%;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        h1 {
            font-size: 2.75rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
            letter-spacing: -0.02em;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--bg-card-border);
            backdrop-filter: blur(16px);
            border-radius: 1.25rem;
            padding: 2rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-gradient);
            opacity: 0.8;
        }

        .card h2 {
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: #f1f5f9;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: #cbd5e1;
        }

        textarea, input[type="text"] {
            background: var(--input-bg);
            border: 1px solid var(--bg-card-border);
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
            color: #fff;
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            outline: none;
            transition: all 0.2s ease;
        }

        textarea:focus, input[type="text"]:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 0 3px var(--border-glow);
        }

        .quick-prompts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #cbd5e1;
            padding: 0.35rem 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: rgba(99, 102, 241, 0.4);
            color: #fff;
        }

        .sliders {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .slider-box {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        input[type="range"] {
            accent-color: var(--accent-glow);
            cursor: pointer;
        }

        button.btn-generate {
            background: var(--accent-gradient);
            border: none;
            color: white;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.9rem 1.5rem;
            border-radius: 0.75rem;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            margin-top: 0.5rem;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
        }

        button.btn-generate:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
        }

        button.btn-generate:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Image Display Area */
        .preview-box {
            min-height: 320px;
            background: #06070a;
            border: 2px dashed rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .preview-box img {
            width: 100%;
            height: auto;
            max-height: 380px;
            object-fit: contain;
            border-radius: 0.75rem;
        }

        .placeholder-text {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-align: center;
        }

        .spinner {
            display: none;
            width: 45px;
            height: 45px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-left-color: var(--accent-glow);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 0.75rem;
            border-radius: 0.6rem;
            text-align: center;
        }

        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: #a5b4fc;
            font-size: 0.95rem;
        }

        .stat-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        footer {
            margin-top: 3rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">🚀 Built 100% From Scratch in PyTorch</div>
            <h1>Multimodal LLM Studio</h1>
            <p class="subtitle">Unified Autoregressive Transformer with Discrete VQ-VAE Visual Tokenizer generating images token-by-token.</p>
        </header>

        <div class="grid">
            <!-- Left Panel: Prompt Controls -->
            <div class="card">
                <h2>✨ Text Prompt & Parameters</h2>
                
                <div class="form-group">
                    <label for="promptInput">Enter Prompt</label>
                    <textarea id="promptInput" rows="3" placeholder="e.g., a vibrant cyan glowing circle on dark sunset background">a vibrant cyan glowing star on sunset gradient background</textarea>
                </div>

                <div class="form-group">
                    <label>Quick Presets</label>
                    <div class="quick-prompts">
                        <span class="chip" onclick="setPrompt('a vibrant red glowing circle on dark background')">🔴 Red Circle</span>
                        <span class="chip" onclick="setPrompt('a glowing yellow star with neon glow background')">⭐ Yellow Star</span>
                        <span class="chip" onclick="setPrompt('a blue square in the center on white background')">🟦 Blue Square</span>
                        <span class="chip" onclick="setPrompt('a green diamond with sunset gradient background')">💎 Green Diamond</span>
                        <span class="chip" onclick="setPrompt('a purple triangle on deep blue background')">🔺 Purple Triangle</span>
                    </div>
                </div>

                <div class="sliders">
                    <div class="slider-box">
                        <div class="slider-header">
                            <span>Temperature</span>
                            <span id="tempVal">0.85</span>
                        </div>
                        <input type="range" id="tempSlider" min="0.1" max="1.5" step="0.05" value="0.85" oninput="document.getElementById('tempVal').innerText = this.value">
                    </div>

                    <div class="slider-box">
                        <div class="slider-header">
                            <span>Top-P (Nucleus)</span>
                            <span id="topPVal">0.92</span>
                        </div>
                        <input type="range" id="topPSlider" min="0.5" max="1.0" step="0.02" value="0.92" oninput="document.getElementById('topPVal').innerText = this.value">
                    </div>
                </div>

                <button class="btn-generate" id="generateBtn" onclick="generateImage()">
                    <span>🎨 Generate Image</span>
                </button>
            </div>

            <!-- Right Panel: Output & Visual Tokens -->
            <div class="card">
                <h2>🖼️ Generated Output</h2>
                
                <div class="preview-box" id="previewBox">
                    <div class="spinner" id="spinner"></div>
                    <div class="placeholder-text" id="placeholder">Click "Generate Image" to create an image with the multimodal LLM.</div>
                    <img id="resultImage" style="display: none;" alt="Generated Result">
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">256</div>
                        <div class="stat-label">Image Tokens (16x16)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">4096</div>
                        <div class="stat-label">Codebook Vocab</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="genTime">-- ms</div>
                        <div class="stat-label">Generation Time</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>Pure Foundation Multimodal Architecture • RoPE Attention • RMSNorm • SwiGLU • Vector Quantization</p>
        </footer>
    </div>

    <script>
        function setPrompt(text) {
            document.getElementById('promptInput').value = text;
        }

        async function generateImage() {
            const prompt = document.getElementById('promptInput').value.trim();
            if (!prompt) return;

            const temp = parseFloat(document.getElementById('tempSlider').value);
            const topP = parseFloat(document.getElementById('topPSlider').value);

            const btn = document.getElementById('generateBtn');
            const spinner = document.getElementById('spinner');
            const placeholder = document.getElementById('placeholder');
            const resultImage = document.getElementById('resultImage');
            const genTime = document.getElementById('genTime');

            btn.disabled = true;
            spinner.style.display = 'block';
            placeholder.style.display = 'none';
            resultImage.style.display = 'none';

            const startTime = performance.now();

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, temperature: temp, top_p: topP })
                });

                const data = await response.json();
                if (data.success) {
                    resultImage.src = 'data:image/png;base64,' + data.image_base64;
                    resultImage.style.display = 'block';
                    const elapsed = Math.round(performance.now() - startTime);
                    genTime.innerText = elapsed + ' ms';
                } else {
                    alert('Error: ' + (data.error || 'Failed to generate image'));
                    placeholder.style.display = 'block';
                }
            } catch (err) {
                alert('Request failed: ' + err.message);
                placeholder.style.display = 'block';
            } finally {
                spinner.style.display = 'none';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/generate", methods=["POST"])
def generate_api():
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "a red glowing circle on dark background")
        temp = float(data.get("temperature", 0.85))
        top_p = float(data.get("top_p", 0.92))

        pipeline = get_pipeline()
        gen_cfg = GenerationConfig(temperature=temp, top_p=top_p)

        img = pipeline.generate_image(prompt=prompt, gen_config=gen_cfg)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"success": True, "image_base64": img_str})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("[*] Starting Multimodal LLM Web Studio on http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
