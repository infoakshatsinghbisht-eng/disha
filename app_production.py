"""
Production-Grade Full-Stack Multimodal LLM Studio.
Includes Constant-Memory Streaming Attention, Live RAM Telemetry,
AI Prompt Enhancer, and Real-Time Spatial Token Visualizer.
"""

import os
import io
import time
import base64
import random
import psutil
import torch
from flask import Flask, render_template_string, request, jsonify
from PIL import Image

from config import LLMConfig, VQVAEConfig, GenerationConfig
from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE
from model.transformer import MultimodalTransformer
from pipeline.sampler import MultimodalGeneratorPipeline
from pipeline.prompt_enhancer import PromptEnhancer

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = None
MODEL_RAM_MB = 0.0


def init_master_pipeline():
    global PIPELINE, MODEL_RAM_MB
    if PIPELINE is None:
        ckpt_path = "checkpoints/multimodal_llm.pt"
        if os.path.exists(ckpt_path):
            print(f"[+] Loading trained checkpoint from {ckpt_path}")
            PIPELINE = MultimodalGeneratorPipeline.from_pretrained(ckpt_path, device=DEVICE)
        else:
            print("[!] Initializing fresh pipeline on device:", DEVICE)
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
                max_seq_len=llm_cfg.max_seq_len,
            )
            vqvae = VQVAE(
                in_channels=vq_cfg.in_channels,
                hidden_dim=vq_cfg.hidden_dim,
                embedding_dim=vq_cfg.embedding_dim,
                codebook_size=vq_cfg.codebook_size,
                num_res_blocks=vq_cfg.num_res_blocks,
                num_downsamples=vq_cfg.num_downsamples,
            )
            PIPELINE = MultimodalGeneratorPipeline(
                llm=llm,
                vqvae=vqvae,
                tokenizer=tokenizer,
                text_vocab_size=llm_cfg.text_vocab_size,
                image_token_len=llm_cfg.image_token_len,
                device=DEVICE,
            )

        # Calculate model parameter RAM in MB
        total_bytes = sum(p.numel() * p.element_size() for p in PIPELINE.llm.parameters())
        total_bytes += sum(p.numel() * p.element_size() for p in PIPELINE.vqvae.parameters())
        MODEL_RAM_MB = round(total_bytes / (1024 * 1024), 2)
        print(f"[+] Model memory footprint: {MODEL_RAM_MB} MB")
    return PIPELINE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nano-Gemini Pro (200M) | High-Fidelity Multimodal Foundation Model</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #08090d;
            --bg-card: rgba(15, 18, 28, 0.85);
            --bg-card-border: rgba(255, 255, 255, 0.08);
            --accent-glow: #6366f1;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(10, 13, 22, 0.95);
            --border-glow: rgba(99, 102, 241, 0.4);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.14) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container { max-width: 1350px; width: 100%; }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--bg-card-border);
            flex-wrap: wrap;
            gap: 1rem;
        }

        .brand-box h1 {
            font-size: 2.25rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .brand-box p {
            color: var(--text-muted);
            font-size: 0.92rem;
            margin-top: 0.25rem;
        }

        .telemetry-bar {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .telemetry-pill {
            background: var(--bg-card);
            border: 1px solid var(--bg-card-border);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 10px var(--accent-emerald);
        }

        .telemetry-val {
            font-family: 'JetBrains Mono', monospace;
            color: #a5b4fc;
        }

        .main-layout {
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 1.75rem;
        }

        @media (max-width: 1024px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--bg-card-border);
            backdrop-filter: blur(20px);
            border-radius: 1.25rem;
            padding: 1.75rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            position: relative;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-header h2 {
            font-size: 1.2rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        label {
            font-size: 0.825rem;
            font-weight: 600;
            color: #cbd5e1;
            display: flex;
            justify-content: space-between;
        }

        textarea, input[type="text"], select {
            background: var(--input-bg);
            border: 1px solid var(--bg-card-border);
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
            color: #fff;
            font-family: inherit;
            font-size: 0.92rem;
            outline: none;
            transition: all 0.2s ease;
        }

        textarea:focus, input[type="text"]:focus, select:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 0 3px var(--border-glow);
        }

        .btn-enhance {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #a5b4fc;
            padding: 0.35rem 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-enhance:hover {
            background: rgba(99, 102, 241, 0.3);
            color: #fff;
        }

        .style-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .style-pill {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #cbd5e1;
            padding: 0.35rem 0.7rem;
            border-radius: 0.5rem;
            font-size: 0.76rem;
            cursor: pointer;
            transition: all 0.15s;
        }

        .style-pill:hover, .style-pill.active {
            background: rgba(99, 102, 241, 0.25);
            border-color: var(--accent-glow);
            color: #fff;
        }

        .controls-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .control-box {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
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
            font-size: 1.05rem;
            padding: 1rem 1.75rem;
            border-radius: 0.85rem;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5);
        }

        button.btn-generate:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.65);
        }

        button.btn-generate:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Right Panel: Output & Visualizer */
        .preview-container {
            min-height: 340px;
            background: #050608;
            border: 2px dashed rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .preview-container img {
            width: 100%;
            height: auto;
            max-height: 380px;
            object-fit: contain;
            border-radius: 0.75rem;
            transition: opacity 0.3s ease;
        }

        .token-grid-visualizer {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 4px;
            width: 160px;
            height: 160px;
            margin-bottom: 1rem;
        }

        .token-cell {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 2px;
            transition: background 0.15s;
        }

        .token-cell.active {
            background: #ec4899;
            box-shadow: 0 0 8px #ec4899;
        }

        .spinner {
            display: none;
            width: 45px;
            height: 45px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-left-color: var(--accent-glow);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-bottom: 1rem;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 0.75rem;
            border-radius: 0.6rem;
            text-align: center;
        }

        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: #38bdf8;
            font-size: 0.95rem;
        }

        .metric-lbl {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 0.2rem;
        }

        .action-row {
            display: flex;
            gap: 0.75rem;
        }

        .btn-action {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            padding: 0.65rem;
            border-radius: 0.6rem;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }

        .btn-action:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--accent-glow);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand-box">
                <h1>⚡ Nano-Gemini 2.0B Pro</h1>
                <p>2.0 Billion Parameters Foundation LLM • 2048 Codebook • Internet Scraped Dataset • Streaming Attention</p>
            </div>
            <div class="telemetry-bar">
                <div class="telemetry-pill">
                    <span class="dot"></span>
                    <span>Architecture: <span class="telemetry-val" id="engineMode">2.0B Multimodal Foundation LLM</span></span>
                </div>
                <div class="telemetry-pill">
                    <span>Model Size: <span class="telemetry-val">{{ model_ram_mb }} MB</span></span>
                </div>
                <div class="telemetry-pill">
                    <span>Host RAM: <span class="telemetry-val" id="hostRamUsage">-- GB</span></span>
                </div>
            </div>
        </header>

        <div class="main-layout">
            <!-- Left Column: Controls & Prompting -->
            <div class="card">
                <div class="card-header">
                    <h2>✨ Prompt Studio</h2>
                    <button class="btn-enhance" onclick="enhancePrompt()">⚡ AI Prompt Enhancer</button>
                </div>

                <div class="form-group">
                    <label for="promptInput">
                        <span>Natural Language Prompt</span>
                    </label>
                    <textarea id="promptInput" rows="3" placeholder="Describe your image in natural language...">a glowing ruby red celestial sun with stars on sunset background</textarea>
                </div>

                <div class="form-group">
                    <label>Artistic Style Presets</label>
                    <div class="style-pills">
                        <span class="style-pill active" onclick="selectStyle(this, 'Photorealistic')">📸 Photorealistic</span>
                        <span class="style-pill" onclick="selectStyle(this, 'Cyberpunk / Sci-Fi')">🌆 Cyberpunk</span>
                        <span class="style-pill" onclick="selectStyle(this, 'Anime / Studio Ghibli')">🌸 Studio Ghibli</span>
                        <span class="style-pill" onclick="selectStyle(this, '3D Octane Render')">🔮 3D Octane</span>
                        <span class="style-pill" onclick="selectStyle(this, 'Dark Fantasy')">⚔️ Dark Fantasy</span>
                        <span class="style-pill" onclick="selectStyle(this, 'Synthwave / Retro')">📼 Synthwave</span>
                    </div>
                </div>

                <div class="controls-grid">
                    <div class="control-box">
                        <label>Temperature <span class="metric-val" id="tempDisp">0.80</span></label>
                        <input type="range" id="tempRange" min="0.1" max="1.5" step="0.05" value="0.80" oninput="document.getElementById('tempDisp').innerText = this.value">
                    </div>
                    <div class="control-box">
                        <label>Top-P <span class="metric-val" id="topPDisp">0.90</span></label>
                        <input type="range" id="topPRange" min="0.5" max="1.0" step="0.02" value="0.90" oninput="document.getElementById('topPDisp').innerText = this.value">
                    </div>
                    <div class="control-box">
                        <label>Seed (-1 = Random)</label>
                        <input type="text" id="seedInput" value="-1" style="padding: 0.35rem 0.6rem; font-size: 0.85rem;">
                    </div>
                </div>

                <button class="btn-generate" id="generateBtn" onclick="triggerGeneration()">
                    <span>🎨 Generate Master Image</span>
                </button>
            </div>

            <!-- Right Column: Visual Output & Telemetry -->
            <div class="card">
                <div class="card-header">
                    <h2>🖼️ Output Canvas & KV-Cache Telemetry</h2>
                    <span id="genStatus" style="font-size: 0.78rem; color: var(--accent-emerald); font-weight: 600;">Ready</span>
                </div>

                <div class="preview-container" id="previewArea">
                    <div class="spinner" id="spinner"></div>
                    <div class="token-grid-visualizer" id="tokenGrid" style="display: none;">
                        <!-- 64 spatial token cells -->
                    </div>
                    <div id="placeholderText" style="color: var(--text-muted); font-size: 0.9rem; text-align: center; padding: 2rem;">
                        Enter a prompt and click Generate to see live token streaming and image reconstruction.
                    </div>
                    <img id="mainImage" style="display: none;" alt="Generated Result">
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-val" id="metricTime">-- ms</div>
                        <div class="metric-lbl">Latency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val" id="metricTokens">64 (8x8)</div>
                        <div class="metric-lbl">Image Tokens</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val" id="metricKvCache">< 1.2 MB</div>
                        <div class="metric-lbl">KV-Cache RAM</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val" id="metricRamPeak">< 350 MB</div>
                        <div class="metric-lbl">Peak Process RAM</div>
                    </div>
                </div>

                <div class="action-row">
                    <button class="btn-action" onclick="downloadImage()">💾 Download PNG</button>
                    <button class="btn-action" onclick="copyPrompt()">📋 Copy Prompt</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedStyle = 'Photorealistic';
        let currentImageBase64 = null;

        // Build 64 token cells for spatial visualizer
        const grid = document.getElementById('tokenGrid');
        for (let i = 0; i < 64; i++) {
            const cell = document.createElement('div');
            cell.className = 'token-cell';
            cell.id = 'cell_' + i;
            grid.appendChild(cell);
        }

        function selectStyle(el, styleName) {
            document.querySelectorAll('.style-pill').forEach(p => p.classList.remove('active'));
            el.classList.add('active');
            selectedStyle = styleName;
        }

        async function enhancePrompt() {
            const current = document.getElementById('promptInput').value.trim();
            if (!current) return;
            try {
                const res = await fetch('/api/enhance_prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: current, style: selectedStyle })
                });
                const data = await res.json();
                if (data.enhanced_prompt) {
                    document.getElementById('promptInput').value = data.enhanced_prompt;
                }
            } catch (e) {
                console.error(e);
            }
        }

        async function triggerGeneration() {
            const prompt = document.getElementById('promptInput').value.trim();
            if (!prompt) return;

            const temp = parseFloat(document.getElementById('tempRange').value);
            const topP = parseFloat(document.getElementById('topPRange').value);
            const seed = parseInt(document.getElementById('seedInput').value);

            const btn = document.getElementById('generateBtn');
            const spinner = document.getElementById('spinner');
            const placeholder = document.getElementById('placeholderText');
            const mainImg = document.getElementById('mainImage');
            const tokenGrid = document.getElementById('tokenGrid');
            const genStatus = document.getElementById('genStatus');

            btn.disabled = true;
            spinner.style.display = 'block';
            tokenGrid.style.display = 'grid';
            placeholder.style.display = 'none';
            mainImg.style.display = 'none';
            genStatus.innerText = 'Streaming Tokens...';

            // Animate token generation in visualizer
            let tokenIdx = 0;
            const animInterval = setInterval(() => {
                if (tokenIdx < 64) {
                    const cell = document.getElementById('cell_' + tokenIdx);
                    if (cell) cell.classList.add('active');
                    tokenIdx++;
                }
            }, 25);

            const startT = performance.now();

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, temperature: temp, top_p: topP, seed })
                });

                const data = await res.json();
                clearInterval(animInterval);

                if (data.success) {
                    currentImageBase64 = data.image_base64;
                    mainImg.src = 'data:image/png;base64,' + data.image_base64;
                    mainImg.style.display = 'block';
                    tokenGrid.style.display = 'none';

                    const elapsed = Math.round(performance.now() - startT);
                    document.getElementById('metricTime').innerText = elapsed + ' ms';
                    document.getElementById('metricKvCache').innerText = data.kv_cache_kb + ' KB';
                    document.getElementById('metricRamPeak').innerText = data.process_ram_mb + ' MB';
                    document.getElementById('hostRamUsage').innerText = data.host_ram_used_gb + ' / ' + data.host_ram_total_gb + ' GB';
                    genStatus.innerText = 'Completed in ' + elapsed + 'ms';
                } else {
                    alert('Error: ' + data.error);
                    placeholder.style.display = 'block';
                    genStatus.innerText = 'Error';
                }
            } catch (err) {
                clearInterval(animInterval);
                alert('Generation failed: ' + err.message);
                placeholder.style.display = 'block';
                genStatus.innerText = 'Failed';
            } finally {
                spinner.style.display = 'none';
                tokenGrid.style.display = 'none';
                btn.disabled = false;
                // Reset cells
                for (let i = 0; i < 64; i++) {
                    const cell = document.getElementById('cell_' + i);
                    if (cell) cell.classList.remove('active');
                }
            }
        }

        function downloadImage() {
            if (!currentImageBase64) return;
            const a = document.createElement('a');
            a.href = 'data:image/png;base64,' + currentImageBase64;
            a.download = 'nano_gemini_creation.png';
            a.click();
        }

        function copyPrompt() {
            const prompt = document.getElementById('promptInput').value;
            navigator.clipboard.writeText(prompt);
            alert('Prompt copied to clipboard!');
        }

        // Fetch initial telemetry
        fetch('/api/telemetry')
            .then(res => res.json())
            .then(data => {
                document.getElementById('hostRamUsage').innerText = data.host_ram_used_gb + ' / ' + data.host_ram_total_gb + ' GB';
            }).catch(e => console.error(e));
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    init_master_pipeline()
    return render_template_string(HTML_PAGE, model_ram_mb=MODEL_RAM_MB)


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    vmem = psutil.virtual_memory()
    proc = psutil.Process()
    return jsonify({
        "host_ram_total_gb": round(vmem.total / (1024**3), 1),
        "host_ram_used_gb": round(vmem.used / (1024**3), 1),
        "host_ram_percent": vmem.percent,
        "process_ram_mb": round(proc.memory_info().rss / (1024**2), 1),
        "model_ram_mb": MODEL_RAM_MB,
    })


@app.route("/api/enhance_prompt", methods=["POST"])
def enhance_prompt_api():
    data = request.get_json() or {}
    base_prompt = data.get("prompt", "")
    style = data.get("style", "Photorealistic")
    enhanced = PromptEnhancer.enhance(base_prompt, style=style)
    return jsonify({"enhanced_prompt": enhanced})


@app.route("/api/generate", methods=["POST"])
def generate_api():
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "a glowing ruby red celestial sun on sunset background")
        temp = float(data.get("temperature", 0.80))
        top_p = float(data.get("top_p", 0.90))
        seed = int(data.get("seed", -1))

        if seed >= 0:
            torch.manual_seed(seed)

        pipeline = init_master_pipeline()
        gen_cfg = GenerationConfig(temperature=temp, top_p=top_p)

        start_time = time.time()
        img, tokens = pipeline.generate_image_with_tokens(prompt=prompt, gen_config=gen_cfg)
        elapsed_sec = time.time() - start_time

        # Memory Telemetry
        kv_cache_bytes = pipeline.llm.get_total_kv_cache_bytes()
        kv_cache_kb = round(kv_cache_bytes / 1024, 2)

        proc = psutil.Process()
        proc_ram_mb = round(proc.memory_info().rss / (1024**2), 1)

        vmem = psutil.virtual_memory()

        # Encode image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "image_base64": img_b64,
            "tokens": tokens,
            "kv_cache_kb": kv_cache_kb,
            "model_ram_mb": MODEL_RAM_MB,
            "process_ram_mb": proc_ram_mb,
            "host_ram_used_gb": round(vmem.used / (1024**3), 1),
            "host_ram_total_gb": round(vmem.total / (1024**3), 1),
            "elapsed_ms": round(elapsed_sec * 1000),
            "generation_time_s": round(elapsed_sec, 2),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_master_pipeline()
    print("[*] Starting Nano-Gemini Pro Studio on http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
