"""
Agentic CoT (Chain-of-Thought) Synthetic Recaptioner.
Converts standard flat captions into rich, multi-stage reasoning traces (<think>)
and structured tool calls (<tool_call>), teaching the foundation model test-time compute.
"""

from typing import Dict, List, Optional, Tuple
import random
import re


class AgenticRecaptioner:
    """
    Synthesizes rich Chain-of-Thought training prompts from standard image-caption pairs.
    Embeds semantic decomposition, artistic planning, lighting direction, and camera optics
    into native <think> ... </think> tokens.
    """

    COMPOSITION_PATTERNS = [
        "Rule-of-thirds balanced composition with deep spatial depth",
        "Centered symmetry with dramatic foreground focus and soft background bokeh",
        "Dynamic diagonal perspective with leading visual lines",
        "Cinematic wide-angle atmospheric vista with expansive negative space",
        "Macro close-up framing with sharp edge separation and shallow depth of field",
    ]

    LIGHTING_PATTERNS = [
        "natural volumetric sunlight with soft ray dispersion and ambient fill",
        "moody chiaroscuro rim lighting with deep contrasted shadows",
        "golden hour warm sunlight with glowing amber highlights",
        "bioluminescent neon glow with wet asphalt specular reflections",
        "diffused studio softbox illumination with smooth tonal gradients",
    ]

    CAMERA_OPTICS = [
        "shot on 35mm prime lens, f/1.8 aperture, crisp focus, 8k resolution",
        "shot on 85mm portrait lens, f/1.4 aperture, creamy bokeh, natural color fidelity",
        "shot on 24mm ultra-wide lens, f/2.8 aperture, high dynamic range",
        "macro 100mm lens, f/2.8, razor-sharp micro-textures and tactile detailing",
    ]

    @classmethod
    def generate_cot_trace(cls, raw_caption: str, category: Optional[str] = None) -> Tuple[str, str]:
        """
        Generates:
        1. Full training prompt containing <think> reasoning and <tool_call>.
        2. Expanded cinematic prompt for image generation.
        """
        clean_prompt = raw_caption.strip().rstrip(".")
        comp = random.choice(cls.COMPOSITION_PATTERNS)
        light = random.choice(cls.LIGHTING_PATTERNS)
        optics = random.choice(cls.CAMERA_OPTICS)

        # Detect genre
        genre = category or "Fine Art Photography"
        lower = clean_prompt.lower()
        if any(w in lower for w in ["cyberpunk", "neon", "future", "sci-fi"]):
            genre = "Cyberpunk / Sci-Fi"
        elif any(w in lower for w in ["mountain", "forest", "beach", "desert", "ocean", "river"]):
            genre = "Scenic Landscape"
        elif any(w in lower for w in ["dog", "cat", "horse", "bird", "tiger", "lion"]):
            genre = "Wildlife & Nature"
        elif any(w in lower for w in ["skyscraper", "building", "city", "bridge", "house"]):
            genre = "Urban Architecture"
        elif any(w in lower for w in ["sun", "galaxy", "star", "space", "nebula"]):
            genre = "Celestial & Deep Space"

        # Expanded generation prompt
        expanded_prompt = f"{clean_prompt}, {light}, {optics}, highly detailed, award-winning visual composition"

        # Synthetic Chain-of-Thought reasoning
        think_trace = (
            f"<think>\n"
            f"User Goal: Synthesize high-fidelity image for '{clean_prompt}'.\n"
            f"1. Genre: {genre}.\n"
            f"2. Framing: {comp}.\n"
            f"3. Lighting: {light}.\n"
            f"4. Optics: {optics}.\n"
            f"5. Strategy: Dispatching generation tool with expanded cinematic parameters.\n"
            f"</think>\n"
            f"<tool_call: generate_image {{\"prompt\": \"{expanded_prompt}\"}}>"
        )

        return think_trace, expanded_prompt
