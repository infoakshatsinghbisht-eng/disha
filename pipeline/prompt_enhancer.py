"""
Autonomous Prompt Engineering & Style Enhancement Engine.
Enriches basic user prompts with professional artistic descriptors, lighting, textures, and camera framing.
"""

from typing import Dict, List, Optional
import random


class PromptEnhancer:
    """
    Intelligent prompt enhancement assistant for text-to-image foundation models.
    Supports artistic presets: Photorealistic, Cyberpunk, Studio Ghibli/Anime, 3D Octane, Dark Fantasy, Synthwave.
    """

    STYLE_MODIFIERS: Dict[str, List[str]] = {
        "Photorealistic": [
            "masterpiece 8k resolution photograph",
            "sharp natural volumetric lighting",
            "hyperdetailed textures",
            "shot on 35mm lens, f/1.8 aperture",
            "cinematic color grading",
            "photorealistic, award winning",
        ],
        "Cyberpunk / Sci-Fi": [
            "futuristic cyberpunk aesthetic",
            "glowing neon reflections",
            "holographic accents in purple and cyan",
            "dense urban atmosphere with wet asphalt",
            "octane render, unreal engine 5",
            "high-tech intricate detailing",
        ],
        "Anime / Studio Ghibli": [
            "studio ghibli art style",
            "vibrant painterly colors",
            "soft sunlit clouds and lush nature",
            "detailed anime key visual",
            "makoto shinkai inspired lighting",
            "nostalgic cinematic frame",
        ],
        "3D Octane Render": [
            "3D digital sculpture render",
            "smooth glossy subsurface scattering",
            "dramatic studio key lighting with soft shadows",
            "raytraced reflections, 8k",
            "minimalist modern art gallery piece",
            "octane render quality",
        ],
        "Dark Fantasy": [
            "epic dark fantasy concept art",
            "moody chiaroscuro lighting",
            "ancient runes and glowing magical aura",
            "misty ethereal backdrop",
            "detailed oil painting on canvas",
            "elden ring inspired atmosphere",
        ],
        "Synthwave / Retro": [
            "retro 80s synthwave aesthetic",
            "neon grid horizon under setting digital sun",
            "magenta and cyan color palette",
            "vhs tape chromatic aberration",
            "nostalgic retro-futurism",
        ],
    }

    LIGHTING_OPTIONS = [
        "dramatic rim lighting",
        "soft golden hour sunlight",
        "neon ambient glow",
        "cinematic volumetric fog lighting",
        "clean studio softbox illumination",
    ]

    @classmethod
    def enhance(
        cls,
        base_prompt: str,
        style: Optional[str] = None,
        lighting: Optional[str] = None,
        add_quality_boosters: bool = True,
    ) -> str:
        """
        Expands a raw prompt into a rich descriptor string.
        """
        prompt_parts = [base_prompt.strip()]

        if style and style in cls.STYLE_MODIFIERS:
            chosen_modifiers = random.sample(cls.STYLE_MODIFIERS[style], min(3, len(cls.STYLE_MODIFIERS[style])))
            prompt_parts.extend(chosen_modifiers)

        if lighting and lighting in cls.LIGHTING_OPTIONS:
            prompt_parts.append(lighting)
        elif not style:
            prompt_parts.append(random.choice(cls.LIGHTING_OPTIONS))

        if add_quality_boosters:
            prompt_parts.append("highly detailed, clean composition, crisp focus")

        return ", ".join(prompt_parts)
