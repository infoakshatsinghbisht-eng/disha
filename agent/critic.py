"""
Visual Critic and Image Quality Inspector.
Analyzes image sharpness, contrast, dynamic range, and color richness to score aesthetic quality
and generate automated refinement guidance for the ReAct reflection loop.
"""

from typing import List, Tuple
import math
import numpy as np
from PIL import Image

from .schema import CriticReport


class VisualCritic:
    """
    Automated Visual Quality & Aesthetic Critic.
    Uses computer vision signal-processing metrics to evaluate synthesized images
    without requiring heavy external neural networks, running in < 5ms.
    """

    def __init__(self, aesthetic_threshold: float = 7.0):
        self.aesthetic_threshold = aesthetic_threshold

    def evaluate(self, image: Image.Image, prompt: str = "") -> CriticReport:
        """
        Evaluates an image and returns a comprehensive CriticReport.
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_np = np.array(image, dtype=np.float32)
        
        # 1. Grayscale luminance calculation: Y = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

        # 2. Sharpness via Discrete Laplacian Variance
        # Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        pad_gray = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        laplacian = (
            pad_gray[:-2, 1:-1]
            + pad_gray[2:, 1:-1]
            + pad_gray[1:-1, :-2]
            + pad_gray[1:-1, 2:]
            - 4.0 * pad_gray[1:-1, 1:-1]
        )
        sharpness = float(np.var(laplacian))

        # 3. Contrast (Standard deviation of luminance)
        contrast = float(np.std(gray))

        # 4. Dynamic Range (99th percentile - 1st percentile)
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p1)

        # 5. Color Richness (Saturation distribution & channel diversity)
        # S = (max(RGB) - min(RGB)) / (max(RGB) + 1e-5)
        rgb_max = np.max(img_np, axis=2)
        rgb_min = np.min(img_np, axis=2)
        saturation = (rgb_max - rgb_min) / (rgb_max + 1e-5)
        mean_saturation = float(np.mean(saturation)) * 10.0  # Scale to 0-10

        # Channel covariance/diversity
        r, g, b = img_np[:, :, 0].flatten(), img_np[:, :, 1].flatten(), img_np[:, :, 2].flatten()
        color_spread = float(np.std(r) + np.std(g) + np.std(b)) / 3.0
        color_richness = min(10.0, (mean_saturation * 0.5 + (color_spread / 25.5) * 0.5))

        # 6. Aggregate Aesthetic Score (0.0 to 10.0)
        # Sharpness weight: 3.5 (optimal range ~ 100-300)
        norm_sharpness = min(sharpness / 220.0, 1.0) * 3.5
        # Contrast weight: 3.0 (optimal std ~ 45-75)
        norm_contrast = min(contrast / 65.0, 1.0) * 3.0
        # Dynamic range weight: 2.0 (optimal range ~ 200-255)
        norm_dr = min(dynamic_range / 220.0, 1.0) * 2.0
        # Color richness weight: 1.5
        norm_color = min(color_richness / 7.0, 1.0) * 1.5

        raw_score = norm_sharpness + norm_contrast + norm_dr + norm_color
        aesthetic_score = round(max(1.0, min(10.0, float(raw_score))), 1)

        # 7. Qualitative Critique & Refinement Suggestions
        critiques: List[str] = []
        suggestions: List[str] = []

        if sharpness < 80.0:
            critiques.append(f"Low edge sharpness ({sharpness:.1f} var); fine textures appear smooth or blurry.")
            suggestions.append("Inject 'crisp hyperdetailed micro-textures, 8k sharp focus, ultra-fine detailing'")
        else:
            critiques.append(f"Excellent edge sharpness ({sharpness:.1f} var).")

        if contrast < 35.0:
            critiques.append(f"Flat contrast profile ({contrast:.1f} std); shadows and highlights are compressed.")
            suggestions.append("Inject 'dramatic chiaroscuro lighting, deep shadows, cinematic volumetric highlights'")
        else:
            critiques.append(f"Strong dynamic contrast ({contrast:.1f} std).")

        if dynamic_range < 140.0:
            critiques.append("Limited dynamic range; image is clipped or washed out.")
            suggestions.append("Inject 'wide tonal latitude, vibrant highlights, rich dark tones'")

        if color_richness < 3.5:
            critiques.append("Monochromatic or muted color palette.")
            suggestions.append("Inject 'cinematic color grading, vibrant harmonious palette, saturated hues'")

        status = "APPROVED" if aesthetic_score >= self.aesthetic_threshold else "NEEDS_REFINEMENT"
        full_critique = " ".join(critiques)

        return CriticReport(
            sharpness=round(sharpness, 2),
            contrast=round(contrast, 2),
            dynamic_range=round(dynamic_range, 2),
            color_richness=round(color_richness, 2),
            aesthetic_score=aesthetic_score,
            status=status,
            critique=full_critique,
            suggested_refinements=suggestions,
        )
