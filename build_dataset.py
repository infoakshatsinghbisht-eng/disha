"""
High-Fidelity Automated Multimodal Dataset Builder with 4x Anti-Aliased Super-sampling.
Generates crystal-clear, visually stunning, multi-category images with precise descriptive prompts.
"""

import os
import random
import math
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFilter


class MultimodalDatasetBuilder:
    """
    Generates high-fidelity, anti-aliased synthetic images paired with rich descriptive captions.
    Uses 4x supersampling + Lanczos filtering for anti-aliasing.
    """

    COLOR_PALETTES = {
        "ruby_red": ((245, 35, 65), "vibrant ruby red"),
        "electric_blue": ((30, 140, 255), "electric cobalt blue"),
        "emerald_green": ((20, 215, 110), "emerald green"),
        "golden_yellow": ((255, 215, 30), "bright golden yellow"),
        "neon_purple": ((180, 50, 255), "neon purple"),
        "cyber_cyan": ((15, 240, 245), "cyber cyan"),
        "sunset_coral": ((255, 110, 60), "sunset coral"),
        "magenta_pink": ((255, 35, 175), "hot magenta pink"),
        "pure_white": ((250, 252, 255), "pure glowing white"),
        "metallic_gold": ((235, 185, 45), "deep metallic gold"),
        "deep_sapphire": ((20, 60, 200), "deep sapphire blue"),
        "jade_teal": ((25, 190, 165), "jade teal"),
    }

    THEMES = ["sunset", "cyberpunk", "deep_space", "anime_dawn", "dark_void", "clean_minimal", "neon_synth"]

    @classmethod
    def _create_super_bg(cls, size: int, theme: str) -> Image.Image:
        """Creates high-resolution background with smooth gradients and atmospheric effects."""
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)

        if theme == "sunset":
            top = (255, 80, 100)
            mid = (160, 45, 140)
            bot = (20, 15, 50)
            for y in range(size):
                t = y / size
                if t < 0.5:
                    st = t * 2
                    r = int(top[0] * (1 - st) + mid[0] * st)
                    g = int(top[1] * (1 - st) + mid[1] * st)
                    b = int(top[2] * (1 - st) + mid[2] * st)
                else:
                    st = (t - 0.5) * 2
                    r = int(mid[0] * (1 - st) + bot[0] * st)
                    g = int(mid[1] * (1 - st) + bot[1] * st)
                    b = int(mid[2] * (1 - st) + bot[2] * st)
                draw.line([(0, y), (size, y)], fill=(r, g, b))

        elif theme == "cyberpunk":
            for y in range(size):
                r = int(12 + 18 * math.sin(y / (size * 0.15)))
                g = int(8 + 12 * math.cos(y / (size * 0.2)))
                b = int(35 + 40 * (y / size))
                draw.line([(0, y), (size, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))

        elif theme == "deep_space":
            for y in range(size):
                r = int(4 + 10 * (y / size))
                g = int(6 + 14 * (y / size))
                b = int(18 + 32 * (y / size))
                draw.line([(0, y), (size, y)], fill=(r, g, b))

        elif theme == "anime_dawn":
            top = (100, 140, 230)
            mid = (250, 175, 180)
            bot = (255, 230, 170)
            for y in range(size):
                t = y / size
                if t < 0.6:
                    st = t / 0.6
                    r = int(top[0] * (1 - st) + mid[0] * st)
                    g = int(top[1] * (1 - st) + mid[1] * st)
                    b = int(top[2] * (1 - st) + mid[2] * st)
                else:
                    st = (t - 0.6) / 0.4
                    r = int(mid[0] * (1 - st) + bot[0] * st)
                    g = int(mid[1] * (1 - st) + bot[1] * st)
                    b = int(mid[2] * (1 - st) + bot[2] * st)
                draw.line([(0, y), (size, y)], fill=(r, g, b))

        elif theme == "neon_synth":
            for y in range(size):
                r = int(25 + 30 * math.sin(y / (size * 0.1)))
                g = int(5)
                b = int(50 + 60 * (1.0 - y / size))
                draw.line([(0, y), (size, y)], fill=(max(0, min(255, r)), g, max(0, min(255, b))))

        elif theme == "clean_minimal":
            for y in range(size):
                val = int(248 - 18 * (y / size))
                draw.line([(0, y), (size, y)], fill=(val, val, val + 4))

        else:  # dark_void
            for y in range(size):
                v = int(12 + 8 * (y / size))
                draw.line([(0, y), (size, y)], fill=(v, v + 2, v + 8))

        return img

    @classmethod
    def generate_sample(cls, target_size: int = 64) -> Tuple[Image.Image, str]:
        """Generates a high-fidelity image by rendering at 4x scale and downsampling with anti-aliasing."""
        scale = 4
        canvas_size = target_size * scale
        cx, cy = canvas_size // 2, canvas_size // 2

        theme = random.choice(cls.THEMES)
        img = cls._create_super_bg(canvas_size, theme)
        draw = ImageDraw.Draw(img)

        palette_key = random.choice(list(cls.COLOR_PALETTES.keys()))
        color_rgb, color_name = cls.COLOR_PALETTES[palette_key]

        scene_type = random.choice([
            "celestial_sun",
            "cosmic_orb",
            "cyber_pyramid",
            "glowing_star",
            "faceted_diamond",
            "neon_rings",
            "anime_landscape",
            "sacred_polygon",
        ])

        if scene_type == "celestial_sun":
            r = canvas_size // 4
            # Outer atmospheric glow rings
            for dr in range(r + 30, r, -4):
                alpha = int(25 * (1.0 - (dr - r) / 30.0))
                glow_color = tuple(min(255, c + 40) for c in color_rgb)
                draw.ellipse([(cx - dr, cy - dr), (cx + dr, cy + dr)], fill=glow_color)
            # Main Sun Body
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color_rgb, outline=(255, 255, 255), width=scale * 2)
            # Twinkling distant stars
            for _ in range(random.randint(12, 28)):
                sx = random.randint(10, canvas_size - 10)
                sy = random.randint(10, canvas_size // 2)
                draw.ellipse([(sx - 2, sy - 2), (sx + 2, sy + 2)], fill=(255, 255, 255))
            caption = f"a glowing {color_name} celestial sun with warm atmospheric flares on {theme} background"

        elif scene_type == "cosmic_orb":
            r = canvas_size // 4
            # Core orb
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color_rgb, outline=(255, 255, 255), width=scale)
            # Orbital ring
            rx, ry = int(r * 1.6), int(r * 0.45)
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=(255, 255, 240), width=scale * 2)
            caption = f"a radiant {color_name} cosmic orb with glowing planetary rings on {theme} background"

        elif scene_type == "cyber_pyramid":
            pw = canvas_size // 3
            ph = canvas_size // 3
            points = [(cx, cy - ph), (cx - pw, cy + ph), (cx + pw, cy + ph)]
            # Shadow facet
            draw.polygon([(cx, cy - ph), (cx, cy + ph), (cx + pw, cy + ph)], fill=tuple(int(c * 0.7) for c in color_rgb))
            # Lit facet
            draw.polygon([(cx, cy - ph), (cx - pw, cy + ph), (cx, cy + ph)], fill=color_rgb, outline=(255, 255, 255), width=scale)
            # Horizon grid lines
            for hy in range(cy + ph, canvas_size, scale * 12):
                draw.line([(0, hy), (canvas_size, hy)], fill=(60, 70, 110), width=scale)
            caption = f"a sharp {color_name} cyber pyramid with illuminated edges on {theme} landscape"

        elif scene_type == "glowing_star":
            points = []
            num_pts = 5
            r_outer = canvas_size // 3.5
            r_inner = r_outer // 2.2
            for i in range(num_pts * 2):
                rad = i * math.pi / num_pts - math.pi / 2
                cur_r = r_outer if i % 2 == 0 else r_inner
                points.append((cx + cur_r * math.cos(rad), cy + cur_r * math.sin(rad)))
            draw.polygon(points, fill=color_rgb, outline=(255, 255, 255), width=scale * 2)
            caption = f"a centered glowing {color_name} star with radiant points on {theme} backdrop"

        elif scene_type == "faceted_diamond":
            dw = canvas_size // 3.5
            dh = canvas_size // 2.8
            top = (cx, cy - dh)
            bottom = (cx, cy + dh)
            left = (cx - dw, cy)
            right = (cx + dw, cy)
            draw.polygon([top, right, bottom, left], fill=color_rgb, outline=(255, 255, 255), width=scale * 2)
            # Inner facet lines
            draw.line([top, bottom], fill=(255, 255, 255), width=scale)
            draw.line([left, right], fill=(255, 255, 255), width=scale)
            caption = f"a sharp {color_name} diamond crystal emblem on {theme} background"

        elif scene_type == "neon_rings":
            r_base = canvas_size // 4
            for i in range(3):
                cur_r = r_base + i * (scale * 8)
                shade = tuple(min(255, int(c * (1.0 - i * 0.2))) for c in color_rgb)
                draw.ellipse([(cx - cur_r, cy - cur_r), (cx + cur_r, cy + cur_r)], outline=shade, width=scale * 2)
            caption = f"concentric glowing {color_name} neon energy rings on {theme} backdrop"

        elif scene_type == "anime_landscape":
            # Mountains in background
            m1 = [(0, canvas_size), (canvas_size * 0.35, cy - scale * 10), (canvas_size * 0.7, canvas_size)]
            m2 = [(canvas_size * 0.3, canvas_size), (canvas_size * 0.65, cy - scale * 20), (canvas_size, canvas_size)]
            draw.polygon(m1, fill=(45, 50, 90))
            draw.polygon(m2, fill=(30, 35, 75))
            # Foreground hills
            draw.ellipse([(-canvas_size * 0.2, cy + scale * 10), (canvas_size * 0.8, canvas_size * 1.5)], fill=(20, 30, 60))
            draw.ellipse([(canvas_size * 0.3, cy + scale * 15), (canvas_size * 1.3, canvas_size * 1.5)], fill=(15, 22, 45))
            # Sun or Moon
            sr = canvas_size // 7
            draw.ellipse([(cx - sr, cy - canvas_size // 3 - sr), (cx + sr, cy - canvas_size // 3 + sr)], fill=color_rgb, outline=(255, 255, 255), width=scale)
            caption = f"a tranquil anime landscape with mountains and a {color_name} sun on {theme} sky"

        else:  # sacred_polygon
            num_sides = random.choice([6, 8])
            poly_r = canvas_size // 3.5
            poly_pts = []
            for i in range(num_sides):
                angle = i * 2 * math.pi / num_sides
                poly_pts.append((cx + poly_r * math.cos(angle), cy + poly_r * math.sin(angle)))
            draw.polygon(poly_pts, fill=color_rgb, outline=(255, 255, 255), width=scale * 2)
            # Inner circle
            ic_r = poly_r // 2
            draw.ellipse([(cx - ic_r, cy - ic_r), (cx + ic_r, cy + ic_r)], fill=(255, 255, 255))
            caption = f"a glowing {color_name} sacred geometric emblem on {theme} background"

        # Downsample using high-quality Lanczos filter for anti-aliasing
        final_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return final_img, caption
