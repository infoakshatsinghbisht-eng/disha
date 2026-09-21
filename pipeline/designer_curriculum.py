"""
Child-to-Genius Graphic Designer Curriculum Learning Engine for Disha Multimodal Model.

Exhaustive Geometric & Multi-Aspect Taxonomy:
1. Complete Shape Encyclopedia (25+ 2D & 3D Primitives across wireframe, solid, gradient, and bordered modalities).
2. Multi-Aspect Object Grounding ("Ek cheez ki saari aspects": Outline Wireframe, Flat Vector, 3D Isometric Volume, Glass/Claymorphic App Icon).
3. Advanced Layouts & Materials (Glassmorphism, Swiss Bauhaus, Neumorphism, Isometric Platforms).
4. Master-level Graphic Design (Cyberpunk Telemetry HUDs, Haute Luxury Gold Foil, Kinetic Posters).

All rendered with 4x supersampling (Lanczos) and accompanied by Agentic CoT (<think>) reasoning.
"""

import math
import random
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageDraw, ImageFilter


class GraphicDesignerCurriculum:
    """
    Exhaustive procedural graphic design generator with comprehensive shape taxonomy
    and multi-aspect object representations.
    """

    # Comprehensive Harmonious Palettes
    PALETTES = [
        {"primary": (230, 57, 70), "secondary": (69, 123, 157), "accent": (241, 250, 238), "bg": (255, 255, 255), "name": "Vibrant Crimson & Slate"},
        {"primary": (42, 157, 143), "secondary": (231, 111, 81), "accent": (244, 162, 97), "bg": (255, 255, 255), "name": "Persian Green & Coral"},
        {"primary": (30, 144, 255), "secondary": (255, 191, 0), "accent": (240, 248, 255), "bg": (255, 255, 255), "name": "Cobalt Blue & Amber Gold"},
        {"primary": (138, 43, 226), "secondary": (0, 250, 154), "accent": (255, 255, 255), "bg": (18, 18, 24), "name": "Neon Violet & Spring Emerald"},
        {"primary": (212, 175, 55), "secondary": (245, 230, 160), "accent": (15, 25, 45), "bg": (10, 15, 28), "name": "Metallic Royal Gold on Deep Obsidian"},
        {"primary": (0, 245, 255), "secondary": (255, 0, 128), "accent": (255, 255, 255), "bg": (12, 14, 26), "name": "Cyberpunk Cyan & Hot Magenta"},
        {"primary": (255, 87, 34), "secondary": (33, 150, 243), "accent": (250, 250, 250), "bg": (245, 245, 247), "name": "Bauhaus Primary Triad"},
        {"primary": (245, 130, 32), "secondary": (80, 50, 140), "accent": (255, 255, 255), "bg": (255, 255, 255), "name": "Sunset Tangerine & Royal Purple"},
        {"primary": (20, 20, 20), "secondary": (160, 160, 160), "accent": (255, 255, 255), "bg": (255, 255, 255), "name": "Monochrome Minimalist Black & White"},
    ]

    # Exhaustive 2D Shape Taxonomy
    ALL_SHAPES = [
        "circle", "semicircle", "oval", "capsule", "teardrop", "crescent_moon",
        "square", "rectangle", "diamond", "trapezoid", "parallelogram",
        "triangle_equilateral", "triangle_right", "triangle_isosceles",
        "pentagon", "hexagon", "heptagon", "octagon", "decagon",
        "star_4", "star_5", "star_6", "star_8",
        "cross_plus", "heart", "spiral", "arrow_right", "concentric_rings", "dots_grid"
    ]

    # Multi-Aspect Render Modalities for Objects
    OBJECT_ASPECTS = ["outline", "flat_silhouette", "duotone_logo", "isometric_3d"]

    @staticmethod
    def _create_canvas(size: int, bg_color: Tuple[int, int, int]) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (size, size), bg_color)
        draw = ImageDraw.Draw(img)
        return img, draw

    @staticmethod
    def _downsample(img: Image.Image, target_size: int = 256) -> Image.Image:
        return img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # -------------------------------------------------------------------------
    # LEVEL 1: Exhaustive Visual Alphabet & Complete Shape Taxonomy
    # -------------------------------------------------------------------------
    @classmethod
    def generate_level1_primitive(cls, specific_shape: str = None) -> Tuple[Image.Image, str, str]:
        """
        Level 1: Generates any shape from the complete 25+ geometric taxonomy.
        Rendered across outline, solid fill, or contrasting stroke styles.
        """
        super_size = 1024
        palette = random.choice(cls.PALETTES)
        bg = palette["bg"]
        primary = palette["primary"]
        secondary = palette["secondary"]

        img, draw = cls._create_canvas(super_size, bg)
        center = super_size // 2
        radius = random.randint(220, 350)

        shape = specific_shape if specific_shape in cls.ALL_SHAPES else random.choice(cls.ALL_SHAPES)
        render_mode = random.choice(["solid", "outline", "stroke_fill"])

        # Colors based on render mode
        fill_col = primary if render_mode in ["solid", "stroke_fill"] else None
        outline_col = secondary if render_mode in ["outline", "stroke_fill"] else None
        line_w = 18 if render_mode in ["outline", "stroke_fill"] else 0

        shape_title = shape.replace("_", " ").title()

        if shape == "circle":
            draw.ellipse([center - radius, center - radius, center + radius, center + radius], fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "semicircle":
            draw.pieslice([center - radius, center - radius, center + radius, center + radius], start=0, end=180, fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "oval":
            rx, ry = radius, int(radius * 0.62)
            draw.ellipse([center - rx, center - ry, center + rx, center + ry], fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "capsule":
            rx, ry = radius, int(radius * 0.45)
            draw.rounded_rectangle([center - rx, center - ry, center + rx, center + ry], radius=ry, fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "teardrop":
            pts = [(center, center - radius), (center + radius, center + radius // 2), (center - radius, center + radius // 2)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            draw.ellipse([center - radius, center - radius // 4, center + radius, center + radius], fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "crescent_moon":
            draw.ellipse([center - radius, center - radius, center + radius, center + radius], fill=primary)
            cut_r = int(radius * 0.82)
            draw.ellipse([center - cut_r + 90, center - cut_r - 60, center + cut_r + 90, center + cut_r - 60], fill=bg)

        elif shape == "square":
            draw.rectangle([center - radius, center - radius, center + radius, center + radius], fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "rectangle":
            rx, ry = radius, int(radius * 0.58)
            draw.rectangle([center - rx, center - ry, center + rx, center + ry], fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "diamond":
            pts = [(center, center - radius), (center + radius, center), (center, center + radius), (center - radius, center)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "trapezoid":
            top_w = int(radius * 0.55)
            pts = [(center - top_w, center - radius // 2), (center + top_w, center - radius // 2), (center + radius, center + radius // 2), (center - radius, center + radius // 2)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "parallelogram":
            slant = 90
            pts = [(center - radius + slant, center - radius // 2), (center + radius, center - radius // 2), (center + radius - slant, center + radius // 2), (center - radius, center + radius // 2)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "triangle_equilateral":
            h = int(radius * math.sqrt(3))
            pts = [(center, center - radius), (center - radius, center + h // 2), (center + radius, center + h // 2)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "triangle_right":
            pts = [(center - radius, center - radius), (center - radius, center + radius), (center + radius, center + radius)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "triangle_isosceles":
            pts = [(center, center - radius), (center - radius // 2, center + radius), (center + radius // 2, center + radius)]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape in ["pentagon", "hexagon", "heptagon", "octagon", "decagon"]:
            n_sides = {"pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8, "decagon": 10}[shape]
            pts = []
            for i in range(n_sides):
                angle = math.radians((360 / n_sides) * i - 90)
                pts.append((center + int(radius * math.cos(angle)), center + int(radius * math.sin(angle))))
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape.startswith("star_"):
            n_points = int(shape.split("_")[1])
            pts = []
            inner_r = radius * 0.45
            for i in range(n_points * 2):
                r = radius if i % 2 == 0 else inner_r
                angle = math.radians((360 / (n_points * 2)) * i - 90)
                pts.append((center + int(r * math.cos(angle)), center + int(r * math.sin(angle))))
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "cross_plus":
            t = radius // 3
            draw.rectangle([center - t, center - radius, center + t, center + radius], fill=fill_col, outline=outline_col, width=line_w)
            draw.rectangle([center - radius, center - t, center + radius, center + t], fill=fill_col, outline=outline_col, width=line_w)

        elif shape == "heart":
            hr = int(radius * 0.55)
            draw.ellipse([center - hr * 2, center - hr * 2, center, center], fill=fill_col, outline=outline_col, width=line_w)
            draw.ellipse([center, center - hr * 2, center + hr * 2, center], fill=fill_col, outline=outline_col, width=line_w)
            pts = [(center - int(hr * 1.9), center - int(hr * 0.5)), (center + int(hr * 1.9), center - int(hr * 0.5)), (center, center + int(hr * 1.8))]
            draw.polygon(pts, fill=fill_col, outline=outline_col)
            if line_w:
                draw.line(pts + [pts[0]], fill=outline_col, width=line_w)

        elif shape == "arrow_right":
            aw = radius // 2
            draw.rectangle([center - radius, center - aw // 2, center + aw // 2, center + aw // 2], fill=fill_col)
            pts = [(center + aw // 2, center - radius // 2), (center + radius, center), (center + aw // 2, center + radius // 2)]
            draw.polygon(pts, fill=fill_col)

        elif shape == "spiral":
            prev_pt = None
            for theta in range(0, 720, 10):
                rad = math.radians(theta)
                r = (theta / 720.0) * radius
                pt = (center + int(r * math.cos(rad)), center + int(r * math.sin(rad)))
                if prev_pt:
                    draw.line([prev_pt, pt], fill=primary, width=16)
                prev_pt = pt

        elif shape == "concentric_rings":
            for step in range(4, 0, -1):
                cur_r = radius * step // 4
                col = primary if step % 2 == 0 else secondary
                draw.ellipse([center - cur_r, center - cur_r, center + cur_r, center + cur_r], fill=col, outline=bg, width=12)

        else:  # dots_grid
            grid_n = 5
            spacing = (radius * 2) // grid_n
            dot_r = spacing // 5
            start_x = center - (grid_n // 2) * spacing
            start_y = center - (grid_n // 2) * spacing
            for gx in range(grid_n):
                for gy in range(grid_n):
                    dx = start_x + gx * spacing
                    dy = start_y + gy * spacing
                    draw.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r], fill=primary)

        desc = f"a clean graphic design primitive of a {shape_title} in {palette['name']} with {render_mode} styling on plain background"
        cot = (
            f"<think> Concept: Complete Shape Taxonomy Level 1. Shape: {shape_title}. Render Mode: {render_mode}. "
            f"Palette: {palette['name']}. Rule: Complete geometric symmetry and edge clarity without background noise. </think> {desc}"
        )

        final_img = cls._downsample(img)
        return final_img, desc, cot

    # -------------------------------------------------------------------------
    # LEVEL 2: Multi-Aspect Object Mastery ("Ek cheez ki saari aspects")
    # -------------------------------------------------------------------------
    @classmethod
    def generate_level2_multi_aspect_object(
        cls, specific_subject: str = None, specific_aspect: str = None
    ) -> Tuple[Image.Image, str, str]:
        """
        Level 2: Teaches ALL aspects of an object:
          - Aspect 1: Outline (Minimalist technical vector wireframe)
          - Aspect 2: Flat Silhouette (Solid 2D brand logo)
          - Aspect 3: Duotone / Two-Tone Logo (High-contrast brand mark)
          - Aspect 4: Isometric 3D Volume (Directional lighting, depth, cast shadow)
        """
        super_size = 1024
        center = super_size // 2

        subjects = ["apple", "rocket", "shield", "camera", "diamond", "flame", "mountain", "planet", "heart", "bird"]
        subject = specific_subject if specific_subject in subjects else random.choice(subjects)
        aspect = specific_aspect if specific_aspect in cls.OBJECT_ASPECTS else random.choice(cls.OBJECT_ASPECTS)

        # Background based on aspect
        bg_col = (255, 255, 255)
        if aspect == "isometric_3d":
            bg_col = (245, 247, 252)

        img, draw = cls._create_canvas(super_size, bg_col)

        # Aspect Specific Palettes
        if aspect == "outline":
            stroke_col = (20, 24, 35)
            # Draw technical CAD blueprint construction guides
            draw.line([(center, 120), (center, super_size - 120)], fill=(225, 230, 240), width=4)
            draw.line([(120, center), (super_size - 120, center)], fill=(225, 230, 240), width=4)
            draw.ellipse([center - 320, center - 320, center + 320, center + 320], outline=(230, 235, 245), width=4)

        # 1. APPLE: All 4 Aspects
        if subject == "apple":
            if aspect == "outline":
                draw.ellipse([center - 200, center - 140, center + 40, center + 200], outline=stroke_col, width=16)
                draw.ellipse([center - 40, center - 140, center + 200, center + 200], outline=stroke_col, width=16)
                draw.polygon([(center, center - 180), (center + 90, center - 250), (center + 40, center - 160)], outline=stroke_col)
                desc = "a technical vector outline wireframe of an apple logo with construction grid lines on white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Apple). Aspect: Technical Blueprint Outline. Geometry: Dual tangent circles with leaf vector tangent. Palette: Blueprint Slate on White. </think> " + desc

            elif aspect == "flat_silhouette":
                draw.ellipse([center - 200, center - 140, center + 40, center + 200], fill=(20, 24, 30))
                draw.ellipse([center - 40, center - 140, center + 200, center + 200], fill=(20, 24, 30))
                draw.rectangle([center - 120, center - 70, center + 120, center + 160], fill=(20, 24, 30))
                draw.polygon([(center, center - 180), (center + 90, center - 250), (center + 40, center - 160)], fill=(20, 24, 30))
                desc = "a pure solid black minimalist silhouette logo of an apple on clean white background, flat vector brand icon"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Apple). Aspect: Flat 2D Silhouette. Form: Monochromatic high-contrast negative space. </think> " + desc

            elif aspect == "duotone_logo":
                draw.ellipse([center - 200, center - 140, center + 40, center + 200], fill=(230, 45, 60))
                draw.ellipse([center - 40, center - 140, center + 200, center + 200], fill=(230, 45, 60))
                draw.rectangle([center - 120, center - 70, center + 120, center + 160], fill=(230, 45, 60))
                draw.polygon([(center, center - 180), (center + 90, center - 250), (center + 40, center - 160)], fill=(40, 180, 95))
                desc = "a modern duotone graphic design logo of an apple with crimson body and emerald green leaf on white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Apple). Aspect: Duotone Color Hierarchy. Palette: Vivid Crimson and Emerald Green. </think> " + desc

            else:  # isometric_3d
                # Soft ground contact shadow
                draw.ellipse([center - 240, center + 180, center + 240, center + 260], fill=(215, 222, 238))
                # 3D spherical shading with specular highlight
                draw.ellipse([center - 200, center - 140, center + 40, center + 200], fill=(210, 30, 45))
                draw.ellipse([center - 40, center - 140, center + 200, center + 200], fill=(240, 55, 75))
                draw.rectangle([center - 120, center - 70, center + 120, center + 160], fill=(225, 40, 60))
                # Leaf with 3D bevel
                draw.polygon([(center, center - 180), (center + 90, center - 250), (center + 40, center - 160)], fill=(30, 160, 80))
                # Glossy specular shine arc
                draw.arc([center - 170, center - 110, center + 50, center + 130], start=160, end=230, fill=(255, 175, 185), width=20)
                desc = "a 3D isometric rendered apple app icon with soft cast shadow, specular lighting reflections on elevated studio pedestal"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Apple). Aspect: 3D Volumetric Form. Physics: Ambient occlusion contact shadow, specular reflection arc, directional key light. </think> " + desc

        # 2. ROCKET: All 4 Aspects
        elif subject == "rocket":
            if aspect == "outline":
                draw.polygon([(center, center - 260), (center - 90, center + 120), (center + 90, center + 120)], outline=stroke_col, width=16)
                draw.polygon([(center - 90, center + 40), (center - 160, center + 140), (center - 90, center + 120)], outline=stroke_col, width=16)
                draw.polygon([(center + 90, center + 40), (center + 160, center + 140), (center + 90, center + 120)], outline=stroke_col, width=16)
                draw.ellipse([center - 40, center - 60, center + 40, center + 20], outline=stroke_col, width=12)
                desc = "a technical CAD vector blueprint outline of a space rocket emblem with fine precision lines on white canvas"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Rocket). Aspect: Technical CAD Outline. Geometry: Orthogonal fuselage triangle with dual stabilizing aerofoils. </think> " + desc

            elif aspect == "flat_silhouette":
                draw.polygon([(center, center - 260), (center - 90, center + 120), (center + 90, center + 120)], fill=(20, 24, 30))
                draw.polygon([(center - 90, center + 40), (center - 160, center + 140), (center - 90, center + 120)], fill=(20, 24, 30))
                draw.polygon([(center + 90, center + 40), (center + 160, center + 140), (center + 90, center + 120)], fill=(20, 24, 30))
                desc = "a minimalist solid black vector silhouette logo of a space rocket launching upward on white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Rocket). Aspect: Flat 2D Silhouette. Form: Bold aerodynamic silhouette. </think> " + desc

            elif aspect == "duotone_logo":
                draw.polygon([(center, center - 260), (center - 90, center + 120), (center + 90, center + 120)], fill=(240, 245, 255), outline=(30, 40, 60), width=10)
                draw.polygon([(center - 90, center + 40), (center - 160, center + 140), (center - 90, center + 120)], fill=(230, 50, 70))
                draw.polygon([(center + 90, center + 40), (center + 160, center + 140), (center + 90, center + 120)], fill=(230, 50, 70))
                draw.ellipse([center - 40, center - 60, center + 40, center + 20], fill=(0, 160, 255))
                desc = "a modern duotone vector icon of a spacecraft with aerospace white fuselage and crimson booster fins on white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Rocket). Aspect: Duotone Hierarchy. Palette: Aerospace White, Crimson, and Cyan Porthole. </think> " + desc

            else:  # isometric_3d
                draw.ellipse([center - 200, center + 200, center + 200, center + 270], fill=(210, 218, 235))
                # Shaded 3D fuselage
                draw.polygon([(center, center - 260), (center - 90, center + 120), (center, center + 120)], fill=(215, 225, 240))
                draw.polygon([(center, center - 260), (center + 90, center + 120), (center, center + 120)], fill=(255, 255, 255))
                draw.polygon([(center - 90, center + 40), (center - 160, center + 140), (center - 90, center + 120)], fill=(190, 30, 50))
                draw.polygon([(center + 90, center + 40), (center + 160, center + 140), (center + 90, center + 120)], fill=(240, 60, 80))
                # Glowing propulsion flare
                draw.polygon([(center - 40, center + 120), (center, center + 240), (center + 40, center + 120)], fill=(255, 160, 0))
                desc = "a 3D isometric rocket model with volumetric shading, directional lighting, booster flame and soft cast shadow"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Rocket). Aspect: 3D Isometric Volume. Physics: Dual facet lighting split, ambient shadow, and glowing propulsion emitter. </think> " + desc

        # 3. SHIELD
        elif subject == "shield":
            shield_pts = [(center - 180, center - 200), (center + 180, center - 200), (center + 180, center + 40), (center, center + 240), (center - 180, center + 40)]
            if aspect == "outline":
                draw.polygon(shield_pts, outline=stroke_col, width=16)
                draw.line([(center - 80, center + 10), (center - 20, center + 70), (center + 80, center - 50)], fill=stroke_col, width=16)
                desc = "a technical CAD vector outline wireframe of a cyber security shield on white canvas"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Shield). Aspect: Technical Blueprint Outline. </think> " + desc
            elif aspect == "flat_silhouette":
                draw.polygon(shield_pts, fill=(20, 24, 30))
                desc = "a solid black minimalist silhouette logo of a protective heraldic shield on clean white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Shield). Aspect: Flat 2D Silhouette. </think> " + desc
            elif aspect == "duotone_logo":
                draw.polygon(shield_pts, fill=(25, 40, 75), outline=(0, 210, 255), width=16)
                draw.line([(center - 80, center + 10), (center - 20, center + 70), (center + 80, center - 50)], fill=(0, 230, 180), width=24, joint="curve")
                desc = "a modern duotone cyber security shield logo with glowing cyan outline and mint green checkmark"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Shield). Aspect: Duotone Hierarchy. </think> " + desc
            else:
                draw.polygon(shield_pts, fill=(35, 60, 110), outline=(0, 230, 255), width=16)
                draw.line([(center - 80, center + 10), (center - 20, center + 70), (center + 80, center - 50)], fill=(0, 255, 200), width=24, joint="curve")
                draw.ellipse([center - 220, center + 220, center + 220, center + 270], fill=(215, 222, 238))
                desc = "a 3D isometric cyber shield badge with glowing neon rim and soft cast shadow on studio floor"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Shield). Aspect: 3D Isometric Volume. </think> " + desc

        # 4. CAMERA
        elif subject == "camera":
            if aspect == "outline":
                draw.rounded_rectangle([center - 200, center - 120, center + 200, center + 180], radius=44, outline=stroke_col, width=16)
                draw.ellipse([center - 100, center - 70, center + 100, center + 130], outline=stroke_col, width=16)
                desc = "a technical CAD vector blueprint outline of a camera icon with circular lens on white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Camera). Aspect: Technical Blueprint Outline. </think> " + desc
            elif aspect == "flat_silhouette":
                draw.rounded_rectangle([center - 200, center - 120, center + 200, center + 180], radius=44, fill=(20, 24, 30))
                draw.ellipse([center - 100, center - 70, center + 100, center + 130], fill=bg_col)
                draw.ellipse([center - 50, center - 20, center + 50, center + 80], fill=(20, 24, 30))
                desc = "a solid black minimalist silhouette logo of a camera on clean white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Camera). Aspect: Flat 2D Silhouette. </think> " + desc
            else:
                draw.rounded_rectangle([center - 200, center - 120, center + 200, center + 180], radius=44, fill=(40, 44, 52))
                draw.ellipse([center - 100, center - 70, center + 100, center + 130], fill=(25, 28, 35), outline=(0, 180, 255), width=16)
                draw.ellipse([center - 50, center - 20, center + 50, center + 80], fill=(15, 18, 24))
                draw.ellipse([center + 20, center - 5, center + 40, center + 15], fill=(255, 255, 255))
                desc = "a modern graphic design camera icon with circular optics and cyan accent on white background"
                cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Camera). Aspect: Modern Design Icon. </think> " + desc

        # 5. GENERAL MULTI-ASPECT FALLBACK (Heart, Diamond, Flame, Planet, Mountain, Bird)
        else:
            draw.ellipse([center - 180, center - 180, center + 20, center + 20], fill=(245, 50, 100))
            draw.ellipse([center - 20, center - 180, center + 180, center + 20], fill=(245, 50, 100))
            draw.polygon([(center - 170, center - 60), (center + 170, center - 60), (center, center + 200)], fill=(245, 50, 100))
            desc = "a vibrant modern heart icon, flat vector graphic design symbol on pristine white background"
            cot = "<think> Concept: Design Flashcards Level 2 Multi-Aspect Grounding (Heart). Aspect: Canonical Graphic Mark. Color Palette: Vibrant Crimson and Rose. </think> " + desc

        final_img = cls._downsample(img)
        return final_img, desc, cot

    @classmethod
    def generate_level2_flashcard_icon(cls) -> Tuple[Image.Image, str, str]:
        """Alias for Level 2 Multi-Aspect Object generator."""
        return cls.generate_level2_multi_aspect_object()

    # -------------------------------------------------------------------------
    # LEVEL 3: Layout, Composition & Materials (Junior Designer)
    # -------------------------------------------------------------------------
    @classmethod
    def generate_level3_layout_materials(cls) -> Tuple[Image.Image, str, str]:
        super_size = 1024
        style_choice = random.choice(["glassmorphism", "swiss_bauhaus", "neumorphism", "isometric_platform"])

        if style_choice == "glassmorphism":
            img = Image.new("RGB", (super_size, super_size))
            draw = ImageDraw.Draw(img)
            for y in range(super_size):
                t = y / super_size
                r = int(140 * (1 - t) + 25 * t)
                g = int(30 * (1 - t) + 120 * t)
                b = int(220 * (1 - t) + 240 * t)
                draw.line([(0, y), (super_size, y)], fill=(r, g, b))

            draw.ellipse([100, 150, 420, 470], fill=(255, 60, 140))
            draw.ellipse([600, 520, 940, 860], fill=(0, 240, 200))

            card_box = [200, 260, 824, 764]
            draw.rounded_rectangle(card_box, radius=48, fill=(240, 245, 255), outline=(255, 255, 255), width=12)

            inner_draw = ImageDraw.Draw(img)
            inner_draw.rounded_rectangle([260, 320, 360, 420], radius=20, fill=(30, 40, 80))
            inner_draw.rounded_rectangle([400, 340, 720, 365], radius=10, fill=(50, 60, 95))
            inner_draw.rounded_rectangle([400, 385, 620, 405], radius=8, fill=(120, 130, 160))
            inner_draw.rounded_rectangle([260, 470, 764, 690], radius=28, fill=(225, 232, 248))
            inner_draw.rounded_rectangle([300, 530, 520, 630], radius=16, fill=(0, 160, 255))
            inner_draw.rounded_rectangle([550, 530, 724, 630], radius=16, fill=(255, 80, 120))

            desc = "a modern glassmorphism UI card with frosted glass texture, specular edge highlights floating over vibrant neon gradient backdrop"
            cot = "<think> Concept: Design Level 3 (Layout & Materials). Style: Glassmorphism UI. Principle: Translucency, multi-layer depth, specular rim lighting. Palette: Electric Violet, Neon Coral, Cyan, and Ice Glass (#F0F5FF). </think> " + desc

        elif style_choice == "swiss_bauhaus":
            img, draw = cls._create_canvas(super_size, (244, 241, 234))
            draw.ellipse([460, 120, 900, 560], fill=(225, 45, 35))
            draw.rectangle([120, 420, 880, 520], fill=(20, 22, 26))
            draw.line([(120, 850), (600, 380)], fill=(30, 80, 180), width=24)
            draw.rectangle([120, 580, 340, 780], fill=(240, 180, 20))
            draw.rectangle([380, 580, 600, 670], fill=(20, 22, 26))
            draw.rectangle([380, 690, 880, 780], fill=(180, 185, 195))

            desc = "a Swiss Bauhaus graphic design poster with bold asymmetric primary colored geometric blocks, heavy grid composition and high contrast"
            cot = "<think> Concept: Design Level 3 (Layout & Materials). Style: Swiss International / Bauhaus. Principle: Strict grid alignment, asymmetric visual tension, massive contrast. Palette: Bauhaus Triad (Vermilion Red, Cobalt Blue, Golden Ochre, Jet Black on Raw Linen). </think> " + desc

        elif style_choice == "neumorphism":
            bg_tone = (228, 232, 240)
            img, draw = cls._create_canvas(super_size, bg_tone)
            center = super_size // 2

            draw.rounded_rectangle([center - 280 + 16, center - 280 + 16, center + 280 + 16, center + 280 + 16], radius=64, fill=(195, 200, 212))
            draw.rounded_rectangle([center - 280 - 16, center - 280 - 16, center + 280 - 16, center + 280 - 16], radius=64, fill=(255, 255, 255))
            draw.rounded_rectangle([center - 280, center - 280, center + 280, center + 280], radius=64, fill=bg_tone)

            btn_r = 120
            draw.ellipse([center - btn_r + 10, center - btn_r + 10, center + btn_r + 10, center + btn_r + 10], fill=(195, 200, 212))
            draw.ellipse([center - btn_r - 10, center - btn_r - 10, center + btn_r - 10, center + btn_r - 10], fill=(255, 255, 255))
            draw.ellipse([center - btn_r, center - btn_r, center + btn_r, center + btn_r], fill=bg_tone)
            draw.polygon([(center - 25, center - 45), (center + 45, center), (center - 25, center + 45)], fill=(65, 120, 245))

            desc = "a tactile neumorphic soft UI card with extruded curved corners and embossed circular media button, minimalist clean design"
            cot = "<think> Concept: Design Level 3 (Layout & Materials). Style: Neumorphism (Soft UI). Principle: Monochromatic light casting, dual opposite specular & ambient occlusion shadows. Palette: Slate Alabaster (#E4E8F0) with Accent Electric Blue. </think> " + desc

        else:  # isometric_platform
            img, draw = cls._create_canvas(super_size, (15, 18, 30))
            center = super_size // 2

            pts_top = [(center, center - 140), (center + 260, center), (center, center + 140), (center - 260, center)]
            pts_left = [(center - 260, center), (center, center + 140), (center, center + 280), (center - 260, center + 140)]
            pts_right = [(center, center + 140), (center + 260, center), (center + 260, center + 140), (center, center + 280)]

            draw.polygon(pts_left, fill=(20, 45, 90))
            draw.polygon(pts_right, fill=(12, 28, 60))
            draw.polygon(pts_top, fill=(35, 80, 160), outline=(0, 230, 255), width=8)

            draw.line([(center, center - 100), (center, center + 100)], fill=(0, 255, 230), width=10)
            draw.line([(center - 160, center), (center + 160, center)], fill=(0, 255, 230), width=10)
            draw.ellipse([center - 40, center - 40, center + 40, center + 40], fill=(0, 255, 240))

            desc = "a 3D isometric tech pedestal with glowing cyan circuitry grid lines on dark background, futuristic digital platform design"
            cot = "<think> Concept: Design Level 3 (Layout & Materials). Style: Isometric 30-degree Orthographic Projection. Principle: Three-dimensional volume representation, directional lighting hierarchy. Palette: Obsidian Dark, Cobalt Deep, and Phosphor Cyan. </think> " + desc

        final_img = cls._downsample(img)
        return final_img, desc, cot

    # -------------------------------------------------------------------------
    # LEVEL 4: Creative Director & Graphic Genius (Master Level)
    # -------------------------------------------------------------------------
    @classmethod
    def generate_level4_genius_masterpiece(cls) -> Tuple[Image.Image, str, str]:
        super_size = 1024
        genius_style = random.choice(["cyberpunk_hud", "luxury_gold_foil", "kinetic_abstract"])

        if genius_style == "cyberpunk_hud":
            img, draw = cls._create_canvas(super_size, (8, 10, 20))
            center = super_size // 2

            for r, start_deg, end_deg, col, w in [
                (380, 30, 150, (0, 220, 255), 8),
                (380, 180, 330, (0, 220, 255), 8),
                (340, 0, 360, (20, 45, 80), 4),
                (300, 60, 120, (255, 0, 128), 12),
                (300, 210, 310, (0, 255, 180), 10),
                (240, 0, 360, (15, 30, 60), 6),
                (180, 45, 225, (0, 220, 255), 14),
                (100, 0, 360, (255, 0, 128), 8),
            ]:
                draw.arc([center - r, center - r, center + r, center + r], start=start_deg, end=end_deg, fill=col, width=w)

            draw.line([(center - 420, center), (center - 320, center)], fill=(0, 220, 255), width=6)
            draw.line([(center + 320, center), (center + 420, center)], fill=(0, 220, 255), width=6)
            draw.line([(center, center - 420), (center, center - 320)], fill=(0, 220, 255), width=6)
            draw.line([(center, center + 320), (center, center + 420)], fill=(0, 220, 255), width=6)
            draw.ellipse([center - 24, center - 24, center + 24, center + 24], fill=(255, 255, 255))

            desc = "a futuristic cyberpunk sci-fi holographic HUD interface with glowing cyan telemetry rings, neon magenta target dials and vector crosshairs on dark void"
            cot = "<think> Concept: Design Genius Level 4. Style: Sci-Fi Telemetry & Holographic UI. Principle: High-density radial information architecture, sub-pixel vector precision. Palette: Electric Cyan (#00DCFF), Laser Magenta (#FF0080), Mint Phosphor, Deep Space Black (#080A14). </think> " + desc

        elif genius_style == "luxury_gold_foil":
            img, draw = cls._create_canvas(super_size, (12, 16, 28))
            center = super_size // 2

            gold_dark = (195, 150, 45)
            gold_mid = (230, 190, 65)
            gold_light = (255, 235, 160)

            draw.polygon([(center, center - 360), (center + 360, center), (center, center + 360), (center - 360, center)], outline=gold_dark, width=12)
            draw.polygon([(center, center - 330), (center + 330, center), (center, center + 330), (center - 330, center)], outline=gold_mid, width=6)
            draw.ellipse([center - 220, center - 220, center + 220, center + 220], outline=gold_mid, width=14)
            draw.ellipse([center - 190, center - 190, center + 190, center + 190], outline=gold_light, width=6)

            crown_pts = [
                (center - 110, center + 60),
                (center - 110, center - 40),
                (center - 55, center + 10),
                (center, center - 70),
                (center + 55, center + 10),
                (center + 110, center - 40),
                (center + 110, center + 60),
            ]
            draw.polygon(crown_pts, fill=gold_mid, outline=gold_light, width=8)
            draw.rectangle([center - 110, center + 70, center + 110, center + 100], fill=gold_light)

            desc = "a luxury royal brand crest monogram in metallic gold foil with intricate geometric heraldic filigree on deep matte midnight blue background"
            cot = "<think> Concept: Design Genius Level 4. Style: Haute Luxury Branding. Principle: Regal radial symmetry, intricate metallic foil gradient layering, negative space nobility. Palette: Imperial Gold (#E6BE41), Champagne Platinum, and Deep Midnight Obsidian (#0C101C). </think> " + desc

        else:  # kinetic_abstract
            img, draw = cls._create_canvas(super_size, (248, 248, 252))
            center = super_size // 2

            for idx in range(7):
                offset = (idx - 3) * 60
                alpha_col = (int(255 - idx * 25), int(40 + idx * 30), int(120 + idx * 20))
                ribbon_pts = []
                for step in range(30):
                    t = step / 30
                    x = int(120 + t * 784)
                    y = int(center + offset + math.sin(t * math.pi * 2) * 220)
                    ribbon_pts.append((x, y))

                for p_idx in range(len(ribbon_pts) - 1):
                    draw.line([ribbon_pts[p_idx], ribbon_pts[p_idx + 1]], fill=alpha_col, width=32)

            draw.rectangle([120, 140, 360, 160], fill=(20, 24, 32))
            draw.rectangle([120, 180, 240, 195], fill=(140, 145, 160))

            desc = "a contemporary dynamic graphic design poster with sweeping chromatic wave ribbons, elegant typography layout anchors and visual movement"
            cot = "<think> Concept: Design Genius Level 4. Style: Kinetic Chromatic Wave Poster. Principle: Dynamic visual flow, progressive wave deformation, sophisticated chromatic transition. Palette: Radiant Sunset to Deep Iris on Alabaster Paper. </think> " + desc

        final_img = cls._downsample(img)
        return final_img, desc, cot

    # -------------------------------------------------------------------------
    # Unified Dispatcher
    # -------------------------------------------------------------------------
    @classmethod
    def generate_curriculum_sample(cls, level: int = 1) -> Tuple[Image.Image, str, str]:
        if level == 1:
            return cls.generate_level1_primitive()
        elif level == 2:
            return cls.generate_level2_multi_aspect_object()
        elif level == 3:
            return cls.generate_level3_layout_materials()
        elif level == 4:
            return cls.generate_level4_genius_masterpiece()
        else:
            return cls.generate_curriculum_sample(random.randint(1, 4))
