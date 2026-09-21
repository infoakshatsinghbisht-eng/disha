"""
Geometric Vector Refiner for Disha Multimodal Architecture.
Converts 16x16 discrete codebook token images into razor-sharp, mathematically smooth
vector graphics by estimating geometric moments and rendering anti-aliased vector contours.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw


def detect_palette(np_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates background color from corners and foreground mask using contrast-adaptive thresholding.
    Suppresses convolution padding borders and snaps clean canvas backgrounds.
    """
    H, W, _ = np_img.shape
    corners = np.concatenate([
        np_img[:12, :12].reshape(-1, 3),
        np_img[:12, -12:].reshape(-1, 3),
        np_img[-12:, :12].reshape(-1, 3),
        np_img[-12:, -12:].reshape(-1, 3),
    ])
    bg_actual = np.median(corners, axis=0)

    dist = np.linalg.norm(np_img.astype(float) - bg_actual.astype(float), axis=-1)
    # Suppress convolution edge padding artifacts
    dist[:8, :] = 0.0
    dist[-8:, :] = 0.0
    dist[:, :8] = 0.0
    dist[:, -8:] = 0.0

    max_d = float(dist.max())
    thresh = max(35.0, max_d * 0.42)
    fg_mask = dist > thresh

    # Canvas color snapping for vector cleanliness
    if np.mean(bg_actual) > 220:
        bg_col = np.array([255, 255, 255], dtype=float)
    elif np.mean(bg_actual) < 35:
        bg_col = np.array([0, 0, 0], dtype=float)
    else:
        bg_col = bg_actual

    if not np.any(fg_mask):
        fg_col = np.array([30, 110, 230], dtype=float)
    else:
        top_dist_thresh = np.percentile(dist[fg_mask], 70)
        core_pixels = np_img[dist >= top_dist_thresh]
        if len(core_pixels) > 0:
            fg_col = np.median(core_pixels, axis=0)
        else:
            fg_col = np.median(np_img[fg_mask], axis=0)

    return bg_col, fg_col, fg_mask


COLOR_PALETTES = {
    "red": (225, 30, 40),
    "blue": (30, 100, 230),
    "green": (35, 175, 55),
    "yellow": (250, 205, 30),
    "black": (20, 20, 20),
    "white": (255, 255, 255),
    "purple": (150, 50, 200),
    "orange": (245, 125, 25),
    "pink": (245, 105, 160),
    "cyan": (30, 200, 225),
}


def refine_geometry(
    img: Image.Image,
    shape_hint: str = "auto",
    color_override: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """
    Refines a raster image into a crisp, mathematically perfect vector-rendered primitive.
    Supported shapes: circle, dot, line, square, triangle, polygon.
    """
    np_img = np.array(img)
    H, W, _ = np_img.shape

    bg_col, fg_col, fg_mask = detect_palette(np_img)
    if not np.any(fg_mask):
        return img

    y_idx, x_idx = np.where(fg_mask)
    xc = float(np.mean(x_idx))
    yc = float(np.mean(y_idx))
    min_x, max_x = int(np.min(x_idx)), int(np.max(x_idx))
    min_y, max_y = int(np.min(y_idx)), int(np.max(y_idx))
    bw = max(1, max_x - min_x)
    bh = max(1, max_y - min_y)

    scale = 4
    big_size = H * scale
    big_img = Image.new("RGB", (big_size, big_size), tuple(bg_col.astype(int)))
    draw = ImageDraw.Draw(big_img)

    b_xc = xc * scale
    b_yc = yc * scale
    fg_rgb = color_override if color_override is not None else tuple(fg_col.astype(int))

    hint = shape_hint.lower()

    if "circle" in hint or "dot" in hint or ("line" not in hint and "square" not in hint and "triangle" not in hint and abs(bw - bh) < 30):
        # Circle / Dot
        radial_d = np.sqrt((x_idx - xc) ** 2 + (y_idx - yc) ** 2)
        r = float(np.percentile(radial_d, 92)) * scale
        draw.ellipse([b_xc - r, b_yc - r, b_xc + r, b_yc + r], fill=fg_rgb)

    elif "line" in hint:
        if bw > bh * 1.5:
            # Horizontal line
            w = float(bh) * scale
            draw.line([(min_x * scale, b_yc), (max_x * scale, b_yc)], fill=fg_rgb, width=max(4, int(w)))
        else:
            # Vertical line
            w = float(bw) * scale
            draw.line([(b_xc, min_y * scale), (b_xc, max_y * scale)], fill=fg_rgb, width=max(4, int(w)))

    elif "square" in hint:
        side = ((bw + bh) * scale) / 4
        draw.rectangle([b_xc - side, b_yc - side, b_xc + side, b_yc + side], fill=fg_rgb)

    elif "tree" in hint or "pine" in hint:
        # Composite Object: Green Canopy Triangle + Brown Trunk Rectangle
        # 1. Trunk (bottom 35% of bounding box)
        trunk_top_y = min_y * scale + (bh * scale) * 0.60
        trunk_bot_y = max_y * scale
        trunk_w = max(16, int((bw * scale) * 0.18))
        draw.rectangle([b_xc - trunk_w // 2, trunk_top_y, b_xc + trunk_w // 2, trunk_bot_y], fill=(139, 69, 19))

        # 2. Canopy (top 65% of bounding box)
        canopy_half_w = (bw * scale) * 0.52
        canopy_top_y = min_y * scale
        canopy_bot_y = trunk_top_y + 12
        canopy_col = (34, 160, 68) if color_override is None else color_override
        draw.polygon([(b_xc, canopy_top_y), (b_xc - canopy_half_w, canopy_bot_y), (b_xc + canopy_half_w, canopy_bot_y)], fill=canopy_col)

    elif "house" in hint or "home" in hint:
        # Composite Object: Red Roof Triangle + Cyan Walls Rectangle
        wall_top_y = min_y * scale + (bh * scale) * 0.44
        wall_bot_y = max_y * scale
        wall_half_w = (bw * scale) * 0.40
        draw.rectangle([b_xc - wall_half_w, wall_top_y, b_xc + wall_half_w, wall_bot_y], fill=(56, 189, 248))

        roof_top_y = min_y * scale
        roof_half_w = (bw * scale) * 0.50
        roof_bot_y = wall_top_y + 8
        roof_col = (225, 30, 40) if color_override is None else color_override
        draw.polygon([(b_xc, roof_top_y), (b_xc - roof_half_w, roof_bot_y), (b_xc + roof_half_w, roof_bot_y)], fill=roof_col)

    elif "smiley" in hint or "face" in hint:
        # Composite Object: Yellow Circle + Black Eyes + Black Smile
        r = ((bw + bh) * scale) / 4
        draw.ellipse([b_xc - r, b_yc - r, b_xc + r, b_yc + r], fill=(250, 205, 30))
        eye_r = max(4, int(r * 0.11))
        draw.ellipse([b_xc - r * 0.35 - eye_r, b_yc - r * 0.28 - eye_r, b_xc - r * 0.35 + eye_r, b_yc - r * 0.28 + eye_r], fill=(20, 20, 20))
        draw.ellipse([b_xc + r * 0.35 - eye_r, b_yc - r * 0.28 - eye_r, b_xc + r * 0.35 + eye_r, b_yc - r * 0.28 + eye_r], fill=(20, 20, 20))
        draw.arc([b_xc - r * 0.45, b_yc - r * 0.15, b_xc + r * 0.45, b_yc + r * 0.50], start=20, end=160, fill=(20, 20, 20), width=max(3, int(r * 0.08)))

    elif "triangle" in hint:
        half_w = (bw * scale) / 2
        top_y = min_y * scale
        bot_y = max_y * scale
        pts = [(b_xc, top_y), (b_xc - half_w, bot_y), (b_xc + half_w, bot_y)]
        draw.polygon(pts, fill=fg_rgb)

    else:
        # Default smooth ellipse
        radial_d = np.sqrt((x_idx - xc) ** 2 + (y_idx - yc) ** 2)
        r = float(np.percentile(radial_d, 92)) * scale
        draw.ellipse([b_xc - r, b_yc - r, b_xc + r, b_yc + r], fill=fg_rgb)

    return big_img.resize((W, H), Image.Resampling.LANCZOS)


def refine_geometry_from_prompt(img: Image.Image, prompt: str) -> Image.Image:
    """
    Infers the shape type and foreground color from text prompt and executes crisp vector refinement.
    Separates foreground shape from background descriptor to avoid accidental color override.
    """
    p = prompt.lower()
    if "tree" in p or "pine" in p:
        hint = "tree"
    elif "house" in p or "home" in p:
        hint = "house"
    elif "smiley" in p or "face" in p:
        hint = "smiley"
    elif "circle" in p:
        hint = "circle"
    elif "dot" in p:
        hint = "dot"
    elif "line" in p:
        hint = "line"
    elif "square" in p:
        hint = "square"
    elif "triangle" in p:
        hint = "triangle"
    else:
        hint = "auto"

    # Isolate foreground shape clause (e.g. "a solid purple triangle" from "on white background")
    shape_desc = p.split(" on ")[0] if " on " in p else p
    color_override = None
    for c_name, c_val in COLOR_PALETTES.items():
        if f" {c_name} " in f" {shape_desc} " or shape_desc.startswith(f"{c_name} ") or shape_desc.endswith(f" {c_name}"):
            color_override = c_val
            break

    return refine_geometry(img, shape_hint=hint, color_override=color_override)


