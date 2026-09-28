"""Generate Basketball Clipper logo assets (PNG + ICO) from vector spec.

Run from repo root:
    python tools/generate_logo_assets.py

Outputs to assets/logo/:
    icon_16.png ... icon_512.png   (dark rounded tile)
    logo_256.png, logo_512.png     (transparent symbol, no tile)
    favicon.ico, installer.ico
"""
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "logo"
OUT.mkdir(parents=True, exist_ok=True)

INK = (20, 23, 28, 255)       # #14171C
ORANGE = (255, 107, 44, 255)  # #FF6B2C
TEAL = (39, 210, 191, 255)    # #27D2BF
TRANSPARENT = (0, 0, 0, 0)

# Base canvas for rendering
BASE = 512
RADIUS = 96
BALL_R = 112
BALL_C = BASE // 2
BRACKET_W = 32
SEAM_W = 12


def qbez(p0, p1, p2, steps=64):
    """Quadratic Bezier points."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x, y))
    return pts


def draw_round_line(draw, p1, p2, width, fill):
    """Draw a thick line with round caps."""
    draw.line([p1, p2], fill=fill, width=width)
    r = width // 2
    for p in (p1, p2):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)


def draw_brackets(draw, offset=0):
    """Four corner brackets, teal."""
    w = BRACKET_W
    L = 80  # bracket arm length
    # coordinates relative to ball center 256, bracket inner corner at distance 144 from center
    d = 144
    c = BALL_C
    corners = [
        # (corner_x, corner_y, dx1, dy1, dx2, dy2)
        (c - d, c - d, L, 0, 0, L),      # top-left
        (c + d, c - d, -L, 0, 0, L),     # top-right
        (c - d, c + d, L, 0, 0, -L),     # bottom-left
        (c + d, c + d, -L, 0, 0, -L),    # bottom-right
    ]
    for x, y, dx1, dy1, dx2, dy2 in corners:
        p1 = (x + dx1, y + dy1)
        p2 = (x, y)
        p3 = (x + dx2, y + dy2)
        draw_round_line(draw, p1, p2, w, TEAL)
        draw_round_line(draw, p2, p3, w, TEAL)


def draw_ball(draw):
    """Orange ball with two seam arcs."""
    c = BALL_C
    r = BALL_R
    draw.ellipse([c - r, c - r, c + r, c + r], fill=ORANGE)
    # seams: two crossing quadratic arcs inside the ball
    # upper arc: from left-upper to right-upper, curving down
    p0 = (c - 88, c - 40)
    p1 = (c, c + 40)
    p2 = (c + 88, c - 40)
    pts = qbez(p0, p1, p2)
    draw.line(pts, fill=INK, width=SEAM_W)
    # lower arc: from left-lower to right-lower, curving up
    p0 = (c - 88, c + 40)
    p1 = (c, c - 40)
    p2 = (c + 88, c + 40)
    pts = qbez(p0, p1, p2)
    draw.line(pts, fill=INK, width=SEAM_W)


def render(with_tile=True, size=BASE):
    img = Image.new("RGBA", (BASE, BASE), TRANSPARENT)
    draw = ImageDraw.Draw(img)
    if with_tile:
        draw.rounded_rectangle([0, 0, BASE, BASE], radius=RADIUS, fill=INK)
    draw_brackets(draw)
    draw_ball(draw)
    if size != BASE:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def main():
    # Tile icons
    for s in (16, 32, 48, 64, 128, 256, 512):
        img = render(with_tile=True, size=s)
        img.save(OUT / f"icon_{s}.png")
        print(f"icon_{s}.png")

    # Transparent symbol (no tile)
    for s in (256, 512):
        img = render(with_tile=False, size=s)
        img.save(OUT / f"logo_{s}.png")
        print(f"logo_{s}.png")

    # ICO files (multi-size)
    ico_sizes = [(16, 16), (32, 32), (48, 48), (256, 256)]
    img = render(with_tile=True, size=256)
    img.save(OUT / "favicon.ico", format="ICO", sizes=ico_sizes)
    print("favicon.ico")
    img.save(OUT / "installer.ico", format="ICO", sizes=ico_sizes)
    print("installer.ico")

    print(f"\nAll assets written to {OUT}")


if __name__ == "__main__":
    main()
