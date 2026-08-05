"""
Build the MOAD wordmark logo by extracting M, O, A, D outlines from Roboto-Bold,
applying the font's GPOS kerning + small custom corrections, then anchoring the
hat / dot / brackets to each glyph's actual rendered ink geometry.

One-off generator.  Run from the repo root:

    python3 dev_scripts/build_logo.py

Writes:
    assets/logo.svg
    docs/src/assets/logo.svg
"""
from __future__ import annotations

import os
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

# ---------------------------------------------------------------------------
# Configuration

# Roboto-Bold (Apache-2.0; see THIRD_PARTY_NOTICES.md). Override with
# MOAD_ROBOTO_BOLD, else try the usual install locations.
_FONT_CANDIDATES = (
    "/Library/Fonts/Roboto-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
    os.path.expanduser("~/Library/Fonts/Roboto-Bold.ttf"),
)
FONT_PATH = os.environ.get("MOAD_ROBOTO_BOLD") or next(
    (p for p in _FONT_CANDIDATES if os.path.isfile(p)), _FONT_CANDIDATES[0]
)
WORD = "MOAD"
INK_DARK = "#1d1d2c"
INK_ACCENT = "#9558B2"

# Vertical layout (px in user-space).  Chosen so the hat fits comfortably
# between the bracket top and the cap line.
TOP_PAD = 15           # viewBox top to bracket top
HAT_BRACKET_GAP = 8    # bracket top to hat peak
HAT_RISE = 11          # hat rise (peak to ends)
HAT_LETTER_GAP = 10    # hat ends to letter cap line
FONT_SIZE_PX = 128     # cap height target
BASELINE_GAP = 10      # baseline to bracket bottom
BOTTOM_PAD = 15        # bracket bottom to viewBox bottom

# Horizontal layout
SIDE_PAD = 30          # viewBox edge to bracket vertical bar
BRACKET_WORDMARK_PAD = 32  # bracket vertical bar to nearest letter ink
BRACKET_SERIF = 25     # length of horizontal serif on bracket
BRACKET_STROKE = 8

# Decoration sizing (relative to glyph ink width)
HAT_REL_HALFWIDTH = 0.30   # half-width as fraction of O ink width
DOT_REL_RADIUS = 0.085     # radius as fraction of D ink width
D_OPTICAL_OFFSET_FRAC = 0.05  # dot shift left of D's geometric centre

# Custom kerning corrections on top of font's GPOS kerning, in em/1000.
# Roboto-Bold's GPOS already does strong OA/AD tightening; leave it alone unless
# we see imbalance after path-rendering.
EXTRA_KERN_EM: dict[tuple[str, str], int] = {}

# Derived layout
BRACKET_TOP = TOP_PAD
HAT_PEAK_Y = BRACKET_TOP + HAT_BRACKET_GAP
HAT_END_Y = HAT_PEAK_Y + HAT_RISE
CAP_TOP_Y = HAT_END_Y + HAT_LETTER_GAP
BASELINE_Y = CAP_TOP_Y + FONT_SIZE_PX
BRACKET_BOT = BASELINE_Y + BASELINE_GAP
VIEWBOX_H = BRACKET_BOT + BOTTOM_PAD


# ---------------------------------------------------------------------------
# Load font

font = TTFont(FONT_PATH)
units_per_em = font["head"].unitsPerEm
glyph_set = font.getGlyphSet()
cmap = font.getBestCmap()
hmtx = font["hmtx"]
glyf = font["glyf"]
os2 = font["OS/2"]
cap_height_units = getattr(os2, "sCapHeight", None) or int(units_per_em * 0.7)
scale = FONT_SIZE_PX / cap_height_units


def glyph_name(ch: str) -> str:
    return cmap[ord(ch)]


def glyph_advance_units(name: str) -> int:
    return hmtx.metrics[name][0]


def glyph_ink_bounds_units(name: str) -> tuple[float, float]:
    """Return (xmin, xmax) of the glyph's actual ink in font units, computed via
    BoundsPen so that off-curve control points don't inflate the bbox."""
    pen = BoundsPen(glyph_set)
    glyph_set[name].draw(pen)
    if pen.bounds is None:  # empty glyph
        return 0.0, float(glyph_advance_units(name))
    xmin, _ymin, xmax, _ymax = pen.bounds
    return float(xmin), float(xmax)


# GPOS kerning extraction
def gpos_kerning(font: TTFont) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if "GPOS" not in font:
        return out
    gpos = font["GPOS"].table
    for lookup in gpos.LookupList.Lookup:
        if lookup.LookupType != 2:
            continue
        for st in lookup.SubTable:
            if st.Format == 1:
                for first_glyph, pair_set in zip(st.Coverage.glyphs, st.PairSet):
                    for pair in pair_set.PairValueRecord:
                        adj = getattr(pair.Value1, "XAdvance", 0) or 0
                        if adj:
                            out[(first_glyph, pair.SecondGlyph)] = adj
            elif st.Format == 2:
                cov = set(st.Coverage.glyphs)
                cls1 = st.ClassDef1.classDefs
                cls2 = st.ClassDef2.classDefs
                for g1 in cov:
                    c1 = cls1.get(g1, 0)
                    for g2, c2 in cls2.items():
                        rec = st.Class1Record[c1].Class2Record[c2]
                        adj = getattr(rec.Value1, "XAdvance", 0) or 0
                        if adj:
                            out[(g1, g2)] = adj
    return out


kerning = gpos_kerning(font)


def extract_glyph_path(name: str, x_offset_px: float) -> str:
    """Return SVG path data for the glyph, scaled and positioned in user space.

    Uses fontTools' TransformPen, which is the canonical way: every drawing
    operation gets the affine transform applied as it's emitted, so the
    resulting path string is already in final coordinates with no further
    manipulation.
    """
    out_pen = SVGPathPen(glyph_set)
    # Affine matrix (xx, xy, yx, yy, dx, dy): scale x, flip+scale y, translate.
    affine = (scale, 0.0, 0.0, -scale, x_offset_px, BASELINE_Y)
    transform_pen = TransformPen(out_pen, affine)
    glyph_set[name].draw(transform_pen)
    return out_pen.getCommands()


# ---------------------------------------------------------------------------
# Lay out the wordmark

names = [glyph_name(c) for c in WORD]
advances_u = [glyph_advance_units(n) for n in names]
bounds_u = [glyph_ink_bounds_units(n) for n in names]
font_kern_u = [kerning.get((names[i], names[i + 1]), 0) for i in range(len(names) - 1)]

# Per-letter advance origin in font units, accumulating advances + kerning.
origins_u = [0]
for i in range(1, len(names)):
    extra_em = EXTRA_KERN_EM.get((WORD[i - 1], WORD[i]), 0)
    extra_units = extra_em * units_per_em / 1000
    origins_u.append(
        origins_u[-1] + advances_u[i - 1] + font_kern_u[i - 1] + extra_units
    )

# Anchor the wordmark so the leftmost ink sits exactly BRACKET_WORDMARK_PAD
# from the [ vertical bar, with [ at SIDE_PAD + BRACKET_STROKE/2.
LBRACKET_X = SIDE_PAD + BRACKET_STROKE / 2
WORDMARK_LEFT_INK_X = LBRACKET_X + BRACKET_WORDMARK_PAD
# First letter origin = leftmost ink x − scale*ink_xmin
M_ORIGIN_PX = WORDMARK_LEFT_INK_X - scale * bounds_u[0][0]


def origin_px(i: int) -> float:
    return M_ORIGIN_PX + scale * origins_u[i]


def ink_left_px(i: int) -> float:
    return origin_px(i) + scale * bounds_u[i][0]


def ink_right_px(i: int) -> float:
    return origin_px(i) + scale * bounds_u[i][1]


# Right bracket: vertical bar BRACKET_WORDMARK_PAD beyond rightmost ink.
WORDMARK_RIGHT_INK_X = ink_right_px(len(names) - 1)
RBRACKET_X = WORDMARK_RIGHT_INK_X + BRACKET_WORDMARK_PAD
VIEWBOX_W = RBRACKET_X + BRACKET_STROKE / 2 + SIDE_PAD


# Decoration anchors
o_idx = WORD.index("O")
d_idx = WORD.index("D")

o_centre_px = (ink_left_px(o_idx) + ink_right_px(o_idx)) / 2
o_width_px = ink_right_px(o_idx) - ink_left_px(o_idx)

d_centre_px = (ink_left_px(d_idx) + ink_right_px(d_idx)) / 2
d_width_px = ink_right_px(d_idx) - ink_left_px(d_idx)
d_optical_centre = d_centre_px - D_OPTICAL_OFFSET_FRAC * d_width_px

HAT_HALF_W = HAT_REL_HALFWIDTH * o_width_px
DOT_R = DOT_REL_RADIUS * d_width_px
DOT_CY = (HAT_PEAK_Y + HAT_END_Y) / 2

hat_d = (
    f"M {o_centre_px - HAT_HALF_W:.2f} {HAT_END_Y:.2f} "
    f"L {o_centre_px:.2f} {HAT_PEAK_Y:.2f} "
    f"L {o_centre_px + HAT_HALF_W:.2f} {HAT_END_Y:.2f}"
)


def bracket_path(x: float, side: str) -> str:
    if side == "L":
        return (
            f"M {x + BRACKET_SERIF:.2f} {BRACKET_TOP} "
            f"L {x:.2f} {BRACKET_TOP} "
            f"L {x:.2f} {BRACKET_BOT} "
            f"L {x + BRACKET_SERIF:.2f} {BRACKET_BOT}"
        )
    return (
        f"M {x - BRACKET_SERIF:.2f} {BRACKET_TOP} "
        f"L {x:.2f} {BRACKET_TOP} "
        f"L {x:.2f} {BRACKET_BOT} "
        f"L {x - BRACKET_SERIF:.2f} {BRACKET_BOT}"
    )


# ---------------------------------------------------------------------------
# Emit SVG

letter_paths = [extract_glyph_path(names[i], origin_px(i)) for i in range(len(names))]

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX_W:.0f} {VIEWBOX_H:.0f}" width="{VIEWBOX_W:.0f}" height="{VIEWBOX_H:.0f}">
  <!-- MOAD wordmark: [MÔAḊ] — commutator brackets (Algebra) wrap operator/dynamics-typed letters.
       (c) Yi Lu, MIT-licensed with the rest of MOAD.jl.  Built by dev_scripts/build_logo.py;
       wordmark set in Roboto Bold (credit; see THIRD_PARTY_NOTICES.md).  Uses the font's
       GPOS kerning for letter spacing; hat / dot / brackets are anchored to each glyph's actual
       ink bounding box (computed via fontTools BoundsPen) so positions are deterministic. -->
  <g fill="{INK_DARK}" fill-rule="evenodd">
    <path d="{letter_paths[0]}"/>
    <path d="{letter_paths[1]}"/>
    <path d="{letter_paths[2]}"/>
    <path d="{letter_paths[3]}"/>
  </g>

  <!-- circumflex over O -->
  <path d="{hat_d}"
        stroke="{INK_ACCENT}" stroke-width="7" fill="none"
        stroke-linecap="square" stroke-linejoin="miter"/>

  <!-- overdot over D, optically left-shifted -->
  <circle cx="{d_optical_centre:.2f}" cy="{DOT_CY:.2f}" r="{DOT_R:.2f}" fill="{INK_ACCENT}"/>

  <!-- [ -->
  <path d="{bracket_path(LBRACKET_X, 'L')}"
        stroke="{INK_ACCENT}" stroke-width="{BRACKET_STROKE}" fill="none"
        stroke-linecap="square" stroke-linejoin="miter"/>

  <!-- ] -->
  <path d="{bracket_path(RBRACKET_X, 'R')}"
        stroke="{INK_ACCENT}" stroke-width="{BRACKET_STROKE}" fill="none"
        stroke-linecap="square" stroke-linejoin="miter"/>
</svg>
"""

repo_root = Path(__file__).resolve().parent.parent
for p in [
    repo_root / "assets" / "logo.svg",
    repo_root / "docs" / "src" / "assets" / "logo.svg",
]:
    p.write_text(svg)
    print(f"wrote {p}")

print("---")
print(f"viewBox: {VIEWBOX_W:.0f} x {VIEWBOX_H:.0f}")
print(f"cap_top={CAP_TOP_Y}, baseline={BASELINE_Y}, bracket_top={BRACKET_TOP}, bracket_bot={BRACKET_BOT}")
print(f"hat: peak_y={HAT_PEAK_Y}, end_y={HAT_END_Y}, half_w={HAT_HALF_W:.2f}")
print(f"O centre={o_centre_px:.2f}, D optical centre={d_optical_centre:.2f}, dot_r={DOT_R:.2f}")
print(f"L bracket x={LBRACKET_X}, R bracket x={RBRACKET_X:.2f}")
for i, ch in enumerate(WORD):
    print(f"  {ch}: origin={origin_px(i):.2f} ink=[{ink_left_px(i):.2f}, {ink_right_px(i):.2f}]")
