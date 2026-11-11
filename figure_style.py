"""Publication figure styling and exports with an unchanged physical canvas.

``fig_width`` and ``fig_height`` in existing projects remain inches. Unit
conversion is only for the controls: store converted values in those legacy
keys, and convert them back when displaying millimetres.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.text import Text


MM_PER_INCH = 25.4
# Liberation Sans and Liberation Serif 2.1.5 (SIL Open Font License 1.1, see fonts/OFL.txt) have
# the same character widths as Arial and Times New Roman and ship with the app: figures never
# depend on fonts installed on the server (Streamlit Cloud has neither Arial nor a substitute,
# so matplotlib used to draw in its own DejaVu Sans there) or on a new PC.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
BUNDLED_FONT_FILES = tuple(f"Liberation{family}-{style}.ttf" for family in ("Sans", "Serif")
                           for style in ("Regular", "Bold", "Italic", "BoldItalic"))
# The same choices as the graph app (figapp), plus Times New Roman kept for earlier projects.
FONT_FAMILIES = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "Times New Roman")
# Installed fonts tried in order, as in the graph app: Arial-compatible fonts first, DejaVu last.
FONT_CANDIDATES = {
    "Arial": ("Arial", "Helvetica", "Liberation Sans", "Arimo", "DejaVu Sans"),
    "Helvetica": ("Helvetica", "Arial", "Liberation Sans", "Arimo", "DejaVu Sans"),
    "Liberation Sans": ("Liberation Sans", "Arimo", "Arial", "Helvetica", "DejaVu Sans"),
    "DejaVu Sans": ("DejaVu Sans",),
    "Times New Roman": ("Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"),
}
# Fonts with the same character widths: text laid out with one fits the other exactly,
# so an editable SVG can name the requested font (Illustrator then opens it in Arial).
METRIC_COMPATIBLE = {
    "Arial": {"Liberation Sans", "Arimo", "Helvetica"},
    "Helvetica": {"Arial", "Liberation Sans", "Arimo"},
    "Times New Roman": {"Liberation Serif", "Times"},
}
# Glyphs missing from the chosen font (Liberation Sans has no △ or ◇) come from DejaVu Sans.
GLYPH_FALLBACK = "DejaVu Sans"
CSS_FONT_STACKS = {
    "Arial": 'Arial, "Liberation Sans", Arimo, Helvetica, sans-serif',
    "Helvetica": 'Helvetica, Arial, "Liberation Sans", Arimo, sans-serif',
    "Liberation Sans": '"Liberation Sans", Arimo, Arial, Helvetica, sans-serif',
    "DejaVu Sans": '"DejaVu Sans", Verdana, sans-serif',
    "Times New Roman": '"Times New Roman", "Liberation Serif", Times, serif',
}
# These are convenient starting sizes, not journal-specific specifications.
FIGURE_PRESETS_MM = {
    "custom": None,
    "single": (85.0, 70.0),
    "double": (180.0, 120.0),
    "figapp": (88.0, 70.0),
}
STYLE_DEFAULTS = {
    "size_unit": "mm",
    "figure_preset": "custom",
    "font_family": "Arial",
    "line_width": 1.2,
    "reference_line_width": 1.0,
    "axes_line_width": 0.8,
    "tick_length": 5.0,
    "tick_direction": "auto",
    "axes_box": True,
    "margin_left": 0.14,
    "margin_right": 0.97,
    "margin_bottom": 0.14,
    "margin_top": 0.96,
    "transparent_background": False,
    "svg_text_editable": True,
}
STYLE_KEYS = frozenset(STYLE_DEFAULTS)
_NUMBER_BOUNDS = {"tick_length": (0.5, 10.0)}
EXPORT_MIME_TYPES = {
    "png": "image/png",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
}


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def mm_to_inches(value: float) -> float:
    return _finite_number(value, "Millimetres") / MM_PER_INCH


def inches_to_mm(value: float) -> float:
    return _finite_number(value, "Inches") * MM_PER_INCH


def _unit(unit: str) -> str:
    if not isinstance(unit, str):
        raise ValueError("Figure size unit must be 'mm' or 'inch'.")
    aliases = {"mm": "mm", "inch": "inch", "in": "inch", "inches": "inch"}
    try:
        return aliases[unit.lower().strip()]
    except KeyError as exc:
        raise ValueError("Figure size unit must be 'mm' or 'inch'.") from exc


def size_to_inches(width: float, height: float, unit: str = "mm") -> tuple[float, float]:
    """Convert control values into the legacy, inch-valued project settings."""
    width, height = _finite_number(width, "Width"), _finite_number(height, "Height")
    if width <= 0 or height <= 0:
        raise ValueError("Figure width and height must be positive.")
    if _unit(unit) == "mm":
        width, height = mm_to_inches(width), mm_to_inches(height)
    if width <= 0 or height <= 0:
        raise ValueError("Figure width and height are too small.")
    return width, height


def size_from_inches(width: float, height: float, unit: str = "mm") -> tuple[float, float]:
    width, height = size_to_inches(width, height, "inch")
    if _unit(unit) == "mm":
        return inches_to_mm(width), inches_to_mm(height)
    return width, height


def preset_size_inches(preset: str) -> tuple[float, float] | None:
    try:
        size = FIGURE_PRESETS_MM[preset]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown figure size preset.") from exc
    return None if size is None else size_to_inches(*size, "mm")


def _font_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Unknown font family.")
    normalized = "".join(name.casefold().split())
    choices = {"".join(font.casefold().split()): font for font in FONT_FAMILIES}
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError("Unknown font family.") from exc


def validate_style_settings(settings: Mapping | None, *, fill_defaults: bool = False) -> dict:
    """Validate only this module's settings, ignoring other project keys.

    Partial legacy projects need no added fields. ``fill_defaults=True`` is
    useful for rendering. Invalid fields raise ValueError before state changes.
    Margins are canvas fractions and are checked together with missing defaults.
    """
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("Figure style settings must be an object.")
    cleaned = {}
    for key in STYLE_KEYS.intersection(settings):
        value = settings[key]
        if key == "size_unit":
            value = _unit(value)
        elif key == "figure_preset":
            if not isinstance(value, str) or value not in FIGURE_PRESETS_MM:
                raise ValueError("Unknown figure size preset.")
        elif key == "font_family":
            value = _font_name(value)
        elif key == "tick_direction":
            # auto: inward without references, outward on the reference axis (the earlier fixed behaviour).
            if value not in ("auto", "in", "out"):
                raise ValueError("Tick direction must be auto, in or out.")
        elif key in ("transparent_background", "svg_text_editable", "axes_box"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false.")
        else:
            value = _finite_number(value, key)
            lower, upper = (0.0, 1.0) if key.startswith("margin_") else _NUMBER_BOUNDS.get(key, (0.1, 5.0))
            if not lower <= value <= upper:
                raise ValueError(f"{key} must be between {lower:g} and {upper:g}.")
        cleaned[key] = value
    merged = {**STYLE_DEFAULTS, **cleaned}
    if merged["margin_left"] >= merged["margin_right"]:
        raise ValueError("The left margin must be smaller than the right margin.")
    if merged["margin_bottom"] >= merged["margin_top"]:
        raise ValueError("The bottom margin must be smaller than the top margin.")
    return merged if fill_defaults else cleaned


def register_bundled_fonts() -> list[str]:
    """Make the bundled fonts known to matplotlib; safe to call again (Streamlit reruns, reloads).

    An installed font of the same family stays usable; matplotlib's font lookup cache is cleared
    by ``addfont``, so earlier "not found" answers in a long-running server are not reused.
    """
    known = {entry.fname for entry in font_manager.fontManager.ttflist}
    added = []
    for name in BUNDLED_FONT_FILES:
        path = FONT_DIR / name
        if not path.is_file() or str(path) in known:
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except (OSError, RuntimeError, ValueError):
            continue
        added.append(name)
    return added


register_bundled_fonts()


def resolve_font_family(requested: str = "Arial") -> str:
    """Resolve an installed font, keeping serif/sans-serif character on fallback.

    Arial falls back to Helvetica, then Liberation Sans or Arimo (same character
    widths as Arial), as in the graph app; DejaVu Sans is the last resort.
    """
    requested = _font_name(requested)
    candidates = FONT_CANDIDATES[requested]
    for family in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False,
            )
            return font_manager.get_font(path).family_name
        except (ValueError, OSError, RuntimeError):
            continue
    # DejaVu is shipped with Matplotlib, so a normal installation always reaches
    # an installed fallback. A damaged installation should report its real error.
    raise ValueError("No usable installed font was found.")


def resolve_font(family: str = "Arial") -> str:
    """Installed family name shared by the Matplotlib and Plotly renderers."""
    return resolve_font_family(family)


def font_family_list(family: str) -> list[str]:
    """The family for Text objects, with the glyph fallback for characters it lacks (△, ◇)."""
    return [family] if family == GLYPH_FALLBACK else [family, GLYPH_FALLBACK]


def css_font_stack(requested: str = "Arial") -> str:
    """CSS font-family for the browser preview: the viewer's own Arial, else a same-width substitute."""
    return CSS_FONT_STACKS[_font_name(requested)]


def font_rcparams(settings: Mapping | None = None) -> dict:
    """Fonts for text and mathtext alike, as in the graph app: italics, subscripts and
    superscripts are drawn in the same font as the rest of the label."""
    style = validate_style_settings(settings, fill_defaults=True)
    family = resolve_font_family(style["font_family"])
    return {
        "font.family": font_family_list(family),
        # Anything that asks for the generic sans-serif family gets the chosen font as well.
        "font.sans-serif": font_family_list(family),
        "mathtext.fontset": "custom",
        "mathtext.default": "it",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.cal": family,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none" if style["svg_text_editable"] else "path",
        "axes.linewidth": style["axes_line_width"],
        "xtick.major.width": style["axes_line_width"],
        "xtick.minor.width": style["axes_line_width"],
        "ytick.major.width": style["axes_line_width"],
        "ytick.minor.width": style["axes_line_width"],
        "xtick.major.size": style["tick_length"],
        "xtick.minor.size": style["tick_length"] / 2,
        "ytick.major.size": style["tick_length"],
        "ytick.minor.size": style["tick_length"] / 2,
    }


def svg_font_names(svg: bytes, requested: str) -> bytes:
    """Name the requested font in an editable SVG drawn with a same-width substitute.

    Liberation Sans / Arimo stand in for Arial where Arial is not installed (Streamlit
    Cloud); their character widths are identical, so the SVG says Arial and opens in
    Arial on the user's computer. A substitute with other widths (DejaVu) is left as is.
    """
    requested = _font_name(requested)
    resolved = resolve_font_family(requested)
    if resolved == requested or resolved not in METRIC_COMPATIBLE.get(requested, ()):
        return svg
    text = svg.decode("utf-8")
    text = text.replace(f"font-family: '{resolved}'", f"font-family: '{requested}'")
    text = text.replace(f"font-family:'{resolved}'", f"font-family:'{requested}'")
    text = text.replace(f"font-family: {resolved};", f"font-family: {requested};")
    return text.encode("utf-8")


def apply_mpl_style(fig, settings: Mapping | None = None):
    """Apply margins, axes and font to an existing figure, then return it.

    Curve/reference widths are intentionally explicit renderer inputs because
    artists such as peak markers and reference sticks have different roles.
    Call this after constructing axes; do not call tight_layout afterwards.
    """
    style = validate_style_settings(settings, fill_defaults=True)
    family = resolve_font_family(style["font_family"])
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    else:
        fig.set_constrained_layout(False)
        fig.set_tight_layout(False)
    fig.subplots_adjust(
        left=style["margin_left"], right=style["margin_right"],
        bottom=style["margin_bottom"], top=style["margin_top"],
    )
    background = "none" if style["transparent_background"] else "white"
    fig.patch.set_facecolor(background)
    for ax in fig.axes:
        ax.set_facecolor(background)
        for spine in ax.spines.values():
            spine.set_linewidth(style["axes_line_width"])
        ax.tick_params(which="both", width=style["axes_line_width"])
    families = font_family_list(family)
    for text in fig.findobj(match=Text):
        text.set_fontfamily(families)
        # Mathtext (italics, sub/superscripts) in the chosen font even for text made before the
        # font settings: each Text keeps the math font set that was current when it was created.
        text.set_math_fontfamily("custom")
    return fig


def export_figure(fig, format: str, dpi: float = 300, settings: Mapping | None = None, *, style: Mapping | None = None) -> bytes:
    """Export PNG/TIFF/PDF/SVG without cropping or resizing the physical canvas.

    Raster dimensions are quantized to integer pixels by the backend. Editable
    SVG text references the resolved font; it does not embed font files. PDF
    uses TrueType embedding. This function does not close the caller's figure.
    ``style`` is an alias for ``settings`` for callers that name the style.
    """
    if settings is not None and style is not None:
        raise ValueError("Specify either settings or style, not both.")
    style_values = validate_style_settings(settings if style is None else style, fill_defaults=True)
    if not isinstance(format, str):
        raise ValueError("Unknown export format.")
    format = format.lower().lstrip(".")
    if format == "tif":
        format = "tiff"
    if format not in EXPORT_MIME_TYPES:
        raise ValueError("Export format must be PNG, TIFF, PDF or SVG.")
    dpi = _finite_number(dpi, "DPI")
    if not 0 < dpi <= 2400:
        raise ValueError("DPI must be greater than zero and at most 2400.")
    size_to_inches(*fig.get_size_inches(), "inch")
    background = "none" if style_values["transparent_background"] else "white"
    rc = {**font_rcparams(style_values), "savefig.bbox": None, "savefig.pad_inches": 0.0}
    result = io.BytesIO()
    save_options = {}
    if format == "tiff":
        from PIL import features
        # Standard Pillow wheels include libtiff. Keep custom minimal Pillow
        # builds usable too: their uncompressed TIFF remains pixel-identical.
        if features.check("libtiff"):
            save_options["pil_kwargs"] = {"compression": "tiff_adobe_deflate"}
    with mpl.rc_context(rc):
        # Inside the font settings, so text made while styling or drawing uses them too.
        apply_mpl_style(fig, style_values)
        fig.savefig(
            result, format=format, dpi=dpi, bbox_inches=None, pad_inches=0,
            facecolor=background, edgecolor=background,
            transparent=style_values["transparent_background"],
            **save_options,
        )
    if format == "svg" and style_values["svg_text_editable"]:
        return svg_font_names(result.getvalue(), style_values["font_family"])
    return result.getvalue()
