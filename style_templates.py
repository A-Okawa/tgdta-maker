"""Portable figure appearance, separate from measurements and sample settings.

The size, font, line and margin keys are the ones XRD Maker saves too, so the appearance of a
paper's figures can be carried between the apps by hand (same values, same look).
"""
from __future__ import annotations

import json

from figure_style import STYLE_KEYS
from project_io import ProjectError, clean_settings

TEMPLATE_KEYS = frozenset(STYLE_KEYS) | {
    "fig_width", "fig_height", "dpi_export", "font_size", "tick_font_size", "legend_font_size",
    "annotation_font_size", "show_legend", "legend_loc", "x_minor", "panel_gap", "color_axes",
    "exo_arrow", "exo_loc", "peak_decimals", "mirror_ticks", "onset_lines", "mass_decimals", "mass_lines", "twin_color2",
}
MAX_TEMPLATE_BYTES = 64 * 1024


def encode_style_template(settings: dict, name: str = "Figure style") -> bytes:
    if not isinstance(name, str) or len(name) > 200:
        raise ProjectError("Invalid style name.")
    cleaned = clean_settings({k: v for k, v in settings.items() if k in TEMPLATE_KEYS})
    return json.dumps({"format": "tgdta-maker-style", "version": 1, "name": name, "settings": cleaned},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def decode_style_template(raw: bytes) -> dict:
    if len(raw) > MAX_TEMPLATE_BYTES:
        raise ProjectError("Style template exceeds the 64 KB limit.")
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ProjectError("This is not a valid figure style JSON file.") from exc
    # XRD Maker's style files carry the same size, font, line and margin keys: they open here too.
    if (not isinstance(document, dict) or document.get("format") not in ("tgdta-maker-style", "xrd-maker-style")
            or document.get("version") != 1 or not isinstance(document.get("settings"), dict)):
        raise ProjectError("Unsupported figure style template.")
    settings = {k: v for k, v in document["settings"].items() if k in TEMPLATE_KEYS}
    return clean_settings(settings)
