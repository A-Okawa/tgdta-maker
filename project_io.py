"""TG-DTA Maker projects (measurements + settings in one JSON) and atomic, local-only autosave files.

A project holds the measured files as they were uploaded (base64) and the screen settings.
Every setting is checked against the list below before it reaches the screen, so a damaged or
edited project cannot put unexpected values into the app.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from figure_style import STYLE_KEYS, validate_style_settings
from palettes import PALETTES

PROJECT_FORMAT = "tgdta-maker-project"
PROJECT_VERSION = 1
MAX_PROJECT_BYTES = 120 * 1024 * 1024
MAX_DATA_BYTES = 90 * 1024 * 1024
MAX_TEXT = 2000
HEX = re.compile(r"#[0-9a-fA-F]{6}")


class ProjectError(ValueError):
    """A project is damaged, unsupported, or contains invalid settings."""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bool(v):
    return isinstance(v, bool)


def _num(lo, hi, integer=False):
    def check(v):
        return _number(v) and lo <= v <= hi and (not integer or isinstance(v, int))
    return check


def _choice(*options):
    return lambda v: v in options


def _text(v):
    return isinstance(v, str) and len(v) <= MAX_TEXT


def _pair(lo, hi):
    def check(v):
        return (isinstance(v, (list, tuple)) and len(v) == 2 and all(_number(x) for x in v)
                and lo <= v[0] < v[1] <= hi)
    return check


SCHEMA = {
    "lang_radio": _choice("日本語", "English", "Tiếng Việt", "ไทย"),
    "layout": _choice("twin", "stack", "single"), "single_signal": _choice("tg", "dtg", "dta"),
    "panel_tg": _bool, "panel_dtg": _bool, "panel_dta": _bool,
    "twin_right": _choice("dta", "dtg"),
    "x_mode": _choice("temp", "time"),
    "segment": _choice("heating", "all", "custom"),
    "time_range": _pair(0.0, 1e5),
    "x_auto": _bool, "x_min": _num(-300.0, 1e5), "x_max": _num(-300.0, 1e5),
    "x_major": _num(0.0, 1e4), "x_minor": _bool,
    "tg_mode": _choice("change_pct", "residual_pct", "change_mg"),
    "tg_ref_on": _bool, "tg_ref_temp": _num(-200.0, 3000.0),
    "dta_exo_positive": _bool, "exo_up": _bool, "dta_per_mass": _bool, "dta_zero": _bool, "dta_au": _bool,
    "dta_smooth": _num(0, 201, True), "tg_smooth": _num(0, 201, True), "dtg_smooth": _num(5, 401, True),
    "dtg_basis": _choice("per_min", "per_K"),
    "y_tg_lo": _text, "y_tg_hi": _text, "y_dta_lo": _text, "y_dta_hi": _text, "y_dtg_lo": _text, "y_dtg_hi": _text,
    "x_label": _text, "tg_label": _text, "dta_label": _text, "dtg_label": _text,
    "show_legend": _bool,
    "legend_loc": _choice("best", "upper right", "upper left", "lower left", "lower right", "center right", "center left"),
    "color_axes": _bool, "twin_color2": lambda v: isinstance(v, str) and HEX.fullmatch(v) is not None,
    "exo_arrow": _bool, "exo_loc": _choice("upper left", "upper right", "lower left", "lower right"),
    "show_peaks": _bool, "peak_kind": _choice("both", "exo", "endo"), "peak_prom": _num(0.01, 0.9),
    "peak_distance": _num(0.0, 1000.0), "peak_decimals": _num(0, 2, True),
    "peak_mode": _choice("peak", "onset", "both"), "onset_lines": _bool, "onset_symbol_exo": _text, "onset_symbol_endo": _text,
    "mirror_ticks": _bool,
    "mass_ranges": _text, "mass_decimals": _num(0, 3, True), "mass_lines": _bool,
    "markers": _text, "panel_gap": _num(0.0, 0.5),
    "fig_width": _num(25 / 25.4, 20.0), "fig_height": _num(25 / 25.4, 20.0),
    "dpi_export": lambda v: type(v) is int and v in (300, 600),
    "font_size": _num(4, 28), "tick_font_size": _num(4, 28), "legend_font_size": _num(4, 28),
    "annotation_font_size": _num(4, 20),
    "palette": lambda v: v in PALETTES, "palette_reverse": _bool,
    "show_panel": _bool,
}
GLOBAL_KEYS = tuple(SCHEMA) + tuple(STYLE_KEYS)
SAMPLE_SCHEMA = {
    "vis": _bool, "ord": _num(1, 10000, True), "lbl": _text,
    "color": lambda v: isinstance(v, str) and HEX.fullmatch(v) is not None,
    "mass": _num(0.0, 1e6), "dtaoff": _num(-1e6, 1e6), "tgoff": _num(-1e6, 1e6),
    "map_time": _num(-1, 999, True), "map_temp": _num(-1, 999, True),
    "map_dta": _num(-1, 999, True), "map_tg": _num(-1, 999, True),
}
SAMPLE_KEY = re.compile(r"(" + "|".join(SAMPLE_SCHEMA) + r")_(\d+)$")


def is_setting_key(key) -> bool:
    return isinstance(key, str) and (key in GLOBAL_KEYS or SAMPLE_KEY.fullmatch(key) is not None)


def clean_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ProjectError("Project settings must be an object.")
    try:
        style_values = validate_style_settings({k: v for k, v in settings.items() if k in STYLE_KEYS})
    except (ValueError, TypeError) as exc:
        raise ProjectError(str(exc)) from exc
    cleaned = {}
    for key, value in settings.items():
        if key in STYLE_KEYS:
            cleaned[key] = style_values[key]
            continue
        if key in SCHEMA:
            check = SCHEMA[key]
        else:
            m = SAMPLE_KEY.fullmatch(str(key))
            if not m:
                continue                      # unknown keys from later versions are ignored
            check = SAMPLE_SCHEMA[m.group(1)]
        if not check(value):
            raise ProjectError(f"Invalid setting: {key}")
        cleaned[key] = tuple(float(v) for v in value) if isinstance(value, list) else value
    return cleaned


def encode_project(files: dict[str, bytes], settings: dict) -> bytes:
    if sum(len(v) for v in files.values()) > MAX_DATA_BYTES:
        raise ProjectError("The measurements exceed 90 MB.")
    document = {
        "format": PROJECT_FORMAT, "version": PROJECT_VERSION,
        "files": {name: base64.b64encode(data).decode("ascii") for name, data in files.items()},
        "settings": clean_settings({k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()
                                    if is_setting_key(k)}),
    }
    return json.dumps(document, ensure_ascii=False, allow_nan=False,
                      default=lambda v: list(v) if isinstance(v, tuple) else str(v)).encode("utf-8")


def decode_project(raw: bytes) -> dict:
    if len(raw) > MAX_PROJECT_BYTES:
        raise ProjectError("The project exceeds 120 MB.")
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ProjectError("This is not a TG-DTA Maker project (.json).") from exc
    if (not isinstance(document, dict) or document.get("format") != PROJECT_FORMAT
            or type(document.get("version")) is not int or document["version"] > PROJECT_VERSION):
        raise ProjectError("Unsupported project file.")
    files = document.get("files")
    if not isinstance(files, dict):
        raise ProjectError("The project has no measurements.")
    decoded = {}
    for name, data in files.items():
        if not isinstance(name, str) or not isinstance(data, str) or len(name) > 255:
            raise ProjectError("Damaged measurement entry.")
        try:
            decoded[name] = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProjectError(f"Damaged measurement: {name}") from exc
    return {"files": decoded, "settings": clean_settings(document.get("settings") or {})}


def project_fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def atomic_save(path: Path, data: bytes) -> None:
    """Write a file so that a crash leaves either the old or the new version, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".autosave-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
