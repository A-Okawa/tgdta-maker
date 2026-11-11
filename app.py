"""TG-DTA Maker — publication figures of TG-DTA (TG, DTG, DTA) in the look of XRD Maker and the graph app."""
import inspect
import io
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from figure_style import (
    FIGURE_PRESETS_MM, FONT_FAMILIES, METRIC_COMPATIBLE, STYLE_DEFAULTS, STYLE_KEYS, export_figure,
    inches_to_mm, mm_to_inches, resolve_font, validate_style_settings,
)
from i18n import INLINE_TEXT, LANGUAGES, color_name
from palettes import (DEFAULT_PALETTE, MAX_SHADE_STEPS, PALETTE_NAMES, PALETTES, color_emoji, load_colors,
                      palette_colors, swatch_colors)
from project_io import (GLOBAL_KEYS, SAMPLE_KEY, ProjectError, atomic_save, clean_settings, decode_project,
                        encode_project, is_setting_key, project_fingerprint)
from rich_text import plain
from style_templates import decode_style_template, encode_style_template
from tgdta_io import (SIGNALS, SUPPORTED_EXTENSIONS, TGDTAParseError, heating_segment, parse_markers,
                      parse_ranges, process, processed_table, read_measurement)
from tgdta_plot import Curve, PlotOptions, auto_label, feature_rows, render_mpl, render_plotly, shown_signals

APP_VERSION = "2026-11-11"
LOCAL_MODE = os.environ.get("TGDTA_MAKER_LOCAL") == "1"
LOCAL_DATA_DIR = (Path(os.environ["TGDTA_MAKER_DATA_DIR"]).expanduser()
                  if LOCAL_MODE and os.environ.get("TGDTA_MAKER_DATA_DIR") else None)
# Named colours in the graph app's format; copy that app's colors.csv here to share its names.
COLORS_FILE = Path(__file__).resolve().parent / "colors.csv"
TWIN_COLOR2 = "#D55E00"          # 朱: the DTA curve and axis of one sample in the two-axis layout
SIGNAL_NAMES = {"time": "Time", "temp": "Temperature", "dta": "DTA / DSC", "tg": "TG"}

st.set_page_config(page_title="TG-DTA Maker", page_icon="🔥", layout="wide")
_IMAGE_PARAMETERS = inspect.signature(st.image).parameters
FULL_WIDTH_IMAGE = ({"width": "stretch"} if "int" not in str(_IMAGE_PARAMETERS["width"].annotation)
                    else {"use_container_width": True} if "use_container_width" in _IMAGE_PARAMETERS
                    else {"use_column_width": True})


def _full_width(widget):
    width = inspect.signature(widget).parameters.get("width")
    return {"width": "stretch"} if width is not None and isinstance(width.default, str) else {"use_container_width": True}


FULL_WIDTH = {name: _full_width(getattr(st, name)) for name in ("button", "download_button", "popover", "plotly_chart",
                                                                 "dataframe")}

# ===== 言語 / Language =====
_lang_option = st.sidebar.radio("🌐 Language / 言語", list(LANGUAGES), horizontal=True, key="lang_radio")
_lang = LANGUAGES.get(_lang_option, "ja")


def ui(ja, en):
    """Screen text. Vietnamese and Thai come from i18n.INLINE_TEXT, looked up by the English text."""
    if _lang == "ja":
        return ja
    if _lang == "en":
        return en
    return INLINE_TEXT.get(_lang, {}).get(en, en)


def _secret_password():
    try:
        return st.secrets.get("password", "")
    except Exception:          # no secrets file: a local run, no password
        return ""


def check_password():
    password = _secret_password()
    if LOCAL_MODE or not password or st.session_state.get("authenticated"):
        return True
    st.title("TG-DTA Maker")
    entered = st.text_input(ui("パスワードを入力してください", "Enter the password"), type="password")
    if st.button(ui("ログイン", "Log in")):
        if entered == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error(ui("パスワードが違います", "Wrong password"))
    return False


if not check_password():
    st.stop()

st.title("TG-DTA Maker")
st.caption(ui("TG・DTG・DTA を論文用の図に · XRD Maker・グラフ描画appと同じ体裁 · PDF / SVG / TIFF / PNG で保存",
              "TG, DTG and DTA as publication figures · the look of XRD Maker and the graph app · PDF / SVG / TIFF / PNG"))
if LOCAL_MODE:
    st.caption(ui("PC 版 · データと自動保存はこの PC の中だけにあります。",
                  "Desktop edition · data and autosave stay on this computer."))

# ===== 初期値 / Defaults =====
DEFAULTS = {
    "layout": "twin", "panel_tg": True, "panel_dtg": False, "panel_dta": True, "twin_right": "dta",
    "x_mode": "temp", "segment": "heating", "time_range": (0.0, 120.0),
    "x_auto": True, "x_min": 0.0, "x_max": 1300.0, "x_major": 0.0, "x_minor": True,
    "tg_mode": "change_pct", "tg_ref_on": False, "tg_ref_temp": 120.0,
    "dta_exo_positive": True, "exo_up": True, "dta_per_mass": False, "dta_zero": False, "dta_au": True,
    "dta_smooth": 0, "tg_smooth": 0, "dtg_smooth": 31, "dtg_basis": "per_min",
    "y_tg_lo": "", "y_tg_hi": "", "y_dta_lo": "", "y_dta_hi": "", "y_dtg_lo": "", "y_dtg_hi": "",
    "x_label": "", "tg_label": "", "dta_label": "", "dtg_label": "",
    "show_legend": True, "legend_loc": "best", "color_axes": True, "twin_color2": TWIN_COLOR2,
    "exo_arrow": True, "exo_loc": "upper right",
    "show_peaks": False, "peak_kind": "both", "peak_prom": 0.1, "peak_distance": 30.0, "peak_decimals": 0,
    "mass_ranges": "", "mass_decimals": 1, "mass_lines": True, "markers": "", "panel_gap": 0.06,
    "palette": DEFAULT_PALETTE, "palette_reverse": False, "show_panel": True,
    "dpi_export": 600, "size_unit": "mm", "annotation_font_size": 6.0,
}
LEGEND_LOCS = ["best", "upper right", "upper left", "lower left", "lower right", "center right", "center left"]
CORNERS = ["upper left", "upper right", "lower left", "lower right"]


def corner_name(value):
    return {"best": ui("自動", "Automatic"), "upper right": ui("右上", "Upper right"), "upper left": ui("左上", "Upper left"),
            "lower left": ui("左下", "Lower left"), "lower right": ui("右下", "Lower right"),
            "center right": ui("右中", "Center right"), "center left": ui("左中", "Center left")}[value]


def layout_margins(layout):
    """Margins for the 88 × 70 mm graph-app canvas: room for the tick labels and titles on both sides of a two-axis plot."""
    return dict(margin_left=0.15, margin_right=0.85 if layout == "twin" else 0.97, margin_bottom=0.14, margin_top=0.97)


def apply_preset(name):
    state = st.session_state
    preset = FIGURE_PRESETS_MM.get(name)
    if preset:
        state["fig_width"], state["fig_height"] = map(mm_to_inches, preset)
    layout = state.get("layout", "twin")
    if name == "figapp":
        # The graph app's defaults (as XRD Maker's preset): Arial 7 pt, 6.5 pt ticks and legend, 0.5 pt axes, 0.9 pt lines.
        state.update(font_family="Arial", font_size=7.0, tick_font_size=6.5, legend_font_size=6.5, annotation_font_size=6.0,
                     line_width=0.9, reference_line_width=0.6, axes_line_width=0.5, tick_length=2.5, tick_direction="out",
                     axes_box=False, **layout_margins(layout))
    elif name in ("single", "double"):
        compact = name == "single"
        size = 9.0 if compact else 11.0
        state.update(font_size=size, tick_font_size=size, legend_font_size=size, annotation_font_size=size - 2,
                     margin_left=.16 if compact else .11, margin_right=(.84 if compact else .89) if layout == "twin" else .97,
                     margin_bottom=.15 if compact else .1, margin_top=.97)
    state.pop("_size_signature", None)


def init_defaults():
    state = st.session_state
    for key, value in DEFAULTS.items():
        state.setdefault(key, value)
    if "fig_width" not in state:
        for key, value in STYLE_DEFAULTS.items():
            state.setdefault(key, value)
        state["figure_preset"] = "figapp"
        apply_preset("figapp")
    for key, value in STYLE_DEFAULTS.items():
        state.setdefault(key, value)
    state.setdefault("figure_preset", "custom")
    state.setdefault("_files", {})


init_defaults()
# Keep the values of controls that are hidden for a while (Streamlit forgets a widget's value when it is not drawn).
for _key in list(st.session_state):
    if _key != "lang_radio" and is_setting_key(_key):
        st.session_state[_key] = st.session_state[_key]


def _int_or(value, default):
    return value if type(value) is int else default


# ===== ファイル / Files =====
@st.cache_data(show_spinner=False, max_entries=64)
def load_measurement(raw: bytes, name: str, mapping: tuple):
    return read_measurement(raw, name, dict(mapping) if mapping else None)


def sample_names():
    return list(st.session_state.get("_files", {}))


def _remap_samples(old_names, new_names):
    """Settings are stored by position; keep each file's settings when files are added or removed."""
    state = st.session_state
    saved = {}
    for i, name in enumerate(old_names):
        saved[name] = {m.group(1): state[k] for k in list(state)
                       if (m := SAMPLE_KEY.fullmatch(str(k))) and int(m.group(2)) == i}
    for k in [k for k in list(state) if SAMPLE_KEY.fullmatch(str(k))]:
        del state[k]
    for i, name in enumerate(new_names):
        for field_name, value in saved.get(name, {}).items():
            state[f"{field_name}_{i}"] = value


def _files_changed(widget_key):
    files = st.session_state.get(widget_key) or []
    names = [f.name for f in files]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        st.session_state["_upload_error"] = ", ".join(duplicates)
        return
    st.session_state.pop("_upload_error", None)
    old = sample_names()
    _remap_samples(old, names)
    st.session_state["_files"] = {f.name: f.getvalue() for f in files}
    st.session_state["_recolor"] = True


def _restore(raw: bytes):
    project = decode_project(raw)
    for key in [k for k in list(st.session_state) if k != "lang_radio" and is_setting_key(k)]:
        del st.session_state[key]
    st.session_state.update({k: v for k, v in project["settings"].items() if k != "lang_radio"})
    st.session_state["_files"] = project["files"]
    st.session_state["_epoch"] = st.session_state.get("_epoch", 0) + 1
    st.session_state["_project_loaded"] = True
    st.session_state.pop("_project_error", None)
    st.session_state.pop("_size_signature", None)


def _open_project():
    upload = st.session_state.get("_project_upload")
    if upload is not None:
        try:
            _restore(upload.getvalue())
        except (ProjectError, ValueError) as exc:
            st.session_state["_project_error"] = str(exc)


def _open_autosave():
    try:
        _restore((LOCAL_DATA_DIR / "autosave.json").read_bytes())
    except (OSError, ProjectError) as exc:
        st.session_state["_project_error"] = str(exc)


def _new_project():
    for key in [k for k in list(st.session_state) if k != "lang_radio" and (is_setting_key(k) or k in STYLE_KEYS)]:
        del st.session_state[key]
    for key in ("fig_width", "fig_height", "figure_preset", "_size_signature", "_project_error"):
        st.session_state.pop(key, None)
    st.session_state["_files"] = {}
    st.session_state["_epoch"] = st.session_state.get("_epoch", 0) + 1
    init_defaults()


def _open_style():
    upload = st.session_state.get("_style_upload")
    if upload is None:
        return
    try:
        settings = decode_style_template(upload.getvalue())
        validate_style_settings({**{k: st.session_state[k] for k in STYLE_KEYS if k in st.session_state},
                                 **{k: v for k, v in settings.items() if k in STYLE_KEYS}})
        st.session_state.update(settings)
        st.session_state["figure_preset"] = "custom"
        st.session_state.pop("_size_signature", None)
        st.session_state["_style_loaded"] = True
        st.session_state.pop("_style_error", None)
    except (ProjectError, ValueError) as exc:
        st.session_state["_style_error"] = str(exc)


def _preset_chosen():
    apply_preset(st.session_state["figure_preset"])


def _layout_changed():
    """The right margin follows the layout: room for the right axis in the two-axis plot."""
    st.session_state["margin_right"] = layout_margins(st.session_state["layout"])["margin_right"]
    # Exo. arrow where the curves usually leave room: upper right of the two-axis plot (TG has fallen there),
    # upper left of the DTA panel (the high-temperature peaks are on the right).
    st.session_state["exo_loc"] = "upper right" if st.session_state["layout"] == "twin" else "upper left"
    st.session_state["_recolor"] = True


def _x_mode_changed():
    st.session_state["x_auto"] = True
    st.session_state["x_major"] = 0.0


def _update_dimension(axis):
    value = st.session_state[f"_figure_{axis}"]
    st.session_state[f"fig_{axis}"] = mm_to_inches(value) if st.session_state["size_unit"] == "mm" else value
    st.session_state["figure_preset"] = "custom"


def _update_margin(key):
    value = st.session_state[f"_percent_{key}"] / 100
    st.session_state[key] = 1 - value if key in ("margin_right", "margin_top") else value


# ===== 色 / Colours =====
OTHER_COLOR = "—"


def display_order():
    state = st.session_state
    n = len(sample_names())
    return sorted(range(n), key=lambda i: (_int_or(state.get(f"ord_{i}"), i + 1), i))


def _apply_palette():
    """Colour the visible samples in display order with the chosen palette (the graph app's colours)."""
    state = st.session_state
    order = [i for i in display_order() if state.get(f"vis_{i}", True) is not False]
    hidden = [i for i in display_order() if state.get(f"vis_{i}", True) is False]
    name = state.get("palette", DEFAULT_PALETTE)
    colors = palette_colors(name if name in PALETTES else DEFAULT_PALETTE, len(order) + len(hidden),
                            bool(state.get("palette_reverse")))
    for rank, i in enumerate(order + hidden):
        state[f"color_{i}"] = colors[rank]
    for key in [k for k in list(state) if str(k).startswith(("_palview_", "_cname_"))]:
        del state[key]


def _set_color(key, value):
    st.session_state[key] = value


def _named_color_chosen(key, name_key):
    value = load_colors(COLORS_FILE).get(st.session_state.get(name_key))
    if value:
        st.session_state[key] = value


def color_picker_popover(key: str, default_hex: str, n_items: int = 1, label=None):
    """Swatches of the graph app's palettes, colours by name (colors.csv) or any colour — as in XRD Maker."""
    st.session_state.setdefault(key, default_hex)
    current = str(st.session_state[key])
    if label:
        st.caption(label)
    st.markdown(f'<div style="background:{current};height:20px;border-radius:5px;border:1px solid #ccc;'
                f'margin-bottom:4px"></div>', unsafe_allow_html=True)
    with st.popover(ui("🎨 色を変える", "🎨 Change colour"), **FULL_WIDTH["popover"]):
        view_key = f"_palview_{key}"
        if st.session_state.get(view_key) not in PALETTES:
            st.session_state[view_key] = st.session_state.get("palette", DEFAULT_PALETTE)
        st.selectbox(ui("色の組み合わせ", "Palette"), list(PALETTES), key=view_key,
                     format_func=lambda k: PALETTE_NAMES.get(_lang, PALETTE_NAMES["en"])[k])
        colors = swatch_colors(st.session_state[view_key], n_items, bool(st.session_state.get("palette_reverse")))
        cols = st.columns(len(colors))
        for j, hex_c in enumerate(colors):
            with cols[j]:
                selected = hex_c.lower() == current.lower()
                st.markdown(f'<div style="background:{hex_c};height:24px;border-radius:3px;'
                            f'border:{"3px solid #333" if selected else "1px solid #ccc"};margin-bottom:2px"></div>',
                            unsafe_allow_html=True)
                st.button("✓" if selected else " ", key=f"{key}_sw_{j}", help=hex_c.upper(),
                          on_click=_set_color, args=(key, hex_c), **FULL_WIDTH["button"])
        names = load_colors(COLORS_FILE)
        name_key = f"_cname_{key}"
        st.session_state[name_key] = next((n for n, h in names.items() if h.lower() == current.lower()), OTHER_COLOR)
        st.selectbox(ui("名前で選ぶ", "By name"), [OTHER_COLOR] + list(names), key=name_key,
                     on_change=_named_color_chosen, args=(key, name_key),
                     format_func=lambda n: n if n == OTHER_COLOR else f"{color_emoji(names[n])} {color_name(n, _lang)}  {names[n]}",
                     help=ui("colors.csv の色（グラフ描画appと同じ書式「名前,#RRGGBB」）。",
                             "Colours from colors.csv (the graph app's format, name,#RRGGBB)."))
        st.color_picker(ui("自由に選ぶ", "Any colour"), key=key)
    return st.session_state[key]


# ===== サイドバー：作業ファイル =====
st.sidebar.caption(f"Last updated: {APP_VERSION}")
st.sidebar.header(ui("💾 作業ファイル", "💾 Project"))
st.sidebar.caption(ui("測定データ・試料の設定・図の設定をまとめて保存し、あとで開けます。",
                      "Save measurements, sample settings and figure settings together and open them later."))
st.sidebar.file_uploader(ui("保存した作業ファイルを開く (.json)", "Open a saved project (.json)"), type=["json"],
                         key="_project_upload", on_change=_open_project)
if st.session_state.pop("_project_loaded", False):
    st.sidebar.success(ui("作業ファイルを開きました。", "Project opened."))
if st.session_state.get("_project_error"):
    st.sidebar.error(ui("作業ファイルを開けません: ", "Cannot open the project: ") + st.session_state["_project_error"])
if LOCAL_DATA_DIR and (LOCAL_DATA_DIR / "autosave.json").is_file() and not st.session_state["_files"]:
    st.sidebar.button(ui("前回の自動保存から再開", "Resume from the last autosave"), on_click=_open_autosave,
                      **FULL_WIDTH["button"])
st.sidebar.button(ui("新しく始める", "Start a new project"), on_click=_new_project, **FULL_WIDTH["button"])
autosave_status = st.sidebar.empty()

st.sidebar.header(ui("📂 TG-DTA データ", "📂 TG-DTA data"))
st.sidebar.caption(ui("装置から書き出した CSV / TXT（時間・温度・DTA・TG の列）。Shift_JIS・UTF-8 のどちらでも読めます。複数ファイルで重ね描き。",
                      "CSV / TXT exported by the instrument (time, temperature, DTA and TG columns), Shift_JIS or UTF-8. "
                      "Several files are overlaid."))
_upload_key = f"upload_{st.session_state.get('_epoch', 0)}"
st.sidebar.file_uploader(ui("ファイルを選ぶ", "Choose files"), type=list(SUPPORTED_EXTENSIONS), accept_multiple_files=True,
                         key=_upload_key, on_change=_files_changed, args=(_upload_key,))
if st.session_state.get("_upload_error"):
    st.sidebar.error(ui("同じ名前のファイルがあります: ", "Files with the same name: ") + st.session_state["_upload_error"])
if st.session_state["_files"] and not st.session_state.get(_upload_key):
    st.sidebar.caption(ui("作業ファイルのデータ: ", "Data from the project: ") + ", ".join(sample_names()))

# ===== 測定データの読み込み =====
names = sample_names()
measurements, read_errors = {}, []
for i, name in enumerate(names):
    mapping = tuple((s, st.session_state[f"map_{s}_{i}"]) for s in SIGNALS
                    if type(st.session_state.get(f"map_{s}_{i}")) is int)
    mapping = tuple((s, v if v >= 0 else None) for s, v in mapping)
    try:
        measurements[i] = load_measurement(st.session_state["_files"][name], name, mapping)
    except TGDTAParseError as exc:
        read_errors.append(f"{name}: {exc}")
for _error in read_errors:
    st.error(ui("読み込めません — ", "Cannot read — ") + _error)
for i in measurements:
    st.session_state.setdefault(f"vis_{i}", True)
    st.session_state.setdefault(f"ord_{i}", i + 1)
    st.session_state.setdefault(f"lbl_{i}", os.path.splitext(names[i])[0])
    st.session_state.setdefault(f"mass_{i}", 0.0)
    st.session_state.setdefault(f"dtaoff_{i}", 0.0)
if st.session_state.pop("_recolor", False) or any(f"color_{i}" not in st.session_state for i in measurements):
    _apply_palette()

# ===== サイドバー：グラフの設定 =====
st.sidebar.header(ui("⚙️ グラフの設定", "⚙️ Graph settings"))
st.sidebar.radio(ui("並べ方", "Layout"), ["twin", "stack"], key="layout", on_change=_layout_changed,
                 format_func=lambda v: {"twin": ui("TG と DTA を左右の軸で 1 枚に", "TG and DTA on left and right axes"),
                                        "stack": ui("上下のパネルに分ける", "Panels one above the other")}[v],
                 help=ui("試料が 1 つなら左右の軸、複数を比べるなら上下のパネルが見やすい図になります。",
                         "Left/right axes suit one sample; panels one above the other suit comparing several."))
if st.session_state["layout"] == "stack":
    _pc = st.sidebar.columns(3)
    for _col, _key, _label in zip(_pc, ("panel_tg", "panel_dtg", "panel_dta"), ("TG", "DTG", "DTA")):
        _col.checkbox(_label, key=_key)
else:
    st.sidebar.radio(ui("右の軸", "Right axis"), ["dta", "dtg"], key="twin_right", horizontal=True,
                     format_func=lambda v: v.upper())
st.sidebar.radio(ui("横軸", "x axis"), ["temp", "time"], key="x_mode", horizontal=True, on_change=_x_mode_changed,
                 format_func=lambda v: {"temp": ui("温度", "Temperature"), "time": ui("時間", "Time")}[v])
st.sidebar.selectbox(ui("描く区間", "Part of the run"), ["heating", "all", "custom"], key="segment",
                     format_func=lambda v: {"heating": ui("昇温部だけ（自動）", "Heating ramp only (automatic)"),
                                            "all": ui("すべて（保持・冷却も）", "All (with hold and cooling)"),
                                            "custom": ui("時間で指定", "Time range")}[v],
                     help=ui("「昇温部だけ」は、昇温速度が設定値の 9 割を下回った所（最高温度での保持の始まり）で切ります。"
                             "温度を横軸にしたとき、保持中の DTA の変化が縦線になって残るのを防ぎます。",
                             "'Heating ramp only' ends where the heating rate falls below 90% of the programmed rate "
                             "(the start of the hold). Against temperature, the DTA drift during the hold would otherwise "
                             "show as a vertical line."))
_t_max = max([float(m.time_min()[-1]) for m in measurements.values() if m.time_min() is not None] or [120.0])
if st.session_state["segment"] == "custom":
    _lo, _hi = st.session_state["time_range"]
    _hi_bound = float(np.ceil(_t_max))
    st.session_state["time_range"] = (min(float(_lo), _hi_bound - 0.1), min(max(float(_hi), float(_lo) + 0.1), _hi_bound))
    st.sidebar.slider(ui("時間の範囲 (min)", "Time range (min)"), 0.0, _hi_bound, step=0.1, key="time_range")

st.sidebar.checkbox(ui("横軸の範囲を自動にする", "Automatic x range"), key="x_auto")
if not st.session_state["x_auto"]:
    _xc = st.sidebar.columns(2)
    _xc[0].number_input(ui("下限", "From"), step=10.0, format="%.1f", key="x_min")
    _xc[1].number_input(ui("上限", "To"), step=10.0, format="%.1f", key="x_max")
    if st.session_state["x_min"] >= st.session_state["x_max"]:
        st.sidebar.warning(ui("下限は上限より小さくしてください。", "Make 'From' smaller than 'To'."))
_xt = st.sidebar.columns([0.55, 0.45])
_xt[0].number_input(ui("目盛りの間隔（0 = 自動）", "Tick spacing (0 = auto)"), min_value=0.0, step=50.0 if
                    st.session_state["x_mode"] == "temp" else 10.0, format="%g", key="x_major")
_xt[1].markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
_xt[1].checkbox(ui("副目盛り", "Minor ticks"), key="x_minor")

signals_shown = shown_signals(PlotOptions(
    layout=st.session_state["layout"], twin_right=st.session_state["twin_right"],
    panels=tuple(s for s in ("tg", "dtg", "dta") if st.session_state[f"panel_{s}"])))

st.sidebar.subheader("TG")
st.sidebar.selectbox(ui("TG の表し方", "TG as"), ["change_pct", "residual_pct", "change_mg"], key="tg_mode",
                     format_func=lambda v: {"change_pct": ui("質量変化 Δm/m₀ (%)・0 から", "Mass change Δm/m₀ (%), from 0"),
                                            "residual_pct": ui("残存質量 (%)・100 から", "Remaining mass (%), from 100"),
                                            "change_mg": ui("質量変化 Δm (mg)", "Mass change Δm (mg)")}[v],
                     help=ui("% にするには、すぐ下で各試料の初期質量 m₀ (mg) を入れてください（データの TG 列は mg のため）。",
                             "For %, enter each sample's initial mass m₀ (mg) just below (the TG column is in mg)."))
if measurements:
    st.sidebar.caption(ui("初期質量 m₀ (mg)（0 = 未入力）。秤量した値を入れてください。", "Initial mass m₀ (mg) (0 = not entered): the weighed value."))
    for i in measurements:
        st.sidebar.number_input(f"m₀ · {os.path.splitext(names[i])[0]}", 0.0, 1e6, step=0.1, format="%.3f", key=f"mass_{i}",
                                help=ui("TG を % で描くとき、DTA を質量で割るときに使います。", "Used for TG in % and DTA per mass."))
st.sidebar.checkbox(ui("基準の温度で 0 にそろえる", "Zero at a reference temperature"), key="tg_ref_on",
                    help=ui("例：120 °C（吸着水を除いたあとの質量を基準にする）。オフなら描く区間の最初の点が基準です。",
                            "e.g. 120 °C (mass after adsorbed water is gone). Off: the first point of the drawn part."))
if st.session_state["tg_ref_on"]:
    st.sidebar.number_input(ui("基準の温度 (°C)", "Reference temperature (°C)"), step=10.0, format="%.1f", key="tg_ref_temp")
st.sidebar.number_input(ui("TG の平滑化（点数、0 = なし）", "TG smoothing (points, 0 = none)"), 0, 201, step=2, key="tg_smooth")

st.sidebar.subheader("DTA")
_dc = st.sidebar.columns(2)
_dc[0].radio(ui("装置の発熱の符号", "Exotherm in the data"), [True, False], key="dta_exo_positive",
             format_func=lambda v: ui("＋（正）", "+ (positive)") if v else ui("−（負）", "− (negative)"),
             help=ui("DTA の発熱ピークがデータで正か負か。多くの TG-DTA は正です。",
                     "Whether exothermic peaks are positive or negative in the data. Most TG-DTA instruments: positive."))
_dc[1].radio(ui("図で発熱を", "Exotherm drawn"), [True, False], key="exo_up",
             format_func=lambda v: ui("上に", "Up") if v else ui("下に", "Down"))
st.sidebar.checkbox(ui("縦軸を a.u. にする（目盛りの数値なし）", "Arbitrary units (a.u., no tick values)"), key="dta_au",
                    help=ui("「DTA (a.u.)」として目盛りの数値を出しません。外すと μV の数値と単位を出します。",
                            "Labels the axis 'DTA (a.u.)' without tick values. Untick for values in μV."))
if not st.session_state["dta_au"]:
    st.sidebar.checkbox(ui("試料の質量で割る (μV mg⁻¹)", "Divide by sample mass (μV mg⁻¹)"), key="dta_per_mass")
st.sidebar.checkbox(ui("始点を 0 にそろえる", "Start at 0"), key="dta_zero",
                    help=ui("各試料の DTA から描く区間の最初の値を引きます。重ねるときの縦の位置は試料パネルの「DTA の縦の位置」で。",
                            "Subtracts each curve's first value. Shift curves with 'DTA offset' in the sample panel."))
st.sidebar.number_input(ui("DTA の平滑化（点数、0 = なし）", "DTA smoothing (points, 0 = none)"), 0, 201, step=2, key="dta_smooth")
if "dtg" in signals_shown:
    st.sidebar.subheader("DTG")
    st.sidebar.radio(ui("DTG の単位", "DTG per"), ["per_min", "per_K"], key="dtg_basis", horizontal=True,
                     format_func=lambda v: {"per_min": ui("時間 (min⁻¹)", "time (min⁻¹)"), "per_K": ui("温度 (K⁻¹)", "temperature (K⁻¹)")}[v])
    st.sidebar.number_input(ui("DTG の平滑化（点数）", "DTG smoothing (points)"), 5, 401, step=2, key="dtg_smooth",
                            help=ui("微分はてんびんの雑音を強めるので、Savitzky–Golay で平滑化してから微分します。0.05 min 間隔なら 31 点 ≈ 1.5 min。",
                                    "Differentiation amplifies balance noise, so TG is smoothed (Savitzky–Golay) first. "
                                    "At 0.05 min spacing, 31 points ≈ 1.5 min."))

with st.sidebar.expander(ui("縦軸の範囲（空欄 = 自動）", "y ranges (empty = automatic)")):
    for _s in signals_shown:
        _yc = st.columns(2)
        _yc[0].text_input(f"{_s.upper()} " + ui("下限", "from"), key=f"y_{_s}_lo")
        _yc[1].text_input(f"{_s.upper()} " + ui("上限", "to"), key=f"y_{_s}_hi")

st.sidebar.subheader(ui("軸のラベル（空欄 = 自動）", "Axis labels (empty = automatic)"))
_axis_help = ui("グラフ描画app・XRD Maker と同じ書き方です。文字を選んで Ctrl+I で斜体、Ctrl+,（カンマ）で下付き、Ctrl+.（ピリオド）で上付き"
                "（Mac は ⌘ でも可）。例：`Δ<i>m</i>/<i>m</i>_{0} (%)`、`DTA (μV mg^-1)`。",
                "Same markup as the graph app and XRD Maker. Select text and press Ctrl+I for italic, Ctrl+, subscript, "
                "Ctrl+. superscript (⌘ on a Mac). e.g. `Δ<i>m</i>/<i>m</i>_{0} (%)`, `DTA (μV mg^-1)`.")
label_slots = {"x": "x_label", "tg": "tg_label", "dta": "dta_label", "dtg": "dtg_label"}
_label_widgets = {}
for _s in ["x"] + signals_shown:
    _label_widgets[_s] = st.sidebar.empty()

st.sidebar.subheader(ui("凡例", "Legend"))
_lc = st.sidebar.columns(2)
_lc[0].checkbox(ui("凡例を表示", "Show legend"), key="show_legend")
_lc[1].selectbox(ui("位置", "Position"), LEGEND_LOCS, key="legend_loc", format_func=corner_name, label_visibility="collapsed")
if st.session_state["layout"] == "twin":
    st.sidebar.checkbox(ui("試料が 1 つのとき、軸を曲線の色にする", "One sample: colour the axes like their curves"),
                        key="color_axes", help=ui("左の軸を TG の色、右の軸を DTA の色にします（このとき凡例は出しません）。",
                                                  "Left axis in the TG colour, right axis in the DTA colour (no legend needed then)."))

with st.sidebar.expander(ui("注記（Exo.・ピーク温度・質量変化・縦線）", "Annotations (Exo., peaks, mass change, lines)")):
    _ec = st.columns([0.45, 0.55])
    _ec[0].checkbox(ui("Exo. の矢印", "Exo. arrow"), key="exo_arrow")
    _ec[1].selectbox(ui("矢印の位置", "Arrow position"), CORNERS, key="exo_loc", format_func=corner_name,
                     label_visibility="collapsed")
    st.checkbox(ui("DTA のピーク温度を書く", "Label DTA peak temperatures"), key="show_peaks")
    if st.session_state["show_peaks"]:
        st.radio(ui("ピーク", "Peaks"), ["both", "exo", "endo"], key="peak_kind", horizontal=True,
                 format_func=lambda v: {"both": ui("両方", "Both"), "exo": ui("発熱", "Exo"), "endo": ui("吸熱", "Endo")}[v])
        st.slider(ui("ピーク検出の感度（曲線の振れ幅に対する割合）", "Peak prominence (fraction of the curve's range)"),
                  0.01, 0.9, step=0.01, key="peak_prom", help=ui("小さいほど小さなピークも拾います。", "Smaller finds smaller peaks."))
        _pk = st.columns(2)
        _pk[0].number_input(ui("ピーク同士の最小間隔", "Min. peak spacing"), 0.0, 1000.0, step=5.0, key="peak_distance",
                            help=ui("横軸の単位（°C または min）。", "In x-axis units (°C or min)."))
        _pk[1].number_input(ui("温度の小数の桁", "Decimals"), 0, 2, step=1, key="peak_decimals")
    st.text_input(ui("質量変化を書く温度範囲 (°C)", "Mass change over (°C)"), key="mass_ranges", placeholder="25-200, 200-800",
                  help=ui("範囲ごとに TG の変化（例：−5.0%）を曲線の下に書きます。カンマ区切り。点線で範囲の境目を示します。",
                          "Writes the TG change of each range (e.g. −5.0%) under the curve. Comma-separated; dotted lines mark the ranges."))
    _mc = st.columns(2)
    _mc[0].number_input(ui("質量変化の小数の桁", "Decimals"), 0, 3, step=1, key="mass_decimals")
    _mc[1].markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
    _mc[1].checkbox(ui("境目の点線", "Range lines"), key="mass_lines")
    st.text_area(ui("縦線（1 行に 1 本：位置 | 文字）", "Vertical lines (one per line: position | text)"), key="markers",
                 placeholder="1014 | crystallization", height=80,
                 help=ui("横軸の単位で位置を書きます。文字は省略できます。全パネルを通る破線になります。",
                         "Position in x-axis units; text is optional. Dashed line through all panels."))
    st.number_input(ui("注記の文字サイズ (pt)", "Annotation font size (pt)"), 4.0, 20.0, step=0.5, key="annotation_font_size")

# ===== サイドバー：図の寸法・体裁（XRD Maker と同じ） =====
st.sidebar.header(ui("🖼️ 図の寸法・体裁", "🖼️ Figure size and style"))
st.sidebar.selectbox(ui("論文図のひな形", "Publication figure preset"), list(FIGURE_PRESETS_MM), key="figure_preset",
                     on_change=_preset_chosen,
                     format_func=lambda key: {"custom": ui("自由指定", "Custom"),
                                              "single": ui("1段幅の例 · 85 × 70 mm", "Single column example · 85 × 70 mm"),
                                              "double": ui("2段幅の例 · 180 × 120 mm", "Double column example · 180 × 120 mm"),
                                              "figapp": ui("グラフ描画appと同じ体裁 · 88 × 70 mm", "Graph app look · 88 × 70 mm")}[key])
st.sidebar.caption(ui("初期値は「グラフ描画appと同じ体裁」（Arial 7 pt・目盛り 6.5 pt・線 0.9 pt・外向きの目盛り）で、XRD Maker・グラフ描画appの図と並べられます。",
                      "The default is the graph app look (Arial 7 pt, 6.5 pt ticks, 0.9 pt lines, outward ticks), matching XRD Maker and the graph app."))
size_unit = st.sidebar.radio(ui("寸法の単位", "Size unit"), ["mm", "inch"], horizontal=True, key="size_unit")
_signature = (size_unit, st.session_state["fig_width"], st.session_state["fig_height"])
if st.session_state.get("_size_signature") != _signature:
    for _axis in ("width", "height"):
        _value = st.session_state[f"fig_{_axis}"]
        st.session_state[f"_figure_{_axis}"] = inches_to_mm(_value) if size_unit == "mm" else _value
    st.session_state["_size_signature"] = _signature
for _axis, _label in (("width", ui("図の幅", "Figure width")), ("height", ui("図の高さ", "Figure height"))):
    st.sidebar.number_input(f"{_label} ({size_unit})", min_value=25.0 if size_unit == "mm" else 25 / 25.4,
                            max_value=508.0 if size_unit == "mm" else 20.0, step=1.0 if size_unit == "mm" else 0.1,
                            format="%.2f", key=f"_figure_{_axis}", on_change=_update_dimension, args=(_axis,))
if st.session_state["layout"] == "stack" and len(signals_shown) >= 3 and inches_to_mm(st.session_state["fig_height"]) < 90:
    st.sidebar.caption(ui("3 段のパネルなら、高さ 100 mm 前後が見やすい目安です。", "With three panels, about 100 mm height reads well."))
st.sidebar.selectbox(ui("出力 DPI", "Output DPI"), [300, 600], key="dpi_export")
st.sidebar.caption(ui("文字の大きさ (pt)", "Text sizes (pt)"))
_sc = st.sidebar.columns(3)
for _col, _key, _label in ((_sc[0], "font_size", ui("軸ラベル", "Axis labels")), (_sc[1], "tick_font_size", ui("目盛り", "Ticks")),
                           (_sc[2], "legend_font_size", ui("凡例", "Legend"))):
    st.session_state[_key] = float(st.session_state.get(_key) or 7.0)
    _col.number_input(_label, min_value=4.0, max_value=28.0, step=0.5, format="%.1f", key=_key)
with st.sidebar.expander(ui("線・フォント・余白", "Lines, fonts and margins")):
    st.selectbox(ui("フォント", "Font family"), list(FONT_FAMILIES), key="font_family",
                 help=ui("グラフ描画appと同じ選択肢です。斜体・下付き・上付きも同じ書体で描きます。",
                         "The graph app's choices. Italics, subscripts and superscripts use the same font."))
    _actual = resolve_font(st.session_state["font_family"])
    if _actual != st.session_state["font_family"]:
        if _actual in METRIC_COMPATIBLE.get(st.session_state["font_family"], ()):
            st.caption(ui("この環境には {r} がないため、字幅が同じ {a} で描きます。SVG には {r} と書き込むので、手元で開けば {r} になります。",
                          "{r} is not installed here, so {a} (same character widths) is used. SVG files name {r}.")
                       .format(r=st.session_state["font_family"], a=_actual))
        else:
            st.caption(ui("この環境では {a} を使用します。", "This environment uses {a}.").format(a=_actual))
    st.number_input(ui("曲線の線幅 (pt)", "Curve line width (pt)"), .1, 5., step=.1, key="line_width")
    st.number_input(ui("軸・目盛りの線幅 (pt)", "Axes and tick width (pt)"), .1, 5., step=.1, key="axes_line_width")
    st.number_input(ui("目盛りの長さ (pt)", "Tick length (pt)"), .5, 10., step=.5, key="tick_length",
                    help=ui("主目盛りの長さ。副目盛りはその半分です。", "Major tick length; minor ticks are half as long."))
    st.selectbox(ui("目盛りの向き", "Tick direction"), ["auto", "in", "out"], key="tick_direction",
                 format_func=lambda v: {"auto": ui("外向き（グラフ描画appと同じ）", "Outward (as the graph app)"),
                                        "in": ui("内向き", "Inward"), "out": ui("外向き", "Outward")}[v])
    st.checkbox(ui("上と右の枠線を描く", "Draw the top and right frame"), key="axes_box")
    if st.session_state["layout"] == "stack":
        st.slider(ui("パネルの間隔", "Space between panels"), 0.0, 0.5, step=0.01, key="panel_gap")
    for _key, _label in (("margin_left", ui("左余白 (%)", "Left margin (%)")), ("margin_right", ui("右余白 (%)", "Right margin (%)")),
                         ("margin_bottom", ui("下余白 (%)", "Bottom margin (%)")), ("margin_top", ui("上余白 (%)", "Top margin (%)"))):
        _fraction = st.session_state[_key]
        st.session_state[f"_percent_{_key}"] = round(100 * (1 - _fraction if _key in ("margin_right", "margin_top") else _fraction), 8)
        st.number_input(_label, 0., 100., step=1., key=f"_percent_{_key}", on_change=_update_margin, args=(_key,))
    st.checkbox(ui("背景を透明にする", "Transparent background"), key="transparent_background")
    st.checkbox(ui("SVGの文字を編集可能にする", "Keep SVG text editable"), key="svg_text_editable")
plot_style = {k: st.session_state[k] for k in STYLE_KEYS}
try:
    plot_style = validate_style_settings(plot_style, fill_defaults=True)
except ValueError:
    st.sidebar.error(ui("余白の合計が100%未満になるよう調整してください。", "Adjust opposing margins so their sum is below 100%."))
    st.stop()
with st.sidebar.expander(ui("作図スタイルの保存・読み込み", "Save and load figure styles")):
    st.caption(ui("寸法・線・フォント・余白・注記の体裁を再利用できます。XRD Maker で保存したスタイルも開けます。",
                  "Reuse dimensions, lines, fonts, margins and annotation style. XRD Maker style files open here too."))
    st.file_uploader(ui("スタイルを開く (.json)", "Open figure style (.json)"), type=["json"], key="_style_upload",
                     on_change=_open_style)
    if st.session_state.pop("_style_loaded", False):
        st.success(ui("作図スタイルを読み込みました。", "Figure style loaded."))
    if st.session_state.get("_style_error"):
        st.error(st.session_state["_style_error"])
    st.download_button(ui("スタイルを保存 (.json)", "Save figure style (.json)"),
                       encode_style_template(dict(st.session_state)), file_name="tgdta_figure_style.json",
                       mime="application/json", **FULL_WIDTH["download_button"])
st.sidebar.divider()


# ===== 処理 / Processing =====
def number_or_none(text):
    try:
        value = float(str(text).replace("−", "-").strip())
    except ValueError:
        return None
    return value if np.isfinite(value) else None


state = st.session_state
time_ranges, processed, notes = {}, {}, []
for i, m in measurements.items():
    t, T = m.time_min(), m.column("temp")
    if state["segment"] == "heating" and t is not None and T is not None:
        time_ranges[i] = heating_segment(t, T)
    elif state["segment"] == "custom":
        time_ranges[i] = tuple(state["time_range"])
    else:
        time_ranges[i] = None
    try:
        processed[i] = process(
            m, time_range=time_ranges[i], tg_mode=state["tg_mode"], mass_mg=float(state.get(f"mass_{i}") or 0) or None,
            tg_ref_temp=state["tg_ref_temp"] if state["tg_ref_on"] else None, dta_per_mass=state["dta_per_mass"] and not state["dta_au"],
            dta_flip=bool(state["dta_exo_positive"]) != bool(state["exo_up"]), dta_zero=state["dta_zero"],
            dta_smooth=state["dta_smooth"], tg_smooth=state["tg_smooth"], dtg_smooth=state["dtg_smooth"],
            dtg_basis=state["dtg_basis"])
    except TGDTAParseError as exc:
        st.error(f"{names[i]}: {exc}")

# Labels (empty = automatic; the automatic text is shown in the box as a hint).
_probe = [Curve("", "#000000", p) for p in processed.values()]
_probe_opts = PlotOptions(x_mode=state["x_mode"], dta_au=state["dta_au"])
for _s, _slot in _label_widgets.items():
    _auto = auto_label(_s, _probe, _probe_opts, state["tg_mode"]) if _probe else ""
    _slot.text_input({"x": ui("横軸", "x axis"), "tg": "TG", "dta": "DTA", "dtg": "DTG"}[_s], key=label_slots[_s],
                     placeholder=_auto, help=_axis_help)

# ===== メイン：試料パネル =====
show_panel = st.toggle(ui("試料の設定パネルを表示", "Show the sample panel"), key="show_panel")
if not measurements:
    st.info(ui("左のサイドバーから TG-DTA のファイル（CSV / TXT）をアップロードしてください。",
               "Upload TG-DTA files (CSV / TXT) in the sidebar."))
    with st.expander(ui("読み込めるファイルの形", "File format")):
        st.code("Time(min),TEMP（℃）,DTA(μV),TG(mg),\n0.000000, 12.970000, 8.686500, 0.000200,\n0.050000, 12.980000, 8.683000, 0.000200,\n…",
                language="text")
        st.caption(ui("列の名前から時間・温度・DTA・TG を見分けます。見分けられないときは、試料パネルの「列の対応」で選べます。"
                      "時間は min または s、温度は °C または K、TG は mg または %、DTA は μV（DSC の mW も可）。",
                      "Time, temperature, DTA and TG are found from the column names; otherwise choose them under 'Columns' "
                      "in the sample panel. Time in min or s, temperature in °C or K, TG in mg or %, DTA in μV (DSC mW also)."))
    col_graph = None
else:
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph, col_settings = st.container(), None
    if col_settings is not None:
        with col_settings, st.container(height=760):
            st.markdown(ui("#### 色", "#### Colours"))
            st.selectbox(ui("色の組み合わせ", "Palette"), list(PALETTES), key="palette", on_change=_apply_palette,
                         format_func=lambda k: PALETTE_NAMES.get(_lang, PALETTE_NAMES["en"])[k],
                         help=ui("グラフ描画app・XRD Maker と同じ色の組み合わせです。選ぶと、表示中の試料に表示順で色が付きます。",
                                 "The palettes of the graph app and XRD Maker. Choosing one colours the visible samples in display order."))
            _cc = st.columns([0.5, 0.5])
            _cc[0].checkbox(ui("色の順を逆に", "Reverse"), key="palette_reverse", on_change=_apply_palette)
            _cc[1].button(ui("色を当てはめ直す", "Re-apply"), on_click=_apply_palette, **FULL_WIDTH["button"])
            if state["layout"] == "twin" and sum(1 for i in measurements if state.get(f"vis_{i}", True)) == 1:
                color_picker_popover("twin_color2", TWIN_COLOR2, 2,
                                     label=ui("{s} の色（試料が 1 つのとき）", "{s} colour (one sample)").format(
                                         s=state["twin_right"].upper()))
            st.markdown(ui("#### 試料", "#### Samples"))
            n_samples = len(measurements)
            for i, m in measurements.items():
                with st.expander(f"**{i + 1}. {os.path.splitext(names[i])[0]}**", expanded=n_samples <= 3):
                    _r = st.columns([0.45, 0.55])
                    _r[0].number_input(ui("表示順", "Order"), 1, 10000, step=1, key=f"ord_{i}")
                    _r[1].markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
                    _r[1].checkbox(ui("表示", "Show"), key=f"vis_{i}")
                    st.text_input(ui("凡例の名前", "Legend label"), key=f"lbl_{i}", help=_axis_help)
                    color_picker_popover(f"color_{i}", palette_colors(state["palette"], n_samples)[i % max(n_samples, 1)],
                                         n_samples)
                    if "dta" in signals_shown:
                        st.number_input(ui("DTA の縦の位置（加える値）", "DTA offset (added)"), step=5.0, format="%.2f",
                                        key=f"dtaoff_{i}", help=ui("重ねた DTA を上下にずらします。", "Shifts overlaid DTA curves up or down."))
                    p = processed.get(i)
                    if p is not None:
                        _seg = time_ranges.get(i)
                        st.caption(ui("昇温速度 ≈ {r:.1f} K/min · {n} 点", "Heating rate ≈ {r:.1f} K/min · {n} points").format(
                            r=p.rate, n=len(p.t)) + (ui(" · 描く区間 {a:.1f}–{b:.1f} min（{Ta:.0f}–{Tb:.0f} °C）",
                                                         " · drawn {a:.1f}–{b:.1f} min ({Ta:.0f}–{Tb:.0f} °C)").format(
                            a=p.t[0], b=p.t[-1], Ta=p.T[0], Tb=p.T[-1])))
                    with st.popover(ui("列の対応", "Columns"), **FULL_WIDTH["popover"]):
                        options = [-1] + list(range(len(m.columns)))
                        for s in SIGNALS:
                            key = f"map_{s}_{i}"
                            if type(state.get(key)) is not int:
                                state[key] = m.mapping.get(s) if m.mapping.get(s) is not None else -1
                            st.selectbox(SIGNAL_NAMES[s], options, key=key,
                                         format_func=lambda k, cols=m.columns: ui("（なし）", "(none)") if k < 0 else cols[k])
                        st.caption(ui("単位: ", "Units: ") + ", ".join(f"{s} = {u}" for s, u in m.units.items()))

# ===== 図 / Figure =====
if measurements and col_graph is not None:
    curves = []
    for i in display_order():
        if i not in processed or state.get(f"vis_{i}", True) is False:
            continue
        curves.append(Curve(label=str(state.get(f"lbl_{i}") or os.path.splitext(names[i])[0]),
                            color=state.get(f"color_{i}", "#000000"), data=processed[i],
                            dta_offset=float(state.get(f"dtaoff_{i}") or 0.0), color2=state.get("twin_color2", TWIN_COLOR2)))
    x_range = None
    if not state["x_auto"] and state["x_min"] < state["x_max"]:
        x_range = (float(state["x_min"]), float(state["x_max"]))
    opts = PlotOptions(
        layout=state["layout"], panels=tuple(s for s in ("tg", "dtg", "dta") if state[f"panel_{s}"]),
        twin_right=state["twin_right"], x_mode=state["x_mode"], x_range=x_range,
        x_major=float(state["x_major"]) or None, x_minor=state["x_minor"],
        y_ranges={s: (number_or_none(state.get(f"y_{s}_lo")), number_or_none(state.get(f"y_{s}_hi"))) for s in ("tg", "dta", "dtg")},
        labels={s: state.get(slot, "") for s, slot in label_slots.items()},
        font_size=state["font_size"], tick_font_size=state["tick_font_size"], legend_font_size=state["legend_font_size"],
        annotation_font_size=float(state["annotation_font_size"]), show_legend=state["show_legend"],
        legend_loc=state["legend_loc"], color_axes=state["color_axes"], dta_au=state["dta_au"], exo_arrow=state["exo_arrow"],
        exo_up=bool(state["exo_up"]), exo_loc=state["exo_loc"], show_peaks=state["show_peaks"],
        peak_kind=state["peak_kind"], peak_prominence=float(state["peak_prom"]), peak_min_distance=float(state["peak_distance"]),
        peak_decimals=int(state["peak_decimals"]), mass_ranges=parse_ranges(state["mass_ranges"]),
        mass_decimals=int(state["mass_decimals"]), mass_lines=state["mass_lines"], markers=parse_markers(state["markers"]),
        panel_gap=float(state["panel_gap"]), fig_width=state["fig_width"], fig_height=state["fig_height"],
    )
    no_mass = [plain(c.label) for c in curves if "no_mass" in c.data.notes and state["tg_mode"] != "change_mg"]
    no_mass_dta = [plain(c.label) for c in curves if "no_mass_dta" in c.data.notes]
    ref_out = [plain(c.label) for c in curves if "ref_out_of_range" in c.data.notes]

    with col_graph:
        if no_mass:
            st.info(ui("初期質量 m₀ が未入力のため、TG を mg で描いています: ", "No initial mass m₀, so TG is drawn in mg: ")
                    + ", ".join(no_mass) + ui("（サイドバーの「TG」で m₀ を入れると % になります）", " (enter m₀ under 'TG' in the sidebar for %)"))
        if no_mass_dta:
            st.info(ui("初期質量が未入力のため、DTA を質量で割っていません: ", "No initial mass, so DTA is not divided by mass: ")
                    + ", ".join(no_mass_dta))
        if ref_out:
            st.warning(ui("基準の温度が描く区間の外なので、最初の点を基準にしました: ", "The reference temperature is outside the drawn part; "
                          "the first point is used: ") + ", ".join(ref_out))
        if not curves:
            st.warning(ui("表示する試料がありません。", "No sample is shown."))
        else:
            tab_live, tab_export, tab_numbers = st.tabs([ui("インタラクティブ", "Interactive"), ui("出力プレビュー", "Export preview"),
                                                         ui("数値の一覧", "Numbers")])
            with tab_live:
                st.caption(ui("ドラッグで拡大、ダブルクリックで元に戻ります。曲線に重ねると温度・時間・値が出ます。",
                              "Drag to zoom, double-click to reset. Hover a curve for temperature, time and value."))
                st.plotly_chart(render_plotly(curves, opts, plot_style, state["tg_mode"]), **FULL_WIDTH["plotly_chart"],
                                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
            fig, render_notes = render_mpl(curves, opts, plot_style, state["tg_mode"])
            dpi = int(state["dpi_export"])
            png_preview = export_figure(fig, "png", 200, style=plot_style)
            pdf_bytes = export_figure(fig, "pdf", dpi, style=plot_style)
            svg_bytes = export_figure(fig, "svg", dpi, style=plot_style)
            raster_ok = state["fig_width"] * state["fig_height"] * dpi ** 2 <= 36_000_000
            png_bytes = export_figure(fig, "png", dpi, style=plot_style) if raster_ok else None
            tiff_bytes = export_figure(fig, "tiff", dpi, style=plot_style) if raster_ok else None
            plt.close(fig)
            for _note in render_notes:
                if _note.startswith("unit_mismatch:"):
                    st.warning(ui("TG の単位（mg と %）が他の試料と違うので描いていません: ",
                                  "Not drawn because its TG unit (mg vs %) differs from the others: ") + _note.split(":", 1)[1])
            with tab_export:
                st.image(png_preview, **FULL_WIDTH_IMAGE)
                st.caption(ui("保存される図そのもの（{w:.0f} × {h:.0f} mm、{dpi} dpi）。軸や文字が端で切れる場合は、余白または文字サイズを調整してください。",
                              "The figure as saved ({w:.0f} × {h:.0f} mm, {dpi} dpi). If text or axes are clipped, adjust the margins or font sizes.")
                           .format(w=inches_to_mm(state["fig_width"]), h=inches_to_mm(state["fig_height"]), dpi=dpi))
            with tab_numbers:
                rows = feature_rows(curves, PlotOptions(**{**opts.__dict__, "peak_kind": "both"}))
                table = pd.DataFrame([{
                    ui("試料", "Sample"): r["sample"],
                    ui("昇温速度 (K/min)", "Heating rate (K/min)"): round(r["rate"], 2),
                    ui("終わりの温度 (°C)", "End temperature (°C)"): round(r["T_end"], 1),
                    ui("全体の質量変化", "Total mass change"): f"{r['tg_total']:+.2f} {r['tg_unit']}".replace("-", "−"),
                    ui("発熱ピーク (°C)", "Exo peaks (°C)"): ", ".join(f"{T:.0f}" for T, k in r["peaks"] if k == "exo"),
                    ui("吸熱ピーク (°C)", "Endo peaks (°C)"): ", ".join(f"{T:.0f}" for T, k in r["peaks"] if k == "endo"),
                    ui("範囲ごとの質量変化", "Mass change by range"): "; ".join(f"{lo:g}–{hi:g} °C: {d:+.2f}".replace("-", "−")
                                                                         for lo, hi, d in r["steps"]),
                } for r in rows])
                st.dataframe(table, hide_index=True, **FULL_WIDTH["dataframe"])
                st.caption(ui("ピークは「注記」の感度・最小間隔で、描く区間・横軸の範囲の中から探します。温度は DTA が極大・極小になる点です"
                              "（補外開始温度ではありません）。",
                              "Peaks use the sensitivity and spacing under 'Annotations', within the drawn part and x range. "
                              "Temperatures are the DTA maxima/minima (not extrapolated onsets)."))
                csv_text = "".join(processed_table(plain(c.label), c.data) if k == 0 else
                                   processed_table(plain(c.label), c.data).split("\n", 1)[1] for k, c in enumerate(curves))
                st.download_button(ui("📄 描いた値を CSV で保存", "📄 Save the drawn values (CSV)"), csv_text.encode("utf-8-sig"),
                                   file_name="tgdta_values.csv", mime="text/csv", **FULL_WIDTH["download_button"])

            st.sidebar.download_button(ui("📥 PDF として保存", "📥 Save as PDF"), pdf_bytes, file_name="tgdta_result.pdf",
                                       mime="application/pdf", **FULL_WIDTH["download_button"])
            st.sidebar.download_button(ui("📥 SVG として保存", "📥 Save as SVG"), svg_bytes, file_name="tgdta_result.svg",
                                       mime="image/svg+xml", **FULL_WIDTH["download_button"])
            if raster_ok:
                st.sidebar.download_button(ui("📥 TIFF として保存（{dpi} dpi）", "📥 Save as TIFF ({dpi} dpi)").format(dpi=dpi), tiff_bytes,
                                           file_name="tgdta_result.tiff", mime="image/tiff", **FULL_WIDTH["download_button"])
                st.sidebar.download_button(ui("📥 PNG として保存（{dpi} dpi）", "📥 Save as PNG ({dpi} dpi)").format(dpi=dpi), png_bytes,
                                           file_name="tgdta_result.png", mime="image/png", **FULL_WIDTH["download_button"])
            else:
                st.sidebar.warning(ui("PNG / TIFFは3,600万画素までです。図の寸法またはDPIを下げてください。PDF / SVGは保存できます。",
                                      "For PNG / TIFF, reduce the dimensions or DPI to stay within 36 megapixels. PDF / SVG are available."))

# ===== 作業ファイルの保存・自動保存 =====
if st.session_state["_files"]:
    try:
        settings = {k: v for k, v in st.session_state.items()
                    if is_setting_key(k) and (k != "lang_radio" or v in ("日本語", "English"))}
        project_bytes = encode_project(st.session_state["_files"], settings)
        st.sidebar.download_button(ui("💾 作業ファイルを保存 (.json)", "💾 Save the project (.json)"), project_bytes,
                                   file_name="tgdta_project.json", mime="application/json", **FULL_WIDTH["download_button"])
        if LOCAL_DATA_DIR:
            try:
                fingerprint = project_fingerprint(project_bytes)
                if fingerprint != st.session_state.get("_autosave_fingerprint"):
                    atomic_save(LOCAL_DATA_DIR / "autosave.json", project_bytes)
                    st.session_state["_autosave_fingerprint"] = fingerprint
                    st.session_state["_autosave_time"] = datetime.now().astimezone().strftime("%H:%M:%S")
                autosave_status.caption(ui("この PC に自動保存しました · {t}", "Autosaved on this computer · {t}")
                                        .format(t=st.session_state.get("_autosave_time", "")))
            except (OSError, ProjectError) as exc:
                autosave_status.warning(ui("自動保存できません。作業ファイルを保存してください: ",
                                           "Autosave failed. Please save the project: ") + str(exc))
    except ProjectError as exc:
        autosave_status.warning(ui("作業ファイルを作れません: ", "Cannot build the project file: ") + str(exc))

st.sidebar.divider()
try:
    from build_bundle import create_bundle_bytes
    st.sidebar.download_button(ui("💻 PC 版をダウンロード (ZIP)", "💻 Download the desktop edition (ZIP)"), create_bundle_bytes(),
                               file_name="TG-DTA-Maker-PC.zip", mime="application/zip", **FULL_WIDTH["download_button"])
    st.sidebar.caption(ui("展開して setup → launch の順に開くと、この PC だけで動きます（データを外に送りません）。",
                          "Unzip, run setup, then launch: it runs on this computer only (no data leaves it)."))
except (ImportError, FileNotFoundError):
    pass

# ===== 書式のショートカット（グラフ描画app・XRD Maker と同じ） =====
_SHORTCUTS_HTML = """<script>
(function () {
  const P = window.parent;
  if (!P || !P.document) return;
  const handler = function (e) {
    const el = e.target;
    if (!el || !(el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || !el.type)))) return;
    if (el.getAttribute('role') === 'combobox' || (el.closest && el.closest('[data-baseweb="select"]'))) return;
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    const k = e.key;
    let open = null, close = null;
    if (k === 'i' || k === 'I') { open = '<i>'; close = '</i>'; }
    else if (k === 'b' || k === 'B') { open = '<b>'; close = '</b>'; }
    else if (k === ',' || k === '=') { open = '_{'; close = '}'; }
    else if (k === '.' || k === '+') { open = '^{'; close = '}'; }
    if (!open) return;
    e.preventDefault();
    e.stopPropagation();
    const v = el.value || '';
    const a = (el.selectionStart == null) ? v.length : el.selectionStart;
    const b = (el.selectionEnd == null) ? a : el.selectionEnd;
    const nv = v.slice(0, a) + open + v.slice(a, b) + close + v.slice(b);
    const proto = el.tagName === 'TEXTAREA' ? P.HTMLTextAreaElement.prototype : P.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, nv);
    el.dispatchEvent(new P.Event('input', { bubbles: true }));
    el.setSelectionRange(a + open.length, b + open.length);
  };
  if (P.__tgdtaMakerKeys) P.document.removeEventListener('keydown', P.__tgdtaMakerKeys, true);
  P.__tgdtaMakerKeys = handler;
  P.document.addEventListener('keydown', handler, true);
})();
</script>"""
if hasattr(st, "iframe"):
    st.iframe(_SHORTCUTS_HTML, height="content")
else:
    import streamlit.components.v1 as components
    components.html(_SHORTCUTS_HTML, height=0)
