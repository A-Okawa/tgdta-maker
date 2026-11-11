"""TG-DTA figures in the look of the graph app (figapp) and XRD Maker.

The same fonts, line widths, tick lengths, margins and palettes as the other apps come from the
shared ``figure_style``, ``palettes`` and ``rich_text`` modules, so a TG-DTA figure sits next to
the XRD, Raman and graph-app figures of the same paper. Two layouts:

* ``twin``  — TG on the left axis and DTA (or DTG) on the right axis of one panel;
* ``stack`` — TG, DTG and DTA in panels one above the other, sharing the x axis.

``render_mpl`` draws the publication figure; ``render_plotly`` the interactive preview with the
same curves, ranges, labels and annotations. Neither depends on Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator, NullLocator
from matplotlib.transforms import blended_transform_factory

from figure_style import apply_mpl_style, css_font_stack, font_rcparams, validate_style_settings
from rich_text import plain, rich, to_html
from tgdta_io import Processed, dta_peaks, mass_change, peak_onset, tg_at


SIGNAL_ORDER = ("tg", "dtg", "dta")
TWIN_STYLES = {"tg": "-", "dta": "--", "dtg": ":"}
PLOTLY_DASH = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
MARKER_COLOR = "#4D4D4D"                 # 濃い灰: thin dashed lines keep > 3:1 contrast against white
# The graph app's spacing and legend look (figcore.set_rc), on top of the shared font settings.
FIGAPP_RC = {
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 2.5,
    "legend.frameon": False, "legend.handlelength": 2.0, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35, "legend.columnspacing": 1.0,
}


@dataclass
class Curve:
    label: str
    color: str
    data: Processed
    dta_offset: float = 0.0
    tg_offset: float = 0.0
    color2: str | None = None            # DTA/DTG colour of a single sample in the twin layout


@dataclass
class PlotOptions:
    layout: str = "twin"                 # twin | stack
    panels: tuple = ("tg", "dta")         # stack: which panels (drawn in SIGNAL_ORDER)
    twin_right: str = "dta"               # twin: dta | dtg on the right axis
    x_mode: str = "temp"                  # temp | time
    x_range: tuple | None = None
    x_major: float | None = None
    x_minor: bool = True
    y_ranges: dict = field(default_factory=dict)      # signal → (lo, hi), either may be None
    labels: dict = field(default_factory=dict)        # x, tg, dta, dtg (empty → automatic)
    font_size: float = 7.0
    tick_font_size: float = 6.5
    legend_font_size: float = 6.5
    annotation_font_size: float = 6.0
    show_legend: bool = True
    legend_loc: str = "best"
    color_axes: bool = True               # twin with one sample: axes in the colours of their curves
    dta_au: bool = True                   # DTA in arbitrary units: "DTA (a.u.)", no tick values
    exo_arrow: bool = True
    exo_up: bool = True
    exo_loc: str = "upper right"
    show_peaks: bool = False
    peak_kind: str = "both"
    peak_prominence: float = 0.1
    peak_min_distance: float = 30.0       # °C (or min) between peaks of the same kind
    peak_decimals: int = 0
    peak_mode: str = "peak"               # peak | onset | both: DTA maximum, extrapolated onset (Tx), or both
    onset_lines: bool = True              # draw the baseline and tangent that define the onset
    onset_symbol_exo: str = "<i>T</i>_{x}"
    onset_symbol_endo: str = ""
    mirror_ticks: bool = False            # with the full frame: ticks on the top (and right) axes too
    mass_ranges: list = field(default_factory=list)  # [(lo °C, hi °C)]
    mass_decimals: int = 1
    mass_lines: bool = True
    markers: list = field(default_factory=list)      # [(x, text)]
    panel_gap: float = 0.06               # stack: space between panels (fraction of a panel's height)
    fig_width: float = 88 / 25.4
    fig_height: float = 70 / 25.4


# ------------------------------------------------------------------------------------------------ helpers
def _unit_markup(unit: str) -> str:
    """'μV/mg' → 'μV mg^-1', '%/min' → '% min^-1' (the markup of the other apps)."""
    if "/" in unit:
        top, bottom = unit.split("/", 1)
        return f"{top} {bottom}^-1"
    return unit


def auto_label(signal: str, curves: list[Curve], opts: PlotOptions, tg_mode: str = "change_pct") -> str:
    if signal == "x":
        return "Temperature (°C)" if opts.x_mode == "temp" else "Time (min)"
    unit = next((getattr(c.data, f"{signal}_unit") for c in curves if getattr(c.data, signal) is not None), "")
    if signal == "tg":
        if unit == "mg":
            return "Mass change (mg)"
        return "Mass (%)" if tg_mode == "residual_pct" else "Mass change (%)"
    if signal == "dtg":
        return f"DTG ({_unit_markup(unit)})"
    if opts.dta_au:
        return "Heat flow (a.u.)" if unit.startswith("mW") else "DTA (a.u.)"
    if unit.startswith("mW"):
        return f"Heat flow ({_unit_markup(unit)})"
    return f"DTA ({_unit_markup(unit)})"


def label_of(signal, curves, opts, tg_mode):
    text = str(opts.labels.get(signal) or "").strip()
    return text if text else auto_label(signal, curves, opts, tg_mode)


def x_values(p: Processed, opts: PlotOptions) -> np.ndarray:
    return p.T if opts.x_mode == "temp" else p.t


def x_of_temperature(p: Processed, temp: float, opts: PlotOptions) -> float:
    """Where a temperature falls on the x axis (the time it was reached, on a time axis)."""
    if opts.x_mode == "temp":
        return temp
    return tg_at(p.T, p.t, temp)


def series(c: Curve, signal: str) -> np.ndarray | None:
    y = getattr(c.data, signal)
    if y is None:
        return None
    if signal == "dta":
        return y + c.dta_offset
    if signal == "tg":
        return y + c.tg_offset
    return y


def shown_signals(opts: PlotOptions) -> list[str]:
    if opts.layout == "twin":
        return ["tg", opts.twin_right if opts.twin_right in ("dta", "dtg") else "dta"]
    chosen = [s for s in SIGNAL_ORDER if s in opts.panels]
    return chosen or ["tg", "dta"]


def consistent_curves(curves: list[Curve], notes: list[str]) -> list[Curve]:
    """Curves that share the TG unit of the first one (mg and % cannot share an axis)."""
    out, unit = [], None
    for c in curves:
        u = c.data.tg_unit if c.data.tg is not None else None
        if u and unit is None:
            unit = u
        if u and unit and u != unit:
            notes.append(f"unit_mismatch:{plain(c.label)}")
            continue
        out.append(c)
    return out


def _fmt_temp(value: float, decimals: int) -> str:
    return f"{value:.{max(0, int(decimals))}f} °C"


def _fmt_change(value: float, unit: str, decimals: int) -> str:
    text = f"{value:+.{max(0, int(decimals))}f}".replace("-", "−")
    return f"{text}{'%' if unit == '%' else ' mg'}"


def peak_list(c: Curve, opts: PlotOptions):
    """[(x on the axis, y, temperature, kind, onset)] of the DTA peaks of one curve within the x range.

    ``onset`` is the extrapolated onset (see tgdta_io.peak_onset) with its temperature under 'T',
    or None when the peak has no clear leading edge.
    """
    y = series(c, "dta")
    if y is None:
        return []
    x = x_values(c.data, opts)
    mask = np.ones(len(x), bool)
    if opts.x_range:
        lo, hi = opts.x_range
        mask = (x >= lo) & (x <= hi)
    idx = np.nonzero(mask)[0]
    if idx.size < 5:
        return []
    signed = y[idx] if opts.exo_up else -y[idx]
    found = dta_peaks(x[idx], signed, opts.peak_prominence, opts.peak_kind, opts.peak_min_distance)
    out = []
    for k, kind in found:
        onset = None
        if opts.peak_mode in ("onset", "both"):
            # Onset on the drawn curve; exo is a maximum of the signed curve, so pass 'exo' and flip back.
            onset = peak_onset(x[idx], signed, k, "exo" if kind == "exo" else "endo")
            if onset is not None:
                if not opts.exo_up:
                    onset = {**onset, "y": -onset["y"], "y_inflection": -onset["y_inflection"], "slope": -onset["slope"],
                             "baseline": tuple(-v for v in onset["baseline"]), "y_peak": -onset["y_peak"]}
                onset["T"] = onset["x"] if opts.x_mode == "temp" else tg_at(c.data.t, c.data.T, onset["x"])
        out.append((float(x[idx[k]]), float(y[idx[k]]), float(c.data.T[idx[k]]), kind, onset))
    return out


def onset_label(T: float, kind: str, opts: PlotOptions) -> str:
    symbol = (opts.onset_symbol_exo if kind == "exo" else opts.onset_symbol_endo).strip()
    text = _fmt_temp(T, opts.peak_decimals)
    return f"{symbol} = {text}" if symbol else text


def mass_list(c: Curve, opts: PlotOptions):
    """[(x of the label, y, Δ, lo, hi)] of the chosen temperature ranges of one curve."""
    tg = series(c, "tg")
    if tg is None:
        return []
    out = []
    for lo, hi in opts.mass_ranges:
        delta = mass_change(c.data.T, tg, lo, hi)
        if not math.isfinite(delta):
            continue
        # The label goes in the middle of the range, just below the curve's lowest point there
        # (above its highest point for a gain), so it never sits on the curve.
        inside = (c.data.T >= lo) & (c.data.T <= hi)
        if not inside.any():
            continue
        xm = x_of_temperature(c.data, 0.5 * (lo + hi), opts)
        ym = float(np.nanmin(tg[inside]) if delta <= 0 else np.nanmax(tg[inside]))
        if math.isfinite(xm) and math.isfinite(ym):
            out.append((xm, ym, delta, lo, hi))
    return out


def feature_rows(curves: list[Curve], opts: PlotOptions) -> list[dict]:
    """The numbers shown in the summary table: heating rate, total mass change, peaks, chosen steps."""
    rows = []
    for c in curves:
        p = c.data
        row = {"sample": plain(c.label), "rate": p.rate, "T_end": float(p.T[-1]) if len(p.T) else float("nan"),
               "tg_total": float(p.tg[-1] - p.tg[0]) if p.tg is not None and len(p.tg) else float("nan"),
               "tg_unit": p.tg_unit, "peaks": [], "steps": []}
        found = peak_list(c, PlotOptions(**{**opts.__dict__, "peak_mode": "both"}))
        row["peaks"] = [(T, kind) for _, _, T, kind, _ in found]
        row["onsets"] = [(o["T"], kind) for _, _, _, kind, o in found if o is not None]
        row["steps"] = [(lo, hi, d) for _, _, d, lo, hi in mass_list(c, opts)]
        rows.append(row)
    return rows


def x_limits(curves: list[Curve], opts: PlotOptions) -> tuple[float, float]:
    """The x range: as set, or the data's extent with the lower end moved down to a tick (13 °C → 0 °C)."""
    if opts.x_range:
        return tuple(opts.x_range)
    xs = [x_values(c.data, opts) for c in curves]
    xs = [v[np.isfinite(v)] for v in xs if len(v)]
    xs = [v for v in xs if len(v)]
    if not xs:
        return (0.0, 1.0)
    lo, hi = min(float(v.min()) for v in xs), max(float(v.max()) for v in xs)
    if hi <= lo:
        return (lo - 0.5, hi + 0.5)
    locator = MultipleLocator(opts.x_major) if opts.x_major else MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
    below = [t for t in locator.tick_values(lo, hi) if t <= lo + 1e-9]
    if below and lo - max(below) <= 0.1 * (hi - lo):
        lo = float(max(below))
    return (lo, hi)


# ------------------------------------------------------------------------------------------------ matplotlib
def _style_axis(ax, opts, style, *, x=True, right=False):
    direction = style["tick_direction"] if style["tick_direction"] in ("in", "out") else "out"
    length = style["tick_length"]
    ax.tick_params(which="major", length=length, direction=direction)
    ax.tick_params(which="minor", length=length / 2, direction=direction)
    if x:
        if opts.x_major:
            ax.xaxis.set_major_locator(MultipleLocator(opts.x_major))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        if opts.x_minor:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if not style["axes_box"]:
        ax.spines["top"].set_visible(False)
        if not right:
            ax.spines["right"].set_visible(False)
    elif opts.mirror_ticks:
        # Full frame with ticks on all sides; the right axis of a two-axis plot keeps its own ticks.
        if x:
            ax.tick_params(axis="x", which="both", top=True)
        if not right:
            ax.tick_params(axis="y", which="both", right=True)


def _y_limits(ax, signal, opts):
    lo, hi = (list(opts.y_ranges.get(signal) or (None, None)) + [None, None])[:2]
    if lo is not None or hi is not None:
        cur = ax.get_ylim()
        ax.set_ylim(lo if lo is not None else cur[0], hi if hi is not None else cur[1])


def _color_axis(ax, side, color):
    ax.spines[side].set_color(color)
    ax.tick_params(axis="y", which="both", colors=color)
    ax.yaxis.label.set_color(color)


def _exo_arrow_mpl(ax, opts, style, size):
    loc = opts.exo_loc
    left = "left" in loc
    top = "upper" in loc
    x = 0.05 if left else 0.95
    y0, y1 = (0.76, 0.94) if top else (0.06, 0.24)
    if not opts.exo_up:
        y0, y1 = y1, y0
    ax.annotate("", xy=(x, y1), xytext=(x, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", lw=style["axes_line_width"],
                                color="black", shrinkA=0, shrinkB=0))
    ax.text(x + (0.02 if left else -0.02), 0.5 * (y0 + y1), "Exo.", transform=ax.transAxes, fontsize=size,
            ha="left" if left else "right", va="center")


def render_mpl(curves: list[Curve], opts: PlotOptions, style: dict, tg_mode: str = "change_pct"):
    """The publication figure. Returns (figure, notes)."""
    style = validate_style_settings(style, fill_defaults=True)
    notes: list[str] = []
    curves = consistent_curves(curves, notes)
    signals = shown_signals(opts)
    rc = {**font_rcparams(style), **FIGAPP_RC, "font.size": opts.font_size, "axes.labelsize": opts.font_size,
          "xtick.labelsize": opts.tick_font_size, "ytick.labelsize": opts.tick_font_size,
          "legend.fontsize": opts.legend_font_size, "lines.linewidth": style["line_width"]}
    lw = style["line_width"]
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(opts.fig_width, opts.fig_height))
        if opts.layout == "twin":
            ax = fig.add_subplot(1, 1, 1)
            ax2 = ax.twinx()
            axes = {"tg": ax, signals[1]: ax2}
            panels = [ax]
        else:
            gs = fig.add_gridspec(len(signals), 1, hspace=opts.panel_gap)
            panels, axes, first = [], {}, None
            for k, s in enumerate(signals):
                a = fig.add_subplot(gs[k, 0], sharex=first)
                first = first or a
                panels.append(a)
                axes[s] = a
        single = opts.layout == "twin" and len(curves) == 1

        handles = []
        for c in curves:
            x = x_values(c.data, opts)
            for s in signals:
                y = series(c, s)
                if y is None:
                    continue
                a = axes[s]
                if opts.layout == "twin":
                    color = c.color if (s == "tg" or not single) else (c.color2 or c.color)
                    ls = TWIN_STYLES[s]
                else:
                    color, ls = c.color, "-"
                line, = a.plot(x, y, color=color, lw=lw, ls=ls)
                if (opts.layout != "twin" and s == signals[0]) or (opts.layout == "twin" and s == "tg" and not single):
                    line.set_label(rich(c.label))
                    handles.append(line)
                elif single and not opts.color_axes:
                    # One sample, axes not coloured: the legend tells which curve belongs to which axis.
                    line.set_label("TG" if s == "tg" else s.upper())
                    handles.append(line)

        # Axis ranges, ticks and labels.
        lo_hi = x_limits(curves, opts)
        for a in panels:
            a.set_xlim(*lo_hi)
        for s, a in axes.items():
            right = opts.layout == "twin" and a is not panels[0]
            _style_axis(a, opts, style, x=not right, right=right or (opts.layout == "twin"))
            _y_limits(a, s, opts)
            a.set_ylabel(rich(label_of(s, curves, opts, tg_mode)))
            if s == "dta" and opts.dta_au:
                # Arbitrary units, as the intensity axis of XRD Maker: no tick marks or values.
                a.yaxis.set_major_locator(NullLocator())
                a.yaxis.set_minor_locator(NullLocator())
        if opts.layout == "twin":
            ax2.spines["right"].set_visible(True)
            if not style["axes_box"]:
                ax2.spines["top"].set_visible(False)
            if single and opts.color_axes and curves:
                _color_axis(panels[0], "left", curves[0].color)
                _color_axis(ax2, "right", curves[0].color2 or curves[0].color)
                ax2.spines["left"].set_visible(False)
        for a in panels[:-1]:
            plt.setp(a.get_xticklabels(), visible=False)
        panels[-1].set_xlabel(rich(label_of("x", curves, opts, tg_mode)))

        # Vertical marker lines through every panel, their text at the top of the first one.
        for value, text in opts.markers:
            for a in panels:
                a.axvline(value, color=MARKER_COLOR, lw=0.6 * lw, ls=(0, (3, 2)), zorder=0.5)
            if text:
                tr = blended_transform_factory(panels[0].transData, panels[0].transAxes)
                panels[0].text(value, 0.98, rich(text), transform=tr, rotation=90, ha="right", va="top",
                               fontsize=opts.annotation_font_size, color=MARKER_COLOR)

        # Mass change over the chosen ranges, next to the TG curve.
        if "tg" in axes and opts.mass_ranges:
            a = axes["tg"]
            for c in curves:
                for xm, ym, delta, lo, hi in mass_list(c, opts):
                    a.annotate(_fmt_change(delta, c.data.tg_unit, opts.mass_decimals), (xm, ym),
                               xytext=(0, -2.5 if delta <= 0 else 2.5), textcoords="offset points", ha="center",
                               va="top" if delta <= 0 else "bottom", fontsize=opts.annotation_font_size, color=c.color)
            if not any(v is not None for v in (opts.y_ranges.get("tg") or (None, None))):
                a.margins(y=0.16)      # room for the mass-change labels under the curve
            if opts.mass_lines and curves:
                edges = sorted({v for r in opts.mass_ranges for v in r})
                for v in edges:
                    xv = x_of_temperature(curves[0].data, v, opts)
                    if math.isfinite(xv):
                        a.axvline(xv, color=MARKER_COLOR, lw=0.5 * lw, ls=(0, (1, 1.5)), zorder=0.5)

        # DTA peak temperatures.
        if opts.show_peaks and "dta" in axes:
            a = axes["dta"]
            for c in curves:
                color = c.color2 if single and c.color2 else c.color
                for xp, yp, T, kind, onset in peak_list(c, opts):
                    up = (kind == "exo") == opts.exo_up
                    if opts.peak_mode in ("peak", "both"):
                        a.annotate(_fmt_temp(T, opts.peak_decimals), (xp, yp), xytext=(0, 2.5 if up else -2.5),
                                   textcoords="offset points", ha="center", va="bottom" if up else "top",
                                   fontsize=opts.annotation_font_size, color=color)
                    if onset is None:
                        continue
                    if opts.onset_lines:
                        # Baseline extended to the onset, and the tangent from the onset up the leading edge.
                        ba, bb = onset["baseline"]
                        xb = np.array([onset["x_base_start"], onset["x"]])
                        a.plot(xb, ba * xb + bb, color=MARKER_COLOR, lw=0.5 * lw, ls=(0, (2, 1.5)), zorder=1.5)
                        x_end = onset["x"] + (onset["y_peak"] - onset["y"]) / onset["slope"] * 0.85
                        xt = np.array([onset["x"], x_end])
                        a.plot(xt, onset["y"] + onset["slope"] * (xt - onset["x"]), color=MARKER_COLOR, lw=0.5 * lw,
                               ls=(0, (2, 1.5)), zorder=1.5)
                    a.plot([onset["x"]], [onset["y"]], marker="o", ms=2.2, mfc="white", mec=color, mew=0.6 * lw, zorder=3)
                    a.annotate(rich(onset_label(onset["T"], kind, opts)), (onset["x"], onset["y"]),
                               xytext=(-3, -3 if up else 3), textcoords="offset points", ha="right",
                               va="top" if up else "bottom", fontsize=opts.annotation_font_size, color=color,
                               bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4), zorder=4)
            if not any(v is not None for v in (opts.y_ranges.get("dta") or (None, None))):
                a.margins(y=0.12)       # room for the temperatures above and below the peaks

        if opts.exo_arrow and "dta" in axes:
            _exo_arrow_mpl(axes["dta"], opts, style, opts.annotation_font_size)

        if opts.show_legend and handles:
            legend_ax = ax2 if opts.layout == "twin" else panels[0]
            extra = []
            if opts.layout == "twin" and not single:
                extra = [Line2D([], [], color="black", lw=lw, ls=TWIN_STYLES[s], label=s.upper()) for s in signals]
            legend_ax.legend(handles=handles + extra, loc=opts.legend_loc)
        if opts.layout != "twin":
            fig.align_ylabels(panels)
        apply_mpl_style(fig, style)
    return fig, notes


# ------------------------------------------------------------------------------------------------ plotly
def render_plotly(curves: list[Curve], opts: PlotOptions, style: dict, tg_mode: str = "change_pct"):
    """The interactive preview: the same curves, ranges, labels and annotations as ``render_mpl``."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    style = validate_style_settings(style, fill_defaults=True)
    curves = consistent_curves(curves, [])
    signals = shown_signals(opts)
    single = opts.layout == "twin" and len(curves) == 1
    font = css_font_stack(style["font_family"])
    if opts.layout == "twin":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        where = {"tg": dict(row=1, col=1, secondary_y=False), signals[1]: dict(row=1, col=1, secondary_y=True)}
        refs = {"tg": ("x", "y"), signals[1]: ("x", "y2")}
    else:
        fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True,
                            vertical_spacing=opts.panel_gap / max(len(signals), 1))
        where = {s: dict(row=k + 1, col=1) for k, s in enumerate(signals)}
        refs = {s: ("x" if k == 0 else f"x{k + 1}", "y" if k == 0 else f"y{k + 1}") for k, s in enumerate(signals)}
    lw = style["line_width"] * 4 / 3
    x_name = plain(label_of("x", curves, opts, tg_mode))
    for c in curves:
        x = x_values(c.data, opts)
        for s in signals:
            y = series(c, s)
            if y is None:
                continue
            if opts.layout == "twin":
                color = c.color if (s == "tg" or not single) else (c.color2 or c.color)
                dash = PLOTLY_DASH[TWIN_STYLES[s]]
            else:
                color, dash = c.color, "solid"
            name = to_html(c.label) if not single else ("TG" if s == "tg" else s.upper())
            show = (opts.show_legend and ((opts.layout != "twin" and s == signals[0])
                                          or (opts.layout == "twin" and not single and s == "tg")
                                          or (single and not opts.color_axes)))
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=name, showlegend=show, legendgroup=c.label,
                line=dict(color=color, width=lw, dash=dash),
                customdata=np.column_stack([c.data.T, c.data.t]),
                hovertemplate=(f"{plain(c.label)}<br>{s.upper()} = %{{y:.4g}}<br>"
                               "T = %{customdata[0]:.1f} °C<br>t = %{customdata[1]:.2f} min<extra></extra>"),
            ), **where[s])

    for value, text in opts.markers:
        for s in signals:
            fig.add_vline(x=value, line=dict(color=MARKER_COLOR, width=0.6 * lw, dash="dash"), **{k: v for k, v in where[s].items() if k != "secondary_y"})
        if text:
            xr, yr = refs[signals[0]]
            fig.add_annotation(x=value, y=1, xref=xr, yref=f"{yr} domain", text=to_html(text), showarrow=False,
                               textangle=-90, xanchor="right", yanchor="top",
                               font=dict(size=opts.annotation_font_size, color=MARKER_COLOR))
    if "tg" in signals and opts.mass_ranges:
        xr, yr = refs["tg"]
        for c in curves:
            for xm, ym, delta, lo, hi in mass_list(c, opts):
                fig.add_annotation(x=xm, y=ym, xref=xr, yref=yr, text=_fmt_change(delta, c.data.tg_unit, opts.mass_decimals),
                                   showarrow=False, yshift=7, font=dict(size=opts.annotation_font_size, color=c.color))
    if opts.show_peaks and "dta" in signals:
        xr, yr = refs["dta"]
        for c in curves:
            color = c.color2 if single and c.color2 else c.color
            for xp, yp, T, kind, onset in peak_list(c, opts):
                up = (kind == "exo") == opts.exo_up
                if opts.peak_mode in ("peak", "both"):
                    fig.add_annotation(x=xp, y=yp, xref=xr, yref=yr, text=_fmt_temp(T, opts.peak_decimals), showarrow=False,
                                       yshift=7 if up else -7, font=dict(size=opts.annotation_font_size, color=color))
                if onset is None:
                    continue
                if opts.onset_lines:
                    ba, bb = onset["baseline"]
                    x_end = onset["x"] + (onset["y_peak"] - onset["y"]) / onset["slope"] * 0.85
                    for xs, ys in (([onset["x_base_start"], onset["x"]], [ba * onset["x_base_start"] + bb, onset["y"]]),
                                   ([onset["x"], x_end], [onset["y"], onset["y"] + onset["slope"] * (x_end - onset["x"])])):
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                                                 line=dict(color=MARKER_COLOR, width=0.5 * lw, dash="dot")), **where["dta"])
                fig.add_trace(go.Scatter(x=[onset["x"]], y=[onset["y"]], mode="markers", showlegend=False,
                                         marker=dict(size=5, color="white", line=dict(color=color, width=1)),
                                         hovertemplate=f"onset {onset['T']:.1f} °C<extra></extra>"), **where["dta"])
                fig.add_annotation(x=onset["x"], y=onset["y"], xref=xr, yref=yr, text=to_html(onset_label(onset["T"], kind, opts)),
                                   showarrow=False, xanchor="right", xshift=-4, yshift=-7 if up else 7,
                                   font=dict(size=opts.annotation_font_size, color=color))
    if opts.exo_arrow and "dta" in signals:
        xr, yr = refs["dta"]
        left, top = "left" in opts.exo_loc, "upper" in opts.exo_loc
        x = 0.05 if left else 0.95
        y0, y1 = (0.76, 0.94) if top else (0.06, 0.24)
        if not opts.exo_up:
            y0, y1 = y1, y0
        fig.add_annotation(x=x, y=y1, ax=x, ay=y0, xref=f"{xr} domain", yref=f"{yr} domain", axref=f"{xr} domain",
                           ayref=f"{yr} domain", text="", showarrow=True, arrowhead=2, arrowwidth=style["axes_line_width"] * 4 / 3)
        fig.add_annotation(x=x + (0.02 if left else -0.02), y=0.5 * (y0 + y1), xref=f"{xr} domain", yref=f"{yr} domain",
                           text="Exo.", showarrow=False, xanchor="left" if left else "right",
                           font=dict(size=opts.annotation_font_size))

    ticks = "outside" if style["tick_direction"] != "in" else "inside"
    tick_px = style["tick_length"] * 4 / 3
    mirror = ("ticks" if opts.mirror_ticks else True) if style["axes_box"] else False
    axis_common = dict(showline=True, linecolor="black", linewidth=style["axes_line_width"] * 4 / 3, mirror=mirror,
                       showgrid=False, zeroline=False, ticks=ticks, ticklen=tick_px,
                       tickfont=dict(color="black", size=opts.tick_font_size), title_font=dict(color="black", size=opts.font_size),
                       minor=dict(ticks=ticks, ticklen=tick_px / 2, showgrid=False))
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)
    fig.update_xaxes(range=list(x_limits(curves, opts)))
    if opts.x_major:
        fig.update_xaxes(dtick=opts.x_major)
    last = where[signals[-1]] if opts.layout != "twin" else dict(row=1, col=1)
    fig.update_xaxes(title_text=to_html(label_of("x", curves, opts, tg_mode)), **{k: v for k, v in last.items() if k != "secondary_y"})
    for s in signals:
        lo, hi = (list(opts.y_ranges.get(s) or (None, None)) + [None, None])[:2]
        kw = dict(title_text=to_html(label_of(s, curves, opts, tg_mode)))
        if lo is not None and hi is not None:
            kw["range"] = [lo, hi]
        if opts.layout == "twin":
            kw.update(mirror=False)          # the right side belongs to the second axis
        if s == "dta" and opts.dta_au:
            kw.update(showticklabels=False, ticks="", minor=dict(ticks=""))
        fig.update_yaxes(**kw, **where[s])
        if single and opts.color_axes and curves:
            color = curves[0].color if s == "tg" else (curves[0].color2 or curves[0].color)
            fig.update_yaxes(linecolor=color, tickfont=dict(color=color, size=opts.tick_font_size),
                             title_font=dict(color=color, size=opts.font_size), **where[s])
    W, H = opts.fig_width * 96, opts.fig_height * 96
    bg = "rgba(255,255,255,0)" if style["transparent_background"] else "white"
    fig.update_layout(
        dragmode="zoom", hovermode="closest",
        font=dict(family=font, size=opts.font_size, color="black"),
        height=max(360, min(1400, int(H * 1.6))),
        plot_bgcolor=bg, paper_bgcolor=bg, showlegend=opts.show_legend,
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.6)",
                    font=dict(color="black", size=opts.legend_font_size)),
        margin=dict(l=int(W * 1.6 * style["margin_left"]) + 10, r=int(W * 1.6 * (1 - style["margin_right"])) + 10,
                    t=int(H * 1.6 * (1 - style["margin_top"])) + 10, b=int(H * 1.6 * style["margin_bottom"]) + 10),
    )
    return fig
