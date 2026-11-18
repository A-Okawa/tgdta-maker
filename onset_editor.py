"""A Streamlit component to drag the baseline and tangent of a Tx / Tg construction.

The page (onset_editor/index.html) is plain HTML + Plotly, served by Streamlit from this folder, so it
updates with the app (git push) and needs no separate build or distribution. Plotly is copied next to it
from the installed Python plotly package, so the desktop edition works offline.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components

FOLDER = Path(__file__).resolve().parent / "onset_editor"
_component = None


def _ensure_plotly_js() -> None:
    target = FOLDER / "plotly.min.js"
    if target.exists():
        return
    try:
        import plotly
        source = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        shutil.copyfile(source, target)
    except (ImportError, OSError):
        pass                      # the page then loads Plotly from its CDN


def _get_component():
    global _component
    if _component is None:
        _ensure_plotly_js()
        _component = components.declare_component("tgdta_onset_editor", path=str(FOLDER))
    return _component


def onset_editor(*, T, y, base, tan, xrange, yrange, label, ylabel, color, token, sample, feature,
                 text_hint, text_parallel, key, height=380):
    """Draw the curve with two draggable lines; returns {token, sample, feature, base, tan, T} after a drag."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    step = max(1, len(T) // 2500)            # keep the page light
    return _get_component()(
        T=T[::step].round(4).tolist(), y=y[::step].round(6).tolist(), base=[float(v) for v in base],
        tan=[float(v) for v in tan], xrange=[float(v) for v in xrange], yrange=[float(v) for v in yrange],
        label=label, ylabel=ylabel, color=color, token=token, sample=sample, feature=feature,
        text_hint=text_hint, text_parallel=text_parallel, height=int(height), key=key, default=None)
