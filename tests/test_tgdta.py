"""Tests of TG-DTA Maker: reading, processing, figures, projects, translations and the running app.

The data are synthetic (Shift_JIS, the instrument's column names), so the tests need no measurement.
"""
import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from figure_style import STYLE_DEFAULTS, export_figure  # noqa: E402
from project_io import ProjectError, clean_settings, decode_project, encode_project  # noqa: E402
from style_templates import decode_style_template, encode_style_template  # noqa: E402
from tgdta_io import (dta_peaks, guess_mapping, heating_rate, heating_segment, parse_markers, parse_ranges,  # noqa: E402
                      process, processed_table, read_measurement)
from tgdta_plot import Curve, PlotOptions, feature_rows, render_mpl, render_plotly  # noqa: E402

# The files XRD Maker, Raman Maker and UV-Vis-NIR Maker share. A change here must be copied to all apps.
SHARED_SHA256 = {
    "figure_style.py": "53abc26e9a1e77b4faaea9e6d9659fec66d6e6ab12d1b2a6bebe4b1f3a8a96ae",
    "palettes.py": "38a8b4eff1fe971652f4b1349c9cba2f8a0813edd5c66cd7ad8df69c30a53314",
    "rich_text.py": "6a3e13a8f6a07e767e49aa13c04c979b54f1f093bffa50498e36dcfc2424d1c5",
    "colors.csv": "540dc8b9d28f68e2f58189acf43de7a4b432b3fe92873e5b1a590357b36d9515",
}
FIGAPP_STYLE = dict(STYLE_DEFAULTS, line_width=0.9, axes_line_width=0.5, tick_length=2.5, tick_direction="out",
                    axes_box=False, margin_left=.15, margin_right=.85, margin_bottom=.14, margin_top=.97)


def synthetic_csv(rate=15.0, top=1260.0, hold=20.0, mass_loss=1.0, exo_at=1000.0, sign=1.0) -> bytes:
    """A run as the instrument writes it: 0.05 min steps, heating at ``rate`` to ``top``, then a hold."""
    t = np.arange(0, (top - 20) / rate + hold, 0.05)
    T = np.minimum(20 + rate * t, top) + 0.01 * np.sin(t)
    tg = -mass_loss / (1 + np.exp(-(T - 200) / 25))                      # one step around 200 °C
    dta = sign * (40 * np.exp(-((T - exo_at) / 8) ** 2) - 15 * np.exp(-((T - 150) / 20) ** 2)) + 0.002 * T
    lines = ["Time(min),TEMP（℃）,DTA(μV),TG(mg),"]
    lines += [f"{a:15.6f},{b:15.6f},{c:15.6f},{d:15.6f}," for a, b, c, d in zip(t, T, dta, tg)]
    return ("\r\n".join(lines) + "\r\n").encode("cp932")


class Reading(unittest.TestCase):
    def test_columns_units_and_encoding(self):
        m = read_measurement(synthetic_csv(), "a.csv")
        self.assertEqual(m.mapping, {"time": 0, "temp": 1, "dta": 2, "tg": 3})
        self.assertEqual(m.units, {"time": "min", "temp": "°C", "dta": "μV", "tg": "mg"})
        self.assertEqual(m.data.shape[1], 4)                     # the trailing comma adds no column

    def test_other_names(self):
        self.assertEqual(guess_mapping(["Time /s", "Temp. /°C", "DSC /(mW/mg)", "Mass /%"]),
                         {"time": 0, "temp": 1, "dta": 2, "tg": 3})
        self.assertEqual(guess_mapping(["時間(min)", "温度(℃)", "DTA(uV)", "TG(mg)"]),
                         {"time": 0, "temp": 1, "dta": 2, "tg": 3})

    def test_manual_mapping_and_header_lines(self):
        text = "Sample: x\nOperator: y\nA\tB\tC\tD\n" + "\n".join(f"{i * .1}\t{20 + i}\t{i % 3}\t{-i * .001}" for i in range(50))
        m = read_measurement(text.encode("utf-8"), "b.txt", {"time": 0, "temp": 1, "dta": 2, "tg": 3})
        self.assertEqual(m.columns, ["A", "B", "C", "D"])
        self.assertAlmostEqual(float(m.column("temp")[-1]), 69.0)

    def test_heating_segment_cuts_the_hold(self):
        m = read_measurement(synthetic_csv(), "a.csv")
        t, T = m.time_min(), m.column("temp")
        self.assertAlmostEqual(heating_rate(t, T), 15.0, delta=0.2)
        lo, hi = heating_segment(t, T)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, (1260 - 20) / 15, delta=0.5)


class Processing(unittest.TestCase):
    def setUp(self):
        self.m = read_measurement(synthetic_csv(mass_loss=1.0), "a.csv")

    def test_mass_change_percent_and_residual(self):
        p = process(self.m, mass_mg=20.0)
        self.assertEqual(p.tg_unit, "%")
        self.assertAlmostEqual(p.tg[0], 0.0)
        self.assertAlmostEqual(p.tg[-1], -5.0, delta=0.1)       # −1 mg of 20 mg
        r = process(self.m, mass_mg=20.0, tg_mode="residual_pct")
        self.assertAlmostEqual(r.tg[-1], 95.0, delta=0.1)
        self.assertTrue(np.all(np.nanmin(p.dtg) < 0))

    def test_without_mass_stays_in_mg(self):
        p = process(self.m)
        self.assertEqual(p.tg_unit, "mg")
        self.assertIn("no_mass", p.notes)

    def test_reference_temperature(self):
        p = process(self.m, mass_mg=20.0, tg_ref_temp=400.0)
        i = int(np.argmin(np.abs(p.T - 400)))
        self.assertAlmostEqual(p.tg[i], 0.0, delta=0.02)

    def test_peaks_and_flip(self):
        for sign in (1.0, -1.0):
            m = read_measurement(synthetic_csv(sign=sign), "a.csv")
            p = process(m, dta_flip=sign < 0)                    # exo drawn upward either way
            found = dta_peaks(p.T, p.dta, 0.1)
            exo = [p.T[i] for i, k in found if k == "exo"]
            endo = [p.T[i] for i, k in found if k == "endo"]
            self.assertTrue(any(abs(T - 1000) < 3 for T in exo), exo)
            self.assertTrue(any(abs(T - 150) < 8 for T in endo), endo)

    def test_text_inputs(self):
        self.assertEqual(parse_ranges("25-200, 200–800\n800 to 1200; x"), [(25, 200), (200, 800), (800, 1200)])
        self.assertEqual(parse_markers("1014 | cryst.\nfoo\n900"), [(1014.0, "cryst."), (900.0, "")])

    def test_drawn_values_table(self):
        text = processed_table("S", process(self.m, mass_mg=20.0))
        self.assertTrue(text.startswith("sample,time_min,temperature_C,TG_%,DTG_%_per_min,DTA_μV"))


class Figures(unittest.TestCase):
    def curves(self):
        p1 = process(read_measurement(synthetic_csv(), "a"), mass_mg=20.0, time_range=(0, 82))
        p2 = process(read_measurement(synthetic_csv(mass_loss=0.6, exo_at=980), "b"), mass_mg=20.0, time_range=(0, 82))
        return [Curve("Na_{2}O 5 mol%", "#0072B2", p1, color2="#D55E00"), Curve("b", "#D55E00", p2, dta_offset=-20)]

    def test_layouts_render_and_export(self):
        curves = self.curves()
        cases = [PlotOptions(), PlotOptions(layout="twin", twin_right="dtg"),
                 PlotOptions(layout="stack", panels=("tg", "dtg", "dta"), show_peaks=True,
                             mass_ranges=[(25, 400)], markers=[(1000, "cryst.")]),
                 PlotOptions(layout="stack", panels=("dta",), x_mode="time", exo_up=False, exo_loc="lower right")]
        for opts in cases:
            for subset in (curves[:1], curves):
                fig, notes = render_mpl(subset, opts, FIGAPP_STYLE)
                self.assertEqual(notes, [])
                for fmt in ("png", "pdf", "svg"):
                    self.assertGreater(len(export_figure(fig, fmt, 150, style=FIGAPP_STYLE)), 1000)
                plt.close(fig)
                self.assertTrue(render_plotly(subset, opts, FIGAPP_STYLE).data)

    def test_canvas_and_fonts_match_the_other_apps(self):
        fig, _ = render_mpl(self.curves()[:1], PlotOptions(), FIGAPP_STYLE)
        self.assertAlmostEqual(fig.get_size_inches()[0] * 25.4, 88.0, places=3)
        ax = fig.axes[0]
        self.assertEqual(ax.xaxis.label.get_fontsize(), 7.0)
        self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 6.5)
        self.assertFalse(ax.spines["top"].get_visible())
        self.assertAlmostEqual(ax.spines["left"].get_linewidth(), 0.5)
        self.assertEqual(ax.lines[0].get_linewidth(), 0.9)
        plt.close(fig)

    def test_feature_rows(self):
        rows = feature_rows(self.curves(), PlotOptions(mass_ranges=[(25, 400)]))
        self.assertAlmostEqual(rows[0]["rate"], 15.0, delta=0.2)
        self.assertTrue(any(abs(T - 1000) < 3 and k == "exo" for T, k in rows[0]["peaks"]))
        self.assertAlmostEqual(rows[0]["steps"][0][2], -5.0, delta=0.2)

    def test_mixed_units_are_reported(self):
        c = self.curves()
        c[1].data = process(read_measurement(synthetic_csv(), "b"))          # mg, no mass
        fig, notes = render_mpl(c, PlotOptions(), FIGAPP_STYLE)
        plt.close(fig)
        self.assertEqual(notes, ["unit_mismatch:b"])


class Projects(unittest.TestCase):
    def test_round_trip(self):
        raw = synthetic_csv()
        settings = {"layout": "stack", "time_range": (0.0, 80.0), "mass_0": 20.0, "lbl_0": "S", "color_0": "#0072B2",
                    "font_family": "Arial", "unknown_key": 1}
        project = decode_project(encode_project({"a.csv": raw}, settings))
        self.assertEqual(project["files"], {"a.csv": raw})
        self.assertEqual(project["settings"]["time_range"], (0.0, 80.0))
        self.assertNotIn("unknown_key", project["settings"])

    def test_invalid_values_are_refused(self):
        for bad in ({"layout": "<script>"}, {"color_0": "red"}, {"dpi_export": 72}, {"mass_0": -1}):
            with self.assertRaises(ProjectError):
                clean_settings(bad)
        with self.assertRaises(ProjectError):
            decode_project(json.dumps({"format": "other", "version": 1}).encode())

    def test_style_templates(self):
        style = decode_style_template(encode_style_template({"font_size": 7.0, "line_width": 0.9, "lbl_0": "x"}))
        self.assertEqual(style, {"font_size": 7.0, "line_width": 0.9})
        xrd = json.dumps({"format": "xrd-maker-style", "version": 1,
                          "settings": {"font_family": "Arial", "fig_width": 3.4646, "show_cif_legend": True}}).encode()
        self.assertEqual(decode_style_template(xrd), {"font_family": "Arial", "fig_width": 3.4646})


class Consistency(unittest.TestCase):
    def test_shared_files_are_unchanged(self):
        for name, digest in SHARED_SHA256.items():
            self.assertEqual(hashlib.sha256((APP_DIR / name).read_bytes()).hexdigest(), digest,
                             f"{name} differs from XRD Maker's copy: copy the change to every app")

    def test_every_screen_text_is_translated(self):
        from i18n import INLINE_TEXT
        tree = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"))
        english = [n.args[1].value for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "ui" and len(n.args) == 2
                   and isinstance(n.args[1], ast.Constant)]
        self.assertGreater(len(english), 150)
        for code in ("vi", "th"):
            self.assertEqual([s for s in english if s not in INLINE_TEXT[code]], [], code)


class RunningApp(unittest.TestCase):
    def test_app_in_every_language_and_layout(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=120)
        at.run()
        self.assertFalse(at.exception)
        at.session_state["_files"] = {"a.csv": synthetic_csv(), "b.csv": synthetic_csv(mass_loss=0.5)}
        at.session_state["_recolor"] = True
        at.run()
        self.assertFalse(at.exception)
        at.session_state["mass_0"], at.session_state["mass_1"] = 20.0, 20.0
        for layout, lang in (("stack", "English"), ("twin", "Tiếng Việt"), ("stack", "ไทย")):
            at.session_state["layout"], at.session_state["lang_radio"] = layout, lang
            at.session_state["show_peaks"] = True
            at.session_state["mass_ranges"] = "25-400"
            at.run()
            self.assertFalse(at.exception, (layout, lang))
            self.assertFalse(at.error, (layout, lang))


if __name__ == "__main__":
    unittest.main()
