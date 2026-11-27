"""Tests of reading without silent loss, of heating-only temperature values and of the quality checks (QC).

All data are synthetic, made here with known answers (no measurement is needed or kept in the repository).
"""
import json
import re
import string
import sys
import unittest
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tgdta import synthetic_csv  # noqa: E402
from tgdta_io import heating_segment, process, read_measurement, temperature_profile  # noqa: E402
from tgdta_plot import Curve, PlotOptions, feature_rows, mass_issues, peak_list  # noqa: E402
import tgdta_qc  # noqa: E402
from tgdta_qc import TEXT, Check, build_report, check_texts, evaluate, noise_sigma, overall, smoothing_span  # noqa: E402


def csv_bytes(t, T, dta, tg) -> bytes:
    lines = ["Time(min),TEMP（℃）,DTA(μV),TG(mg),"]
    lines += [f"{a:15.6f},{b:15.6f},{c:15.6f},{d:15.6f}," for a, b, c, d in zip(t, T, dta, tg)]
    return ("\r\n".join(lines) + "\r\n").encode("cp932")


def noisy_run(amplitude=40.0, noise=0.05, seed=0, rate=10.0, dip=0.0):
    """A heating run, 0.05 min steps, 20 → 1220 °C: an exothermic Gaussian peak of ``amplitude`` μV at 1000 °C
    (σ 8 K) on a sloping baseline, white noise of σ = ``noise`` μV (fixed seed), a 1 mg loss around 200 °C.
    ``dip``: an endothermic undershoot of that size right after the exotherm (its return to the baseline)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 120.0 + 1e-9, 0.05)
    T = 20 + rate * t
    dta = amplitude * np.exp(-((T - 1000) / 8) ** 2) + 0.002 * T + noise * rng.standard_normal(len(T))
    dta -= dip * np.exp(-((T - 1022) / 5) ** 2)
    tg = -1.0 / (1 + np.exp(-(T - 200) / 25))
    return t, T, dta, tg


def qc(raw, *, mass=20.0, smooth=0, opts=None, segment="heating", tg_mode="change_pct"):
    m = read_measurement(raw, "x.csv")
    tr = heating_segment(m.time_min(), m.column("temp")) if segment == "heating" else None
    kw = dict(time_range=tr, tg_mode=tg_mode, mass_mg=mass, dta_smooth=smooth)
    p = process(m, **kw)
    raw_p = process(m, **{**kw, "dta_smooth": 0})
    c, c_raw = Curve("x", "#000", p, key="x.csv"), Curve("x", "#000", raw_p, key="x.csv")
    o = opts or PlotOptions()
    return m, c, evaluate(m, c, c_raw, o, mass_mg=mass, mass_source="entered", tg_mode=tg_mode, dta_smooth=smooth)


def by_code(result, code):
    return [chk for chk in result["checks"] if chk.code == code]


class ReadingKeepsEveryRow(unittest.TestCase):
    """P0-1: a bad cell no longer cuts the table; the row is left out and reported with its line number."""

    def setUp(self):
        self.lines = synthetic_csv().decode("cp932").split("\r\n")
        self.clean = read_measurement(synthetic_csv(), "a.csv")

    def read(self, lines):
        return read_measurement("\r\n".join(lines).encode("cp932"), "a.csv")

    def test_nan_blank_text_and_short_rows(self):
        lines = list(self.lines)
        lines[800] = lines[800].replace(lines[800].split(",")[2].strip(), "NaN")     # file line 801
        cells = lines[200].split(",")
        cells[1] = ""
        lines[200] = ",".join(cells)                                                # line 201: an empty cell
        lines[300] = ",".join(lines[300].split(",")[:2])                            # line 301: two cells only
        lines[400] = lines[400].replace(lines[400].split(",")[3].strip(), "err")    # line 401: text
        m = self.read(lines)
        n = self.clean.data.shape[0]
        self.assertEqual(m.data.shape, (n - 4, 4))                                  # before: 199 of 2054 rows
        self.assertEqual([e["line"] for e in m.read_report["excluded"]], [201, 301, 401, 801])
        self.assertEqual([e["reason"] for e in m.read_report["excluded"]], ["not_numeric", "width", "not_numeric", "not_numeric"])
        # The other rows are exactly the clean file's rows: nothing interpolated or shifted.
        keep = np.ones(n, bool)
        keep[[199, 299, 399, 799]] = False                                         # data rows of those lines
        np.testing.assert_array_equal(m.data, self.clean.data[keep])

    def test_footer_is_counted_not_excluded(self):
        m = self.read(list(self.lines) + ["End of data", "Operator, someone"])
        np.testing.assert_array_equal(m.data, self.clean.data)
        self.assertEqual(m.read_report["excluded"], [])
        self.assertEqual(m.read_report["footer_lines"], 2)

    def test_clean_file_is_read_as_before(self):
        self.assertEqual(self.clean.read_report["excluded"], [])
        self.assertEqual(self.clean.read_report["rows"], self.clean.data.shape[0])
        self.assertEqual(self.clean.read_report["first_line"], 2)


class TemperatureValuesOnHeatingOnly(unittest.TestCase):
    """P0-2: with cooling (or a hold) a temperature picks no single point; values read at it are not given."""

    def setUp(self):
        t = np.arange(0, 140.0, 0.05)
        heat = 20 + 10 * t
        T = np.where(t <= 60, heat, 620 - 10 * (t - 60))                    # 20 → 620 °C, then back to 20 °C
        tg = np.where(t <= 60, -1.0 / (1 + np.exp(-(T - 300) / 20)), -1.0)  # 1 mg lost on heating, none on cooling
        dta = 20 * np.exp(-((t - 40) / 1.0) ** 2) + 0.001 * T
        self.raw = csv_bytes(t, T, dta, tg)
        self.m = read_measurement(self.raw, "c.csv")

    def test_profile(self):
        p = process(self.m, mass_mg=20.0)
        self.assertEqual(p.profile["status"], "cooling")
        heating = heating_segment(self.m.time_min(), self.m.column("temp"))
        self.assertEqual(process(self.m, time_range=heating, mass_mg=20.0).profile["status"], "heating")
        t = np.arange(0, 10, 0.1)
        self.assertEqual(temperature_profile(t, 20 + 5 * t, 5.0)["status"], "heating")
        self.assertEqual(temperature_profile(t, np.minimum(20 + 50 * t, 200), 50.0)["status"], "hold")

    def test_mass_change_is_not_given_when_cooling(self):
        o = PlotOptions(mass_ranges=[(100.0, 500.0)], peak_mode="both")
        cooling = Curve("c", "#000", process(self.m, mass_mg=20.0), key="c.csv")
        self.assertEqual(feature_rows([cooling], o)[0]["steps"], [])          # before: −3.4 % instead of −5.0 %
        self.assertEqual(peak_list(cooling, o), [])
        self.assertEqual(mass_issues(cooling, o), [(100.0, 500.0, "not_heating")])
        heating = heating_segment(self.m.time_min(), self.m.column("temp"))
        c = Curve("c", "#000", process(self.m, time_range=heating, mass_mg=20.0), key="c.csv")
        (lo, hi, d), = feature_rows([c], o)[0]["steps"]
        self.assertAlmostEqual(d, -5.0, delta=0.05)                            # 1 mg of 20 mg


class Checks(unittest.TestCase):
    def test_overall_rule(self):
        ok, caution = Check("a", "PASS", "ok"), Check("b", "CAUTION", "ok")
        missing = Check("c", "NOT EVALUATED", "x", required=True)
        optional = Check("d", "NOT EVALUATED", "x")
        self.assertEqual(overall([ok, optional]), "PASS")
        self.assertEqual(overall([ok, caution]), "CAUTION")
        self.assertEqual(overall([ok, caution, missing]), "NOT EVALUATED")    # never PASS without a required check
        self.assertEqual(overall([missing, Check("e", "WARNING", "x")]), "WARNING")
        self.assertEqual(overall([missing, Check("e", "FAIL", "x")]), "FAIL")

    def test_clean_run_passes_and_values_are_unchanged(self):
        raw = csv_bytes(*noisy_run())
        m, c, result = qc(raw)
        self.assertEqual(result["overall"], "PASS", [(k.code, k.status) for k in result["checks"]])
        # QC only reads: the curve and the numbers are those without QC.
        m2 = read_measurement(raw, "x.csv")
        p2 = process(m2, time_range=heating_segment(m2.time_min(), m2.column("temp")), mass_mg=20.0)
        np.testing.assert_array_equal(c.data.dta, p2.dta)

    def test_peak_signal_to_noise(self):
        for amplitude, noise, expected in ((40.0, 0.05, "PASS"), (0.5, 0.08, "CAUTION"), (0.15, 0.08, "FAIL")):
            _, _, result = qc(csv_bytes(*noisy_run(amplitude, noise)), opts=PlotOptions(peak_prominence=0.01))
            near = [k for k in by_code(result, "peak_sn") if abs(k.target_params["T"] - 1000) < 5]
            self.assertTrue(near, (amplitude, noise))
            self.assertEqual(near[0].status, expected, (amplitude, noise, near[0].params))
        # S/N is the peak height over the noise σ, both known here: the estimate follows the truth, also near
        # the limits, for several noise draws.
        for amplitude, noise in ((40.0, 0.05), (1.0, 0.08), (0.5, 0.08), (0.3, 0.08)):
            for seed in range(5):
                _, _, result = qc(csv_bytes(*noisy_run(amplitude, noise, seed=seed)), opts=PlotOptions(peak_prominence=0.01))
                sn = [k for k in by_code(result, "peak_sn") if abs(k.target_params["T"] - 1000) < 5][0].params["sn"]
                self.assertAlmostEqual(sn / (amplitude / noise), 1.0, delta=0.2, msg=(amplitude, noise, seed))

    def test_noise_sigma(self):
        rng = np.random.default_rng(1)
        y = 0.3 * rng.standard_normal(20000) + np.linspace(0, 50, 20000)    # drift is not noise
        self.assertAlmostEqual(noise_sigma(y), 0.3, delta=0.02)

    def test_onset_robustness_and_smoothing(self):
        _, _, result = qc(csv_bytes(*noisy_run()))
        (robust,) = by_code(result, "onset_robust")
        self.assertEqual(robust.status, "PASS", robust.params)
        self.assertEqual(robust.params["ok"], 9)
        _, _, smoothed = qc(csv_bytes(*noisy_run()), smooth=61)
        (effect,) = by_code(smoothed, "onset_smoothing")
        self.assertIn(effect.status, ("PASS", "CAUTION"))
        self.assertGreaterEqual(effect.params["d"], 0)

    def test_mistyped_mass_fails(self):
        _, _, result = qc(csv_bytes(*noisy_run()), mass=0.5)                  # 1 mg lost of "0.5 mg": −200 %
        (phys,) = by_code(result, "tg_physical")
        self.assertEqual(phys.status, "FAIL")
        self.assertAlmostEqual(phys.params["loss"], 200.0, delta=1.0)
        self.assertEqual(result["overall"], "FAIL")

    def test_missing_mass_is_not_evaluated(self):
        _, _, result = qc(csv_bytes(*noisy_run()), mass=None)
        (m0,) = by_code(result, "m0")
        self.assertEqual((m0.status, m0.required), ("NOT EVALUATED", True))
        self.assertEqual(result["overall"], "NOT EVALUATED")

    def test_excluded_rows_and_time_going_back(self):
        lines = csv_bytes(*noisy_run()).decode("cp932").split("\r\n")
        lines[500] = lines[500].replace(lines[500].split(",")[2].strip(), "NaN")
        _, _, result = qc("\r\n".join(lines).encode("cp932"))
        self.assertEqual(by_code(result, "input_rows")[0].status, "WARNING")
        self.assertEqual(result["overall"], "WARNING")
        t, T, dta, tg = noisy_run()
        t = t.copy()
        t[1000:1010] = t[990:1000]                                            # a repeated block of time
        _, _, result = qc(csv_bytes(t, T, dta, tg))
        (order,) = by_code(result, "time_order")
        self.assertEqual(order.status, "WARNING")
        self.assertGreater(order.params["back"], 0)

    def test_uneven_sampling(self):
        t, T, dta, tg = noisy_run()
        keep = np.ones(len(t), bool)
        keep[1000:1400:2] = False                                             # every other point gone for a while
        _, _, result = qc(csv_bytes(t[keep], T[keep], dta[keep], tg[keep]))
        (sampling,) = by_code(result, "sampling")
        self.assertEqual(sampling.status, "CAUTION")
        self.assertAlmostEqual(sampling.params["ratio"], 2.0, delta=0.01)

    def test_artefact_after_an_exotherm(self):
        _, _, result = qc(csv_bytes(*noisy_run(dip=12.0)), opts=PlotOptions(peak_prominence=0.05))
        flagged = by_code(result, "peak_after_exo")
        self.assertTrue(flagged)
        self.assertAlmostEqual(flagged[0].params["exo"], 1000, delta=3)

    def test_hand_drawn_line_is_caution_with_the_difference(self):
        _, c, result = qc(csv_bytes(*noisy_run()), opts=PlotOptions(peak_mode="onset", show_peaks=True))
        auto_T = [o["T"] for _, _, _, kind, o in peak_list(c, PlotOptions(peak_mode="onset")) if o is not None][0]
        # A baseline y = 0.002 T + 0 and a steeper tangent through (985, 0.002·985 + 0) … crossing at 985 °C.
        y0 = 0.002 * 985
        line = {"base": [900, 0.002 * 900, 985, y0], "tan": [985, y0, 995, y0 + 30]}
        o = PlotOptions(peak_mode="onset", show_peaks=True, overrides={"x.csv": {"exo@1000": line}})
        _, _, result = qc(csv_bytes(*noisy_run()), opts=o)
        (manual,) = by_code(result, "manual")
        self.assertEqual(manual.status, "CAUTION")
        self.assertAlmostEqual(manual.params["man"], 985.0, delta=0.01)
        self.assertAlmostEqual(manual.params["auto"], auto_T, delta=0.01)
        self.assertEqual(by_code(result, "onset_robust"), [])               # hand-drawn: no variation check

    def test_glass_transition_checks(self):
        rng = np.random.default_rng(2)
        t = np.arange(0, 120.0, 0.05)
        T = 20 + 10 * t
        dta = -3.0 / (1 + np.exp(-(T - 950) / 4)) + 0.02 * rng.standard_normal(len(T))   # endothermic step at 950 °C
        _, _, result = qc(csv_bytes(t, T, dta, -0.0 * T), opts=PlotOptions(tg_range=(900.0, 1000.0)))
        (sn,) = by_code(result, "tg_sn")
        (robust,) = by_code(result, "tg_robust")
        self.assertEqual(sn.status, "PASS")
        self.assertEqual(robust.params["n"], 15)
        _, _, flat = qc(csv_bytes(t, T, 0.02 * rng.standard_normal(len(T)), -0.0 * T), opts=PlotOptions(tg_range=(900.0, 1000.0)))
        self.assertEqual([k.status for k in by_code(flat, "tg_found")], ["NOT EVALUATED"])

    def test_degenerate_data_do_not_break_the_checks(self):
        t = np.arange(0, 1.0, 0.1)                                            # 10 points, constant temperature, flat DTA
        _, _, result = qc(csv_bytes(t, np.full(10, 25.0), np.zeros(10), np.zeros(10)), segment="all")
        self.assertIn(result["overall"], tgdta_qc.STATUSES)
        self.assertEqual(by_code(result, "temp_direction")[0].status, "CAUTION")

    def test_smoothing_span(self):
        t = np.arange(0, 10, 0.05)
        span = smoothing_span(31, t, 15.0)
        self.assertEqual(span["points"], 31)
        self.assertAlmostEqual(span["min"], 1.5)
        self.assertAlmostEqual(span["K"], 22.5)
        self.assertEqual(smoothing_span(30, t, 15.0)["points"], 31)          # made odd, as the smoothing does
        self.assertIsNone(smoothing_span(0, t, 15.0))
        self.assertEqual(smoothing_span(0, t, 15.0, minimum=5)["points"], 5)  # DTG: at least 5 points


class Texts(unittest.TestCase):
    def test_every_text_in_four_languages_with_the_same_placeholders(self):
        names = lambda s: sorted({f for _, f, _, _ in string.Formatter().parse(s) if f})
        for key, fields in TEXT.items():
            for field_name, values in fields.items():
                self.assertEqual(len(values), 4, (key, field_name))
                self.assertTrue(all(values), (key, field_name))
                self.assertEqual({tuple(names(v)) for v in values}, {tuple(names(values[1]))}, (key, field_name))

    def test_every_check_is_written_out(self):
        results = [qc(csv_bytes(*noisy_run()))[2], qc(csv_bytes(*noisy_run(dip=12.0)), mass=0.5, smooth=31,
                                                       opts=PlotOptions(peak_prominence=0.05, tg_range=(900.0, 990.0),
                                                                        mass_ranges=[(10.0, 2000.0)]))[2],
                   qc(csv_bytes(*noisy_run()), mass=None, segment="all")[2]]
        for result in results:
            for chk in result["checks"]:
                for lang in ("ja", "en", "vi", "th"):
                    texts = check_texts(chk, lang)
                    self.assertTrue(texts["item"] and texts["reason"] and texts["value"], (chk.code, chk.reason, lang))
                    self.assertFalse(re.search(r"\{[a-zA-Z_]", " ".join(texts.values())), (chk.code, lang, texts))

    def test_report(self):
        raw = csv_bytes(*noisy_run())
        m, c, result = qc(raw)
        result["processing"] = tgdta_qc.processing_steps(m, c.data, segment="heating", time_range=None, mass_mg=20.0,
                                                         mass_source="entered", tg_ref_temp=None, dta_flip=False,
                                                         dta_zero=False, dta_per_mass=False, tg_smooth=0, dta_smooth=0,
                                                         dtg_smooth=31)
        row = feature_rows([c], PlotOptions(peak_mode="both"))[0]
        js, cs = build_report("test 0000000", {"segment": "heating", "x": float("nan")},
                              [{"label": "x", "file": "x.csv", "raw": raw, "measurement": m, "row": row, "result": result}],
                              "2026-01-01T00:00:00+09:00")
        report = json.loads(js)                                               # strict JSON (no NaN)
        self.assertEqual(report["version"], "test 0000000")
        self.assertIsNone(report["settings"]["x"])
        self.assertEqual(len(report["samples"][0]["file_sha256"]), 64)
        self.assertEqual(report["samples"][0]["overall"], "PASS")
        self.assertTrue(all("reason_text" in k for k in report["samples"][0]["checks"]))
        self.assertIn("check", cs.decode("utf-8-sig"))


class RunningApp(unittest.TestCase):
    def test_app_reports_left_out_rows_and_cooling(self):
        from streamlit.testing.v1 import AppTest
        lines = synthetic_csv().decode("cp932").split("\r\n")
        lines[800] = lines[800].replace(lines[800].split(",")[2].strip(), "NaN")
        at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=180)
        at.run()
        at.session_state["_files"] = {"bad.csv": "\r\n".join(lines).encode("cp932"), "ok.csv": synthetic_csv()}
        at.session_state["_recolor"] = True
        at.run()
        at.session_state["mass_0"], at.session_state["mass_1"] = 20.0, 20.0
        at.session_state["mass_ranges"] = "100-600"
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("801" in w.value for w in at.warning))                     # the line left out is named
        self.assertTrue(any("WARNING" in c.value for c in at.caption))                  # its QC mark in the sample panel
        self.assertTrue(any("WARNING" in w.value for w in at.sidebar.warning))          # next to the save buttons
        at.session_state["segment"] = "all"                                              # the hold at the top is included
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Tp・Tx・Tg" in w.value for w in at.warning))
        for lang in ("English", "Tiếng Việt", "ไทย"):
            at.session_state["lang_radio"] = lang
            at.run()
            self.assertFalse(at.exception, lang)


if __name__ == "__main__":
    unittest.main()
