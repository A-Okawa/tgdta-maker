"""Quality checks of a TG-DTA analysis: how far each number the app gives can be trusted, and why.

Each check has a status:

    PASS           meets the criterion
    CAUTION        usable, with the reason in mind
    WARNING        the number may be wrong; check before use
    FAIL           the number cannot be used as it is
    NOT EVALUATED  the check could not be made (the reason says why)

The overall status of a sample is its worst check. A required check that could not be made (input
completeness, time order, m₀ when TG is drawn in %) makes the overall status NOT EVALUATED unless a
WARNING or FAIL is already there; it is never counted as PASS.

The checks look at the data and the constructions; they change nothing. The thresholds marked
provisional in THRESHOLDS are working values, not values from a standard.

Nothing here depends on Streamlit. The texts in four languages are in TEXT, keyed by check code.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import peak_prominences, peak_widths

from tgdta_io import COOLING_TOLERANCE_K, HOLD_TOLERANCE_MIN, _odd_window, glass_transition, peak_onset, smooth
from tgdta_plot import Curve, PlotOptions, glass_info, mass_issues, peak_inputs, peak_list, series

STATUSES = ("PASS", "CAUTION", "WARNING", "FAIL", "NOT EVALUATED")
ICONS = {"PASS": "🟢", "CAUTION": "🟡", "WARNING": "🟠", "FAIL": "🔴", "NOT EVALUATED": "⚪"}
_RANK = {"PASS": 0, "CAUTION": 1, "NOT EVALUATED": 2, "WARNING": 3, "FAIL": 4}

THRESHOLDS = {
    # Detection and quantitation limits, 3σ and 10σ (the IUPAC / ICH Q2 convention).
    "sn_fail": 3.0,
    "sn_pass": 10.0,
    # Provisional: spread of an onset when the construction is varied (no standard value exists).
    "robust_pass_K": 2.0,
    "robust_caution_K": 5.0,
    # Provisional: change of an onset when the DTA is not smoothed.
    "smoothing_pass_K": 1.0,
    # Provisional: largest / smallest time step.
    "sampling_ratio": 1.5,
    # Physical limit: a sample cannot lose more than its mass.
    "tg_loss_fail_pct": 100.0,
    # A construction counts as unstable when fewer than this fraction of the varied ones can be drawn.
    "min_variant_fraction": 0.5,
    # Endothermic peak this many exothermic half-widths after an exothermic peak: possibly its return to the baseline.
    "after_exo_widths": 2.0,
}
PROVISIONAL = ("robust_pass_K", "robust_caution_K", "smoothing_pass_K", "sampling_ratio", "after_exo_widths")

# Varied constructions. Tx: foot at 15 / 10 / 5 % of the peak height × baseline 6–3, 5–2, 4–1.5 widths before it.
ONSET_VARIANTS = [(f, b) for f in (0.85, 0.9, 0.95) for b in ((6.0, 3.0), (5.0, 2.0), (4.0, 1.5))]
# Tg: baseline 30 / 40 / 50 % of the range × the range as given, either end moved in by 5 %, both in, both out.
TG_FRACTIONS = (0.3, 0.4, 0.5)
TG_SHIFTS = ((0.0, 0.0), (0.05, 0.0), (0.0, -0.05), (0.05, -0.05), (-0.05, 0.05))


@dataclass
class Check:
    code: str                    # what is checked (TEXT key)
    status: str                  # one of STATUSES
    reason: str                  # TEXT key of the reason under the code ("ok", "bad", …)
    params: dict = field(default_factory=dict)      # numbers for the texts
    target: str = ""             # TEXT key of what the check is about ("", "target_exo", …)
    target_params: dict = field(default_factory=dict)
    required: bool = False
    value: str = "value"         # TEXT key of the value text


def overall(checks: list[Check]) -> str:
    """The sample's status: the worst check; a required check not made gives NOT EVALUATED (never PASS)."""
    made = [c.status for c in checks if c.status != "NOT EVALUATED"]
    worst = max(made, key=_RANK.get) if made else "PASS"
    if any(c.required and c.status == "NOT EVALUATED" for c in checks) and _RANK[worst] < _RANK["WARNING"]:
        return "NOT EVALUATED"
    if not made and checks:
        return "NOT EVALUATED"
    return worst


def noise_sigma(y: np.ndarray) -> float:
    """White-noise σ of a signal from its point-to-point differences (robust: 1.4826 × MAD / √2).
    Slow baseline drift barely enters a difference of neighbours, so it is not counted as noise."""
    d = np.diff(np.asarray(y, float))
    d = d[np.isfinite(d)]
    if d.size < 5:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2))


def _climb(y: np.ndarray, k: int) -> int:
    """From k, uphill to the local maximum it belongs to (-1 when it runs off an end)."""
    j = int(k)
    while 0 < j < len(y) - 1:
        if y[j + 1] > y[j] and y[j + 1] >= y[j - 1]:
            j += 1
        elif y[j - 1] > y[j]:
            j -= 1
        else:
            return j
    return -1


def peak_height(y: np.ndarray, k: int, wlen: int | None) -> float:
    """Height of the peak at ``k`` above its local baseline (prominence within ``wlen`` points), read on the
    signal smoothed over half the peak's half-width (at least 1 % of the points), so that a noise spike on
    top of a small peak does not add to its height and raise its S/N."""
    w1 = max(5, (len(y) // 100) | 1)
    y1 = smooth(y, w1)
    lo, hi = max(0, k - w1), min(len(y) - 1, k + w1)
    j1 = _climb(y1, lo + int(np.argmax(y1[lo:hi + 1])))        # the highest point within one window, then uphill
    if j1 < 0:
        return 0.0
    fwhm = float(peak_widths(y1, [j1], rel_height=0.5, prominence_data=peak_prominences(y1, [j1], wlen=wlen))[0][0])
    w = max(w1, int(fwhm / 2) | 1)
    ys = smooth(y, w) if w != w1 else y1
    lo, hi = max(0, j1 - w), min(len(y) - 1, j1 + w)
    j = _climb(ys, lo + int(np.argmax(ys[lo:hi + 1])))
    return float(peak_prominences(ys, [j], wlen=wlen)[0][0]) if j >= 0 else 0.0


def _sn_status(sn: float) -> str:
    if not math.isfinite(sn):
        return "PASS"                         # noise below the resolution of the record
    if sn < THRESHOLDS["sn_fail"]:
        return "FAIL"
    return "CAUTION" if sn < THRESHOLDS["sn_pass"] else "PASS"


def _spread_status(spread: float) -> str:
    if spread <= THRESHOLDS["robust_pass_K"]:
        return "PASS"
    return "CAUTION" if spread <= THRESHOLDS["robust_caution_K"] else "WARNING"


def smoothing_span(points: int, t_min: np.ndarray, rate: float, minimum: int = 0) -> dict | None:
    """The Savitzky–Golay window actually used for ``points`` (made odd, at least ``minimum``) as time and
    temperature: {"points", "min", "K"}; None for no smoothing."""
    n = len(t_min) if t_min is not None else 0
    w = _odd_window(max(int(points or 0), minimum), n, 2) if (points and points > 1) or minimum else 0
    if not w or n < 2:
        return None
    dt = float(np.median(np.diff(t_min)))
    span = (w - 1) * dt
    return {"points": int(w), "min": span, "K": span * rate if math.isfinite(rate) else float("nan")}


# ------------------------------------------------------------------------------------------------ checks
def _input_checks(m) -> list[Check]:
    rep = m.read_report or {}
    excluded = rep.get("excluded") or []
    lines = ", ".join(str(e["line"]) for e in excluded[:10]) + (" …" if len(excluded) > 10 else "")
    params = {"rows": int(rep.get("rows", m.data.shape[0])), "excluded": len(excluded), "lines": lines,
              "footer": int(rep.get("footer_lines") or 0)}
    out = [Check("input_rows", "WARNING" if excluded else "PASS", "bad" if excluded else "ok", params, required=True)]

    t = m.time_min()
    if t is None:
        out.append(Check("time_order", "NOT EVALUATED", "no_time", {}, required=True, value="value_none"))
        out.append(Check("sampling", "NOT EVALUATED", "no_time", {}, value="value_none"))
        return out
    dt = np.diff(t)
    back, dup = int(np.sum(dt < 0)), int(np.sum(dt == 0))
    if m.source == "rigaku-mes":
        assumed = "mes_time_assumed" in m.notes
        out.append(Check("time_order", "CAUTION" if assumed else "PASS", "mes_assumed" if assumed else "mes",
                         {"back": back, "dup": dup}, required=True))
        out.append(Check("sampling", "NOT EVALUATED", "mes", {}, value="value_none"))
        return out
    bad = back or dup
    first = int(np.nonzero(dt <= 0)[0][0]) + 2 if bad else 0
    out.append(Check("time_order", "WARNING" if bad else "PASS", "bad" if bad else "ok",
                     {"back": back, "dup": dup, "first": first}, required=True))
    pos = dt[dt > 0]
    if pos.size < 2:
        out.append(Check("sampling", "NOT EVALUATED", "no_time", {}, value="value_none"))
    else:
        ratio = float(pos.max() / pos.min())
        out.append(Check("sampling", "PASS" if ratio <= THRESHOLDS["sampling_ratio"] else "CAUTION",
                         "ok" if ratio <= THRESHOLDS["sampling_ratio"] else "bad",
                         {"min": 60 * float(pos.min()), "max": 60 * float(pos.max()), "ratio": ratio}))
    return out


def _mass_checks(p, mass_mg, mass_source, tg_mode) -> list[Check]:
    out = []
    if p.tg is None:
        return out
    wants_pct = tg_mode != "change_mg"
    if wants_pct and p.tg_unit == "mg" and "no_mass" in p.notes:
        out.append(Check("m0", "NOT EVALUATED", "missing", {}, required=True, value="value_none"))
    elif wants_pct and p.tg_unit == "%" and mass_mg:
        out.append(Check("m0", "PASS", "ok", {"m": float(mass_mg), "source": mass_source}, required=True))
    if p.tg_unit == "%":
        change = p.tg if tg_mode == "change_pct" else p.tg - 100.0
        loss = float(-np.nanmin(change))
        fail = loss > THRESHOLDS["tg_loss_fail_pct"]
        out.append(Check("tg_physical", "FAIL" if fail else "PASS", "bad" if fail else "ok", {"loss": loss}))
    else:
        out.append(Check("tg_physical", "NOT EVALUATED", "mg", {}, value="value_none"))
    return out


def _temperature_check(p) -> list[Check]:
    prof = p.profile or {}
    status = prof.get("status", "heating")
    params = {"drop": float(prof.get("drop_K", 0.0)), "after": float(prof.get("after_ramp_min", 0.0)),
              "cool": COOLING_TOLERANCE_K, "hold": HOLD_TOLERANCE_MIN}
    if status == "heating":
        return [Check("temp_direction", "PASS", "ok", params, value="value_heating")]
    return [Check("temp_direction", "CAUTION", "bad", params, value=f"value_{status}"),
            Check("temp_values", "NOT EVALUATED", "not_heating", {}, value="value_none")]


def _onset_variants(T, signed, k, kind):
    values = []
    for foot, widths in ONSET_VARIANTS:
        o = peak_onset(T, signed, k, kind, foot_height=foot, baseline_widths=widths)
        if o is not None:
            values.append(o["x"])
    return values


def _peak_checks(c: Curve, c_raw: Curve, opts: PlotOptions, dta_smooth: int) -> list[Check]:
    """S/N of each peak, artefacts after an exotherm, and robustness, smoothing and hand-drawn lines of each onset."""
    o = PlotOptions(**{**opts.__dict__, "peak_kind": "both", "peak_mode": "both"})
    inputs = peak_inputs(c, o)
    if inputs is None:
        return []
    idx, signed, found = inputs
    T = c.data.T[idx]
    raw = series(c_raw, "dta")
    raw_signed = (raw[idx] if opts.exo_up else -raw[idx]) if raw is not None else signed
    sigma = noise_sigma(raw_signed)
    listed = {(round(T_peak, 6), kind): onset for _, _, T_peak, kind, onset in peak_list(c, o)}
    auto = {(round(T_peak, 6), kind): onset
            for _, _, T_peak, kind, onset in peak_list(c, PlotOptions(**{**o.__dict__, "overrides": {}}))}
    step = float(np.median(np.diff(T))) if len(T) > 1 else 1.0
    # The same local window as the onset construction (10 % of the x span): a peak's width is that of the
    # peak itself, not of the whole curve around it.
    wlen = max(5, int(2 * 0.1 * float(T[-1] - T[0]) / max(step, 1e-9)) | 1) if len(T) > 1 else None
    out = []
    exo_peaks = []
    for k, kind in found:
        T_peak = float(T[k])
        target = "target_exo" if kind == "exo" else "target_endo"
        tp = {"T": T_peak}
        y = signed if kind == "exo" else -signed
        height = peak_height(raw_signed if kind == "exo" else -raw_signed, k, wlen)
        sn = height / sigma if sigma and math.isfinite(sigma) and sigma > 0 else float("inf")
        status = _sn_status(sn)
        out.append(Check("peak_sn", status, {"PASS": "ok", "CAUTION": "low", "FAIL": "bad"}[status] if math.isfinite(sn)
                         else "resolution", {"sn": sn}, target, tp, value="value" if math.isfinite(sn) else "value_resolution"))
        if kind == "exo":
            local = peak_prominences(y, [k], wlen=wlen)
            width = float(peak_widths(y, [k], rel_height=0.5, prominence_data=local)[0][0]) * step
            exo_peaks.append((T_peak, width))
        elif exo_peaks:
            Te, w = exo_peaks[-1]            # the exothermic peak just before this one
            if 0 < T_peak - Te <= THRESHOLDS["after_exo_widths"] * w:
                out.append(Check("peak_after_exo", "CAUTION", "bad", {"exo": Te, "d": T_peak - Te}, target, tp))

        onset = listed.get((round(T_peak, 6), kind))
        if onset is None:
            continue
        otarget = "target_tx" if kind == "exo" else "target_onset_endo"
        if onset.get("manual"):
            a = auto.get((round(T_peak, 6), kind))
            if a is not None:
                out.append(Check("manual", "CAUTION", "ok", {"man": onset["T"], "auto": a["T"], "d": onset["T"] - a["T"]},
                                 otarget, tp))
            else:
                out.append(Check("manual", "CAUTION", "ok", {"man": onset["T"]}, otarget, tp, value="value_no_auto"))
            continue
        values = _onset_variants(T, signed, k, kind)
        made, n = len(values), len(ONSET_VARIANTS)
        if made < THRESHOLDS["min_variant_fraction"] * n:
            out.append(Check("onset_robust", "CAUTION", "unstable", {"ok": made, "n": n,
                                                                    "spread": float(np.ptp(values)) if values else float("nan")},
                             otarget, tp, value="value_unstable"))
        else:
            spread = float(np.ptp(values))
            status = _spread_status(spread)
            out.append(Check("onset_robust", status, "ok" if status == "PASS" else "bad",
                             {"spread": spread, "ok": made, "n": n, "lo": min(values), "hi": max(values)}, otarget, tp))
        if dta_smooth and dta_smooth > 1:
            # The same peak in the unsmoothed DTA: its extreme within one smoothing window of the smoothed one.
            reach = int(dta_smooth)
            lo_k, hi_k = max(1, k - reach), min(len(T) - 2, k + reach)
            seg = raw_signed[lo_k:hi_k + 1] if kind == "exo" else -raw_signed[lo_k:hi_k + 1]
            k_raw = lo_k + int(np.argmax(seg)) if seg.size else k
            r = peak_onset(T, raw_signed, k_raw, kind)
            if r is None:
                out.append(Check("onset_smoothing", "CAUTION", "no_raw", {"pts": int(dta_smooth)}, otarget, tp,
                                 value="value_no_raw"))
            else:
                d = abs(onset["T"] - r["x"])
                ok = d <= THRESHOLDS["smoothing_pass_K"]
                out.append(Check("onset_smoothing", "PASS" if ok else "CAUTION", "ok" if ok else "bad",
                                 {"d": d, "raw": r["x"], "pts": int(dta_smooth)}, otarget, tp))
    return out


def _tg_checks(c: Curve, opts: PlotOptions) -> list[Check]:
    if not opts.tg_range or c.data.dta is None:
        return []
    lo, hi = opts.tg_range
    if not (c.data.profile or {}).get("status", "heating") == "heating":
        return [Check("tg_found", "NOT EVALUATED", "not_heating", {"lo": lo, "hi": hi}, "target_tg", value="value_none")]
    g = glass_info(c, opts)
    if g is None:
        return [Check("tg_found", "NOT EVALUATED", "none", {"lo": lo, "hi": hi}, "target_tg", value="value_none")]
    out = []
    if g.get("manual"):
        a = glass_info(c, PlotOptions(**{**opts.__dict__, "overrides": {}}))
        if a is not None:
            out.append(Check("manual", "CAUTION", "ok", {"man": g["T"], "auto": a["T"], "d": g["T"] - a["T"]}, "target_tg"))
        else:
            out.append(Check("manual", "CAUTION", "ok", {"man": g["T"]}, "target_tg", value="value_no_auto"))
        return out
    y = c.data.dta
    base = glass_transition(c.data.T, y, lo, hi, exo_up=opts.exo_up)
    if base is not None:
        sn = abs(base["depth"]) / base["noise"] if base["noise"] > 0 else float("inf")
        status = _sn_status(sn)
        out.append(Check("tg_sn", status, {"PASS": "ok", "CAUTION": "low", "FAIL": "bad"}[status] if math.isfinite(sn)
                         else "resolution", {"sn": sn}, "target_tg",
                         value="value" if math.isfinite(sn) else "value_resolution"))
    span = hi - lo
    values = []
    for fraction in TG_FRACTIONS:
        for a, b in TG_SHIFTS:
            v = glass_transition(c.data.T, y, lo + a * span, hi + b * span, exo_up=opts.exo_up, baseline_fraction=fraction)
            if v is not None:
                values.append(v["x"])
    n = len(TG_FRACTIONS) * len(TG_SHIFTS)
    if len(values) < THRESHOLDS["min_variant_fraction"] * n:
        out.append(Check("tg_robust", "CAUTION", "unstable", {"ok": len(values), "n": n,
                                                             "spread": float(np.ptp(values)) if values else float("nan")},
                         "target_tg", value="value_unstable"))
    else:
        spread = float(np.ptp(values))
        status = _spread_status(spread)
        out.append(Check("tg_robust", status, "ok" if status == "PASS" else "bad",
                         {"spread": spread, "ok": len(values), "n": n, "lo": min(values), "hi": max(values)}, "target_tg"))
    return out


def _mass_range_checks(c: Curve, opts: PlotOptions) -> list[Check]:
    return [Check("mass_range", "NOT EVALUATED", reason, {"lo": lo, "hi": hi}, "target_range", {"lo": lo, "hi": hi},
                  value="value_none") for lo, hi, reason in mass_issues(c, opts)]


def processing_steps(m, p, *, segment, time_range, mass_mg, mass_source, tg_ref_temp, dta_flip, dta_zero,
                     dta_per_mass, tg_smooth, dta_smooth, dtg_smooth) -> list[tuple[str, dict]]:
    """What the app did to the data before drawing it: [(TEXT key, params)]."""
    steps = []
    n = len(p.t)
    if segment == "heating":
        steps.append(("proc_segment_heating", {"t0": float(p.t[0]), "t1": float(p.t[-1]), "T0": float(p.T[0]),
                                               "T1": float(p.T[-1]), "n": n}))
    elif segment == "custom" and time_range:
        steps.append(("proc_segment_custom", {"t0": float(time_range[0]), "t1": float(time_range[1]), "n": n}))
    else:
        steps.append(("proc_segment_all", {"n": n}))
    rep = m.read_report or {}
    if rep.get("excluded"):
        lines = ", ".join(str(e["line"]) for e in rep["excluded"][:10]) + (" …" if len(rep["excluded"]) > 10 else "")
        steps.append(("proc_excluded", {"n": len(rep["excluded"]), "lines": lines}))
    if rep.get("footer_lines"):
        steps.append(("proc_footer", {"n": int(rep["footer_lines"])}))
    if m.units.get("temp") == "K":
        steps.append(("proc_temp_K", {}))
    if "no_time" in p.notes:
        steps.append(("proc_no_time", {}))
    elif m.source == "rigaku-mes":
        steps.append(("proc_time_assumed" if "mes_time_assumed" in m.notes else "proc_time_mes", {}))
    elif m.units.get("time") == "s":
        steps.append(("proc_time_s", {}))
    if p.tg is not None:
        if mass_mg:
            steps.append(("proc_m0", {"m": float(mass_mg), "source": mass_source}))
        else:
            steps.append(("proc_m0_none", {}))
        if tg_ref_temp is None:
            steps.append(("proc_ref_first", {}))
        elif "ref_not_evaluated" in p.notes:
            steps.append(("proc_ref_fallback_heating", {"T": float(tg_ref_temp)}))
        elif "ref_out_of_range" in p.notes:
            steps.append(("proc_ref_fallback_range", {"T": float(tg_ref_temp)}))
        else:
            steps.append(("proc_ref_temp", {"T": float(tg_ref_temp)}))
    if p.dta is not None:
        if dta_flip:
            steps.append(("proc_flip", {}))
        if dta_zero:
            steps.append(("proc_zero", {}))
        if dta_per_mass and "no_mass_dta" not in p.notes:
            steps.append(("proc_per_mass", {}))
    t = None if "no_time" in p.notes else p.t
    for what, points, minimum, present in (("TG", tg_smooth, 0, p.tg is not None), ("DTA", dta_smooth, 0, p.dta is not None),
                                            ("DTG", dtg_smooth, 5, p.dtg is not None)):
        if not present:
            continue
        span = smoothing_span(points, t, p.rate, minimum) if t is not None else None
        if span is None and (points and points > 1 or minimum):
            steps.append(("proc_smooth_points", {"what": what, "pts": max(int(points or 0), minimum)}))
        elif span is None:
            steps.append(("proc_smooth_none", {"what": what}))
        else:
            steps.append(("proc_smooth", {"what": what, "pts": span["points"], "min": span["min"], "K": span["K"]}))
    return steps


def evaluate(m, c: Curve, c_raw: Curve, opts: PlotOptions, *, mass_mg, mass_source, tg_mode, dta_smooth) -> dict:
    """All checks of one sample: {"checks": [Check], "overall": status}.

    ``c`` is the curve as drawn; ``c_raw`` the same with the DTA not smoothed (the noise is measured on it).
    ``opts`` the figure's options: the peaks are those of the numbers table (both kinds, within the x range).
    """
    checks = _input_checks(m)
    checks += _temperature_check(c.data)
    checks += _mass_checks(c.data, mass_mg, mass_source, tg_mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # scipy: "some peaks have a prominence of 0" for a varied construction
        checks += _peak_checks(c, c_raw, opts, dta_smooth)
        checks += _tg_checks(c, opts)
    checks += _mass_range_checks(c, opts)
    return {"checks": checks, "overall": overall(checks)}


# ------------------------------------------------------------------------------------------------ report
def _clean_json(o):
    """JSON has no NaN or infinity: write null. Arrays become lists, other objects their text."""
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean_json(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o if o is None or isinstance(o, (str, int, float, bool)) else str(o)


def build_report(version: str, settings: dict, samples: list[dict], generated: str) -> tuple[bytes, bytes]:
    """The QC report as JSON (codes, numbers and English texts) and CSV (one row per setting, step and check).

    ``samples``: [{"label", "file", "raw" (file bytes), "measurement", "row" (tgdta_plot.feature_rows), "result"
    (evaluate, with "processing")}].
    """
    import csv
    import hashlib
    import io
    import json
    out_samples = []
    for s in samples:
        m, row, result = s["measurement"], s["row"], s["result"]
        out_samples.append({
            "sample": s["label"], "file": s["file"], "file_sha256": hashlib.sha256(s["raw"]).hexdigest(),
            "file_bytes": len(s["raw"]), "source": m.source, "columns": m.columns, "units": m.units, "read_report": m.read_report,
            "processing": [{"step": k, **prm, "text": text(k, "text", "en", prm)} for k, prm in result.get("processing", [])],
            "values": {"heating_rate_K_per_min": row["rate"], "end_temperature_C": row["T_end"],
                       "total_mass_change": row["tg_total"], "tg_unit": row["tg_unit"],
                       "peaks_C": [{"T": T, "kind": k} for T, k in row["peaks"]],
                       "onsets_C": [{"T": T, "kind": k, "manual": (T, k) in row["manual"]} for T, k in row["onsets"]],
                       "glass_transition_C": ({"onset": row["tg"][0], "inflection": row["tg"][1], "manual": row["tg_manual"]}
                                              if row["tg"] else None),
                       "mass_change_by_range": [{"from_C": lo, "to_C": hi, "change": d} for lo, hi, d in row["steps"]]},
            "overall": result["overall"], "checks": [check_record(chk) for chk in result["checks"]],
        })
    report = {"app": "TG-DTA Maker", "version": version, "generated": generated, "thresholds": THRESHOLDS,
              "provisional_thresholds": list(PROVISIONAL), "settings": settings, "samples": out_samples}
    json_bytes = json.dumps(_clean_json(report), ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["section", "sample", "item", "target", "value", "criterion", "status", "reason"])
    writer.writerow(["app", "", "version", "", version, "", "", ""])
    writer.writerow(["app", "", "generated", "", generated, "", "", ""])
    for key in sorted(settings):
        writer.writerow(["setting", "", key, "", json.dumps(_clean_json(settings[key]), ensure_ascii=False), "", "", ""])
    for s in out_samples:
        writer.writerow(["file", s["sample"], s["file"], "", s["file_sha256"], "", "", ""])
        for step in s["processing"]:
            writer.writerow(["processing", s["sample"], step["step"], "", step["text"], "", "", ""])
        writer.writerow(["overall", s["sample"], "overall", "", "", "", s["overall"], ""])
        for chk in s["checks"]:
            writer.writerow(["check", s["sample"], chk["item_text"], chk["target_text"], chk["value_text"],
                             chk["criterion_text"], chk["status"], chk["reason_text"]])
    return json_bytes, buffer.getvalue().encode("utf-8-sig")


# ------------------------------------------------------------------------------------------------ text
def _fmt(template: str, params: dict) -> str:
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template


LANG_INDEX = {"ja": 0, "en": 1, "vi": 2, "th": 3}


def text(key: str, field_name: str, lang: str = "en", params: dict | None = None) -> str:
    """TEXT[key][field_name] in ``lang`` (English when missing), with ``params`` filled in."""
    entry = TEXT.get(key, {}).get(field_name)
    if entry is None:
        return ""
    s = entry[LANG_INDEX.get(lang, 1)] or entry[1]
    return _fmt(s, {**(params or {}), **_source_words(params or {}, lang)})


def _source_words(params: dict, lang: str) -> dict:
    src = params.get("source")
    if src in ("file", "entered"):
        return {"source": text("words", src, lang)}
    return {}


def check_texts(c: Check, lang: str = "en") -> dict:
    """The check as text: item, target, value, criterion, status, reason, action."""
    p = {**THRESHOLDS, **c.params}
    return {"item": text(c.code, "item", lang, p), "target": text(c.target, "text", lang, c.target_params) if c.target else "",
            "value": text(c.code, c.value, lang, p), "criterion": text(c.code, "criterion", lang, p), "status": c.status,
            "reason": text(c.code, c.reason, lang, p),
            "action": text(c.code, f"action_{c.reason}", lang, p) if c.status != "PASS" else ""}


def check_record(c: Check) -> dict:
    """A check for the JSON report: the codes and numbers, and the English texts."""
    return {**asdict(c), **{f"{k}_text": v for k, v in check_texts(c, "en").items() if k != "status"}}


# (ja, en, vi, th). Placeholders are the same in every language.
TEXT = {
    "words": {
        "file": ("ファイルの値", "value in the file", "giá trị trong tệp", "ค่าในไฟล์"),
        "entered": ("手入力", "entered by hand", "nhập tay", "ป้อนเอง"),
    },
    "input_rows": {
        "item": ("入力の完全性", "Input completeness", "Tính đầy đủ của dữ liệu vào", "ความครบถ้วนของข้อมูลนำเข้า"),
        "criterion": ("除外した行が 0 行なら PASS、1 行以上なら WARNING", "PASS if no row was left out, WARNING otherwise",
                      "PASS nếu không bỏ dòng nào, ngược lại WARNING", "PASS ถ้าไม่มีแถวที่ถูกตัดออก ไม่เช่นนั้น WARNING"),
        "value": ("{rows} 行を読み込み・{excluded} 行を除外", "{rows} rows read, {excluded} left out",
                  "Đọc {rows} dòng, bỏ {excluded} dòng", "อ่าน {rows} แถว ตัดออก {excluded} แถว"),
        "ok": ("表の途中で除外した行はありません。", "No row inside the table was left out.",
               "Không có dòng nào trong bảng bị bỏ.", "ไม่มีแถวใดในตารางถูกตัดออก"),
        "bad": ("数値でない値（空欄・NaN・文字）や列の数が違う行を除外しました（行 {lines}）。値の補間はしていません。",
                "Rows with a value that is not a number (empty, NaN, text) or another number of cells were left out "
                "(lines {lines}). Nothing was interpolated.",
                "Đã bỏ các dòng có giá trị không phải số (trống, NaN, chữ) hoặc số ô khác (dòng {lines}). Không nội suy.",
                "ตัดแถวที่มีค่าที่ไม่ใช่ตัวเลข (ว่าง NaN ข้อความ) หรือจำนวนช่องต่างออกไป (แถว {lines}) ไม่ได้ประมาณค่าแทน"),
        "action_bad": ("元のファイルの該当行を確かめてください。", "Check those lines in the original file.",
                       "Hãy kiểm tra các dòng đó trong tệp gốc.", "โปรดตรวจแถวเหล่านั้นในไฟล์ต้นฉบับ"),
    },
    "time_order": {
        "item": ("時間の向き", "Time order", "Thứ tự thời gian", "ลำดับเวลา"),
        "criterion": ("時間の逆行・重複が 0 なら PASS、あれば WARNING", "PASS with no step back or repeat in time, WARNING otherwise",
                      "PASS nếu thời gian không lùi hay lặp, ngược lại WARNING", "PASS ถ้าเวลาไม่ย้อนหรือซ้ำ ไม่เช่นนั้น WARNING"),
        "value": ("逆行 {back}・重複 {dup}", "{back} back, {dup} repeated", "lùi {back}, lặp {dup}", "ย้อน {back} ซ้ำ {dup}"),
        "value_none": ("—", "—", "—", "—"),
        "ok": ("時間は単調に増えています。", "Time increases throughout.", "Thời gian tăng liên tục.", "เวลาเพิ่มขึ้นตลอด"),
        "bad": ("時間が逆行または重複しています（{first} 点目から）。ファイルの継ぎ足しなどが考えられます。",
                "Time steps back or repeats (from point {first}), e.g. files joined together.",
                "Thời gian lùi hoặc lặp (từ điểm {first}), ví dụ do ghép tệp.",
                "เวลาย้อนหรือซ้ำ (ตั้งแต่จุดที่ {first}) เช่น จากการต่อไฟล์"),
        "action_bad": ("元のファイルの時間の列を確かめてください。DTG と昇温速度は時間で微分しています。",
                       "Check the time column of the file; DTG and the heating rate are derivatives over time.",
                       "Hãy kiểm tra cột thời gian; DTG và tốc độ gia nhiệt là đạo hàm theo thời gian.",
                       "โปรดตรวจคอลัมน์เวลา DTG และอัตราให้ความร้อนเป็นอนุพันธ์ตามเวลา"),
        "no_time": ("時間の列がありません。昇温速度と DTG（min⁻¹）は評価できません。",
                    "No time column: the heating rate and DTG per minute cannot be evaluated.",
                    "Không có cột thời gian: không đánh giá được tốc độ gia nhiệt và DTG theo phút.",
                    "ไม่มีคอลัมน์เวลา จึงประเมินอัตราให้ความร้อนและ DTG ต่อนาทีไม่ได้"),
        "action_no_time": ("「列の対応」で時間の列を選べるか確かめてください。", "See whether a time column can be chosen under 'Columns'.",
                           "Xem có thể chọn cột thời gian trong 'Cột' không.", "ดูว่าเลือกคอลัมน์เวลาใน 'คอลัมน์' ได้หรือไม่"),
        "mes": ("装置記録の測定時間の範囲を点数で割って作った時間です（等間隔）。",
                "Time made by dividing the recorded time range over the points (equal steps).",
                "Thời gian được tạo bằng cách chia khoảng thời gian ghi lại cho số điểm (bước đều).",
                "เวลาสร้างจากช่วงเวลาที่บันทึกหารด้วยจำนวนจุด (ช่วงเท่ากัน)"),
        "mes_assumed": ("測定時間の範囲が見つからず、1 秒ごとの記録と仮定して時間を作りました。",
                        "The time range was not found; 1 s sampling was assumed.",
                        "Không tìm thấy khoảng thời gian; giả định ghi mỗi 1 s.",
                        "ไม่พบช่วงเวลา จึงสมมติว่าบันทึกทุก 1 วินาที"),
        "action_mes_assumed": ("時間を横軸にするとき、DTG を min⁻¹ で使うときは装置の記録と照らしてください。",
                               "Compare with the instrument's record before using time on the x axis or DTG per minute.",
                               "Hãy đối chiếu với bản ghi của máy trước khi dùng trục thời gian hoặc DTG theo phút.",
                               "โปรดเทียบกับบันทึกของเครื่องก่อนใช้แกนเวลาหรือ DTG ต่อนาที"),
    },
    "sampling": {
        "item": ("サンプリング間隔", "Sampling interval", "Khoảng lấy mẫu", "ช่วงการเก็บข้อมูล"),
        "criterion": ("間隔の最大 ÷ 最小 ≤ {sampling_ratio} なら PASS、超えれば CAUTION（暫定）",
                      "PASS if largest ÷ smallest step ≤ {sampling_ratio}, CAUTION otherwise (provisional)",
                      "PASS nếu bước lớn nhất ÷ nhỏ nhất ≤ {sampling_ratio}, ngược lại CAUTION (tạm thời)",
                      "PASS ถ้าช่วงมากสุด ÷ น้อยสุด ≤ {sampling_ratio} ไม่เช่นนั้น CAUTION (ชั่วคราว)"),
        "value": ("最小 {min:.3g} s・最大 {max:.3g} s（比 {ratio:.3g}）", "{min:.3g} s to {max:.3g} s (ratio {ratio:.3g})",
                  "{min:.3g} s đến {max:.3g} s (tỉ số {ratio:.3g})", "{min:.3g} s ถึง {max:.3g} s (อัตราส่วน {ratio:.3g})"),
        "value_none": ("—", "—", "—", "—"),
        "ok": ("間隔はほぼ一定です。", "The steps are nearly equal.", "Các bước gần như đều.", "ช่วงเกือบเท่ากัน"),
        "bad": ("間隔がばらついています。点数で決まる平滑化・微分の幅（時間）が場所によって変わります。",
                "The steps vary, so a smoothing width set in points covers different times along the run.",
                "Các bước không đều, nên độ rộng làm mịn tính theo điểm ứng với thời gian khác nhau.",
                "ช่วงไม่เท่ากัน ความกว้างการทำให้เรียบที่กำหนดเป็นจุดจึงครอบคลุมเวลาต่างกัน"),
        "action_bad": ("DTG の比較には注意してください。", "Take care when comparing DTG.", "Cẩn thận khi so sánh DTG.",
                       "ระวังเมื่อเปรียบเทียบ DTG"),
        "no_time": ("時間の列がないため検査できません。", "No time column to check.", "Không có cột thời gian để kiểm tra.",
                    "ไม่มีคอลัมน์เวลาให้ตรวจ"),
        "mes": ("時間を等間隔で作っているため検査できません。", "Time was made in equal steps, so it cannot be checked.",
                "Thời gian được tạo với bước đều nên không kiểm tra được.", "เวลาสร้างเป็นช่วงเท่ากัน จึงตรวจไม่ได้"),
    },
    "temp_direction": {
        "item": ("描く区間の温度の向き", "Temperature in the drawn part", "Nhiệt độ trong đoạn được vẽ", "อุณหภูมิในช่วงที่วาด"),
        "criterion": ("昇温だけなら PASS。冷却（{cool:g} K 超の低下）か保持（昇温の後 {hold:g} min 超）を含めば CAUTION",
                      "PASS for heating only; CAUTION with cooling (a fall of more than {cool:g} K) or a hold "
                      "(more than {hold:g} min after the ramp)",
                      "PASS nếu chỉ gia nhiệt; CAUTION nếu có làm nguội (giảm hơn {cool:g} K) hoặc giữ nhiệt (hơn {hold:g} min sau đoạn gia nhiệt)",
                      "PASS ถ้าเป็นการให้ความร้อนอย่างเดียว CAUTION ถ้ามีการทำให้เย็น (ลดลงเกิน {cool:g} K) หรือคงอุณหภูมิ (เกิน {hold:g} min หลังช่วงให้ความร้อน)"),
        "value_heating": ("昇温のみ", "Heating only", "Chỉ gia nhiệt", "ให้ความร้อนอย่างเดียว"),
        "value_cooling": ("冷却を含む（最大 {drop:.1f} K 低下）", "Includes cooling (falls up to {drop:.1f} K)",
                          "Có làm nguội (giảm tới {drop:.1f} K)", "มีการทำให้เย็น (ลดลงสูงสุด {drop:.1f} K)"),
        "value_hold": ("保持を含む（昇温の後 {after:.1f} min）", "Includes a hold ({after:.1f} min after the ramp)",
                       "Có giữ nhiệt ({after:.1f} min sau đoạn gia nhiệt)", "มีการคงอุณหภูมิ ({after:.1f} min หลังช่วงให้ความร้อน)"),
        "ok": ("温度は区間を通して上がっています。", "Temperature rises through the part.", "Nhiệt độ tăng suốt đoạn.",
               "อุณหภูมิเพิ่มขึ้นตลอดช่วง"),
        "bad": ("1 つの温度が 1 点に決まらないため、温度で読む値（Tp・Tx・Tg・質量変化）は出していません。図はそのまま描いています。",
                "A temperature does not pick one point, so values read at a temperature (Tp, Tx, Tg, mass change) are "
                "not given. The curves are drawn as they are.",
                "Một nhiệt độ không ứng với một điểm, nên các giá trị đọc theo nhiệt độ (Tp, Tx, Tg, thay đổi khối lượng) "
                "không được đưa ra. Đường cong vẫn được vẽ.",
                "อุณหภูมิหนึ่งไม่ได้ตรงกับจุดเดียว จึงไม่ให้ค่าที่อ่านตามอุณหภูมิ (Tp, Tx, Tg, การเปลี่ยนแปลงมวล) แต่ยังวาดเส้นตามเดิม"),
        "action_bad": ("「描く区間」を「昇温部だけ」にするか、時間で区間を指定してください。",
                       "Set 'Part of the run' to the heating ramp, or choose a time range.",
                       "Đặt 'Phần dữ liệu được vẽ' là 'Chỉ đoạn gia nhiệt (tự động)' hoặc chọn khoảng thời gian.",
                       "ตั้ง 'ช่วงของการวัดที่วาด' เป็น 'เฉพาะช่วงให้ความร้อน (อัตโนมัติ)' หรือเลือกช่วงเวลา"),
    },
    "temp_values": {
        "item": ("温度で読む値（Tp・Tx・Tg・質量変化）", "Values read at a temperature (Tp, Tx, Tg, mass change)",
                 "Giá trị đọc theo nhiệt độ (Tp, Tx, Tg, thay đổi khối lượng)",
                 "ค่าที่อ่านตามอุณหภูมิ (Tp, Tx, Tg, การเปลี่ยนแปลงมวล)"),
        "criterion": ("昇温だけの区間で評価", "Evaluated on a heating run only", "Chỉ đánh giá trên đoạn gia nhiệt",
                      "ประเมินเฉพาะช่วงให้ความร้อน"),
        "value_none": ("—", "—", "—", "—"),
        "not_heating": ("区間が冷却または保持を含むため評価していません。", "Not evaluated: the part includes cooling or a hold.",
                        "Không đánh giá: đoạn có làm nguội hoặc giữ nhiệt.", "ไม่ได้ประเมิน: ช่วงนี้มีการทำให้เย็นหรือคงอุณหภูมิ"),
    },
    "m0": {
        "item": ("初期質量 m₀", "Initial mass m₀", "Khối lượng ban đầu m₀", "มวลเริ่มต้น m₀"),
        "criterion": ("TG を % で描くには必須", "Required for TG in %", "Bắt buộc cho TG theo %", "จำเป็นสำหรับ TG เป็น %"),
        "value": ("{m:g} mg（{source}）", "{m:g} mg ({source})", "{m:g} mg ({source})", "{m:g} mg ({source})"),
        "value_none": ("未入力", "not entered", "chưa nhập", "ยังไม่ได้ป้อน"),
        "ok": ("Δm/m₀ はこの値で計算しています。", "Δm/m₀ is computed with this value.", "Δm/m₀ được tính với giá trị này.",
               "Δm/m₀ คำนวณด้วยค่านี้"),
        "missing": ("m₀ が未入力のため、TG を % にできず mg で描いています。", "No m₀, so TG is drawn in mg instead of %.",
                    "Chưa có m₀ nên TG được vẽ theo mg thay vì %.", "ไม่มี m₀ จึงวาด TG เป็น mg แทน %"),
        "action_missing": ("サイドバーの「TG」で秤量した値を入れてください。", "Enter the weighed mass under 'TG' in the sidebar.",
                           "Nhập khối lượng đã cân ở mục 'TG' trong thanh bên.", "ป้อนมวลที่ชั่งไว้ในส่วน 'TG' ของแถบข้าง"),
    },
    "tg_physical": {
        "item": ("TG の物理的妥当性", "TG physically possible", "TG hợp lý về vật lý", "TG เป็นไปได้ทางกายภาพ"),
        "criterion": ("質量の減少が {tg_loss_fail_pct:g}% を超えたら FAIL", "FAIL if the loss exceeds {tg_loss_fail_pct:g}%",
                      "FAIL nếu hao hụt vượt {tg_loss_fail_pct:g}%", "FAIL ถ้ามวลลดลงเกิน {tg_loss_fail_pct:g}%"),
        "value": ("最大の減少 {loss:.1f}%", "Largest loss {loss:.1f}%", "Hao hụt lớn nhất {loss:.1f}%", "ลดลงมากสุด {loss:.1f}%"),
        "value_none": ("—", "—", "—", "—"),
        "ok": ("物理的にあり得る範囲です。", "Within what is physically possible.", "Nằm trong giới hạn vật lý.",
               "อยู่ในช่วงที่เป็นไปได้ทางกายภาพ"),
        "bad": ("減少が m₀ を超えています。m₀ の入力ミス（桁違いなど）が考えられます。",
                "The loss exceeds m₀: probably a mistyped m₀ (e.g. a wrong decimal place).",
                "Hao hụt vượt m₀: có thể nhập sai m₀ (ví dụ sai chữ số thập phân).",
                "มวลที่ลดลงเกิน m₀ น่าจะป้อน m₀ ผิด (เช่น ทศนิยมผิดตำแหน่ง)"),
        "action_bad": ("m₀ を確かめてください。", "Check m₀.", "Hãy kiểm tra m₀.", "โปรดตรวจ m₀"),
        "mg": ("TG が mg のため m₀ と比べられません。", "TG is in mg, so it cannot be compared with m₀.",
               "TG theo mg nên không so được với m₀.", "TG เป็น mg จึงเทียบกับ m₀ ไม่ได้"),
    },
    "peak_sn": {
        "item": ("ピークの S/N", "Peak S/N", "S/N của đỉnh", "S/N ของพีก"),
        "criterion": ("S/N（ピークの高さ ÷ 雑音 σ）{sn_fail:g} 未満 FAIL・{sn_fail:g}〜{sn_pass:g} CAUTION・{sn_pass:g} 以上 PASS",
                      "S/N (peak height ÷ noise σ): below {sn_fail:g} FAIL, {sn_fail:g}–{sn_pass:g} CAUTION, {sn_pass:g} or more PASS",
                      "S/N (chiều cao đỉnh ÷ nhiễu σ): dưới {sn_fail:g} FAIL, {sn_fail:g}–{sn_pass:g} CAUTION, từ {sn_pass:g} PASS",
                      "S/N (ความสูงพีก ÷ สัญญาณรบกวน σ): ต่ำกว่า {sn_fail:g} FAIL, {sn_fail:g}–{sn_pass:g} CAUTION, ตั้งแต่ {sn_pass:g} PASS"),
        "value": ("S/N = {sn:.3g}", "S/N = {sn:.3g}", "S/N = {sn:.3g}", "S/N = {sn:.3g}"),
        "value_resolution": ("雑音が記録の分解能以下", "Noise below the record's resolution", "Nhiễu dưới độ phân giải của bản ghi",
                             "สัญญาณรบกวนต่ำกว่าความละเอียดของการบันทึก"),
        "ok": ("雑音より十分大きいピークです。", "Well above the noise.", "Cao hơn nhiều so với nhiễu.", "สูงกว่าสัญญาณรบกวนมาก"),
        "low": ("定量限界（10σ）に届きません。ピーク温度の不確かさが大きくなります。",
                "Below the quantitation limit (10σ): the peak temperature is less certain.",
                "Dưới giới hạn định lượng (10σ): nhiệt độ đỉnh kém chắc chắn.", "ต่ำกว่าขีดจำกัดการวัดเชิงปริมาณ (10σ) อุณหภูมิพีกจึงไม่แน่นอนมากขึ้น"),
        "bad": ("検出限界（3σ）未満で、雑音と区別できません。", "Below the detection limit (3σ): not distinguishable from noise.",
                "Dưới giới hạn phát hiện (3σ): không phân biệt được với nhiễu.", "ต่ำกว่าขีดจำกัดการตรวจพบ (3σ) แยกจากสัญญาณรบกวนไม่ได้"),
        "resolution": ("雑音が小さすぎて測れません（記録の分解能以下）。", "The noise is too small to measure (below the resolution).",
                       "Nhiễu quá nhỏ để đo (dưới độ phân giải).", "สัญญาณรบกวนเล็กเกินกว่าจะวัดได้ (ต่ำกว่าความละเอียด)"),
        "action_low": ("値を使うときは不確かさに注意してください。", "Mind the uncertainty when using the value.",
                       "Lưu ý độ bất định khi dùng giá trị.", "โปรดคำนึงถึงความไม่แน่นอนเมื่อใช้ค่านี้"),
        "action_bad": ("ピーク検出の感度を上げすぎていないか確かめてください。", "Check that the peak sensitivity is not set too high.",
                       "Kiểm tra độ nhạy dò đỉnh có đặt quá cao không.", "ตรวจว่าตั้งความไวในการหาพีกสูงเกินไปหรือไม่"),
    },
    "peak_after_exo": {
        "item": ("偽ピークの候補", "Possible artefact peak", "Đỉnh có thể là giả", "พีกที่อาจเป็นสิ่งแปลกปลอม"),
        "criterion": ("発熱ピークの直後（半値幅の {after_exo_widths:g} 倍以内）の吸熱ピークは CAUTION（暫定）",
                      "CAUTION for an endothermic peak within {after_exo_widths:g} half-widths after an exothermic peak (provisional)",
                      "CAUTION cho đỉnh thu nhiệt trong vòng {after_exo_widths:g} lần nửa độ rộng sau đỉnh tỏa nhiệt (tạm thời)",
                      "CAUTION สำหรับพีกดูดความร้อนภายใน {after_exo_widths:g} เท่าของความกว้างครึ่งหลังพีกคายความร้อน (ชั่วคราว)"),
        "value": ("発熱ピーク {exo:.0f} °C の {d:.0f} K 後", "{d:.0f} K after the exothermic peak at {exo:.0f} °C",
                  "{d:.0f} K sau đỉnh tỏa nhiệt {exo:.0f} °C", "{d:.0f} K หลังพีกคายความร้อนที่ {exo:.0f} °C"),
        "bad": ("発熱ピークから基線へ戻る部分を吸熱ピークとして拾った可能性があります。",
                "It may be the return from the exothermic peak to the baseline, counted as an endotherm.",
                "Có thể là đoạn trở về đường nền sau đỉnh tỏa nhiệt, bị tính là thu nhiệt.",
                "อาจเป็นช่วงที่กลับสู่เส้นฐานหลังพีกคายความร้อน ซึ่งถูกนับเป็นพีกดูดความร้อน"),
        "action_bad": ("図で確かめ、不要ならピークを「発熱」だけにしてください。",
                       "Check it on the figure; if it is not real, set the peaks to 'Exo' only.",
                       "Kiểm tra trên hình; nếu không thật, chỉ chọn đỉnh 'Tỏa nhiệt'.",
                       "ตรวจในรูป ถ้าไม่ใช่พีกจริงให้เลือกพีกเป็น 'คายความร้อน' อย่างเดียว"),
    },
    "onset_robust": {
        "item": ("補外開始温度の頑健性", "Robustness of the onset", "Độ ổn định của điểm bắt đầu", "ความคงทนของอุณหภูมิเริ่มต้น"),
        "criterion": ("作図条件を変えたときの値の幅 ≤ {robust_pass_K:g} K PASS・{robust_pass_K:g}〜{robust_caution_K:g} K CAUTION・"
                      "{robust_caution_K:g} K 超 WARNING（暫定）",
                      "Spread when the construction is varied: ≤ {robust_pass_K:g} K PASS, {robust_pass_K:g}–{robust_caution_K:g} K "
                      "CAUTION, over {robust_caution_K:g} K WARNING (provisional)",
                      "Độ trải khi thay đổi cách dựng: ≤ {robust_pass_K:g} K PASS, {robust_pass_K:g}–{robust_caution_K:g} K CAUTION, "
                      "trên {robust_caution_K:g} K WARNING (tạm thời)",
                      "ช่วงของค่าเมื่อเปลี่ยนวิธีสร้าง: ≤ {robust_pass_K:g} K PASS, {robust_pass_K:g}–{robust_caution_K:g} K CAUTION, "
                      "เกิน {robust_caution_K:g} K WARNING (ชั่วคราว)"),
        "value": ("幅 {spread:.1f} K（{ok}/{n} 通りで作図）", "Spread {spread:.1f} K ({ok}/{n} constructions)",
                  "Độ trải {spread:.1f} K ({ok}/{n} cách dựng)", "ช่วง {spread:.1f} K ({ok}/{n} แบบ)"),
        "value_unstable": ("{ok}/{n} 通りでしか作図できない", "Only {ok}/{n} constructions possible", "Chỉ dựng được {ok}/{n} cách",
                           "สร้างได้เพียง {ok}/{n} แบบ"),
        "ok": ("ピークの足の位置（高さの 5〜15%）とベースラインの範囲を変えても、値はほぼ変わりません。",
               "Moving the foot of the peak (5–15 % of its height) and the baseline range hardly changes the value.",
               "Dịch chân đỉnh (5–15 % chiều cao) và khoảng đường nền hầu như không đổi giá trị.",
               "การเลื่อนฐานพีก (5–15 % ของความสูง) และช่วงเส้นฐานแทบไม่เปลี่ยนค่า"),
        "bad": ("ピークの足の位置（高さの 5〜15%）とベースラインの範囲を変えると、値が {lo:.1f}〜{hi:.1f} °C の間で動きます。",
                "Moving the foot of the peak (5–15 % of its height) and the baseline range moves the value between {lo:.1f} and {hi:.1f} °C.",
                "Dịch chân đỉnh (5–15 % chiều cao) và khoảng đường nền làm giá trị thay đổi trong {lo:.1f}–{hi:.1f} °C.",
                "การเลื่อนฐานพีก (5–15 % ของความสูง) และช่วงเส้นฐานทำให้ค่าเปลี่ยนระหว่าง {lo:.1f} ถึง {hi:.1f} °C"),
        "unstable": ("作図できた条件が半分に届かず、作図が不安定です。", "Fewer than half of the constructions can be drawn: unstable.",
                     "Dựng được chưa tới một nửa: không ổn định.", "สร้างได้ไม่ถึงครึ่ง จึงไม่เสถียร"),
        "action_bad": ("「Tx・Tg の接線を直す」で線を確かめ、値には作図による幅があることを示してください。",
                       "Check the lines under 'Adjust Tx / Tg tangents' and give the value with its construction spread.",
                       "Kiểm tra các đường ở 'Chỉnh tiếp tuyến Tx / Tg' và ghi giá trị kèm độ trải.",
                       "ตรวจเส้นใน 'ปรับเส้นสัมผัส Tx / Tg' และระบุค่าพร้อมช่วงจากการสร้าง"),
        "action_unstable": ("「Tx・Tg の接線を直す」で線を確かめてください。", "Check the lines under 'Adjust Tx / Tg tangents'.",
                            "Kiểm tra các đường ở 'Chỉnh tiếp tuyến Tx / Tg'.", "ตรวจเส้นใน 'ปรับเส้นสัมผัส Tx / Tg'"),
    },
    "onset_smoothing": {
        "item": ("平滑化の影響", "Effect of smoothing", "Ảnh hưởng của làm mịn", "ผลของการทำให้เรียบ"),
        "criterion": ("DTA の平滑化あり・なしの差 ≤ {smoothing_pass_K:g} K なら PASS、超えれば CAUTION（暫定）",
                      "PASS if the value with and without DTA smoothing differs by ≤ {smoothing_pass_K:g} K, CAUTION otherwise (provisional)",
                      "PASS nếu chênh lệch có và không làm mịn DTA ≤ {smoothing_pass_K:g} K, ngược lại CAUTION (tạm thời)",
                      "PASS ถ้าค่าที่ทำให้เรียบและไม่ทำต่างกัน ≤ {smoothing_pass_K:g} K ไม่เช่นนั้น CAUTION (ชั่วคราว)"),
        "value": ("差 {d:.1f} K（平滑化なし {raw:.1f} °C）", "Differs by {d:.1f} K ({raw:.1f} °C unsmoothed)",
                  "Chênh {d:.1f} K ({raw:.1f} °C khi không làm mịn)", "ต่างกัน {d:.1f} K ({raw:.1f} °C เมื่อไม่ทำให้เรียบ)"),
        "value_no_raw": ("平滑化なしでは作図できない", "No construction without smoothing", "Không dựng được khi không làm mịn",
                         "สร้างไม่ได้เมื่อไม่ทำให้เรียบ"),
        "ok": ("平滑化による値の偏りは小さいです。", "Smoothing barely shifts the value.", "Làm mịn hầu như không dịch giá trị.",
               "การทำให้เรียบแทบไม่เลื่อนค่า"),
        "bad": ("DTA の平滑化（{pts} 点）で値が {d:.1f} K 動いています。", "DTA smoothing ({pts} points) shifts the value by {d:.1f} K.",
                "Làm mịn DTA ({pts} điểm) dịch giá trị {d:.1f} K.", "การทำให้ DTA เรียบ ({pts} จุด) เลื่อนค่าไป {d:.1f} K"),
        "no_raw": ("平滑化しない DTA では接線とベースラインを引けません。", "Without smoothing the tangent and baseline cannot be drawn.",
                   "Không làm mịn thì không dựng được tiếp tuyến và đường nền.", "ถ้าไม่ทำให้เรียบจะลากเส้นสัมผัสและเส้นฐานไม่ได้"),
        "action_bad": ("平滑化を小さくするか、平滑化の条件と一緒に値を示してください。",
                       "Use less smoothing, or give the value with the smoothing used.",
                       "Giảm làm mịn hoặc ghi giá trị kèm điều kiện làm mịn.", "ลดการทำให้เรียบ หรือระบุค่าพร้อมเงื่อนไขการทำให้เรียบ"),
        "action_no_raw": ("平滑化の条件と一緒に値を示してください。", "Give the value with the smoothing used.",
                          "Ghi giá trị kèm điều kiện làm mịn.", "ระบุค่าพร้อมเงื่อนไขการทำให้เรียบ"),
    },
    "manual": {
        "item": ("手で引いた線", "Hand-drawn lines", "Đường vẽ tay", "เส้นที่ลากเอง"),
        "criterion": ("手で引いた線は CAUTION（自動の値との差を示す）", "CAUTION for hand-drawn lines (difference from the automatic value shown)",
                      "CAUTION cho đường vẽ tay (hiện chênh lệch với giá trị tự động)", "CAUTION สำหรับเส้นที่ลากเอง (แสดงผลต่างจากค่าอัตโนมัติ)"),
        "value": ("手動 {man:.1f} °C・自動 {auto:.1f} °C（差 {d:+.1f} K）", "Hand {man:.1f} °C, automatic {auto:.1f} °C ({d:+.1f} K)",
                  "Tay {man:.1f} °C, tự động {auto:.1f} °C ({d:+.1f} K)", "ลากเอง {man:.1f} °C อัตโนมัติ {auto:.1f} °C ({d:+.1f} K)"),
        "value_no_auto": ("手動 {man:.1f} °C・自動では作図できない", "Hand {man:.1f} °C, no automatic construction",
                          "Tay {man:.1f} °C, không dựng tự động được", "ลากเอง {man:.1f} °C สร้างอัตโนมัติไม่ได้"),
        "ok": ("作図が使う人の判断によります。", "The construction rests on the user's judgement.", "Cách dựng dựa vào phán đoán của người dùng.",
               "การสร้างขึ้นกับการตัดสินของผู้ใช้"),
        "action_ok": ("論文では手で引いたことと作図法を明記してください。", "State in the paper that the lines were drawn by hand, and how.",
                      "Ghi rõ trong bài báo rằng các đường được vẽ tay và cách vẽ.", "ระบุในบทความว่าลากเส้นเองและลากอย่างไร"),
    },
    "tg_found": {
        "item": ("Tg の段", "Tg step", "Bậc Tg", "ขั้น Tg"),
        "criterion": ("指定範囲で段が雑音（3σ）より深く、作図が成り立つこと", "A step deeper than the noise (3σ) in the range, and a valid construction",
                      "Có bậc sâu hơn nhiễu (3σ) trong khoảng và dựng được", "มีขั้นลึกกว่าสัญญาณรบกวน (3σ) ในช่วง และสร้างได้"),
        "value_none": ("—", "—", "—", "—"),
        "none": ("{lo:g}–{hi:g} °C に段が見つからない、または作図が成り立たないため、Tg を出していません。",
                 "No Tg: no step found in {lo:g}–{hi:g} °C, or the construction fails.",
                 "Không có Tg: không thấy bậc trong {lo:g}–{hi:g} °C hoặc không dựng được.",
                 "ไม่มี Tg: ไม่พบขั้นใน {lo:g}–{hi:g} °C หรือสร้างไม่ได้"),
        "not_heating": ("区間が冷却または保持を含むため評価していません。", "Not evaluated: the part includes cooling or a hold.",
                        "Không đánh giá: đoạn có làm nguội hoặc giữ nhiệt.", "ไม่ได้ประเมิน: ช่วงนี้มีการทำให้เย็นหรือคงอุณหภูมิ"),
    },
    "tg_sn": {
        "item": ("Tg の段の S/N", "Tg step S/N", "S/N của bậc Tg", "S/N ของขั้น Tg"),
        "criterion": ("S/N（段の深さ ÷ ベースラインのばらつき σ）{sn_fail:g} 未満 FAIL・{sn_fail:g}〜{sn_pass:g} CAUTION・{sn_pass:g} 以上 PASS",
                      "S/N (step depth ÷ baseline scatter σ): below {sn_fail:g} FAIL, {sn_fail:g}–{sn_pass:g} CAUTION, {sn_pass:g} or more PASS",
                      "S/N (độ sâu bậc ÷ độ tán xạ đường nền σ): dưới {sn_fail:g} FAIL, {sn_fail:g}–{sn_pass:g} CAUTION, từ {sn_pass:g} PASS",
                      "S/N (ความลึกขั้น ÷ การกระจายของเส้นฐาน σ): ต่ำกว่า {sn_fail:g} FAIL, {sn_fail:g}–{sn_pass:g} CAUTION, ตั้งแต่ {sn_pass:g} PASS"),
        "value": ("S/N = {sn:.3g}", "S/N = {sn:.3g}", "S/N = {sn:.3g}", "S/N = {sn:.3g}"),
        "value_resolution": ("雑音が記録の分解能以下", "Noise below the record's resolution", "Nhiễu dưới độ phân giải của bản ghi",
                             "สัญญาณรบกวนต่ำกว่าความละเอียดของการบันทึก"),
        "ok": ("ベースラインのばらつきより十分深い段です。", "Well deeper than the baseline scatter.", "Sâu hơn nhiều so với tán xạ đường nền.",
               "ลึกกว่าการกระจายของเส้นฐานมาก"),
        "low": ("定量限界（10σ）に届かない浅い段です。Tg の不確かさが大きくなります。",
                "A shallow step below the quantitation limit (10σ): Tg is less certain.",
                "Bậc nông dưới giới hạn định lượng (10σ): Tg kém chắc chắn.", "ขั้นตื้นต่ำกว่าขีดจำกัดเชิงปริมาณ (10σ) Tg จึงไม่แน่นอนมากขึ้น"),
        "bad": ("検出限界（3σ）未満です。", "Below the detection limit (3σ).", "Dưới giới hạn phát hiện (3σ).", "ต่ำกว่าขีดจำกัดการตรวจพบ (3σ)"),
        "resolution": ("ばらつきが小さすぎて測れません。", "The scatter is too small to measure.", "Tán xạ quá nhỏ để đo.",
                       "การกระจายเล็กเกินกว่าจะวัดได้"),
        "action_low": ("値を使うときは不確かさに注意してください。", "Mind the uncertainty when using the value.",
                       "Lưu ý độ bất định khi dùng giá trị.", "โปรดคำนึงถึงความไม่แน่นอนเมื่อใช้ค่านี้"),
    },
    "tg_robust": {
        "item": ("Tg の頑健性", "Robustness of Tg", "Độ ổn định của Tg", "ความคงทนของ Tg"),
        "criterion": ("作図条件を変えたときの値の幅 ≤ {robust_pass_K:g} K PASS・{robust_pass_K:g}〜{robust_caution_K:g} K CAUTION・"
                      "{robust_caution_K:g} K 超 WARNING（暫定）",
                      "Spread when the construction is varied: ≤ {robust_pass_K:g} K PASS, {robust_pass_K:g}–{robust_caution_K:g} K "
                      "CAUTION, over {robust_caution_K:g} K WARNING (provisional)",
                      "Độ trải khi thay đổi cách dựng: ≤ {robust_pass_K:g} K PASS, {robust_pass_K:g}–{robust_caution_K:g} K CAUTION, "
                      "trên {robust_caution_K:g} K WARNING (tạm thời)",
                      "ช่วงของค่าเมื่อเปลี่ยนวิธีสร้าง: ≤ {robust_pass_K:g} K PASS, {robust_pass_K:g}–{robust_caution_K:g} K CAUTION, "
                      "เกิน {robust_caution_K:g} K WARNING (ชั่วคราว)"),
        "value": ("幅 {spread:.1f} K（{ok}/{n} 通りで作図）", "Spread {spread:.1f} K ({ok}/{n} constructions)",
                  "Độ trải {spread:.1f} K ({ok}/{n} cách dựng)", "ช่วง {spread:.1f} K ({ok}/{n} แบบ)"),
        "value_unstable": ("{ok}/{n} 通りでしか作図できない", "Only {ok}/{n} constructions possible", "Chỉ dựng được {ok}/{n} cách",
                           "สร้างได้เพียง {ok}/{n} แบบ"),
        "ok": ("探す範囲の両端（±5%）とベースラインの割合（30〜50%）を変えても、値はほぼ変わりません。",
               "Moving the ends of the range (±5 %) and the baseline share (30–50 %) hardly changes the value.",
               "Dịch hai đầu khoảng (±5 %) và tỉ lệ đường nền (30–50 %) hầu như không đổi giá trị.",
               "การเลื่อนปลายช่วง (±5 %) และสัดส่วนเส้นฐาน (30–50 %) แทบไม่เปลี่ยนค่า"),
        "bad": ("探す範囲の両端（±5%）とベースラインの割合（30〜50%）を変えると、値が {lo:.1f}〜{hi:.1f} °C の間で動きます。",
                "Moving the ends of the range (±5 %) and the baseline share (30–50 %) moves the value between {lo:.1f} and {hi:.1f} °C.",
                "Dịch hai đầu khoảng (±5 %) và tỉ lệ đường nền (30–50 %) làm giá trị thay đổi trong {lo:.1f}–{hi:.1f} °C.",
                "การเลื่อนปลายช่วง (±5 %) และสัดส่วนเส้นฐาน (30–50 %) ทำให้ค่าเปลี่ยนระหว่าง {lo:.1f} ถึง {hi:.1f} °C"),
        "unstable": ("作図できた条件が半分に届かず、作図が不安定です。", "Fewer than half of the constructions can be drawn: unstable.",
                     "Dựng được chưa tới một nửa: không ổn định.", "สร้างได้ไม่ถึงครึ่ง จึงไม่เสถียร"),
        "action_bad": ("探す範囲を見直し、値には作図による幅があることを示してください。",
                       "Review the search range and give the value with its construction spread.",
                       "Xem lại khoảng tìm và ghi giá trị kèm độ trải.", "ทบทวนช่วงค้นหาและระบุค่าพร้อมช่วงจากการสร้าง"),
        "action_unstable": ("探す範囲を見直してください。", "Review the search range.", "Xem lại khoảng tìm.", "ทบทวนช่วงค้นหา"),
    },
    "mass_range": {
        "item": ("質量変化", "Mass change", "Thay đổi khối lượng", "การเปลี่ยนแปลงมวล"),
        "criterion": ("範囲の両端が描く区間（昇温）の中にあること", "Both ends within the drawn (heating) part",
                      "Hai đầu nằm trong đoạn được vẽ (gia nhiệt)", "ปลายทั้งสองอยู่ในช่วงที่วาด (ให้ความร้อน)"),
        "value_none": ("—", "—", "—", "—"),
        "outside": ("範囲の端が描く区間の温度の外にあるため、ラベルを書いていません。",
                    "No label: an end of the range lies outside the drawn temperatures.",
                    "Không có nhãn: một đầu khoảng nằm ngoài nhiệt độ được vẽ.", "ไม่มีป้าย: ปลายช่วงอยู่นอกอุณหภูมิที่วาด"),
        "not_heating": ("区間が冷却または保持を含むため評価していません。", "Not evaluated: the part includes cooling or a hold.",
                        "Không đánh giá: đoạn có làm nguội hoặc giữ nhiệt.", "ไม่ได้ประเมิน: ช่วงนี้มีการทำให้เย็นหรือคงอุณหภูมิ"),
        "no_tg": ("TG の列がありません。", "No TG column.", "Không có cột TG.", "ไม่มีคอลัมน์ TG"),
    },
    "target_exo": {"text": ("発熱ピーク {T:.0f} °C", "Exo peak {T:.0f} °C", "Đỉnh tỏa nhiệt {T:.0f} °C", "พีกคายความร้อน {T:.0f} °C")},
    "target_endo": {"text": ("吸熱ピーク {T:.0f} °C", "Endo peak {T:.0f} °C", "Đỉnh thu nhiệt {T:.0f} °C", "พีกดูดความร้อน {T:.0f} °C")},
    "target_tx": {"text": ("Tx（発熱ピーク {T:.0f} °C）", "Tx (exo peak {T:.0f} °C)", "Tx (đỉnh tỏa nhiệt {T:.0f} °C)",
                           "Tx (พีกคายความร้อน {T:.0f} °C)")},
    "target_onset_endo": {"text": ("吸熱の補外開始（ピーク {T:.0f} °C）", "Endo onset (peak {T:.0f} °C)",
                                   "Bắt đầu thu nhiệt (đỉnh {T:.0f} °C)", "จุดเริ่มดูดความร้อน (พีก {T:.0f} °C)")},
    "target_tg": {"text": ("Tg", "Tg", "Tg", "Tg")},
    "target_range": {"text": ("{lo:g}–{hi:g} °C", "{lo:g}–{hi:g} °C", "{lo:g}–{hi:g} °C", "{lo:g}–{hi:g} °C")},
    # What was done to the data (processing_steps).
    "proc_segment_heating": {"text": ("描く区間：昇温部だけ（自動）{t0:.1f}–{t1:.1f} min・{T0:.0f}–{T1:.0f} °C・{n} 点",
                                      "Drawn part: heating ramp only (automatic), {t0:.1f}–{t1:.1f} min, {T0:.0f}–{T1:.0f} °C, {n} points",
                                      "Đoạn vẽ: chỉ đoạn gia nhiệt (tự động), {t0:.1f}–{t1:.1f} min, {T0:.0f}–{T1:.0f} °C, {n} điểm",
                                      "ช่วงที่วาด: เฉพาะช่วงให้ความร้อน (อัตโนมัติ) {t0:.1f}–{t1:.1f} min, {T0:.0f}–{T1:.0f} °C, {n} จุด")},
    "proc_segment_custom": {"text": ("描く区間：時間で指定 {t0:.1f}–{t1:.1f} min・{n} 点", "Drawn part: time range {t0:.1f}–{t1:.1f} min, {n} points",
                                     "Đoạn vẽ: khoảng thời gian {t0:.1f}–{t1:.1f} min, {n} điểm", "ช่วงที่วาด: ช่วงเวลา {t0:.1f}–{t1:.1f} min, {n} จุด")},
    "proc_segment_all": {"text": ("描く区間：すべて・{n} 点", "Drawn part: all, {n} points", "Đoạn vẽ: tất cả, {n} điểm",
                                  "ช่วงที่วาด: ทั้งหมด {n} จุด")},
    "proc_excluded": {"text": ("読み込みで {n} 行を除外（行 {lines}）", "{n} rows left out on reading (lines {lines})",
                               "Bỏ {n} dòng khi đọc (dòng {lines})", "ตัด {n} แถวออกตอนอ่าน (แถว {lines})")},
    "proc_footer": {"text": ("表の後ろの {n} 行は表の外として読まない", "{n} lines after the table not read (outside the table)",
                             "{n} dòng sau bảng không được đọc (ngoài bảng)", "{n} บรรทัดหลังตารางไม่ได้อ่าน (อยู่นอกตาราง)")},
    "proc_temp_K": {"text": ("温度を K から °C に換算", "Temperature converted from K to °C", "Đổi nhiệt độ từ K sang °C",
                             "แปลงอุณหภูมิจาก K เป็น °C")},
    "proc_time_s": {"text": ("時間を s から min に換算", "Time converted from s to min", "Đổi thời gian từ s sang min",
                             "แปลงเวลาจาก s เป็น min")},
    "proc_time_mes": {"text": ("時間を装置記録の測定時間の範囲から等間隔で作成", "Time made in equal steps over the recorded time range",
                               "Thời gian tạo theo bước đều trên khoảng thời gian ghi lại", "สร้างเวลาเป็นช่วงเท่ากันตามช่วงเวลาที่บันทึก")},
    "proc_time_assumed": {"text": ("時間を 1 秒ごとの記録と仮定して作成", "Time made assuming 1 s sampling", "Thời gian tạo giả định ghi mỗi 1 s",
                                   "สร้างเวลาโดยสมมติว่าบันทึกทุก 1 วินาที")},
    "proc_no_time": {"text": ("時間の列なし（点の番号を時間の代わりに使用）", "No time column (point numbers used instead)",
                              "Không có cột thời gian (dùng số thứ tự điểm)", "ไม่มีคอลัมน์เวลา (ใช้ลำดับจุดแทน)")},
    "proc_m0": {"text": ("m₀ = {m:g} mg（{source}）", "m₀ = {m:g} mg ({source})", "m₀ = {m:g} mg ({source})", "m₀ = {m:g} mg ({source})")},
    "proc_m0_none": {"text": ("m₀ 未入力", "m₀ not entered", "Chưa nhập m₀", "ยังไม่ได้ป้อน m₀")},
    "proc_ref_first": {"text": ("TG の基準：描く区間の最初の点", "TG reference: first point of the drawn part",
                                "Mốc TG: điểm đầu của đoạn vẽ", "จุดอ้างอิง TG: จุดแรกของช่วงที่วาด")},
    "proc_ref_temp": {"text": ("TG の基準：{T:g} °C", "TG reference: {T:g} °C", "Mốc TG: {T:g} °C", "จุดอ้างอิง TG: {T:g} °C")},
    "proc_ref_fallback_range": {"text": ("TG の基準：{T:g} °C が描く区間の外のため最初の点", "TG reference: first point ({T:g} °C is outside the drawn part)",
                                         "Mốc TG: điểm đầu ({T:g} °C nằm ngoài đoạn vẽ)", "จุดอ้างอิง TG: จุดแรก ({T:g} °C อยู่นอกช่วงที่วาด)")},
    "proc_ref_fallback_heating": {"text": ("TG の基準：区間が冷却・保持を含み {T:g} °C が 1 点に決まらないため最初の点",
                                           "TG reference: first point ({T:g} °C does not pick one point: the part cools or holds)",
                                           "Mốc TG: điểm đầu ({T:g} °C không ứng với một điểm: đoạn có làm nguội hoặc giữ nhiệt)",
                                           "จุดอ้างอิง TG: จุดแรก ({T:g} °C ไม่ตรงกับจุดเดียว เพราะช่วงนี้มีการทำให้เย็นหรือคงอุณหภูมิ)")},
    "proc_flip": {"text": ("DTA の符号を反転（発熱を上に）", "DTA sign reversed (exotherm upward)", "Đảo dấu DTA (tỏa nhiệt hướng lên)",
                           "กลับเครื่องหมาย DTA (คายความร้อนขึ้นบน)")},
    "proc_zero": {"text": ("DTA の始点を 0 にそろえる", "DTA shifted to start at 0", "DTA dịch để bắt đầu từ 0", "เลื่อน DTA ให้เริ่มที่ 0")},
    "proc_per_mass": {"text": ("DTA を m₀ で割る", "DTA divided by m₀", "Chia DTA cho m₀", "หาร DTA ด้วย m₀")},
    "proc_smooth": {"text": ("{what} の平滑化：Savitzky–Golay {pts} 点（2 次）＝ 約 {min:.2f} min・約 {K:.1f} K",
                             "{what} smoothing: Savitzky–Golay, {pts} points (2nd order) ≈ {min:.2f} min ≈ {K:.1f} K",
                             "Làm mịn {what}: Savitzky–Golay, {pts} điểm (bậc 2) ≈ {min:.2f} min ≈ {K:.1f} K",
                             "การทำให้ {what} เรียบ: Savitzky–Golay {pts} จุด (อันดับ 2) ≈ {min:.2f} min ≈ {K:.1f} K")},
    "proc_smooth_points": {"text": ("{what} の平滑化：Savitzky–Golay {pts} 点（2 次）", "{what} smoothing: Savitzky–Golay, {pts} points (2nd order)",
                                    "Làm mịn {what}: Savitzky–Golay, {pts} điểm (bậc 2)", "การทำให้ {what} เรียบ: Savitzky–Golay {pts} จุด (อันดับ 2)")},
    "proc_smooth_none": {"text": ("{what} の平滑化：なし", "{what} smoothing: none", "Làm mịn {what}: không", "การทำให้ {what} เรียบ: ไม่มี")},
}
