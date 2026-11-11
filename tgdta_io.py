"""Reading TG-DTA (TG-DSC) text exports and turning them into the quantities drawn in papers.

A measurement is a table of time, temperature, DTA (or DSC) and TG columns, such as the
instrument's CSV export::

    Time(min),TEMP（℃）,DTA(μV),TG(mg),
           0.000000,      12.970000,       8.686500,       0.000200,

Files may be Shift_JIS (cp932) or UTF-8, separated by commas, tabs or semicolons, with
header lines before the column names. Columns are found by their names; when a name is
not recognised the column can be chosen by hand (``mapping``).

Nothing here depends on Streamlit, so the processing can be tested on its own.
"""

from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter


SUPPORTED_EXTENSIONS = ("csv", "txt", "tsv", "dat")
SIGNALS = ("time", "temp", "dta", "tg")
MAX_FILE_BYTES = 40 * 1024 * 1024


class TGDTAParseError(ValueError):
    """The file is not a TG-DTA table this reader understands."""


@dataclass
class Measurement:
    """One measurement: the numeric columns, which column is which, and the units read from the header."""
    name: str
    columns: list[str]
    data: np.ndarray                      # rows × columns, float
    mapping: dict[str, int | None]        # signal → column index
    units: dict[str, str] = field(default_factory=dict)

    def column(self, signal: str) -> np.ndarray | None:
        index = self.mapping.get(signal)
        if index is None or not 0 <= index < self.data.shape[1]:
            return None
        return self.data[:, index]

    def time_min(self) -> np.ndarray | None:
        t = self.column("time")
        if t is None:
            return None
        return t / 60.0 if self.units.get("time") == "s" else t.copy()


# ------------------------------------------------------------------------------------------------ reading
def decode_text(raw: bytes) -> str:
    """Text of a file in UTF-8 (with or without BOM) or Shift_JIS, the encodings instruments write."""
    for encoding in ("utf-8-sig", "cp932", "utf-16"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text
    return raw.decode("latin-1")


def _norm(text: str) -> str:
    """Header text without width/case differences: 'TEMP（℃）' → 'temp(°c)', 'μV' → 'μv'."""
    text = unicodedata.normalize("NFKC", str(text)).replace("℃", "°c").replace("µ", "μ")
    return text.strip().lower()


_PATTERNS = {
    "time": re.compile(r"time|時間|経過|\bt\s*[\(/\[]\s*(min|s)\b|^t$|^min$"),
    "temp": re.compile(r"temp|温度|°c|deg\s*c|\btc\b|\bts\b|^t\s*[\(/\[]\s*°"),
    "dta": re.compile(r"dta|dsc|μv|uv\b|heat\s*flow|mw|熱流"),
    "tg": re.compile(r"\btg\b|^tg|tga|mass|weight|質量|重量|\bmg\b|^tg\("),
}


def _unit_of(signal: str, header: str) -> str:
    h = _norm(header)
    if signal == "time":
        return "s" if re.search(r"\(s\)|\[s\]|/s\b|\bsec", h) else "min"
    if signal == "temp":
        return "K" if re.search(r"\(k\)|\[k\]|/k\b", h) else "°C"
    if signal == "dta":
        if "mw/mg" in h or "w/g" in h:
            return "mW/mg"
        if "mw" in h:
            return "mW"
        return "μV"
    if signal == "tg":
        return "%" if "%" in h else "mg"
    return ""


def guess_mapping(columns: list[str]) -> dict[str, int | None]:
    """Which column is time, temperature, DTA and TG, from the column names (None where nothing fits)."""
    mapping: dict[str, int | None] = {s: None for s in SIGNALS}
    taken: set[int] = set()
    normalized = [_norm(c) for c in columns]
    # The specific names first (DTA before a generic 'μV', TG before 'mg'), so one column is not claimed twice.
    for signal in ("dta", "tg", "temp", "time"):
        for index, name in enumerate(normalized):
            if index in taken or not name:
                continue
            if signal == "time" and ("temp" in name or "°c" in name):
                continue
            if signal == "tg" and ("dta" in name or "dsc" in name):
                continue
            if _PATTERNS[signal].search(name):
                mapping[signal] = index
                taken.add(index)
                break
    return mapping


def _is_number(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    try:
        return math.isfinite(float(token))
    except ValueError:
        return False


def _split(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return line.split()
    return next(csv.reader([line], delimiter=delimiter))


def read_measurement(raw: bytes, name: str = "", mapping: dict | None = None) -> Measurement:
    """Parse one exported table. ``mapping`` (signal → column index) overrides the automatic choice."""
    if len(raw) > MAX_FILE_BYTES:
        raise TGDTAParseError("The file is larger than 40 MB.")
    text = decode_text(raw)
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise TGDTAParseError("The file is empty.")
    sample = "\n".join(lines[:60])
    counts = {d: sample.count(d) for d in (",", "\t", ";")}
    delimiter = max(counts, key=counts.get) if max(counts.values()) else None

    rows, header = [], None
    for k, line in enumerate(lines):
        fields = [f.strip() for f in _split(line, delimiter)]
        while fields and not fields[-1]:          # the trailing comma of the export
            fields.pop()
        numeric = [_is_number(f) for f in fields]
        if len(fields) >= 2 and all(numeric):
            rows.append([float(f) for f in fields])
        elif rows:
            break                                   # a footer after the data
        elif len(fields) >= 2 and not any(numeric):
            header = fields
    if len(rows) < 5:
        raise TGDTAParseError("No numeric table with at least 5 rows was found.")
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    data = np.asarray(rows, dtype=float)
    columns = list(header[:width]) if header and len(header) >= width else [f"column {i + 1}" for i in range(width)]
    columns += [f"column {i + 1}" for i in range(len(columns), width)]
    auto = guess_mapping(columns)
    if mapping:
        for signal in SIGNALS:
            if signal in mapping:
                value = mapping[signal]
                auto[signal] = value if isinstance(value, int) and 0 <= value < width else None
    units = {signal: _unit_of(signal, columns[i]) for signal, i in auto.items() if i is not None}
    return Measurement(name=name, columns=columns, data=data, mapping=auto, units=units)


# ------------------------------------------------------------------------------------------------ processing
def _odd_window(points: int, n: int, order: int = 2) -> int:
    """A Savitzky–Golay window: odd, larger than the order, not larger than the data."""
    w = int(points)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    return w if w > order else 0


def smooth(y: np.ndarray, points: int, order: int = 2) -> np.ndarray:
    """Savitzky–Golay smoothing over ``points`` samples (0 or 1: unchanged)."""
    w = _odd_window(points, len(y), order) if points and points > 1 else 0
    return savgol_filter(y, w, order) if w else np.asarray(y, float).copy()


def heating_rate(t_min: np.ndarray, temp: np.ndarray) -> float:
    """The programmed heating rate (K/min): the typical slope over the ramp, ignoring holds and the start."""
    if len(t_min) < 5:
        return float("nan")
    rate = np.gradient(smooth(temp, 21), t_min)
    positive = rate[rate > 0]
    if positive.size == 0:
        return float("nan")
    high = np.percentile(positive, 95)
    return float(np.median(positive[positive > 0.3 * high]))


def heating_segment(t_min: np.ndarray, temp: np.ndarray, fraction: float = 0.9) -> tuple[float, float]:
    """Time range (min) of the heating ramp: from the start to the last point still heating at ≥ ``fraction`` of the rate.

    An isothermal hold after the ramp (temperature creeping a few degrees over many minutes) is
    left out, so a plot against temperature does not end in a vertical line.
    """
    rate = heating_rate(t_min, temp)
    if not math.isfinite(rate) or rate <= 0:
        return float(t_min[0]), float(t_min[-1])
    slope = np.gradient(smooth(temp, 9), t_min)
    fast = np.nonzero(slope >= fraction * rate)[0]
    if fast.size == 0:
        return float(t_min[0]), float(t_min[-1])
    return float(t_min[0]), float(t_min[fast[-1]])


@dataclass
class Processed:
    """The curves of one measurement, ready to draw (all arrays the same length)."""
    t: np.ndarray            # min
    T: np.ndarray            # °C
    tg: np.ndarray | None    # in tg_unit
    dta: np.ndarray | None   # in dta_unit
    dtg: np.ndarray | None   # in dtg_unit
    tg_unit: str
    dta_unit: str
    dtg_unit: str
    rate: float              # K/min
    notes: list[str] = field(default_factory=list)


TG_MODES = ("change_pct", "residual_pct", "change_mg")
DTG_BASES = ("per_min", "per_K")


def tg_at(x: np.ndarray, y: np.ndarray, value: float) -> float:
    """y at x = value (linear interpolation over the measured order; x need not be sorted)."""
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    if value < xs[0] or value > xs[-1]:
        return float("nan")
    return float(np.interp(value, xs, ys))


def process(m: Measurement, *, time_range=None, tg_mode="change_pct", mass_mg=None, tg_ref_temp=None,
            dta_per_mass=False, dta_flip=False, dta_zero=False, dta_smooth=0, tg_smooth=0,
            dtg_smooth=31, dtg_basis="per_min") -> Processed:
    """Cut the time range and compute TG (mass change or residual mass), DTA and DTG.

    tg_mode: change_pct = Δm/m₀ × 100 (starts at 0), residual_pct = 100 + Δm/m₀ × 100,
    change_mg = Δm. The percentages need the initial mass ``mass_mg`` unless the TG column is
    already in %; without it the curve stays in mg and a note says so. Δm is taken from the
    first point of the range, or from ``tg_ref_temp`` (°C) when given (for example after drying).
    """
    notes: list[str] = []
    t = m.time_min()
    T = m.column("temp")
    if T is None:
        raise TGDTAParseError("No temperature column.")
    T = T - 273.15 if m.units.get("temp") == "K" else T.copy()
    if t is None:
        t = np.arange(len(T), dtype=float)
        notes.append("no_time")
    tg_raw, dta_raw = m.column("tg"), m.column("dta")
    rate = heating_rate(t, T) if "no_time" not in notes else float("nan")
    keep = np.ones(len(t), bool)
    if time_range is not None:
        lo, hi = time_range
        keep = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if keep.sum() < 3:
        raise TGDTAParseError("Fewer than 3 points in the chosen time range.")
    t, T = t[keep], T[keep]

    tg = dtg = None
    tg_unit = dtg_unit = ""
    if tg_raw is not None:
        y = smooth(tg_raw[keep], tg_smooth)
        in_pct = m.units.get("tg") == "%"
        ref = y[0] if tg_ref_temp is None else tg_at(T, y, float(tg_ref_temp))
        if not math.isfinite(ref):
            ref = y[0]
            notes.append("ref_out_of_range")
        delta = y - ref
        mode = tg_mode if tg_mode in TG_MODES else "change_pct"
        if mode == "change_mg" and in_pct:
            mode = "change_pct"
        if mode == "change_mg":
            tg, tg_unit = delta, "mg"
        elif in_pct:
            tg, tg_unit = (delta if mode == "change_pct" else 100.0 + delta), "%"
        elif mass_mg and mass_mg > 0:
            pct = delta / float(mass_mg) * 100.0
            tg, tg_unit = (pct if mode == "change_pct" else 100.0 + pct), "%"
        else:
            tg, tg_unit = delta, "mg"
            notes.append("no_mass")
        # DTG from the same curve, smoothed (a derivative amplifies the noise of the balance).
        basis = dtg_basis if dtg_basis in DTG_BASES else "per_min"
        x = t if basis == "per_min" else T
        w = _odd_window(max(int(dtg_smooth or 0), 5), len(tg), 2)
        ys = savgol_filter(tg, w, 2) if w else tg
        with np.errstate(divide="ignore", invalid="ignore"):
            dtg = np.gradient(ys, t)
            if basis == "per_K":
                dTdt = np.gradient(smooth(T, w or 5), t)
                dtg = np.where(np.abs(dTdt) > 1e-6, dtg / dTdt, np.nan)
        dtg_unit = f"{tg_unit}/min" if basis == "per_min" else f"{tg_unit}/K"
        del x

    dta, dta_unit = None, ""
    if dta_raw is not None:
        y = smooth(dta_raw[keep], dta_smooth)
        unit = m.units.get("dta", "μV")
        if dta_flip:
            y = -y
        if dta_zero:
            y = y - y[0]
        if dta_per_mass:
            if mass_mg and mass_mg > 0:
                y = y / float(mass_mg)
                unit = f"{unit}/mg" if "/mg" not in unit else unit
            else:
                notes.append("no_mass_dta")
        dta, dta_unit = y, unit
    return Processed(t=t, T=T, tg=tg, dta=dta, dtg=dtg, tg_unit=tg_unit, dta_unit=dta_unit,
                     dtg_unit=dtg_unit, rate=rate, notes=notes)


def dta_peaks(x: np.ndarray, y: np.ndarray, prominence: float = 0.1, kind: str = "both",
              min_distance: float | None = None) -> list[tuple[int, str]]:
    """Exothermic (maxima, 'exo') and endothermic (minima, 'endo') peaks of a DTA curve.

    ``prominence`` is a fraction of the curve's range (0.1 = 10 %), as the XRD Maker peak sensitivity.
    ``y`` must already have exo upward (``process`` flips it when the instrument's sign is opposite).
    """
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    if finite.sum() < 5:
        return []
    span = float(np.nanmax(y) - np.nanmin(y))
    if span <= 0:
        return []
    distance = None
    if min_distance and len(x) > 1:
        step = float(np.nanmedian(np.abs(np.diff(x)))) or 1.0
        distance = max(1, int(min_distance / step))
    out: list[tuple[int, str]] = []
    filled = np.where(finite, y, np.nanmean(y))
    if kind in ("both", "exo"):
        idx, _ = find_peaks(filled, prominence=prominence * span, distance=distance)
        out += [(int(i), "exo") for i in idx]
    if kind in ("both", "endo"):
        idx, _ = find_peaks(-filled, prominence=prominence * span, distance=distance)
        out += [(int(i), "endo") for i in idx]
    return sorted(out)


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """'25-200, 200-800' or one range per line ('25–200', '25 ~ 200', '25 to 200') → [(25, 200), (200, 800)]."""
    out = []
    for part in re.split(r"[,\n;、]+", str(text or "")):
        m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|—|~|〜|～|to)\s*(-?\d+(?:\.\d+)?)\s*$", part)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if a != b:
                out.append((min(a, b), max(a, b)))
    return out


def parse_markers(text: str) -> list[tuple[float, str]]:
    """Vertical marker lines: one per line, 'value' or 'value | text' (the graph app's '|' format)."""
    out = []
    for line in str(text or "").splitlines():
        parts = [p.strip() for p in line.split("|", 1)]
        try:
            value = float(parts[0])
        except ValueError:
            continue
        if math.isfinite(value):
            out.append((value, parts[1] if len(parts) > 1 else ""))
    return out


def mass_change(T: np.ndarray, tg: np.ndarray, lo: float, hi: float) -> float:
    """TG(hi) − TG(lo) in the TG unit (negative for a loss)."""
    return tg_at(T, tg, hi) - tg_at(T, tg, lo)


def processed_table(name: str, p: Processed) -> str:
    """CSV text of the drawn values of one measurement (what is in the figure, for the paper's data file)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    head = ["sample", "time_min", "temperature_C"]
    cols = [p.t, p.T]
    for arr, label in ((p.tg, f"TG_{p.tg_unit}"), (p.dtg, f"DTG_{p.dtg_unit}"), (p.dta, f"DTA_{p.dta_unit}")):
        if arr is not None:
            head.append(label.replace("/", "_per_"))
            cols.append(arr)
    writer.writerow(head)
    for k in range(len(p.t)):
        writer.writerow([name] + [f"{c[k]:.6g}" for c in cols])
    return out.getvalue()
