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
import struct
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter


SUPPORTED_EXTENSIONS = ("csv", "txt", "tsv", "dat", "xlsx", "mes")
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
    mass_mg: float | None = None          # sample mass written in the file (Rigaku: "Weight"), if any
    source: str = "text"                  # text | xlsx | rigaku-mes
    notes: list[str] = field(default_factory=list)

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


_TAG = re.compile(r"#[A-Za-z]{1,4}$")          # Rigaku TPASCII row tags: #GD (data), #GI, #MY, …
_WEIGHT = re.compile(r"weight\s*[:：]?[\s,\t]*([0-9]+(?:\.[0-9]+)?)\s*mg", re.IGNORECASE)


def _table_from_lines(lines: list[str]) -> tuple[list[str], np.ndarray]:
    """The numeric table of a text export, with the column names of the header line above it."""
    sample = "\n".join(lines[:60])
    counts = {d: sample.count(d) for d in (",", "\t", ";")}
    delimiter = max(counts, key=counts.get) if max(counts.values()) else None
    rows, header = [], None
    for line in lines:
        fields = [f.strip() for f in _split(line, delimiter)]
        while fields and not fields[-1]:          # the trailing comma of the export
            fields.pop()
        while fields and (not fields[0] or _TAG.match(fields[0])):
            fields.pop(0)                         # Rigaku's '#GD' tag and an empty first cell
        if not fields:
            continue
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
    columns = list(header[:width]) if header and len(header) >= width else [f"column {i + 1}" for i in range(width)]
    columns += [f"column {i + 1}" for i in range(len(columns), width)]
    return columns, np.asarray(rows, dtype=float)


def _mass_from_text(lines: list[str]) -> float | None:
    for line in lines[:200]:
        m = _WEIGHT.search(line)
        if m:
            value = float(m.group(1))
            if 0 < value < 1e4:
                return value
    return None


def _xlsx_sheets(raw: bytes) -> list[list[str]]:
    """Every sheet of a workbook as tab-separated lines (Rigaku's Excel export: Sheet1 conditions, Sheet2 data)."""
    try:
        import openpyxl
    except ImportError as exc:             # pragma: no cover - listed in requirements.txt
        raise TGDTAParseError("Reading .xlsx needs openpyxl.") from exc
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # e.g. "wmf image format is not supported"
            book = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except Exception as exc:
        raise TGDTAParseError("This .xlsx file cannot be opened.") from exc
    sheets = []
    for sheet in book.worksheets:
        lines = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if v is None else (repr(v) if isinstance(v, float) else str(v)).replace("\t", " ") for v in row]
            if any(c.strip() for c in cells):
                lines.append("\t".join(cells))
        sheets.append(lines)
    book.close()
    return sheets


# Rigaku ThermoPlus EVO measurement file (.MES, "TPE MeasFile"): binary. Each measured series is an int32
# point count followed by that many float32 values; 60 bytes before the count sits the series' type code.
MES_SIGNATURE = b"ThermoPlusE"
MES_TYPES = {1: ("Temp.(deg C)", "temp"), 3: ("TG(mg)", "tg"), 5: ("DTA(uV)", "dta")}


def _mes_series(raw: bytes) -> list[tuple[int, int, np.ndarray]]:
    """(offset of the count, type code, values) of every float32 series of the same, longest length."""
    found = []
    for align in range(4):
        n_words = (len(raw) - align) // 4
        words = np.frombuffer(raw, "<i4", count=n_words, offset=align)
        for k in np.nonzero((words >= 20) & (words <= 5_000_000))[0]:
            count = int(words[k])
            p = align + 4 * int(k)
            if p + 4 + 4 * count > len(raw) or p < 64:
                continue
            values = np.frombuffer(raw, "<f4", count=count, offset=p + 4)
            if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1e7:
                continue
            code = struct.unpack_from("<i", raw, p - 60)[0]
            found.append((p, code, values.astype(float)))
    if not found:
        return []
    lengths = {}
    for p, code, v in found:
        if code in MES_TYPES:
            lengths[len(v)] = lengths.get(len(v), 0) + 1
    if not lengths:
        return []
    n = max(lengths, key=lambda length: (lengths[length], length))
    return [(p, code, v) for p, code, v in found if len(v) == n and code in MES_TYPES]


def _mes_time_range(raw: bytes, temp: np.ndarray) -> tuple[float, float] | None:
    """Start and end time (min) from the x-axis record: [t min][t max][…][T min][T max], T as in the series."""
    pattern = struct.pack("<ff", float(np.float32(temp.min())), float(np.float32(temp.max())))
    start = 0
    while True:
        q = raw.find(pattern, start)
        if q < 0:
            return None
        start = q + 1
        if q < 12:
            continue
        t0, t1 = struct.unpack_from("<f", raw, q - 12)[0], struct.unpack_from("<f", raw, q - 8)[0]
        if math.isfinite(t0) and math.isfinite(t1) and 0 <= t0 < t1 < 1e5:
            step_s = (t1 - t0) * 60 / max(len(temp) - 1, 1)
            if 0.01 <= step_s <= 600:
                return t0, t1


def _mes_mass(raw: bytes) -> float | None:
    """The sample weight: a float32 after the condition field 0x01B6 (followed by the pan name)."""
    q = raw.find(b"\x02\x00\xb6\x01")
    while q >= 0:
        value = struct.unpack_from("<f", raw, q + 4)[0] if q + 8 <= len(raw) else float("nan")
        if math.isfinite(value) and 1e-3 <= value < 1e4:      # skips the same bytes elsewhere (denormal floats)
            return round(float(value), 4)
        q = raw.find(b"\x02\x00\xb6\x01", q + 1)
    return None


def _read_mes(raw: bytes) -> tuple[list[str], np.ndarray, float | None, list[str]]:
    series = _mes_series(raw)
    by_signal = {}
    for p, code, values in series:
        by_signal.setdefault(MES_TYPES[code][1], (MES_TYPES[code][0], values))
    if "temp" not in by_signal:
        raise TGDTAParseError("No temperature series was found in this Rigaku .MES file.")
    temp = by_signal["temp"][1]
    notes = []
    span = _mes_time_range(raw, np.asarray(temp, np.float32))
    if span is None:
        t = np.arange(len(temp)) / 60.0                 # 1 s sampling, the instrument default
        notes.append("mes_time_assumed")
    else:
        t = np.linspace(span[0], span[1], len(temp))
    columns, cols = ["Time(min)"], [t]
    for signal in ("temp", "tg", "dta"):
        if signal in by_signal:
            columns.append(by_signal[signal][0])
            cols.append(by_signal[signal][1])
    return columns, np.column_stack(cols), _mes_mass(raw), notes


def read_measurement(raw: bytes, name: str = "", mapping: dict | None = None) -> Measurement:
    """Parse one file: a text export (CSV / TXT, Rigaku TPASCII), an Excel export (.xlsx) or a Rigaku
    ThermoPlus EVO measurement file (.MES). ``mapping`` (signal → column index) overrides the automatic choice."""
    if len(raw) > MAX_FILE_BYTES:
        raise TGDTAParseError("The file is larger than 40 MB.")
    mass, notes = None, []
    if raw.startswith(MES_SIGNATURE):
        source = "rigaku-mes"
        columns, data, mass, notes = _read_mes(raw)
    elif raw[:2] == b"PK":
        source = "xlsx"
        sheets = _xlsx_sheets(raw)
        table = None
        for lines in sheets:
            try:
                table = _table_from_lines(lines)
                break
            except TGDTAParseError:
                continue
        if table is None:
            raise TGDTAParseError("No numeric table with at least 5 rows was found in the workbook.")
        columns, data = table
        mass = next((m for m in (_mass_from_text(lines) for lines in sheets) if m), None)
    else:
        source = "text"
        text = decode_text(raw)
        lines = [line.rstrip("\r\n") for line in text.splitlines()]
        lines = [line for line in lines if line.strip()]
        if not lines:
            raise TGDTAParseError("The file is empty.")
        columns, data = _table_from_lines(lines)
        mass = _mass_from_text(lines)
    width = data.shape[1]
    auto = guess_mapping(columns)
    if mapping:
        for signal in SIGNALS:
            if signal in mapping:
                value = mapping[signal]
                auto[signal] = value if isinstance(value, int) and 0 <= value < width else None
    units = {signal: _unit_of(signal, columns[i]) for signal, i in auto.items() if i is not None}
    return Measurement(name=name, columns=columns, data=data, mapping=auto, units=units, mass_mg=mass,
                       source=source, notes=notes)


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


def peak_onset(x: np.ndarray, y: np.ndarray, p: int, kind: str = "exo", window: float | None = None) -> dict | None:
    """Extrapolated onset of a DTA peak (Tx of a crystallization peak): where the tangent at the steepest
    point of the leading edge crosses the baseline extrapolated from before the peak (ISO 11357 / ASTM E794).

    ``x`` must increase (a heating run); ``kind`` 'exo' for a maximum, 'endo' for a minimum.
    ``window`` (x units) bounds the search around the peak; by default 10 % of the x span.
    Returns the onset and the construction lines, or None when the peak has no clear leading edge.
    """
    from scipy.signal import peak_prominences, peak_widths
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or not 0 < p < len(x) - 1 or x[-1] <= x[0]:
        return None
    # The recorded temperature repeats a value now and then (12.98, 12.98): use a smoothed, rising copy.
    x = np.maximum.accumulate(smooth(x, 9)) + 1e-9 * np.arange(len(x))
    s = 1.0 if kind == "exo" else -1.0
    ys = s * y
    step = float(np.median(np.diff(x)))
    window = window or 0.1 * float(x[-1] - x[0])
    wlen = max(5, int(2 * window / step) | 1)
    prom = peak_prominences(ys, [p], wlen=wlen)
    if prom[0][0] <= 0:
        return None
    # Foot of the peak: where the leading edge has risen 10 % of the (local) peak height.
    foot_i = peak_widths(ys, [p], rel_height=0.9, prominence_data=prom)[2][0]
    foot = float(np.interp(foot_i, np.arange(len(x)), x))
    width = max(float(x[p]) - foot, 3 * step)
    # Leading edge: the steepest point between the foot and the peak.
    lo = int(np.searchsorted(x, foot))
    if p - lo < 2:
        return None
    dxdi = np.maximum(np.gradient(smooth(x, 21)), 1e-12)
    slope = np.gradient(smooth(ys, 5)) / dxdi
    i = lo + int(np.argmax(slope[lo:p + 1]))
    m = float(slope[i])
    if m <= 0:
        return None
    # Baseline: straight line on the flat part well before the foot (5 to 2 foot-to-peak widths before it),
    # clear of the slow rise that often precedes a crystallization peak.
    base = (x >= foot - 5 * width) & (x <= foot - 2 * width)
    if base.sum() < 3:
        return None
    a, b = np.polyfit(x[base], ys[base], 1)
    if abs(a - m) < 1e-12:
        return None
    onset = (ys[i] - m * x[i] - b) / (a - m)
    if not (foot - 3 * width <= onset <= x[p]):
        return None
    return {"x": float(onset), "y": float(s * (a * onset + b)), "x_inflection": float(x[i]), "y_inflection": float(y[i]),
            "slope": s * m, "baseline": (s * float(a), s * float(b)), "x_base_start": float(x[base][0]),
            "x_peak": float(x[p]), "y_peak": float(y[p])}


def glass_transition(x: np.ndarray, y: np.ndarray, lo: float, hi: float, exo_up: bool = True,
                     baseline_fraction: float = 0.4) -> dict | None:
    """Glass transition from a DTA step within [lo, hi] (x units): the extrapolated onset, where the baseline
    (straight line fitted to the first ``baseline_fraction`` of the range) meets the tangent at the steepest
    endothermic point (the inflection, Ti) — the construction of ISO 11357-2 and ASTM E1356.

    ``exo_up``: exotherm drawn upward, so the endothermic step goes down. Returns None when no step is found.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 20 or x[-1] <= x[0] or not lo < hi:
        return None
    x = np.maximum.accumulate(smooth(x, 9)) + 1e-9 * np.arange(len(x))
    inside = np.nonzero((x >= lo) & (x <= hi))[0]
    if inside.size < 20:
        return None
    s = -1.0 if exo_up else 1.0                 # endothermic direction as positive
    step = float(np.median(np.diff(x[inside])))
    # Smooth over ~1/12 of the range (at least 5 points): the step is slow, the noise of the DTA is not.
    w = max(5, int((hi - lo) / 12 / max(step, 1e-9)))
    ys = s * smooth(y, w)
    split = lo + baseline_fraction * (hi - lo)
    base = inside[x[inside] <= split]
    rest = inside[x[inside] > split]
    if base.size < 5 or rest.size < 5:
        return None
    a, b = np.polyfit(x[base], ys[base], 1)
    slope = np.gradient(ys) / np.maximum(np.gradient(smooth(x, w)), 1e-12)
    i = int(rest[np.argmax(slope[rest])])
    m = float(slope[i])
    if m <= a or abs(m - a) < 1e-12:
        return None
    # The step must stand out from the scatter of the unsmoothed signal about the baseline.
    noise = float(np.std(s * y[base] - (a * x[base] + b))) or 1e-12
    depth = float(np.max(ys[rest] - (a * x[rest] + b)))
    if depth < 3 * noise:
        return None
    onset = (ys[i] - m * x[i] - b) / (a - m)
    if not lo <= onset <= x[i]:
        return None
    return {"x": float(onset), "y": float(s * (a * onset + b)), "x_inflection": float(x[i]),
            "y_inflection": float(s * ys[i]), "slope": s * m, "baseline": (s * float(a), s * float(b)),
            "x_base_start": float(x[base][0]), "depth": s * depth, "noise": noise}
