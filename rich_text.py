"""Text markup shared with the graph app (figapp).

Labels are written as in the graph app:
    <i>…</i> italic, <b>…</b> bold, _{…} or <sub>…</sub> subscript, ^{…} or <sup>…</sup> superscript,
    ^2 / ^-1 (a number right after ^) superscript, $...$ matplotlib mathtext as it is.
For example ``Diffraction angle, 2<i>θ</i> (deg.)``, ``Yb_{2}Si_{2}O_{7}``, ``Eu^{3+}``, ``cm^-1``.
A plain ``_`` or ``^`` (file names such as ``5SiC_750C``) stays as written; ``\\_`` and ``\\^`` force it.

``rich`` makes the matplotlib text (the same function as figapp's), ``to_html`` the text for
the Plotly preview, ``plain`` text without markup. Labels saved by earlier XRD Maker versions
(``$\\it{TEXT}$``, ``$_{N}$``, ``$^{N}$``) are mathtext and keep working in all three.
"""

from __future__ import annotations

import html
import re


_TAG = re.compile(r"<(/?)(i|b|sub|sup)>", re.I)
_SUPNUM = re.compile(r"\^([+\-−]?\d+(?:\.\d+)?)")


def _runs(s, it=0, bd=0, script=None):
    """Split marked-up text into ('math', text) and ('text', text, italic, bold, script) pieces."""
    out, buf, i = [], [], 0

    def flush():
        if buf:
            out.append(("text", "".join(buf), it > 0, bd > 0, script))
            buf.clear()
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in "_^$<":
            buf.append(s[i + 1] if s[i + 1] != "$" else "\\$")
            i += 2
            continue
        if c == "$":
            j = i + 1
            while j < len(s) and not (s[j] == "$" and s[j - 1] != "\\"):
                j += 1
            if j < len(s):
                flush()
                out.append(("math", s[i:j + 1]))
                i = j + 1
                continue
        m = _TAG.match(s, i)
        if m:
            flush()
            tag, closing = m.group(2).lower(), m.group(1) == "/"
            if tag == "i":
                it = max(0, it - 1) if closing else it + 1
            elif tag == "b":
                bd = max(0, bd - 1) if closing else bd + 1
            elif script is None or closing:
                k = s.lower().find(f"</{tag}>", m.end()) if not closing else -1
                if not closing:
                    inner_end = k if k >= 0 else len(s)
                    out.extend(_runs(s[m.end():inner_end], it, bd, tag))
                    i = inner_end + (len(tag) + 3 if k >= 0 else 0)
                    continue
            i = m.end()
            continue
        if c in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            j = s.find("}", i + 2)
            if j > i + 2:
                flush()
                out.extend(_runs(s[i + 2:j], it, bd, "sub" if c == "_" else "sup"))
                i = j + 1
                continue
        m = _SUPNUM.match(s, i) if c == "^" and script is None else None
        if m:
            flush()
            out.append(("text", m.group(1), it > 0, bd > 0, "sup"))
            i = m.end()
            continue
        buf.append(c)
        i += 1
    flush()
    return out


def _mathrm(t):
    """Text inside \\mathrm{} etc.: keep spaces, escape % # & { }, and write + and − without the extra space mathtext puts
    around binary operators (Eu^{3+} → 3+, not 3 +)."""
    t = t.replace("−", "-")
    t = re.sub(r"([ %#&{}])", lambda m: "\\ " if m.group(1) == " " else "\\" + m.group(1), t)
    return re.sub(r"([+\-])", r"{\1}", t)


def rich(s):
    """Text for the matplotlib figure: 'Yb_{2}Si_{2}O_{7}', '2<i>θ</i> (deg.)', 'cm^-1', '<b>a</b>' → mathtext."""
    s = "" if s is None else str(s)
    if not any(ch in s for ch in "_^<"):
        return s
    if len(re.findall(r"(?<!\\)\$", s)) % 2:          # an odd number of $: leave it as it is
        return s
    out, group, gscript = [], [], None

    def close():
        if group:
            body = "".join(group)
            out.append("$" + {"sub": "_{", "sup": "^{"}.get(gscript, "") + body + ("}" if gscript else "") + "$")
            group.clear()
    for r in _runs(s):
        if r[0] == "math":
            close()
            out.append(r[1])
            continue
        _, t, it, bd, script = r
        if not it and not bd and script is None:
            close()
            out.append(t)
            continue
        cmd = "\\mathbf" if bd else ("\\mathit" if it else "\\mathrm")
        if script != gscript:
            close()
            gscript = script
        group.append(f"{cmd}{{{_mathrm(t)}}}")
    close()
    return "".join(out)


# ------------------------------------------------------------------------------------------------ mathtext → HTML
_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ",
    "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_SYMBOLS = {
    "circ": "∘", "degree": "°", "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "AA": "Å",
    "sim": "∼", "approx": "≈", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "infty": "∞", "prime": "′", "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "uparrow": "↑", "downarrow": "↓", "partial": "∂", "nabla": "∇", "langle": "⟨", "rangle": "⟩",
    "perp": "⊥", "parallel": "∥", "propto": "∝", "ell": "ℓ", "hbar": "ℏ", "dots": "…", "ldots": "…",
    "cdots": "⋯", "bullet": "•", "star": "⋆", "dagger": "†", "minus": "−",
    ",": " ", ";": " ", ":": " ", ">": " ", "!": "", " ": " ", "quad": " ", "qquad": "  ",
    "%": "%", "#": "#", "&": "&", "$": "$", "{": "{", "}": "}", "_": "_", "^": "^", "|": "‖", "backslash": "\\",
}
_STYLE_COMMANDS = {
    "it": "it", "mathit": "it", "textit": "it", "rm": "rm", "mathrm": "rm", "mathregular": "rm", "mathsf": "rm",
    "mathtt": "rm", "mathcal": "rm", "operatorname": "rm", "mathdefault": "default", "bf": "bf", "mathbf": "bf",
    "textbf": "bf", "boldsymbol": "bf", "mathbfit": "bfit", "text": "text", "textrm": "text",
}


def _styled(text, style):
    """One piece of math text in HTML; letters are italic in default math style, as matplotlib draws them."""
    escaped = html.escape(text, quote=False)
    if style in ("it", "bfit") or (style == "default" and text.isalpha()):
        escaped = f"<i>{escaped}</i>"
    if style in ("bf", "bfit"):
        escaped = f"<b>{escaped}</b>"
    return escaped


def _math_token(s, i, style):
    """HTML for one token at s[i] (a group, a command with its argument, or a character) and the next index."""
    c = s[i]
    if c == "{":
        return _math_group(s, i + 1, style, "}")
    if c == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        name = m.group(1) if m else ""
        i += len(m.group(0)) if m else 1
        if name in _STYLE_COMMANDS:
            return _math_argument(s, i, _STYLE_COMMANDS[name])
        if name in _GREEK:
            return _styled(_GREEK[name], style), i
        if name in _SYMBOLS:
            return html.escape(_SYMBOLS[name], quote=False), i
        if name in ("frac", "dfrac", "tfrac"):
            num, i = _math_argument(s, i, style)
            den, i = _math_argument(s, i, style)
            return f"{num}/{den}", i
        if name == "sqrt":
            body, i = _math_argument(s, i, style)
            return f"√({body})", i
        if name in ("left", "right", "mathbb", "overline", "bar", "hat", "tilde", "vec", "dot", "widehat", "widetilde"):
            return "", i
        return html.escape(name, quote=False), i
    if c in " \t\n" and style != "text":
        return "", i + 1                      # mathtext ignores plain spaces
    return _styled(c, "rm" if style == "text" else style), i + 1


def _math_argument(s, i, style):
    """The argument of a command or of _ / ^: a {group} or a single token."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _math_group(s, i + 1, style, "}")
    return _math_token(s, i, style)


def _math_group(s, i, style, stop):
    out = []
    while i < len(s):
        c = s[i]
        if stop and c == stop:
            return "".join(out), i + 1
        if c in "_^":
            body, i = _math_argument(s, i + 1, style)
            tag = "sub" if c == "_" else "sup"
            out.append(f"<{tag}>{body}</{tag}>")
            continue
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            name = m.group(1) if m else ""
            # \it, \rm, \bf without braces switch the style for the rest of the group (TeX style).
            if name in ("it", "rm", "bf") and not re.match(r"\s*\{", s[i + len(m.group(0)):]):
                style = _STYLE_COMMANDS[name]
                i += len(m.group(0))
                continue
        piece, i = _math_token(s, i, style)
        out.append(piece)
    return "".join(out), i


def math_to_html(expr):
    """The inside of a $...$ mathtext expression as HTML for Plotly (common commands only)."""
    return _math_group(str(expr), 0, "default", None)[0].replace("</b><b>", "").replace("</i><i>", "")


def to_html(s):
    """Text for the Plotly preview (HTML subset: <i>, <b>, <sub>, <sup>), from the same markup as ``rich``."""
    s = "" if s is None else str(s)
    out = []
    for r in _runs(s):
        if r[0] == "math":
            out.append(math_to_html(r[1][1:-1]))
            continue
        _, t, it, bd, script = r
        h = html.escape(t.replace("\\$", "$"), quote=False)
        if it:
            h = f"<i>{h}</i>"
        if bd:
            h = f"<b>{h}</b>"
        if script:
            h = f"<{script}>{h}</{script}>"
        out.append(h)
    return "".join(out)


def plain(s):
    """Text without markup (file names, hover text): 'Yb_{2}Si_{2}O_{7}' → 'Yb2Si2O7', '2<i>θ</i>' → '2θ'."""
    return html.unescape(re.sub(r"<[^>]+>", "", to_html(s)))
