from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["sanitize_markdown", "sanitize_markdown_text"]

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F100-\U0001F1FF\U00002700-\U000027BF\U00002600-\U000026FF]"
)
_INLINE_CODE_RE = re.compile(r"(`[^`]*`)")
_TOKEN_RE = re.compile(
    r"(?<!\\)\b(?P<name>alpha|beta|gamma|delta|theta|lambda|sigma|eta|mu|nu|phi|psi|omega|kappa|rho|tau|pi|epsilon|varepsilon|chi|zeta|iota|upsilon|xi)"
    r"(?P<suffix>(?:_[A-Za-z0-9]+|\^[A-Za-z0-9]+)*)\b"
)
_VARIABLE_RE = re.compile(
    r"(?<!\\)\b(?P<name>[A-Za-z])(?P<suffix>(?:_[A-Za-z0-9]+|\^[A-Za-z0-9]+)+)\b"
)
_SQRT_RE = re.compile(r"sqrt\((?P<body>[^`$]*?)\)")
_LATEX_INLINE_RE = re.compile(r"\\\((?P<body>.+?)\\\)")
_LATEX_DISPLAY_RE = re.compile(r"\\\[(?P<body>.+?)\\\]")

_GREEK_LATEX = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "theta": "theta",
    "lambda": "lambda",
    "sigma": "sigma",
    "eta": "eta",
    "mu": "mu",
    "nu": "nu",
    "phi": "phi",
    "psi": "psi",
    "omega": "omega",
    "kappa": "kappa",
    "rho": "rho",
    "tau": "tau",
    "pi": "pi",
    "varepsilon": "varepsilon",
    "chi": "chi",
    "zeta": "zeta",
    "iota": "iota",
    "upsilon": "upsilon",
    "xi": "xi",
}
_IMAGE_LINK_RE = re.compile(r"(!?\[[^\]]*\]\()(?P<url>[^)]+)(\))")
_HTML_IMG_RE = re.compile(r'(<img[^>]*?src=")(?P<url>[^"<>]+)(")', re.IGNORECASE)


def sanitize_markdown(md_path: str | os.PathLike[str], report_root: str | os.PathLike[str]) -> None:
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    text = path.read_text(encoding="utf-8")
    sanitized = sanitize_markdown_text(text, report_root)
    path.write_text(sanitized, encoding="utf-8")


def sanitize_markdown_text(text: str, report_root: str | os.PathLike[str]) -> str:
    root = Path(report_root)
    lines = text.splitlines()
    sanitized_lines: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            sanitized_lines.append(line)
            in_code_block = not in_code_block
            continue
        if in_code_block:
            sanitized_lines.append(line)
            continue

        updated = _replace_emoji(line)
        updated = _wrap_math_tokens(updated)
        sanitized_lines.append(updated)

    sanitized_text = "\n".join(sanitized_lines)
    sanitized_text = _relativize_paths(sanitized_text, root)
    return sanitized_text


def _replace_emoji(text: str) -> str:
    if not text:
        return text
    text = text.replace("⚠️", "[Warning]")
    text = text.replace("⚠", "[Warning]")
    return _EMOJI_RE.sub("[Warning]", text)


def _wrap_math_tokens(text: str) -> str:
    if not text:
        return text

    segments = _INLINE_CODE_RE.split(text)
    for idx in range(0, len(segments), 2):
        segments[idx] = _wrap_math_segment(segments[idx])
    return "".join(segments)


def _wrap_math_segment(segment: str) -> str:
    if not segment:
        return segment

    placeholders: list[str] = []

    def store(value: str) -> str:
        placeholders.append(value)
        return f"@@LATEX{len(placeholders) - 1}@@"

    segment = _LATEX_INLINE_RE.sub(
        lambda m: store(f"\\({_convert_math_body(m.group('body'))}\\)"),
        segment,
    )
    segment = _LATEX_DISPLAY_RE.sub(
        lambda m: store(f"\\[{_convert_math_body(m.group('body'))}\\]"),
        segment,
    )

    def wrap(match: re.Match[str]) -> str:
        token = match.group(0)
        return f"${token}$"

    segment = _apply_sqrt(segment, wrap=True)
    segment = _wrap_standalone_tokens(segment)
    segment = _wrap_variable_tokens(segment)
    for idx, value in enumerate(placeholders):
        segment = segment.replace(f"@@LATEX{idx}@@", value)
    return segment


def _apply_sqrt(text: str, *, wrap: bool) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        converted = _convert_plain_tokens(body)
        if wrap:
            return f"$\\sqrt{{{converted}}}$"
        return f"\\sqrt{{{converted}}}"

    return _SQRT_RE.sub(repl, text)


def _convert_math_body(body: str) -> str:
    converted = _apply_sqrt(body, wrap=False)
    return _convert_plain_tokens(converted)


def _wrap_standalone_tokens(text: str) -> str:
    segments = text.split("$")
    if len(segments) == 1:
        return _TOKEN_RE.sub(_token_to_math, text)

    rebuilt: list[str] = []
    for idx, segment in enumerate(segments):
        if idx % 2 == 1:
            rebuilt.append(segment)
            continue
        rebuilt.append(_TOKEN_RE.sub(_token_to_math, segment))
    return "$".join(rebuilt)


def _convert_plain_tokens(text: str) -> str:
    return _TOKEN_RE.sub(lambda m: _format_token(m.group("name"), m.group("suffix")), text)


def _token_to_math(match: re.Match[str]) -> str:
    latex = _format_token(match.group("name"), match.group("suffix"))
    return f"${latex}$"


def _wrap_variable_tokens(text: str) -> str:
    segments = text.split("$")
    if len(segments) == 1:
        return _VARIABLE_RE.sub(_variable_to_math, text)

    rebuilt: list[str] = []
    for idx, segment in enumerate(segments):
        if idx % 2 == 1:
            rebuilt.append(segment)
            continue
        rebuilt.append(_VARIABLE_RE.sub(_variable_to_math, segment))
    return "$".join(rebuilt)


def _variable_to_math(match: re.Match[str]) -> str:
    latex = _format_variable(match.group("name"), match.group("suffix"))
    return f"${latex}$"


def _format_token(name: str, suffix: str) -> str:
    base = f"\\{_GREEK_LATEX.get(name, name)}"
    if not suffix:
        return base
    formatted = []
    for marker, value in re.findall(r"([_^])([A-Za-z0-9]+)", suffix):
        if len(value) > 1:
            formatted.append(f"{marker}{{{value}}}")
        else:
            formatted.append(f"{marker}{value}")
    return base + "".join(formatted)


def _format_variable(name: str, suffix: str) -> str:
    formatted = name
    for marker, value in re.findall(r"([_^])([A-Za-z0-9]+)", suffix):
        if len(value) > 1:
            formatted += f"{marker}{{{value}}}"
        else:
            formatted += f"{marker}{value}"
    return formatted


def _relativize_paths(text: str, report_root: Path) -> str:
    def _adjust_url(url: str) -> str:
        if not url:
            return url
        stripped = url.strip()
        if stripped.startswith(("http://", "https://", "data:")):
            return stripped
        if stripped.startswith("#"):
            return stripped
        path = Path(stripped)
        if path.is_absolute():
            try:
                rel = os.path.relpath(path, report_root)
            except ValueError:
                rel = path.name
            return rel
        return stripped

    def md_repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        url = match.group("url")
        suffix = match.group(3)
        adjusted = _adjust_url(url)
        return f"{prefix}{adjusted}{suffix}"

    text = _IMAGE_LINK_RE.sub(md_repl, text)

    def html_repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        url = match.group("url")
        suffix = match.group(3)
        adjusted = _adjust_url(url)
        return f"{prefix}{adjusted}{suffix}"

    return _HTML_IMG_RE.sub(html_repl, text)
