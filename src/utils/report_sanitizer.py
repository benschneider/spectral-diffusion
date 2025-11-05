from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["sanitize_markdown", "sanitize_markdown_text"]

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F100-\U0001F1FF\U00002700-\U000027BF\U00002600-\U000026FF]"
)
_INLINE_CODE_RE = re.compile(r"(`[^`]*`)")
_MATH_TOKEN_RE = re.compile(
    r"(?<!\$)\b(?:alpha|beta|gamma|delta|theta|lambda|sigma|eta|mu|nu|phi|psi|omega|kappa|rho|tau|pi)"
    r"(?:_[A-Za-z0-9]+|\^[A-Za-z0-9]+)\b(?!\$)"
)
_SQRT_RE = re.compile(r"(?<!\$)(sqrt\([^`$]*?\))(?!\$)")
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

    def wrap(match: re.Match[str]) -> str:
        token = match.group(0)
        return f"${token}$"

    segment = _SQRT_RE.sub(wrap, segment)
    segment = _MATH_TOKEN_RE.sub(wrap, segment)
    return segment


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
