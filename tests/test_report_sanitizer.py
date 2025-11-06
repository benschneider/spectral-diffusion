from pathlib import Path

from src.utils.report_sanitizer import sanitize_markdown, sanitize_markdown_text


def test_sanitize_markdown_text_handles_emoji_math_and_paths(tmp_path):
    figure = tmp_path / "figures" / "example.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"\x89PNG")

    absolute_path = str(figure.resolve())
    text = "Warning ⚠️\n![Example](%s)\nalpha_t = sqrt(beta_t)" % absolute_path

    sanitized = sanitize_markdown_text(text, tmp_path)
    assert "[Warning]" in sanitized
    assert "![Example](figures/example.png)" in sanitized
    assert "$\\alpha_t$" in sanitized
    assert "$\\sqrt{\\beta_t}$" in sanitized


def test_sanitize_markdown_file_round_trip(tmp_path):
    md_path = tmp_path / "report.md"
    md_path.write_text("![Img](/tmp/nonexistent.png)", encoding="utf-8")
    sanitize_markdown(md_path, tmp_path)
    content = md_path.read_text(encoding="utf-8")
    assert "nonexistent.png)" in content


def test_inline_latex_remains_valid(tmp_path):
    text = r"Formulation: \(x_t = sqrt(alpha_t)\)"
    sanitized = sanitize_markdown_text(text, tmp_path)
    assert "\\(x_t = \\sqrt{\\alpha_t}\\)" in sanitized
