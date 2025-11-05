import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils.plot_style import (
    declutter_texts,
    is_duplicate,
    reduce_tick_density,
    shorten_label,
)


def test_shorten_label_handles_paths_and_timestamps(tmp_path):
    metrics_file = tmp_path / "metrics" / "sample_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text("{}")

    assert shorten_label(str(metrics_file)) == "sample_metrics"
    assert shorten_label("20251102_142304") == "2025-11-02 14:23"


def test_reduce_tick_density_hides_labels():
    fig, ax = plt.subplots()
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(i) for i in range(20)])
    reduce_tick_density(ax, max_ticks=5)
    visible = sum(1 for lbl in ax.get_xticklabels() if lbl.get_visible())
    assert visible <= 6
    plt.close(fig)


def test_declutter_texts_hides_overlapping():
    fig, ax = plt.subplots()
    ax.text(0.1, 0.1, "A")
    ax.text(0.11, 0.1, "B")
    ax.text(0.5, 0.5, "C")
    declutter_texts(ax, min_dist=0.05)
    visible = [txt.get_visible() for txt in ax.texts]
    assert visible.count(True) == 2
    plt.close(fig)


def test_is_duplicate_detects_identical(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    content = b"binary"
    path_a.write_bytes(content)
    path_b.write_bytes(content)

    seen = set()
    assert not is_duplicate(path_a, seen)
    assert is_duplicate(path_b, seen)
