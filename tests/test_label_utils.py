from src.utils.label_utils import shorten_label, shorten_labels

def test_shorten_label_path_and_metric_stem(tmp_path):
    metrics_path = tmp_path / "metrics" / "run_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_label = f"{metrics_path}"
    assert shorten_label(metrics_label) == "run_metrics"

    nested_path = tmp_path / "artifacts" / "long" / "config.yaml"
    nested_label = str(nested_path)
    assert shorten_label(nested_label) == "config.yaml"


def test_shorten_label_timestamp_and_truncation():
    timestamp = "20251102_142304"
    assert shorten_label(timestamp) == "2025-11-02 14:23"

    long_label = "a" * 40
    shortened = shorten_label(long_label, max_len=20)
    assert shortened.endswith("…")
    assert len(shortened) <= 20


def test_shorten_labels_vectorised_handles_duplicates():
    labels = ["/tmp/a/b/c.json", "/tmp/a/b/c.json", "metrics/run_A.json"]
    shortened = shorten_labels(labels)
    assert shortened == ["c", "c", "run_A"]
