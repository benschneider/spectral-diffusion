import torch.nn as nn


def _batchnorm_layers(module: nn.Module):
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            yield child


def test_batchnorm_running_stats_change(spectral_model, cifar_batch):
    """
    Ensure BatchNorm statistics actually update when the model processes data.
    If no BatchNorm layers are present, the test passes trivially.
    """
    bn_layers = list(_batchnorm_layers(spectral_model))
    if not bn_layers:
        return

    spectral_model.train()
    before_means = [layer.running_mean.clone() for layer in bn_layers]
    before_vars = [layer.running_var.clone() for layer in bn_layers]

    for _ in range(5):
        _ = spectral_model(cifar_batch)

    after_means = [layer.running_mean for layer in bn_layers]
    after_vars = [layer.running_var for layer in bn_layers]

    deltas = [
        (after - before).abs().mean().item()
        for before, after in zip(before_means, after_means)
    ]
    var_deltas = [
        (after - before).abs().mean().item()
        for before, after in zip(before_vars, after_vars)
    ]

    changed = any(delta > 1e-5 for delta in deltas) or any(
        delta > 1e-5 for delta in var_deltas
    )
    assert changed, "BatchNorm running statistics did not change after updates."
