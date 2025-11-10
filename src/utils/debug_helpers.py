import torch
import torch.nn.functional as F

def _fft_band_means(fft_data):
    """Compute low/mid/high band means for amplitude and phase, returned as a dict."""
    n = fft_data.shape[-1]
    low = torch.mean(fft_data[..., :n // 4])
    mid = torch.mean(fft_data[..., n // 4:n // 2])
    high = torch.mean(fft_data[..., n // 2:])
    return {
        "fft_low": low.item(),
        "fft_mid": mid.item(),
        "fft_high": high.item(),
    }

def _grad_norm(model):
    """Compute total gradient norm for diagnostic purposes."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def _parameter_delta(model, prev_params):
    """Return mean absolute delta between current and previous parameters."""
    deltas = []
    for (name, p), prev_p in zip(model.named_parameters(), prev_params):
        if p.requires_grad:
            deltas.append((p.data - prev_p).abs().mean().item())
    return sum(deltas) / len(deltas) if deltas else 0.0

def _structure_correlation(a, b):
    """Compute normalized correlation coefficient between tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    num = torch.dot(a_flat, b_flat)
    denom = torch.norm(a_flat) * torch.norm(b_flat)
    return (num / (denom + 1e-8)).item()

def _phase_rms(a, b=None, norm=None):
    """Compute RMS of phase differences between one or two tensors.
    If two tensors are provided, computes RMS of their phase difference.
    Optionally normalizes by `norm` (scalar or tensor)."""
    if b is not None:
        # Compute phase difference between tensors a and b
        phase_diff = torch.angle(torch.exp(1j * a) * torch.conj(torch.exp(1j * b)))
    else:
        # Single tensor: use adjacent differences
        phase_diff = torch.diff(a, dim=-1)

    rms = torch.sqrt(torch.mean(phase_diff ** 2))
    if norm is not None:
        if isinstance(norm, str):
            if norm.lower() == 'ortho':
                norm = 1.0
            else:
                try:
                    norm = float(norm)
                except ValueError:
                    raise TypeError("norm value must be numeric or 'ortho', got string that cannot be converted to float")
        norm = torch.tensor(norm, dtype=torch.float32, device=rms.device)
        rms = rms / (norm + 1e-8)
    return rms.item()

def _predict_x0(x_t, noise, sqrt_alpha_t, sqrt_one_minus_alpha_t, *args):
    """Predict x0 given noisy input x_t and noise ε.
    Accepts optional extra args for backward compatibility (e.g., snr_raw or clip flag).
    Automatically converts stringified tensor representations back to tensors if necessary."""
    import re

    def _ensure_tensor(v):
        if isinstance(v, str):
            # Try to parse numeric value from a string like "tensor(0.98, device='cuda:0')"
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", v)
            if match:
                return torch.tensor(float(match.group(0)), dtype=torch.float32)
            else:
                raise TypeError(f"_predict_x0: invalid string input {v!r}")
        elif isinstance(v, (tuple, list)):
            v = v[0]
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32)
        return v

    x_t = _ensure_tensor(x_t)
    noise = _ensure_tensor(noise)
    sqrt_alpha_t = _ensure_tensor(sqrt_alpha_t)
    sqrt_one_minus_alpha_t = _ensure_tensor(sqrt_one_minus_alpha_t)

    sqrt_alpha_t = sqrt_alpha_t.to(x_t.device)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.to(x_t.device)
    while sqrt_alpha_t.ndim < x_t.ndim:
        sqrt_alpha_t = sqrt_alpha_t[..., None]
    while sqrt_one_minus_alpha_t.ndim < x_t.ndim:
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t[..., None]

    return (x_t - sqrt_one_minus_alpha_t * noise) / sqrt_alpha_t

def _centered_rms(tensor):
    """Compute RMS of a tensor after mean-centering."""
    centered = tensor - tensor.mean()
    return torch.sqrt(torch.mean(centered ** 2)).item()

def _summarise_snr_spikes(
    snr_vals,
    sqrt_alpha_t=None,
    sqrt_one_minus_t=None,
    timesteps=None,
    clean=None,
    noisy=None,
    noise=None,
    target=None,
    prediction=None,
):
    """Summarise SNR tensor spikes and optionally log top spikes and correlation context."""
    snr_vals = snr_vals.detach()
    summary = {
        "mean_snr": float(snr_vals.mean().item()),
        "std_snr": float(snr_vals.std().item()),
        "max_snr": float(snr_vals.max().item()),
        "count": int(snr_vals.numel()),
    }

    # Optional: compute top spikes
    topk = torch.topk(snr_vals.flatten(), k=min(3, snr_vals.numel()))
    summary["top_timesteps"] = (
        timesteps[topk.indices].tolist() if timesteps is not None else []
    )
    summary["max_snr"] = float(topk.values.max().item())

    return summary

def _log_snr_spike(step, snr_stats):
    """Pretty-print SNR summary for diagnostics."""
    # Handle both legacy and new key names
    mean_val = snr_stats.get("mean", snr_stats.get("mean_snr", float("nan")))
    std_val = snr_stats.get("std", snr_stats.get("std_snr", float("nan")))
    max_val = snr_stats.get("max", snr_stats.get("max_snr", float("nan")))
    count_val = snr_stats.get("count", "?")
    top_ts = snr_stats.get("top_timesteps", [])

    msg = (
        f"[SNR] step={step:04d} mean={mean_val:.4f} "
        f"std={std_val:.4f} max={max_val:.4f} count={count_val}"
    )
    if top_ts:
        msg += f" top_timesteps={top_ts}"
    print(msg)
