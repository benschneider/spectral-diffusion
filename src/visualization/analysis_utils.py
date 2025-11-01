import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

def exp_decay(t, a, b, c):
    """
    Exponential decay function: a * exp(-b * t) + c
    """
    return a * np.exp(-b * t) + c

def fit_loss_curve(steps, loss):
    """
    Fit an exponential decay curve to the loss data.

    Parameters:
    steps (array-like): The steps or epochs.
    loss (array-like): The loss values.

    Returns:
    tuple: (fit_k, r2_score, fitted_curve)
        fit_k (float): The decay rate parameter b from the fit.
        r2_score (float): The coefficient of determination for the fit.
        fitted_curve (np.ndarray): The fitted loss curve values.
    """
    popt, _ = curve_fit(exp_decay, steps, loss, bounds=(0, [np.inf, np.inf, np.inf]))
    fitted_curve = exp_decay(steps, *popt)
    residuals = loss - fitted_curve
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((loss - np.mean(loss))**2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    fit_k = popt[1]
    return fit_k, r2_score, fitted_curve

def _safe_ratio(a, b):
    """
    Safely compute the ratio a/b, returning np.nan if b is zero.

    Parameters:
    a (float): Numerator.
    b (float): Denominator.

    Returns:
    float: a / b or np.nan if b == 0.
    """
    return a / b if b != 0 else np.nan

def _benchmark_takeaways(df):
    """
    Generate a brief summary of benchmark results from a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing benchmark data with columns like 'runtime', 'loss', etc.

    Returns:
    dict: Summary dict with min/max runtime, loss, and other key metrics.
    """
    summary = {}
    if 'runtime' in df.columns:
        summary['min_runtime'] = df['runtime'].min()
        summary['max_runtime'] = df['runtime'].max()
    if 'loss' in df.columns:
        summary['min_loss'] = df['loss'].min()
        summary['max_loss'] = df['loss'].max()
    if 'steps' in df.columns:
        summary['total_steps'] = df['steps'].max()
    if 'efficiency' in df.columns:
        summary['max_efficiency'] = df['efficiency'].max()
        summary['min_efficiency'] = df['efficiency'].min()
    return summary

def collect_loss_histories(df):
    """
    Collect loss history data from metrics files referenced in a summary DataFrame.

    Parameters:
    df (pd.DataFrame): Summary DataFrame with 'metrics_path' column pointing to JSON files.

    Returns:
    list[dict]: List of dictionaries with loss history data and labels.
    """
    import json
    import os
    from pathlib import Path

    histories = []
    if df is None or df.empty or "metrics_path" not in df.columns:
        return histories

    for _, row in df.iterrows():
        metrics_path = Path(row["metrics_path"])
        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            loss_history = metrics.get("loss_history", [])
            if loss_history:
                label = row.get("display_name", row.get("run_id", f"run_{len(histories)+1}"))
                histories.append({
                    "label": label,
                    "loss_history": loss_history,
                    "mae_history": metrics.get("mae_history", []),
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return histories
    """
    Collect loss history CSV files from a given path into a single DataFrame.

    Parameters:
    path (str): Directory path containing loss CSV files.

    Returns:
    pd.DataFrame: Combined DataFrame of all loss histories.
    """
    import os
    all_dfs = []
    for fname in os.listdir(path):
        if fname.endswith('.csv'):
            full_path = os.path.join(path, fname)
            df = pd.read_csv(full_path)
            df['source_file'] = fname
            all_dfs.append(df)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def _fft_benchmark_snapshot(path='results/fft_sweep/fft_scaling.csv'):
    """
    Load FFT scaling benchmark data from CSV if present.

    Parameters:
    path (str): Path to the FFT scaling CSV file.

    Returns:
    pd.DataFrame or None: DataFrame with FFT benchmark data, or None if file not found.
    """
    import os
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return None

def compute_fft_corrected(df):
    """
    Compute FFT-corrected runtime and efficiency metrics in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'runtime', 'fft_runtime', 'efficiency', 'fft_efficiency'.

    Returns:
    pd.DataFrame: DataFrame with added 'runtime_corrected' and 'efficiency_corrected' columns.
    """
    df = df.copy()
    if 'runtime' in df.columns:
        df['runtime_corrected'] = df['runtime'] - df.get('fft_runtime', 0)
        df['runtime_corrected'] = df['runtime_corrected'].replace(0, np.nan)
        df['runtime_corrected'] = df['runtime_corrected'].clip(lower=1e-9)
    if 'efficiency' in df.columns:
        df['efficiency_corrected'] = df['efficiency'] - df.get('fft_efficiency', 0)

    if 'runtime' in df.columns and 'runtime_corrected' in df.columns:
        ratio = df['runtime'] / df['runtime_corrected']
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        if 'images_per_second' in df.columns:
            df['images_per_second_corrected'] = df['images_per_second'] * ratio

        if 'loss_drop_per_second' in df.columns:
            df['loss_drop_per_second_corrected'] = df['loss_drop_per_second'] * ratio
        elif {'loss_drop', 'runtime_corrected'}.issubset(df.columns):
            df['loss_drop_per_second_corrected'] = df['loss_drop'] / df['runtime_corrected']

    return df
