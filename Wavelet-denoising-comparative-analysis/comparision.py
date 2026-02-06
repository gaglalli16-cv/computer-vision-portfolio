"""
comparison.py
--------------
Unified Results Analysis Script
Summarizes denoising results across all noise types:
Gaussian, Speckle, and Salt & Pepper.
"""

import os
import pandas as pd

# ==============================================================
# Setup paths
# ==============================================================

results_dir = "results"
noise_types = ["gaussian", "speckle", "saltpepper"]
dfs = {}

# ==============================================================
# Load all CSV files
# ==============================================================

for noise in noise_types:
    csv_path = os.path.join(results_dir, f"metrics_wavelet_comparison_{noise}.csv")
    if os.path.exists(csv_path):
        dfs[noise] = pd.read_csv(csv_path)
    else:
        print(f"Missing file: {csv_path}")

if not dfs:
    print("No CSV files found in 'results' directory. Run your experiments first.")
    exit()

# ==============================================================
# Summarize and Transpose Best Methods
# ==============================================================

def summarize_and_transpose(df, noise_label):
    # Identify best performing method per dataset
    best = (
        df.sort_values(["Dataset", "PSNR", "SSIM"], ascending=[True, False, False])
          .groupby("Dataset")
          .head(1)
          .reset_index(drop=True)
    )

    # Round values neatly
    best["PSNR"] = best["PSNR"].round(2)
    best["SSIM"] = best["SSIM"].round(4)
    best["MSE"] = best["MSE"].apply(lambda x: f"{x:.6e}")

    # Include Noise_Type metadata
    best["Noise_Type"] = noise_label.capitalize()

    # Keep only relevant columns
    cols = ["Dataset", "Wavelet", "Mode", "PSNR", "SSIM", "MSE", "Winner", "Noise_Type"]
    best = best[cols]

    # Transpose key metrics (PSNR, SSIM, MSE)
    metric_table = best.set_index("Dataset")[["PSNR", "SSIM", "MSE"]].T
    metric_table.reset_index(inplace=True)
    metric_table.rename(columns={"index": "Metric"}, inplace=True)

    # Append metadata rows
    metadata = pd.DataFrame({
        "Metric": ["Winner", "Noise_Type"],
    })
    for col in best["Dataset"]:
        metadata[col] = [
            best.loc[best["Dataset"] == col, "Winner"].values[0],
            best.loc[best["Dataset"] == col, "Noise_Type"].values[0],
        ]

    final_transposed = pd.concat([metric_table, metadata], ignore_index=True)
    return final_transposed


# ==============================================================
# Generate and Save Tables
# ==============================================================

for noise, df in dfs.items():
    transposed_df = summarize_and_transpose(df, noise)
    out_path = os.path.join(results_dir, f"Table_Best_{noise}.csv")
    transposed_df.to_csv(out_path, index=False)
    print(f" Saved Table: {os.path.basename(out_path)}")

