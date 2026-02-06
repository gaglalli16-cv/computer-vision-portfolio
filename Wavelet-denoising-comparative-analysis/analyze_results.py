"""
analyze.py
-----------
Wavelet-Only Denoising Summary and Visualization

This script:
1. Displays best-performing Wavelet families and modes per noise type/dataset.
2. Transposes results for IEEE-style narrow tables.
3. Plots the Average PSNR Comparison across wavelet families.

Gaussian vs Wavelet comparisons are handled in comparision.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# Setup paths
# ==============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(PROJECT_ROOT, "results")

# ==============================================================
# Load all results
# ==============================================================

noise_focus = ["gaussian", "speckle", "saltpepper"]
all_dfs = []

for file in os.listdir(results_dir):
    if file.startswith("metrics_wavelet_comparison_") and file.endswith(".csv"):
        noise_type = file.replace("metrics_wavelet_comparison_", "").replace(".csv", "").lower()
        if noise_type not in noise_focus:
            continue
        path = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(path)
            df["Noise_Type"] = noise_type.capitalize()
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if not all_dfs:
    print("No results found. Run 'wavelet_denoising.py' first.")
    exit()

df = pd.concat(all_dfs, ignore_index=True)

# ==============================================================
# SECTION 1 — Wavelet-only summary (no Gaussian blur)
# ==============================================================

wavelet_df = df[df["Wavelet"] != "Gaussian_Blur"].copy()

best_wavelet = (
    wavelet_df.sort_values(["Noise_Type", "Dataset", "PSNR", "SSIM"],
                           ascending=[True, True, False, False])
              .groupby(["Noise_Type", "Dataset"])
              .head(1)
              .reset_index(drop=True)
)

print("\n=== WAVELET-ONLY BEST METHODS (PER DATASET & NOISE TYPE) ===")
print("==============================================================")
print(best_wavelet[["Noise_Type", "Dataset", "Wavelet", "Mode", "PSNR", "SSIM", "MSE"]].to_string(index=False))

# ==============================================================
# SECTION 2 — Transposed Tables (for IEEE narrow format)
# ==============================================================

print("\n=== TRANSPOSED TABLES (IEEE Two-Column Friendly Format) ===")
print("==============================================================")

def transpose_table(df, noise):
    # Select only columns relevant for display
    t_df = df[["Dataset", "PSNR", "SSIM", "MSE"]].set_index("Dataset").T
    print(f"\nTable — {noise} Noise (Transposed)")
    print(t_df.to_string(float_format="%.6f"))
    return t_df

for noise in best_wavelet["Noise_Type"].unique():
    subset = best_wavelet[best_wavelet["Noise_Type"] == noise]
    transpose_table(subset, noise)

# ==============================================================
# SECTION 3 — PSNR Comparison Visualization
# ==============================================================

plt.figure(figsize=(8, 5))
for noise in noise_focus:
    subset = df[df["Noise_Type"].str.lower() == noise]
    avg_psnr = subset.groupby("Wavelet")["PSNR"].mean().reset_index()
    plt.plot(avg_psnr["Wavelet"], avg_psnr["PSNR"], marker='o', label=noise.capitalize())

plt.title("Average PSNR Comparison Across Wavelet Families")
plt.xlabel("Wavelet Family")
plt.ylabel("Average PSNR (dB)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nAnalysis complete. Wavelet-only summaries and visualization generated successfully.")
