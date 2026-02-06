"""
Wavelet Denoising Comparison — Multi-Noise Evaluation
------------------------------------------------------
Applies Gaussian, Speckle, and Salt & Pepper noise to images,
performs denoising using multiple wavelet families and compares
the results against a Gaussian blur baseline.

Wavelet families: db4, db7, haar, sym4, coif2
Modes: hard, soft
Level: 4
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import pywt
from scipy.ndimage import gaussian_filter
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# Fix randomness for reproducibility
np.random.seed(42)

# ==============================================================
# Setup paths
# ==============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
clean_base = os.path.join(PROJECT_ROOT, "images")
noisy_base = os.path.join(clean_base, "noisy_images")
denoised_base = os.path.join(clean_base, "denoised_images")
results_dir = os.path.join(PROJECT_ROOT, "results")

os.makedirs(noisy_base, exist_ok=True)
os.makedirs(denoised_base, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

datasets = ["histology", "products", "radiology"]
wavelet_families = ["db4", "db7", "haar", "sym4", "coif2"]
modes = ["hard", "soft"]

# ==============================================================
# Utility functions
# ==============================================================

def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def save_image(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

def load_gray_resized(path, target_size=None):
    img = Image.open(path).convert("L")
    if target_size:
        img = img.resize(target_size)
    return np.asarray(img, dtype=np.float32) / 255.0

# Noise functions
def add_gaussian_noise(img, sigma=0.0001):
    return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)

def add_speckle_noise(img):
    noise = np.random.randn(*img.shape)
    return np.clip(img + img * noise * 0.2, 0, 1)

def add_salt_pepper_noise(img, amount=0.02):
    noisy = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 1
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

# Denoising function
def wavelet_denoise(img, wavelet="db4", mode="soft", level=4, thresh_scale=0.3):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    detail_flat = np.concatenate([c.flatten() for (cH, cV, cD) in details for c in (cH, cV, cD)])
    sigma_est = np.median(np.abs(detail_flat)) / 0.6745
    T = sigma_est * np.sqrt(2.0 * np.log(img.size)) * thresh_scale
    new_coeffs = [cA]
    for (cH, cV, cD) in details:
        new_coeffs.append(tuple(pywt.threshold(c, T, mode=mode) for c in (cH, cV, cD)))
    return np.clip(pywt.waverec2(new_coeffs, wavelet=wavelet), 0, 1)

# ==============================================================
# Noise & Denoising Processing
# ==============================================================

def process_noise(noise_type, add_noise_func):
    print(f"\nProcessing {noise_type.upper()} noise...")
    noise_dir = os.path.join(noisy_base, f"{noise_type}_noise")
    os.makedirs(noise_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Step 1: Add noise to all datasets
    # ----------------------------------------------------------
    for folder in datasets:
        input_path = os.path.join(clean_base, folder)
        if not os.path.exists(input_path):
            continue
        for file in os.listdir(input_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = load_image(os.path.join(input_path, file))
            noisy = add_noise_func(img)
            save_image(os.path.join(noise_dir, file), noisy)

    # ----------------------------------------------------------
    # Step 2: Denoise with all wavelet families
    # ----------------------------------------------------------
    wavelet_dir = os.path.join(denoised_base, f"{noise_type}_wavelets")
    os.makedirs(wavelet_dir, exist_ok=True)

    for wv in wavelet_families:
        wv_dir = os.path.join(wavelet_dir, wv)
        os.makedirs(wv_dir, exist_ok=True)
        for file in os.listdir(noise_dir):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = load_gray(os.path.join(noise_dir, file))
            for mode in modes:
                denoised = wavelet_denoise(img, wavelet=wv, mode=mode)
                save_image(os.path.join(wv_dir, f"{wv}_{mode}_{file}"), denoised)

    # ----------------------------------------------------------
    # Step 3: Gaussian blur baseline
    # ----------------------------------------------------------
    blur_dir = os.path.join(denoised_base, f"{noise_type}_blur")
    os.makedirs(blur_dir, exist_ok=True)
    for file in os.listdir(noise_dir):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = load_gray(os.path.join(noise_dir, file))
        save_image(os.path.join(blur_dir, file), gaussian_filter(img, sigma=1))

    # ----------------------------------------------------------
    # Step 4: Metric evaluation (compare all)
    # ----------------------------------------------------------
    results = []
    for dataset in datasets:
        clean_dir = os.path.join(clean_base, dataset)
        if not os.path.exists(clean_dir):
            continue
        for file in os.listdir(clean_dir):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            clean_path = os.path.join(clean_dir, file)
            clean = load_gray_resized(clean_path)
            size = Image.open(clean_path).size

            # Gaussian baseline
            blur_path = os.path.join(blur_dir, file)
            img_blur = load_gray_resized(blur_path, target_size=size)
            g_mse = mean_squared_error(clean, img_blur)
            g_psnr = peak_signal_noise_ratio(clean, img_blur, data_range=1)
            g_ssim = structural_similarity(clean, img_blur, data_range=1)

            # Save Gaussian result too
            results.append({
                "Dataset": dataset,
                "Image": file,
                "Noise_Type": noise_type.capitalize(),
                "Wavelet": "Gaussian_Blur",
                "Mode": "-",
                "PSNR": g_psnr,
                "SSIM": g_ssim,
                "MSE": g_mse,
                "Winner": "Gaussian"
            })

            # Best wavelet initialization
            best_wavelet, best_mode = None, None
            best_metrics = {"PSNR": 0, "SSIM": 0, "MSE": 1e9}

            # Evaluate all wavelets
            for wv in wavelet_families:
                for mode in modes:
                    den_path = os.path.join(wavelet_dir, wv, f"{wv}_{mode}_{file}")
                    if not os.path.exists(den_path):
                        continue
                    img = load_gray_resized(den_path, target_size=size)
                    mse = mean_squared_error(clean, img)
                    psnr = peak_signal_noise_ratio(clean, img, data_range=1)
                    ssim = structural_similarity(clean, img, data_range=1)

                    results.append({
                        "Dataset": dataset,
                        "Image": file,
                        "Noise_Type": noise_type.capitalize(),
                        "Wavelet": wv,
                        "Mode": mode,
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "MSE": mse,
                        "Winner": "Wavelet"
                    })

    # ----------------------------------------------------------
    # Step 5: Save results
    # ----------------------------------------------------------
    df = pd.DataFrame(results)
    out_csv = os.path.join(results_dir, f"metrics_wavelet_comparison_{noise_type}.csv")
    df.to_csv(out_csv, index=False)
    print(f" Saved metrics: {os.path.basename(out_csv)}")


# ==============================================================
# Run all noise types
# ==============================================================

process_noise("gaussian", add_gaussian_noise)
process_noise("speckle", add_speckle_noise)
process_noise("saltpepper", add_salt_pepper_noise)

print("\n All noise-type metrics successfully saved to the 'results' folder.")


import matplotlib.pyplot as plt

def visualize_denoising(clean, noisy, gaussian, wavelet, title="Denoising Comparison"):
    plt.figure(figsize=(10, 6))

    images = [clean, noisy, gaussian, wavelet]
    titles = ["Original", "Noisy", "Gaussian Blur", "Wavelet Denoised"]

    for i, (img, t) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap="gray")
        plt.title(t)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
