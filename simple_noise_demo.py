#!/usr/bin/env python3
"""
Simple Noise Demonstration tool

This module generates different types of noise and gives some graphical output:
- Time Domain plot
- Histogram of amplitude distribution
- Power Spectral Density (PSD) using Welch's method
- Frequency Domain Magnitude using FFT

Supported noise types:
- gaussian-random: Gaussian random noise (samples randomly drawn from normal distribution).
- uniform-random: Uniform random noise (samples randomly drawn from uniform distribution).
- linspace-shuffle: Random noise by shuffling a linspace from -1 to 1
- white-pn: Periodic white noise (constant spectrum, random phases, generated via inverse FFT)
- pink-pn: Periodic pink noise (1/f spectrum, random phases, generated via inverse FFT)

Usage:
    python3 simple_noise_demo.py --type gaussian-random

    python3 simple_noise_demo.py --type pink-pn --samples 10000 --seed 42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import signal as sig


def generate_gaussian_random_noise(n_samples, seed=12345):
    """
    Generate Gaussian random noise (samples randomly drawn from normal distribution).
    
    Parameters:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Normalized Gaussian noise signal
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n_samples)
    noise /= np.max(np.abs(noise))
    return noise

def generate_uniform_random_noise(n_samples, seed=12345):
    """
    Generate uniform random noise (samples randomlydrawn from uniform distribution).
    
    Parameters:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Normalized white noise signal
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1, 1, n_samples)
    noise /= np.max(np.abs(noise))
    return noise

def generate_linspace_shuffle_noise(n_samples, seed=12345):
    """
    Generate random noise by randomly shuffling a linspace from -1 to 1.
    
    Parameters:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Normalized white noise signal
    """
    rng = np.random.default_rng(seed)
    noise = np.linspace(-1, 1, n_samples)
    rng.shuffle(noise)
    noise /= np.max(np.abs(noise))
    return noise

def generate_periodic_white_noise(n_samples, seed=42):
    """
    Generate periodic white noise using inverse FFT.
    Creates white noise with constant spectrum magnitude and random phases.
    
    Parameters:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Normalized periodic white noise signal
    """
    rng = np.random.default_rng(seed)
    
    # For real FFT, we need N//2 + 1 frequency bins
    n_freqs = n_samples // 2 + 1
    
    # Create constant amplitude spectrum
    magnitudes = np.ones(n_freqs)
    
    # Generate random phases (-π to π)
    phases = rng.uniform(-np.pi, np.pi, n_freqs)
    
    # Create complex spectrum
    spectrum = magnitudes * np.exp(1j * phases)
    
    # Perform inverse real FFT to get time domain signal
    noise = np.fft.irfft(spectrum)
    noise /= np.max(np.abs(noise))
    
    return noise


def generate_periodic_pink_noise(n_samples, seed=42):
    """
    Generate periodic pink noise (1/f noise) using inverse FFT.
    
    Parameters:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Normalized periodic pink noise signal
    """
    rng = np.random.default_rng(seed)
    
    # Create frequency array (positive frequencies only for real signal)
    freqs = np.fft.rfftfreq(n_samples)
    
    # Pink noise has 1/f power spectral density, normalize to 1 at 1kHz
    # Therefore we need 1/sqrt(f) amplitude spectral density, because
    # psd = asd^2, so if psd ~ 1/f, then asd ~ 1/sqrt(f)
    # DC is always 0

    # Generate the amplitudes, prevent divison by zero at DC
    amplitudes = np.concatenate([[0.0], np.sqrt(1000.0) / np.sqrt(np.abs(freqs[1:]))])
    
    # Generate random phases (-π to π)
    phases = rng.uniform(-np.pi, np.pi, len(freqs))
    
    # Create complex spectrum
    spectrum = amplitudes * np.exp(1j * phases)
    
    # Generate pink noise using inverse real FFT
    noise = np.fft.irfft(spectrum)
    noise /= np.max(np.abs(noise))
    
    return noise


def calculate_crest_factor(signal):
    """
    Calculate the crest factor of a signal.
    Crest factor = Peak value / RMS value

    Parameters:
        signal (numpy.ndarray): Input signal

    Returns:
        float: Crest factor
    """
    peak_value = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))
    crest_factor = peak_value / rms_value
    return crest_factor


def print_statistics(noise, noise_type, n_samples):
    """Print signal statistics to console."""
    crest_factor = calculate_crest_factor(noise)
    
    print(f"Generated {n_samples} samples of {noise_type} noise")
    print(f"Crest Factor: {crest_factor:.4f}")
    print(f"Crest Factor (dB): {20 * np.log10(crest_factor):.2f} dB")
    
    print(f"\nSignal Statistics:")
    print(f"Mean: {np.mean(noise):.6f}")
    print(f"Standard Deviation: {np.std(noise):.6f}")
    print(f"Peak Value: {np.max(np.abs(noise)):.6f}")
    print(f"RMS Value: {np.sqrt(np.mean(noise**2)):.6f}")
    
    return crest_factor


def plot_time_and_histogram(noise, noise_type, n_samples):
    """
    Plot time domain signal and amplitude histogram in first window.
    
    Parameters:
        noise: The noise signal array
        noise_type: String describing the noise type
        n_samples: Number of samples in the signal
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'{noise_type.replace("-", " ").title()} Noise Analysis - Time Domain', fontsize=14)
    
    # Time domain plot (first 500 samples)
    time = np.arange(min(500, n_samples))
    ax1.plot(time, noise[:len(time)])
    ax1.set_title(f'Time Domain (First {len(time)} samples)')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Histogram of signal values
    ax2.hist(noise, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax2.set_title('Amplitude Distribution')
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True)
    
    mean_val = np.mean(noise)
    std_val = np.std(noise)
    ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax2.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'+1σ: {mean_val + std_val:.4f}')
    ax2.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1, label=f'-1σ: {mean_val - std_val:.4f}')
    
    # Fit and overlay Gaussian distribution
    mu, sigma = norm.fit(noise)
    x = np.linspace(noise.min(), noise.max(), 1000)
    gaussian_fit = norm.pdf(x, mu, sigma)
    ax2.plot(x, gaussian_fit, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu:.4f}, σ={sigma:.4f})')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_frequency_analysis(noise, noise_type, n_samples, sample_rate=1.0):
    """
    Plot Power Spectral Density and Frequency Domain magnitude in second window.
    
    Parameters:
        noise: The noise signal array
        noise_type: String describing the noise type
        n_samples: Number of samples in the signal
        sample_rate: Sample rate for frequency axis scaling
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'"{noise_type}" Noise Analysis - Frequency Domain', fontsize=14)
    
    # Power Spectral Density using Welch's method
    freqs_psd, psd = sig.welch(noise, fs=sample_rate, nperseg=min(1024, n_samples // 4))
    ax1.semilogy(freqs_psd, psd)
    ax1.set_title('Power Spectral Density (Welch)')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Power/Frequency (dB/Hz)')
    ax1.grid(True)
    
    # Frequency domain magnitude (FFT)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sample_rate)
    fft_magnitude = np.abs(np.fft.rfft(noise))
    
    # Use semilogx for pink noise to better show 1/f slope, linear for white noise
    ax2.semilogx(freqs[1:], 20*np.log10(fft_magnitude[1:] + 1e-10))
    
    ax2.set_title('Magnitude Spectrum (FFT)')
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate and analyze different types of noise signals.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '-t', '--type',
        type=str,
        choices=['gaussian-random', 'uniform-random', 'linspace-shuffle', 'white-pn', 'pink-pn'],
        default='gaussian-random',
        help='Type of noise to generate'
    )
    
    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=10000,
        help='Number of samples to generate (default: 10000)'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=96000.0,
        help='Sample rate for frequency axis scaling (default: 96000.0)'
    )
    
    args = parser.parse_args()
    
    # Generate noise based on type
    if args.type == 'gaussian-random':
        noise = generate_gaussian_random_noise(args.samples, args.seed)
    elif args.type == 'uniform-random':
        noise = generate_uniform_random_noise(args.samples, args.seed)
    elif args.type == 'linspace-shuffle':
        noise = generate_linspace_shuffle_noise(args.samples, args.seed)
    elif args.type == 'white-pn':
        noise = generate_periodic_white_noise(args.samples, args.seed)
    elif args.type == 'pink-pn':
        noise = generate_periodic_pink_noise(args.samples, args.seed)
    else:
        raise ValueError(f"Unknown noise type: {args.type}")
    
    # Print statistics
    crest_factor = print_statistics(noise, args.type, args.samples)
    
    # Create plots in two separate windows
    fig1 = plot_time_and_histogram(noise, args.type, args.samples)
    fig2 = plot_frequency_analysis(noise, args.type, args.samples, args.sample_rate)
    
    plt.show()
    
    return noise, crest_factor


if __name__ == "__main__":
    main()
