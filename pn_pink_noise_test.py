import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_periodic_pink_noise(length, seed=42):
    rng = np.random.default_rng(seed)

    # Create frequency array (positive frequencies only for real signal)
    freqs = np.fft.rfftfreq(length)

    # np.sqrt -> Normalize to 0.5 (normalized freq 0.5)
    amplitudes = np.concatenate([[0.0], np.sqrt(0.5) / np.sqrt(np.abs(freqs[1:]))])

    # Generate random phases (-π to π)
    phases = np.random.uniform(-np.pi, np.pi, len(freqs))

    # Create complex spectrum
    spectrum = amplitudes * np.exp(1j * phases)

    # Generate pink noise using inverse real FFT
    pink_noise = np.fft.irfft(spectrum)

    # Normalize to prevent clipping
    pink_noise = pink_noise / np.max(np.abs(pink_noise))

    return pink_noise

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

def main():
     # Parameters
    N = 10000  # Number of samples
    seed = 42  # Random seed for reproducibility

    # Generate  noise
    noise = generate_periodic_pink_noise(N, seed)

    # Calculate crest factor
    crest_factor = calculate_crest_factor(noise)

    # Print results
    print(f"Generated {N} samples of periodic pink noise")
    print(f"Crest Factor: {crest_factor:.4f}")
    print(f"Crest Factor (dB): {20 * np.log10(crest_factor):.2f} dB")

    # Additional statistics
    print(f"\nSignal Statistics:")
    print(f"Mean: {np.mean(noise):.6f}")
    print(f"Standard Deviation: {np.std(noise):.6f}")
    print(f"Peak Value: {np.max(np.abs(noise)):.6f}")
    print(f"RMS Value: {np.sqrt(np.mean(noise**2)):.6f}")

    # Plot the signal and its spectrum
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Time domain plot
    time = np.arange(N) / N
    ax1.plot(time[:1000], noise[:1000])  # Show first 1000 samples
    ax1.set_title('Periodic Pink Noise Signal (First 1000 samples)')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Frequency domain magnitude
    freqs = np.fft.rfftfreq(N)
    fft_magnitude = np.abs(np.fft.rfft(noise))
    ax2.semilogx(freqs, 20*np.log10(fft_magnitude))
    ax2.set_title('Magnitude Spectrum')
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)

    # Histogram of signal values
    ax3.hist(noise, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax3.set_title('Amplitude Distribution')
    ax3.set_xlabel('Amplitude')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True)

    mean_val = np.mean(noise)
    std_val = np.std(noise)
    ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax3.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'+1σ: {mean_val + std_val:.4f}')
    ax3.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1, label=f'-1σ: {mean_val - std_val:.4f}')
    
    # Fit and overlay Gaussian distribution
    mu, sigma = norm.fit(noise)
    x = np.linspace(noise.min(), noise.max(), 1000)
    gaussian_fit = norm.pdf(x, mu, sigma)
    ax3.plot(x, gaussian_fit, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu:.4f}, σ={sigma:.4f})')
    
    ax3.legend()

    plt.tight_layout()
    plt.show()

    return noise, crest_factor

if __name__ == "__main__":
    signal, crest = main()
