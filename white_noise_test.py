import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_white_noise_uniform(n_samples, seed=12345):
    rng = np.random.default_rng(seed)

    # Generate uniform white noise between -1 and 1
    noise = rng.normal(0, 1, n_samples)
    #noise = rng.random.uniform(-1, 1, n_samples)

    noise /= np.max(np.abs(noise))  # Normalize to prevent clipping
    
    return noise

def calculate_crest_factor(signal):
    """
    Calculate the crest factor of a signal.
    Crest factor = peak amplitude / RMS amplitude

    Parameters:
    signal (numpy.ndarray): Input signal

    Returns:
    float: Crest factor
    """
    peak_amplitude = np.max(np.abs(signal))
    rms_amplitude = np.sqrt(np.mean(signal**2))
    crest_factor = peak_amplitude / rms_amplitude
    return crest_factor

def main():
    # Parameters
    N = 100000  # Number of samples
    seed = 12345        # For reproducibility

    # Generate white noise
    noise = generate_white_noise_uniform(N, seed)

    # Calculate crest factor
    crest_factor = calculate_crest_factor(noise)

    # Print results
    print(f"Generated {N} samples of white noise")
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
    ax1.set_title('White Noise Signal (First 1000 samples)')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Frequency domain magnitude
    freqs = np.fft.rfftfreq(N)
    fft_magnitude = np.abs(np.fft.rfft(noise))
    ax2.plot(freqs, 20*np.log10(fft_magnitude))
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
    main()
