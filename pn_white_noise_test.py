import numpy as np
import matplotlib.pyplot as plt

def generate_white_noise(N, seed=42):
    """
    Generate reproducible white noise using inverse FFT with constant amplitude and random phase.

    Parameters:
    N (int): Number of samples
    seed (int): Random seed for reproducibility

    Returns:
    numpy.ndarray: Generated white noise signal
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # For real FFT, we need N//2 + 1 frequency bins
    n_freqs = N // 2 + 1

    # Create constant amplitude spectrum
    amplitude = 1.0
    magnitudes = np.full(n_freqs, amplitude)

    # Generate random phases (0 to 2π)
    phases = np.random.uniform(0, 2*np.pi, n_freqs)

    # Special handling for DC and Nyquist components (must be real)
    phases[0] = 0  # DC component
    if N % 2 == 0:  # If N is even, set Nyquist frequency phase to 0
        phases[-1] = 0

    # Create complex spectrum
    spectrum = magnitudes * np.exp(1j * phases)

    # Perform inverse real FFT to get time domain signal
    signal = np.fft.irfft(spectrum, n=N)

    return signal

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
    N = 8192  # Number of samples
    seed = 42  # Random seed for reproducibility

    # Generate white noise
    noise = generate_white_noise(N, seed)

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
    ax2.plot(freqs, fft_magnitude)
    ax2.set_title('Magnitude Spectrum')
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)

    # Histogram of signal values
    ax3.hist(noise, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax3.set_title('Amplitude Distribution')
    ax3.set_xlabel('Amplitude')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return noise, crest_factor

if __name__ == "__main__":
    signal, cf = main()
