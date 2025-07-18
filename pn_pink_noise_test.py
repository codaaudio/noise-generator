import numpy as np
import matplotlib.pyplot as plt

def generate_pink_noise(length, sample_rate=44100, seed=42):
    """
    Generate pink noise using inverse FFT with 1/sqrt(freq) amplitude and random phase.

    Parameters:
    - length: number of samples
    - sample_rate: sampling rate in Hz
    - seed: random seed for reproducibility

    Returns:
    - pink_noise: generated pink noise signal
    - freqs: frequency array
    - spectrum: complex spectrum used
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create frequency array (positive frequencies only for real signal)
    freqs = np.fft.rfftfreq(length, 1/sample_rate)

    # Avoid division by zero at DC (0 Hz)
    freqs[0] = 1e-10

    # Create amplitude spectrum: 1/sqrt(freq)
    amplitudes = 1 / np.sqrt(freqs)

    # Set DC component to zero to remove any offset
    amplitudes[0] = 0

    # Generate random phases (0 to 2π)
    phases = np.random.uniform(0, 2*np.pi, len(freqs))

    # Create complex spectrum
    spectrum = amplitudes * np.exp(1j * phases)

    # Generate pink noise using inverse real FFT
    pink_noise = np.fft.irfft(spectrum, n=length)

    # Normalize to prevent clipping
    pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.9

    return pink_noise, freqs, spectrum

def calculate_crest_factor(signal):
    """
    Calculate the crest factor (peak-to-RMS ratio) of a signal.

    Parameters:
    - signal: input signal array

    Returns:
    - crest_factor_linear: crest factor in linear scale
    - crest_factor_db: crest factor in dB
    """
    peak_value = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))

    crest_factor_linear = peak_value / rms_value
    crest_factor_db = 20 * np.log10(crest_factor_linear)

    return crest_factor_linear, crest_factor_db

def main():
    # Parameters
    duration = 5.0  # seconds
    sample_rate = 44100  # Hz
    length = int(duration * sample_rate)

    print(f"Generating pink noise...")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Length: {length} samples")
    print(f"Random seed: 42 (for reproducibility)")
    print("-" * 50)

    # Generate pink noise
    pink_noise, freqs, spectrum = generate_pink_noise(length, sample_rate, seed=42)

    # Calculate crest factor
    crest_linear, crest_db = calculate_crest_factor(pink_noise)

    # Print results
    print(f"Crest Factor (linear): {crest_linear:.3f}")
    print(f"Crest Factor (dB): {crest_db:.1f} dB")
    print("-" * 50)

    # Additional statistics
    print(f"Peak amplitude: {np.max(np.abs(pink_noise)):.4f}")
    print(f"RMS amplitude: {np.sqrt(np.mean(pink_noise**2)):.4f}")
    print(f"Mean: {np.mean(pink_noise):.6f}")
    print(f"Standard deviation: {np.std(pink_noise):.4f}")

    # Optional: Plot the results
    plot_results = True
    if plot_results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Time domain plot
        time = np.linspace(0, duration, length)
        ax1.plot(time[:sample_rate], pink_noise[:sample_rate])  # First second only
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Pink Noise (First 1 second)')
        ax1.grid(True)

        # Frequency domain plot
        fft_pink = np.fft.rfft(pink_noise)
        freqs_plot = np.fft.rfftfreq(length, 1/sample_rate)
        ax2.loglog(freqs_plot[1:], np.abs(fft_pink[1:]))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Pink Noise Spectrum')
        ax2.grid(True)

        # Histogram
        ax3.hist(pink_noise, bins=50, density=True, alpha=0.7)
        ax3.set_xlabel('Amplitude')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Amplitude Distribution')
        ax3.grid(True)

        # Theoretical vs actual spectrum comparison
        theoretical_slope = freqs_plot[1:] ** (-0.5)
        theoretical_slope = theoretical_slope / theoretical_slope[0] * np.abs(fft_pink[1])
        ax4.loglog(freqs_plot[1:], np.abs(fft_pink[1:]), label='Generated', alpha=0.7)
        ax4.loglog(freqs_plot[1:], theoretical_slope, 'r--', label='Theoretical 1/√f', alpha=0.7)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('Spectrum Comparison')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    return pink_noise, crest_linear, crest_db

if __name__ == "__main__":
    pink_noise, crest_linear, crest_db = main()
