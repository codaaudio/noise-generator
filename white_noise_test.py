import numpy as np

def generate_white_noise_uniform(n_samples, seed=12345):
    """
    Generate reproducible white noise from uniform distribution.

    Parameters:
    n_samples (int): Number of samples to generate
    seed (int): Random seed for reproducibility

    Returns:
    numpy.ndarray: White noise samples
    """
    np.random.seed(seed)
    # Generate uniform white noise between -1 and 1
    noise = np.random.normal(0, 1, n_samples)
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
    n_samples = 100000  # Number of samples
    seed = 12345        # For reproducibility

    # Generate white noise
    noise = generate_white_noise_uniform(n_samples, seed)

    # Calculate crest factor
    crest_factor = calculate_crest_factor(noise)

    # Print results
    print(f"White Noise Statistics:")
    print(f"Number of samples: {n_samples}")
    print(f"Random seed: {seed}")
    print(f"Peak amplitude: {np.max(np.abs(noise)):.6f}")
    print(f"RMS amplitude: {np.sqrt(np.mean(noise**2)):.6f}")
    print(f"Crest factor: {crest_factor:.6f}")
    print(f"Crest factor (dB): {20 * np.log10(crest_factor):.2f} dB")

    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Mean: {np.mean(noise):.6f}")
    print(f"Standard deviation: {np.std(noise):.6f}")

if __name__ == "__main__":
    main()
