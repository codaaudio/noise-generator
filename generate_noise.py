import os
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import soxr
from scipy.io import wavfile
import argparse
from scipy.interpolate import CubicSpline

def crest_factor(signal):
    """Calculate the crest factor of a signal."""
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    return peak / rms if rms > 0 else np.inf

def crest_factor_to_dB(cf):
    """Convert crest factor to decibels."""
    return 20 * np.log10(cf)

def db_to_crest_factor(dB):
    """Convert decibels to crest factor."""
    return 10 ** (dB / 20)

def generate_pink_amplitudes(freqs, lf_cutoff = 10, normalization_freq=1000.0):
    # Pink noise has 1/f power spectral density, normalize to 1 at 1kHz
    # Therefore we need 1/sqrt(f) amplitude spectral density, because
    # psd = asd^2, so if psd ~ 1/f, then asd ~ 1/sqrt(f)
    # DC is always 0

    # Generate the amplitudes, prevent divison by zero at DC
    ampls = np.concatenate([[0.0], np.sqrt(normalization_freq) / np.sqrt(np.abs(freqs[1:]))])

    # Set values below lf_cutoff to 0
    return np.where(freqs < lf_cutoff, np.zeros(len(ampls)), ampls)

def generate_white_amplitudes(freqs, lf_cutoff = 10):
    # White noise has flat amplitude spectral density and power spectral density
    ampls = np.concatenate([[0.0], np.ones(len(freqs) - 1)])

    # Set values below lf_cutoff to 0
    return np.where(freqs < lf_cutoff, np.zeros(len(ampls)), ampls)

def generate_brown_amplitudes(freqs, lf_cutoff = 10, normalization_freq=1000.0):
    # Brown noise has 1/f^2 power spectral density, therefore 1/f amplitude spectral density
    # Normalize to 1 at 1kHz
    ampls = np.concatenate([[0.0], np.sqrt(normalization_freq) / np.abs(freqs[1:])])

    # Set values below lf_cutoff to 0
    return np.where(freqs < lf_cutoff, np.zeros(len(ampls)), ampls)

def generate_speech_amplitudes(freqs, lf_cutoff = 10):

    # Second order high-pass filter
    # resonant frequency fh= 142 Hz Q = 0.58
    fh = 142  # Hz
    Q1 = 0.58
    fac = 1.0 / (2*np.pi*fh)
    num1, den1 = [fac**2, 0.0, 0.0], [fac**2, fac/Q1, 1.0]

    #Biquadratic peaking filter
    # Centre frequency fc = 500 Hz Q = 1.78 Gain g = 2.7 dB
    gain2 = 2.7
    Q2 = 1.78
    fc = 500
    GainFac2 = 10**(gain2 / 20)
    W = 2.0*np.asinh(1.0/(2.0*Q2))/np.log(2)
    w0 = 2.0*np.pi*fc
    dW = w0*(2**(W/2)-2**(-(W/2)))
    A = dW*np.sqrt(1/GainFac2)
    B = GainFac2*A
    num2, den2 = [1.0, B, w0**2], [1.0, A, w0**2]


    # First order low-pass filter
    # Turnover frequency f l = 315 Hz
    fl = 315
    num3, den3 = [1], [1.0 / (2*np.pi*fl), 1.0]

    # Gain
    gain4 = 4.0
    GainFac4 = 10**(gain4/20)
    num4, den4 = [GainFac4], [1.0]

    # Get individual responses
    w1, h1 = signal.freqs(num1, den1, freqs)
    w2, h2 = signal.freqs(num2, den2, freqs)
    w3, h3 = signal.freqs(num3, den3, freqs)
    w4, h4 = signal.freqs(num4, den4, freqs)

    h5 = generate_pink_amplitudes(freqs)

    # Combine by multiplying transfer functions
    h_combined = h1 * h2 * h3 * h4 * h5

    ampls = np.abs(h_combined)

    return np.where(freqs < lf_cutoff, np.zeros(len(ampls)), ampls)


def generate_pink_a_weighted(freqs, lf_cutoff = 10):
    a_weighting_fun = lambda f: (12194**2 * f**4) / ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) * (f**2 + 12194**2))

    pink_amplitudes = generate_pink_amplitudes(freqs)
    ampls = pink_amplitudes * a_weighting_fun(freqs)

    return np.where(freqs < lf_cutoff, np.zeros(len(ampls)), ampls)

def generate_amplitudes_like(freqs, source_wav):
    """Generate amplitudes for a noise signal based on the spectrum of a source WAV file."""
    sample_rate, data = wavfile.read(source_wav)
    # if multi channel, use first channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Split in blocks of size (freqs.size-1)*2 and compute average spectrum
    block_size = (len(freqs) - 1) * 2
    num_blocks = len(data) // block_size
    spectrum_blocks = []
    for i in range(num_blocks):
        block = data[i * block_size:(i + 1) * block_size]
        if len(block) < block_size:
            continue
        # Compute FFT and take magnitudes
        spectrum = np.fft.rfft(block)
        magnitudes = np.abs(spectrum)
        spectrum_blocks.append(magnitudes)
    # Average the magnitudes across all blocks
    avg_magnitudes = np.mean(spectrum_blocks, axis=0)

    #spectrum = np.fft.rfft(data)
    #magnitudes = np.abs(spectrum)
    
    # Interpolate magnitudes to match the frequency bins
    freqs_source = np.fft.rfftfreq(block_size, d=1/sample_rate)
    cs = CubicSpline(freqs_source, avg_magnitudes)
    return cs(freqs)

def optimize_noise(
        target_noise_type='pink',
        target_crest_factor_dB = 12,
        internal_sample_rate=48000,
        internal_num_samples=32768,
        target_sample_rate=96000,
        external_wav=None
    ):
    
    freqs = np.fft.rfftfreq(internal_num_samples, 1/internal_sample_rate)
    num_freqs = freqs.size
    
    amplitudes = None
    if target_noise_type == 'pink':
        amplitudes = generate_pink_amplitudes(freqs)
    elif target_noise_type == 'white':
        amplitudes = generate_white_amplitudes(freqs)
    elif target_noise_type == 'brown':
        amplitudes = generate_brown_amplitudes(freqs)
    elif target_noise_type == 'speech':
        amplitudes = generate_speech_amplitudes(freqs)
    elif target_noise_type == 'pink_a_weighted':
        amplitudes = generate_pink_a_weighted(freqs)
    elif target_noise_type == 'external':
        if external_wav is None:
            raise ValueError("External WAV file must be provided for 'external' noise type.")
        amplitudes = generate_amplitudes_like(freqs, external_wav)
    else:
        raise ValueError(f"Unknown noise type: {target_noise_type}. Use 'pink', 'white', 'brown', 'speech', 'pink_a_weighted' or 'external'.")
    
    # Initial random phases, reproducible
    rng = np.random.default_rng(12345)
    base_phases = rng.uniform(-np.pi, np.pi, num_freqs)
    
    n_objective_fun_calls = 0

    def objective_function(phases):
        """Objective function to minimize difference from target crest factor."""
    
        nonlocal n_objective_fun_calls
        n_objective_fun_calls += 1

        # Combine amplitudes and phase
        spectrum = amplitudes * np.exp(1j * phases)
        
        # Generate time-domain signal
        signal = np.fft.irfft(spectrum)

        if internal_sample_rate != target_sample_rate:
            upsampled_signal = soxr.resample(signal, internal_sample_rate, target_sample_rate, quality='VHQ')
        else:
            upsampled_signal = signal
        
        # Calculate crest factor
        cf = crest_factor(upsampled_signal)
        cfdB = crest_factor_to_dB(cf)
        
        # Minimize difference from target
        target_fun = np.abs((cfdB - target_crest_factor_dB))
        #print(f"diff: {cfdB - target_crest_factor_dB}dB")
        return target_fun
    
    # Initial phase values for components to be optimized
    initial_phases = base_phases
    
    # Optimize
    result = optimize.minimize(
        objective_function,
        initial_phases,
        method='COBYLA',
        bounds=[(-np.pi, np.pi)] * num_freqs,
        options={'maxiter': 800},
    )

    print(f"Optimization completed in {n_objective_fun_calls} calls.")
    
    # Generate final signal with optimized phases
    final_phases = result.x
    
    return amplitudes, final_phases


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate noise with specified crest factor')
    parser.add_argument('noise_type', choices=['white', 'pink', 'brown', 'speech', 'pink_a_weighted', 'external'],
                        help='Type of noise to generate')
    parser.add_argument('crest_factor_dB', type=float)
    parser.add_argument('--external-wav', type=str,
                        help='Path to external WAV file (required when noise_type is "external")')
    args = parser.parse_args()
    
    # Validate that external-wav is provided when noise_type is 'external'
    if args.noise_type == 'external' and args.external_wav is None:
        parser.error("--external-wav is required when noise_type is 'external'")
    
    # Parameters
    internal_sample_rate = 48000  # Hz
    internal_num_samples = 32768
    target_sample_rate = 96000
    target_crest_factor_dB = args.crest_factor_dB
    
    target_noise_type = args.noise_type

    print(f"Generating {target_noise_type} noise...")
    print(f"Target crest factor: {target_crest_factor_dB}dB")
    
    # Generate optimized noise
    target_magnitudes, target_phases = optimize_noise(
        target_noise_type=target_noise_type,
        target_crest_factor_dB=target_crest_factor_dB,
        internal_sample_rate=internal_sample_rate,
        internal_num_samples=internal_num_samples,
        target_sample_rate=target_sample_rate,
        external_wav=args.external_wav 
    )
    final_signal = np.fft.irfft(target_magnitudes * np.exp(1j * target_phases))
    final_signal = final_signal / np.max(np.abs(final_signal))  # Normalize to [-1, 1]


    freqs = np.fft.rfftfreq(internal_num_samples, 1/internal_sample_rate)
    actual_spectrum = np.fft.rfft(final_signal)
    actual_magnitudes = np.abs(actual_spectrum)
    actual_phases = np.angle(actual_spectrum)

    actual_cf = crest_factor(final_signal)
    actual_cf_dB = crest_factor_to_dB(actual_cf)
    
    print(f"\nOptimization complete!")
    
    # Calculate some statistics
    print(f"\nInternal Signal statistics ({internal_sample_rate/1000:.1f}kHz):")
    print(f"Duration: {len(final_signal) / internal_sample_rate:.3f} s ({len(final_signal)} samples)")
    print(f"Crest factor: {actual_cf_dB:.3f}dB ({actual_cf:.3}x)")
    print(f"Peak value: {np.max(np.abs(final_signal)):.3f}")
    print(f"RMS value: {np.sqrt(np.mean(final_signal**2)):.3f}")
    print(f"Mean: {np.mean(final_signal):.6f}")
    print(f"Std dev: {np.std(final_signal):.3f}")

    # Upsample the final signal to 96kHz
    upsampled_signal = soxr.resample(final_signal, internal_sample_rate, target_sample_rate, quality='VHQ')
    upsampled_signal = upsampled_signal / np.max(np.abs(upsampled_signal))  # Normalize to [-1, 1]
    num_upsampled_samples = len(upsampled_signal)

    upsampled_freqs = np.fft.rfftfreq(num_upsampled_samples, 1/target_sample_rate)
    upsampled_spectrum = np.fft.rfft(upsampled_signal)
    upsampled_magnitudes = np.abs(upsampled_spectrum)
    upsampled_phases = np.angle(upsampled_spectrum)

    upsampled_cf = crest_factor(upsampled_signal)
    upsampled_cf_dB = crest_factor_to_dB(upsampled_cf)

    print(f"\nUpsampled signal statistics ({target_sample_rate/1000.0:.1f}kHz):")
    print(f"Upsampled sample rate: {target_sample_rate} Hz")
    print(f"Upsampled duration: {len(upsampled_signal) / target_sample_rate:.3f} s ({len(upsampled_signal)} samples)")
    print(f"Upsampled crest factor: {upsampled_cf_dB:.3f}dB (target: {target_crest_factor_dB}dB, Error {abs(upsampled_cf_dB - target_crest_factor_dB):.3f}dB)")
    print(f"Upsampled peak value: {np.max(np.abs(upsampled_signal)):.3f}")
    print(f"Upsampled RMS value: {np.sqrt(np.mean(upsampled_signal**2)):.3f}")
    print(f"Upsampled mean: {np.mean(upsampled_signal):.6f}")
    print(f"Upsampled std dev: {np.std(upsampled_signal):.3f}")

    # Save upsampled signal as WAV file
    wavfile.write(f'generated_{target_noise_type}_noise_{target_sample_rate/1000.0:.1f}kHz.wav', target_sample_rate, upsampled_signal.astype(np.float32))

    # Save amplitudes to text file with comment
    with open(f'generated_{target_noise_type}_noise_{target_sample_rate/1000.0:.1f}kHz_amplitudes.txt', 'w') as f:
        f.write(f"# Amplitude Spectrum for {target_sample_rate/1000.0:.1f}kHz {target_noise_type} noise with crest factor {upsampled_cf_dB:.3}dB\n")
        for amp in upsampled_magnitudes:
            f.write(f"{amp}\n")

    # Save phases to text file with comment
    with open(f'generated_{target_noise_type}_noise_{target_sample_rate/1000.0:.1f}kHz_phases.txt', 'w') as f:
        f.write(f"# Phases (radians) for {target_sample_rate/1000.0:.1f}kHz {target_noise_type} noise with crest factor {upsampled_cf_dB:.3}dB\n")
        for phase in upsampled_phases:
            f.write(f"{phase}\n")

    
    # Plot time and frequency domain
    plt.figure(figsize=(12, 8))

    # Time domain plot
    plt.subplot(2, 1, 1)
    time_axis = np.arange(num_upsampled_samples) / target_sample_rate
    plt.plot(time_axis, upsampled_signal)
    plt.title(f'{target_noise_type} Noise Time Domain (CF = {upsampled_cf_dB:.2f} dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Frequency domain plot with magnitude and phase
    plt.subplot(2, 1, 2)
    # We'll use twinx to have two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot magnitude on left axis
    ax1.semilogx(upsampled_freqs[1:], 20 * np.log10(upsampled_magnitudes[1:]), 'b-', label='Magnitude')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.set_xlim([20, target_sample_rate/2])

    # Plot phase on right axis
    ax2.semilogx(upsampled_freqs[1:], np.unwrap(upsampled_phases[1:]), 'r-', alpha=0.6, label='Phase')
    ax2.set_ylabel('Phase (rad)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Frequency Domain')
    plt.tight_layout()
    plt.show()
