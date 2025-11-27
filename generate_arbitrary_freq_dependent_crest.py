#!/bin/env python3

"""
Advanced Arbitrary Crest Noise Generator

!!WARNING!!
This is research code, here be dragons!
Convergence is NOT guaranteed
Assume everything is evil and wants to eat your cats and dogs
!!WARNING!!

Contrary to the name ("Consistent Crest Noise Generator"), this tool allows specifying arbitrary
frequency-dependent crest factor targets simultaneously across multiple frequency bands.

Those CAN all be identical (consistent crest factors), but they do not have to be (Music Noise)

The script uses gradient-based optimization with JAX for automatic differentiation.

Unlike simpler noise generators that only target a single broadband crest factor,
this generator optimizes the phase spectrum to achieve specific crest factors in
octave bands, third-octave bands, and 1/24-octave bands ALL AT THE SAME TIME.

The optimization process works as follows:
1. Generate the desired amplitude spectrum (from file or analytical function)
2. Build a bank of fractional octave filters (1, 1/3, and 1/24 octave) only once for performance reasons
3. Generate an initial phase spectrum (random or from a starting WAV file)
4. Synthesize the noise signal with the current phase spectrum
5. Apply all filters to get band-limited signals
6. Calculate the crest factor for each filtered signal
7. Compute the weighted mean absolute error from target crest factors
8. Use L-BFGS-B optimization with JAX gradients to adjust phases
9. Repeat until convergence or user interruption (SIGINT)

The target crest factors can be specified per frequency band, allowing generation
of noise signals that match the statistical properties of real-world signals like
music or speech.

WARNING: While in theory, --starting-point sounds like a good idea (take an existing
good noise signal as starting point and just optimize the phases), it doesn't really work



Usage examples:
  # Generate pink noise with uniform 12dB crest factor and ~11 seconds run time (SLOW!! Will take hours to complete)
  python3 generate_arbitrary_freq_dependent_crest.py --noise-type pink

  # Generate music noise (frequency-dependent crest factors) and ~11 seconds run time (SLOW!! Will take hours to complete)
  # requires Music-Noise_96kHz.wav to derive amplitude spectrum
  python3 generate_arbitrary_freq_dependent_crest.py --noise-type external-ampl-target --external-ampl-wav Music-Noise_96kHz.wav --crest-targets music

  # Generate pink noise with consistent crest factors (absolute level not controlled, just consistency)
  # Will take around an hour to complete
  python3 generate_arbitrary_freq_dependent_crest.py --noise-type pink --objective-mode consistent

  # Generate pink noise with only broadband crest factor control and small number of samples, very quick
  python3 generate_arbitrary_freq_dependent_crest.py --noise-type pink --objective-mode broadband-only --num-samples 32768

"""

import math
import sys
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from freq_dependent_crest_factor import (
    get_fractional_octave_center_frequencies,
    design_fractional_octave_fir_filter,
    design_fractional_octave_butterworth_filter,
)
import signal as signal_handling
import scipy.signal as signal
import scipy.interpolate
import argparse

# global
abort_calculation = False
optimization_running = False


def handle_siginit(sig, frame):
    if optimization_running:
        global abort_calculation
        print(f"Received SIGINT, aborting optimization ASAP...")
        abort_calculation = True
    else:
        print(f"Received SIGINT, exiting...")
        sys.exit(0)


def smooth_fractional_octave(data, fraction):
    """
    Data smoothing using fractional octave smoothing
    Inspired by the paper "Increasing the Audio Measurement Capability of FFT Analysers by Microcomputer Post-Processing"
    by Lipshitz, Scott and Vanderkooy
    """

    num_freqs = len(data)
    lin_freqs = np.arange(num_freqs)
    log_freqs = num_freqs ** (lin_freqs / (num_freqs - 1))

    log_freqs_fractional_spacing = np.log2(log_freqs[1] / log_freqs[0])

    window_width = int(2 * np.floor(1 / (fraction * log_freqs_fractional_spacing * 2)))

    if window_width <= 1:
        raise ValueError(
            (
                "Resulting smoothing window has length 1. Make smoothing wider (Decrease fraction) or use a longer signal. "
            )
        )

    # Interpolate from lin frequency scale to log frequency scale
    cs_lin_to_log = scipy.interpolate.CubicSpline(lin_freqs, data)
    log_data = cs_lin_to_log(log_freqs)

    # Fractional octave smoothing by constant width moving average on log frequency scale
    log_smoothed = np.convolve(
        log_data, np.ones(window_width) / window_width, mode="same"
    )

    # Interpolate from log frequency scale back to lin frequency scale
    cs_log_to_lin = scipy.interpolate.CubicSpline(log_freqs, log_smoothed)
    smoothed_data_lin = cs_log_to_lin(lin_freqs)

    return smoothed_data_lin


def generate_pink_amplitudes(freqs, normalization_freq=1000.0):
    # Pink noise has 1/f power spectral density, normalize to 1 at 1kHz
    # Therefore we need 1/sqrt(f) amplitude spectral density, because
    # psd = asd^2, so if psd ~ 1/f, then asd ~ 1/sqrt(f)
    # DC is always 0

    # Generate the amplitudes, prevent divison by zero at DC
    ampls = np.concatenate(
        [[0.0], np.sqrt(normalization_freq) / np.sqrt(np.abs(freqs[1:]))]
    )

    return ampls


def generate_white_amplitudes(freqs):
    # White noise has flat amplitude spectral density and power spectral density
    ampls = np.concatenate([[0.0], np.ones(len(freqs) - 1)])

    # Set values below lf_cutoff to 0
    return ampls


def generate_brown_amplitudes(freqs, normalization_freq=1000.0):
    # Brown noise has 1/f^2 power spectral density, therefore 1/f amplitude spectral density
    # Normalize to 1 at 1kHz
    ampls = np.concatenate([[0.0], np.sqrt(normalization_freq) / np.abs(freqs[1:])])

    # Set values below lf_cutoff to 0
    return ampls


def generate_speech_amplitudes(freqs):

    # Second order high-pass filter
    # resonant frequency fh= 142 Hz Q = 0.58
    fh = 142  # Hz
    Q1 = 0.58
    fac = 1.0 / (2 * np.pi * fh)
    num1, den1 = [fac**2, 0.0, 0.0], [fac**2, fac / Q1, 1.0]

    # Biquadratic peaking filter
    # Centre frequency fc = 500 Hz Q = 1.78 Gain g = 2.7 dB
    gain2 = 2.7
    Q2 = 1.78
    fc = 500
    GainFac2 = 10 ** (gain2 / 20)
    W = 2.0 * np.asinh(1.0 / (2.0 * Q2)) / np.log(2)
    w0 = 2.0 * np.pi * fc
    dW = w0 * (2 ** (W / 2) - 2 ** (-(W / 2)))
    A = dW * np.sqrt(1 / GainFac2)
    B = GainFac2 * A
    num2, den2 = [1.0, B, w0**2], [1.0, A, w0**2]

    # First order low-pass filter
    # Turnover frequency f l = 315 Hz
    fl = 315
    num3, den3 = [1], [1.0 / (2 * np.pi * fl), 1.0]

    # Gain
    gain4 = 4.0
    GainFac4 = 10 ** (gain4 / 20)
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

    return ampls


def generate_pink_a_weighted_amplitudes(freqs):
    a_weighting_fun = lambda f: (12194**2 * f**4) / (
        (f**2 + 20.6**2)
        * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
        * (f**2 + 12194**2)
    )

    pink_amplitudes = generate_pink_amplitudes(freqs)
    ampls = pink_amplitudes * a_weighting_fun(freqs)

    return ampls


def generate_amplitudes_like(freqs, source_wav, block_size=65536):
    """Generate amplitudes for a noise signal based on the spectrum of a source WAV file."""
    sample_rate, data = wavfile.read(source_wav)
    # if multi channel, use first channel
    if data.ndim > 1:
        data = data[:, 0]

    # Split in blocks of size (freqs.size-1)*2 and compute average spectrum
    num_blocks = len(data) // block_size
    freqs_source = np.fft.rfftfreq(block_size, d=1 / sample_rate)

    spectrum_blocks = []
    print(f"Using FFT length {block_size}")
    for i in range(num_blocks):
        block = data[i * block_size : (i + 1) * block_size]
        if len(block) < block_size:
            continue
        # Compute FFT and take magnitudes
        spectrum = np.fft.rfft(block)
        magnitudes = np.abs(spectrum)
        spectrum_blocks.append(magnitudes)
    # Average the magnitudes across all blocks
    avg_magnitudes = np.mean(spectrum_blocks, axis=0)

    # pre smooth for LF range
    smoothed_magnitudes = signal.savgol_filter(avg_magnitudes, 15, 3, mode="nearest")

    # postsmooth with fractional octave smoothing for hf range
    smoothed_magnitudes = smooth_fractional_octave(smoothed_magnitudes, fraction=6)

    # spectrum = np.fft.rfft(data)
    # magnitudes = np.abs(spectrum)

    # Interpolate magnitudes to target frequencies
    cs = scipy.interpolate.CubicSpline(freqs_source, smoothed_magnitudes)

    return cs(freqs)


def generate_music_noise_crests(freqs):

    if freqs is None:
        return 18.06  # broadband crest factor

    one_third_crest_factors = np.array(
        [
            [25, 12.5],
            [31.5, 12.5],
            [40, 12.5],
            [50, 12.5],
            [63, 12.5],
            [80, 12.5],
            [100, 12.5],
            [125, 12.5],
            [160, 12.5],
            [200, 12.5],
            [250, 12.6],
            [315, 12.7],
            [400, 12.8],
            [500, 12.9],
            [630, 13],
            [800, 13.15],
            [1000, 13.343],
            [1250, 13.478],
            [1600, 13.935],
            [2000, 14.5],
            [2500, 14.962],
            [3150, 15.503],
            [4000, 16.334],
            [5000, 17],
            [6300, 18],
            [8000, 18.726],
            [10000, 19.462],
            [12500, 19.986],
            [16000, 20.7],
            [20000, 21.506],
            [24000, 22.3],
        ]
    )

    # Interpolate the crest factors to the frequencies
    cs = scipy.interpolate.CubicSpline(
        one_third_crest_factors[:, 0], one_third_crest_factors[:, 1]
    )
    interpolated_crests = cs(freqs)

    return interpolated_crests


def generate_uniform_crests(freqs, target_crest_dB):
    """Generate uniform crest factor targets for all frequency bands."""
    if freqs is None:
        return target_crest_dB  # broadband crest factor
    return np.full(len(freqs), target_crest_dB)


friendly_objective_names = {
    "target": "Frequency-Dependent Crest Factor Targeting",
    "consistent": "Consistent Crest Factors (Minimize StdDev)",
    "broadband-only": "Broadband Crest Factor Only",
}

friendly_noise_names = {
    "pink": "Periodic Pink Noise",
    "white": "Periodic White Noise",
    "brown": "Periodic Brown Noise",
    "speech": "Periodic IEC60268-16:2020 Speech shaped noise",
    "pink_a_weighted": "Periodic A-weighted Pink Noise",
    "external-ampl-target": "Periodic External Spectrum Noise",
}

friendly_crest_targets_names = {
    "uniform": "Uniform Crest Factor",
    "music": "Music Noise (Frequency-Dependent Crest Factors)",
    "external-file": "External Crest Factor File",
}


@jax.jit
def crest_factor(sig):
    peak = jnp.max(jnp.abs(sig))
    rms = jnp.sqrt(jnp.mean(sig**2))

    return jnp.where(rms <= 0.0, jnp.inf, peak / rms)


@jax.jit
def crest_factor_to_dB(cf):
    return 20 * jnp.log10(cf)


def noise_signal_objective_single_broadband(
    phases, num_phases_lf_pad, num_phases_hf_pad, amplitudes, target_broadband_crest
):

    padded_phases = jnp.pad(
        phases,
        (num_phases_lf_pad, num_phases_hf_pad),
        mode="constant",
        constant_values=0.0,
    )

    curr_spectrum = amplitudes * jnp.exp(1j * padded_phases)

    curr_signal = jnp.fft.irfft(curr_spectrum)

    curr_broadband_crest_factor_dB = crest_factor_to_dB(crest_factor(curr_signal))

    curr_obj_fun = jnp.abs(curr_broadband_crest_factor_dB - target_broadband_crest)

    return curr_obj_fun


@jax.jit
def crest_factor_mtx(signal_mtx):
    # Signals are in each row
    peak = jnp.max(jnp.abs(signal_mtx), axis=1)
    rms = jnp.sqrt(jnp.mean(signal_mtx**2, axis=1))

    return jnp.where(rms <= 0.0, jnp.inf, peak / rms)


def noise_signal_objective_multi_filter_target(
    phases,
    num_phases_lf_pad,
    num_phases_hf_pad,
    amplitudes,
    filters,
    target_crests,
    target_crest_weightings,
):

    padded_phases = jnp.pad(
        phases,
        (num_phases_lf_pad, num_phases_hf_pad),
        mode="constant",
        constant_values=0.0,
    )

    spectrum_row = amplitudes * jnp.exp(1j * padded_phases)

    # spectrum is in each row
    spectrum_matrix = jnp.tile(spectrum_row, (filters.shape[0], 1))

    # apply the filters, arranged as matrix
    spectrum_matrix = jnp.multiply(spectrum_matrix, filters)

    # perform inverse FFT along the rows
    # each row is a signal
    signal_matrix = jnp.fft.irfft(spectrum_matrix, axis=1)

    crest_factors = crest_factor_mtx(signal_matrix)

    crest_factors_dB = crest_factor_to_dB(crest_factors)

    curr_obj_fun = jnp.mean(
        jnp.abs(crest_factors_dB - target_crests) * target_crest_weightings
    )
    # curr_obj_fun = jnp.std(crest_factors_dB)

    return curr_obj_fun


def noise_signal_objective_multi_filter_consistent(
    phases, num_phases_lf_pad, num_phases_hf_pad, amplitudes, filters
):

    padded_phases = jnp.pad(
        phases,
        (num_phases_lf_pad, num_phases_hf_pad),
        mode="constant",
        constant_values=0.0,
    )

    spectrum_row = amplitudes * jnp.exp(1j * padded_phases)

    # spectrum is in each row
    spectrum_matrix = jnp.tile(spectrum_row, (filters.shape[0], 1))

    # apply the filters, arranged as matrix
    spectrum_matrix = jnp.multiply(spectrum_matrix, filters)

    # perform inverse FFT along the rows
    # each row is a signal
    signal_matrix = jnp.fft.irfft(spectrum_matrix, axis=1)

    crest_factors = crest_factor_mtx(signal_matrix)

    crest_factors_dB = crest_factor_to_dB(crest_factors)

    curr_obj_fun = jnp.std(crest_factors_dB)

    return curr_obj_fun


def eval_g(_x, _out):
    return


def eval_jac_g(_x, _out):
    return


# define the nonzero slots in the jacobian
# there are no nonzeros in the constraint jacobian
eval_jac_g_sparsity_indices = (np.array([]), np.array([]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Arbitrary Crest Noise Generator - Generate noise with frequency-dependent crest factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate pink noise with uniform 12dB crest factor and ~11 seconds run time (SLOW!! Will take hours to complete)
  %(prog)s --noise-type pink

  # Generate music noise (frequency-dependent crest factors) and ~11 seconds run time (SLOW!! Will take hours to complete)
  # requires Music-Noise_96kHz.wav to derive amplitude spectrum
  %(prog)s --noise-type external-ampl-target --external-ampl-wav Music-Noise_96kHz.wav --crest-targets music

  # Generate pink noise with consistent crest factors (absolute level not controlled, just consistency)
  # Will take around an hour to complete
  %(prog)s --noise-type pink --objective-mode consistent

  # Generate pink noise with only broadband crest factor control and small number of samples, very quick
  %(prog)s --noise-type pink --objective-mode broadband-only --num-samples 32768
        """,
    )

    parser.add_argument(
        "--noise-type",
        choices=[
            "pink",
            "white",
            "speech",
            "pink_a_weighted",
            "brown",
            "external-ampl-target",
        ],
        help="Type of noise spectrum to generate",
        required=True,
    )

    parser.add_argument(
        "--crest-targets",
        choices=["uniform", "music", "external-file"],
        default="uniform",
        help="Crest factor target mode: uniform (same for all bands), music (frequency-dependent), or external-file (not yet supported)",
    )

    parser.add_argument(
        "--crest-filter-type",
        choices=["fir", "iir"],
        default="fir",
        help="Type of filter used to determine fractional octave crest factors (default: fir, IIR NOT YET SUPPORTED)",
    )

    parser.add_argument(
        "--uniform-crest-factor",
        type=float,
        default=12.0,
        help='Target crest factor in dB (used when crest-targets is "uniform", default: 12.0)',
    )

    parser.add_argument(
        "--external-ampl-wav",
        type=str,
        help='Path to external WAV file for amplitude spectrum (required when noise_type is "external-ampl-target")',
    )

    parser.add_argument(
        "--crest-file",
        type=str,
        help='Path to external crest factor file (required when crest-targets is "external-file", NOT YET SUPPORTED)',
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=96000,
        help="Output sample rate in Hz (default: 96000)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=65536 * 2 * 2 * 2 * 2,
        help="Number of samples in output signal (default: 1048576)",
    )

    parser.add_argument(
        "--lf-cutoff",
        type=float,
        default=10.0,
        help="Low frequency cutoff in Hz (default: 10.0)",
    )

    parser.add_argument(
        "--hf-cutoff",
        type=float,
        default=22400.0,
        help="High frequency cutoff in Hz (default: 22400.0)",
    )

    parser.add_argument(
        "--starting-point",
        type=str,
        help="Path to WAV file for initial phases",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: auto-generated based on noise type and crest mode)",
    )

    parser.add_argument(
        "--broadband-weight",
        type=float,
        default=20.0,
        help="Weight for broadband crest factor in optimization (default: 20.0)",
    )

    parser.add_argument(
        "--objective-mode",
        choices=["target", "consistent", "broadband-only"],
        default="target",
        help="Objective function mode: target (match frequency-dependent crest factors), consistent (minimize stddev of crest factors), broadband-only (optimize only broadband crest factor)",
    )

    parser.add_argument(
        "--no-plot", action="store_true", help="Disable plotting of results"
    )

    args = parser.parse_args()

    # Some validation
    if args.noise_type == "external-ampl-target" and args.external_ampl_wav is None:
        parser.error(
            "--external-ampl-wav is required when noise_type is 'external-ampl-target'"
        )

    if args.crest_targets == "uniform":
        if args.uniform_crest_factor is None:
            parser.error(
                "--uniform-crest-factor is required when crest-targets is 'uniform'"
            )
        if args.uniform_crest_factor <= 0:
            parser.error("--uniform-crest-factor must be positive")

    if args.crest_targets == "external-file":
        if args.crest_file is None:
            parser.error(
                "--crest-file is required when crest-targets is 'external-file'"
            )
        parser.error("crest-targets 'external-file' is not yet supported")

    if args.crest_filter_type == "iir":
        parser.error("crest-filter-type 'iir' is not yet supported")

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")

    if args.lf_cutoff < 0:
        parser.error("--lf-cutoff must be non-negative")

    if args.hf_cutoff <= args.lf_cutoff:
        parser.error("--hf-cutoff must be greater than --lf-cutoff")

    return args


def get_crest_factor_targets(args, octave_freqs, third_octave_freqs, tf_oct_freqs):
    """Get crest factor targets based on the crest mode."""
    if args.crest_targets == "uniform":
        broadband_crest = args.uniform_crest_factor
        octave_crests = generate_uniform_crests(octave_freqs, args.uniform_crest_factor)
        third_octave_crests = generate_uniform_crests(
            third_octave_freqs, args.uniform_crest_factor
        )
        tf_oct_crests = generate_uniform_crests(tf_oct_freqs, args.uniform_crest_factor)
    elif args.crest_targets == "music":
        broadband_crest = generate_music_noise_crests(None)
        octave_crests = generate_music_noise_crests(octave_freqs)
        third_octave_crests = generate_music_noise_crests(third_octave_freqs)
        tf_oct_crests = generate_music_noise_crests(tf_oct_freqs)
    else:
        raise ValueError(f"Unsupported crest mode: {args.crest_targets}")

    return np.concatenate(
        [[broadband_crest], octave_crests, third_octave_crests, tf_oct_crests]
    ).astype(np.float32)


def get_amplitudes(args, freqs):
    """Get amplitude spectrum based on noise type."""
    if args.noise_type == "pink":
        return generate_pink_amplitudes(freqs)
    elif args.noise_type == "white":
        return generate_white_amplitudes(freqs)
    elif args.noise_type == "speech":
        return generate_speech_amplitudes(freqs)
    elif args.noise_type == "pink_a_weighted":
        return generate_pink_a_weighted_amplitudes(freqs)
    elif args.noise_type == "brown":
        return generate_brown_amplitudes(freqs)
    elif args.noise_type == "external-ampl-target":
        return generate_amplitudes_like(freqs, args.external_ampl_wav)
    else:
        raise ValueError(f"Unknown noise type: {args.noise_type}")


def get_output_prefix(args):
    """Generate output prefix based on arguments."""
    if args.output_prefix is not None:
        return args.output_prefix

    if args.objective_mode == "target":
        if args.crest_targets == "uniform":
            return f"{args.noise_type}_crest_target_uniform_{args.uniform_crest_factor:.1f}dB"
        elif args.crest_targets == "music":
            return f"{args.noise_type}_crest_target_music"
        elif args.crest_targets == "external-file":
            return f"{args.noise_type}_crest_target_external"

    elif args.objective_mode == "consistent":
        return f"{args.noise_type}_crest_consistent"

    elif args.objective_mode == "broadband-only":
        return (
            f"{args.noise_type}_crest_broadband_only_{args.uniform_crest_factor:.1f}dB"
        )

    else:
        raise ValueError(f"Unknown objective mode: {args.objective_mode}")


def main():
    signal_handling.signal(signal_handling.SIGINT, handle_siginit)

    args = parse_args()

    # Extract parameters from args
    lf_cutoff = args.lf_cutoff
    hf_cutoff = args.hf_cutoff
    sample_rate = args.sample_rate
    nSamples = args.num_samples
    output_prefix = get_output_prefix(args)

    print(f"\n{'='*60}")
    print(f"Advanced Arbitrary Crest Noise Generator")
    print(f"{'='*60}")
    print(
        f"Objective Mode: {friendly_objective_names.get(args.objective_mode, args.objective_mode)}"
    )
    print(f"Noise Type: {friendly_noise_names.get(args.noise_type, args.noise_type)}")
    if args.noise_type == "external-ampl-target":
        print(f"External Amplitude Target WAV: {args.external_ampl_wav}")

    print(
        f"Crest Targets: {friendly_crest_targets_names.get(args.crest_targets, args.crest_targets)}"
    )
    if args.crest_targets == "uniform":
        print(f"Target Crest Factors: {args.uniform_crest_factor:.1f} dB (uniform)")
    if args.crest_targets == "external-file":
        print(f"Target Crest Factors: From External file {args.crest_file}")

    print(f"{'='*60}\n")

    freqs = np.fft.rfftfreq(nSamples, 1 / sample_rate)
    num_freqs = len(freqs)

    # Get amplitudes based on noise type
    amplitudes = get_amplitudes(args, freqs)
    amplitudes = np.where(
        (freqs < lf_cutoff) | (freqs > hf_cutoff), np.zeros(len(amplitudes)), amplitudes
    )

    rng = np.random.default_rng(12345)

    if args.starting_point is not None:
        if not os.path.exists(args.starting_point):
            raise ValueError(
                f"Starting point file '{args.starting_point}' does not exist"
            )
        # Load starting point from WAV file
        starting_point_sample_rate, starting_signal = wavfile.read(args.starting_point)
        if starting_point_sample_rate != sample_rate:
            raise ValueError(
                f"Expected sample rate of {sample_rate}Hz, but got {starting_point_sample_rate}Hz"
            )
        starting_point_n_samples = len(starting_signal)
        if starting_point_n_samples != nSamples:
            raise ValueError(
                f"Expected {nSamples} samples, but got {starting_point_n_samples} samples"
            )

        starting_signal = starting_signal.astype(np.float64)
        starting_point_spectrum = np.fft.rfft(starting_signal)

        base_phases = np.angle(starting_point_spectrum)

        print(f"Using initial phases from file '{args.starting_point}'")
    else:
        base_phases = rng.uniform(-np.pi, np.pi, num_freqs)

        print(f"Using random initial phases in range [-pi, pi]")

    # Adjust the number of phases to optimize (don't optimize where amplitude is zero / very small)
    # This appears to massively improve stability and convergence speed
    num_phases_lf_pad = 0
    num_phases_hf_pad = 0
    lf_cutoff_index = 0
    hf_cutoff_index = len(amplitudes)

    if lf_cutoff is not None:
        lf_cutoff_index = np.argmax(amplitudes > 1e-03)
        num_phases_lf_pad = lf_cutoff_index

    if hf_cutoff is not None:
        hf_cutoff_index = len(amplitudes) - np.argmax(amplitudes[::-1] > 1e-03)
        num_phases_hf_pad = len(amplitudes) - hf_cutoff_index

    num_phases_to_optimize = num_freqs - num_phases_lf_pad - num_phases_hf_pad

    print(
        f"Removing {num_phases_lf_pad + num_phases_hf_pad} of {num_freqs} phases from optimization, {num_phases_to_optimize} phases remaining"
    )
    print(f"LF cutoff freq: {lf_cutoff}Hz, HF cutoff freq: {hf_cutoff}Hz")
    print(
        f"Optimization Indices: [{lf_cutoff_index};{hf_cutoff_index-1}] ->  [{freqs[lf_cutoff_index]};{freqs[hf_cutoff_index-1]}]Hz"
    )

    base_phases = base_phases[lf_cutoff_index:hf_cutoff_index]
    if len(base_phases) != num_phases_to_optimize:
        raise ValueError(
            f"Internal error, Expected {num_phases_to_optimize} phases to optimize, but got {len(base_phases)} phases"
        )

    octave_freqs = get_fractional_octave_center_frequencies(1)
    num_octave_freqs = len(octave_freqs)

    third_octave_freqs = get_fractional_octave_center_frequencies(3)
    num_third_octave_freqs = len(third_octave_freqs)

    tf_oct_freqs = get_fractional_octave_center_frequencies(24)
    num_tf_oct_freqs = len(tf_oct_freqs)

    num_filters = 1 + num_octave_freqs + num_third_octave_freqs + num_tf_oct_freqs

    filters = np.zeros((num_filters, num_freqs), dtype=np.complex64)

    print(f"Building {num_filters} filters...", end=" ", flush=True)
    filters[0] = np.ones(num_freqs, dtype=np.complex64)  # Base filter (no filtering)
    curr_filter_index = 1
    for fc in octave_freqs:
        curr_filter_taps = design_fractional_octave_fir_filter(
            f_center=fc, fraction=1, fs=sample_rate
        )
        curr_filter_response = np.fft.rfft(curr_filter_taps, nSamples)
        filters[curr_filter_index] = curr_filter_response
        curr_filter_index += 1

    for fc in third_octave_freqs:
        curr_filter_taps = design_fractional_octave_fir_filter(
            f_center=fc, fraction=3, fs=sample_rate
        )
        curr_filter_response = np.fft.rfft(curr_filter_taps, nSamples)
        filters[curr_filter_index] = curr_filter_response
        curr_filter_index += 1

    for fc in tf_oct_freqs:
        curr_filter_taps = design_fractional_octave_fir_filter(
            f_center=fc, fraction=24, fs=sample_rate
        )
        curr_filter_response = np.fft.rfft(curr_filter_taps, nSamples)
        filters[curr_filter_index] = curr_filter_response
        curr_filter_index += 1

    print("Done.", flush=True)

    # Get crest factor targets based on mode
    target_crests = get_crest_factor_targets(
        args, octave_freqs, third_octave_freqs, tf_oct_freqs
    )

    target_crest_weightings = np.ones((num_filters,), dtype=np.float32)
    target_crest_weightings[0] = args.broadband_weight

    opt_fun_jit = None

    if args.objective_mode == "target":
        print("Using target crest factor optimization mode.")

        opt_fun_jit = jax.jit(
            partial(
                noise_signal_objective_multi_filter_target,
                amplitudes=amplitudes,
                num_phases_lf_pad=num_phases_lf_pad,
                num_phases_hf_pad=num_phases_hf_pad,
                filters=filters,
                target_crests=target_crests,
                target_crest_weightings=target_crest_weightings,
            )
        )
    elif args.objective_mode == "consistent":
        print(
            "Using consistent crest factor optimization mode (only stddev of crests)!!"
        )

        opt_fun_jit = jax.jit(
            partial(
                noise_signal_objective_multi_filter_consistent,
                amplitudes=amplitudes,
                num_phases_lf_pad=num_phases_lf_pad,
                num_phases_hf_pad=num_phases_hf_pad,
                filters=filters,
            )
        )
    elif args.objective_mode == "broadband-only":
        print(
            "Using broadband-only optimization (no frequency-dependent crest factors)!!"
        )

        opt_fun_jit = jax.jit(
            partial(
                noise_signal_objective_single_broadband,
                amplitudes=amplitudes,
                num_phases_lf_pad=num_phases_lf_pad,
                num_phases_hf_pad=num_phases_hf_pad,
                target_broadband_crest=target_crests[0],
            )
        )

    else:
        raise ValueError(f"Unknown objective mode: {args.objective_mode}")

    opt_grad_fun_jit = jax.jit(jax.grad(opt_fun_jit))

    best_solution = base_phases.copy()
    best_objective = math.inf

    def opt_fun(phases):
        obj = opt_fun_jit(phases)

        nonlocal best_solution, best_objective
        if obj < best_objective:
            best_objective = obj
            best_solution = phases.copy()

        return obj

    def opt_grad_fun(phases):
        grad = opt_grad_fun_jit(phases)
        return grad

    num_iters = 0

    def intermediate_callback(intermediate_result: scipy.optimize.OptimizeResult):
        nonlocal num_iters
        num_iters += 1

        obj_value = intermediate_result.fun

        print(
            f"Iteration {num_iters}, Objective: {obj_value:.6f}dB, Best: {best_objective:.6f}dB",
            flush=True,
        )

        # global
        if abort_calculation:
            print(
                f"Aborting optimization at iteration {num_iters} due to user request."
            )
            raise StopIteration

        # Terminate, 0.0001dB is good enough.
        if obj_value < 1e-03:
            raise StopIteration

    print("\nStarting optimization:")
    print(f"Sample Rate: {sample_rate}Hz")
    print(f"Signal length {nSamples / sample_rate}s (Number of Samples: {nSamples})")
    print(f"Num Frequencies: {num_freqs}")
    print(f"Num Filters: {num_filters}")
    sys.stdout.flush()

    x0 = base_phases

    global optimization_running
    optimization_running = True

    scipy.optimize.minimize(
        fun=opt_fun,
        x0=x0,
        method="L-BFGS-B",
        jac=opt_grad_fun,
        bounds=[
            (-np.pi, np.pi),
        ]
        * num_phases_to_optimize,
        options={},
        callback=intermediate_callback,
    )

    optimization_running = False

    final_obj = best_objective

    print(
        f"\nOptimization finished after {num_iters} iterations, best objective: {final_obj:.6f}dB"
    )

    final_phases = np.pad(
        best_solution,
        (num_phases_lf_pad, num_phases_hf_pad),
        mode="constant",
        constant_values=0.0,
    )

    final_signal = np.fft.irfft(amplitudes * np.exp(1j * final_phases))
    final_signal = final_signal / np.max(np.abs(final_signal))

    actual_cf = crest_factor(final_signal)
    actual_cf_dB = crest_factor_to_dB(actual_cf)

    print("\nSUMMARY")
    print(f"=========================================")
    print(f"Noise Type: {friendly_noise_names.get(args.noise_type, args.noise_type)}")
    print(
        f"Crest Mode: {friendly_crest_targets_names.get(args.crest_targets, args.crest_targets)}"
    )
    print(
        f"Optimized {num_freqs} frequencies, reduced to {num_phases_to_optimize} phases"
    )
    print(f"Achieved error of {final_obj:.6f}dB after {num_iters} iterations")
    print(f"Signal statistics ({sample_rate/1000:.1f}kHz):")
    print(
        f"Duration: {len(final_signal) / sample_rate:.3f} s ({len(final_signal)} samples)"
    )
    print(f"Broadband Crest factor: {actual_cf_dB:.3f}dB ({actual_cf:.3}x)")
    print(f"Peak value: {np.max(np.abs(final_signal)):.3f}")
    print(f"RMS value: {np.sqrt(np.mean(final_signal**2)):.3f}")
    print(f"Mean: {np.mean(final_signal):.6f}")
    print(f"Std dev: {np.std(final_signal):.3f}")

    # Save signal as WAV file
    output_wav = f"generated_{output_prefix}_noise_{sample_rate/1000.0:.1f}kHz.wav"
    wavfile.write(output_wav, sample_rate, final_signal.astype(np.float32))
    print(f"\nOutput WAV: {output_wav}")

    # Save amplitudes to text file with comment
    output_amp = (
        f"generated_{output_prefix}_noise_{sample_rate/1000.0:.1f}kHz_amplitudes.txt"
    )
    with open(output_amp, "w") as f:
        f.write(
            f"# Amplitude Spectrum for {sample_rate/1000.0:.1f}kHz {args.noise_type} noise with crest factor {actual_cf_dB:.3}dB\n"
        )
        for amp in amplitudes:
            f.write(f"{amp}\n")
    print(f"Output Amplitudes: {output_amp}")

    # Save phases to text file with comment
    output_phase = (
        f"generated_{output_prefix}_noise_{sample_rate/1000.0:.1f}kHz_phases.txt"
    )
    with open(output_phase, "w") as f:
        f.write(
            f"# Phases (radians) for {sample_rate/1000.0:.1f}kHz {args.noise_type} noise with crest factor {actual_cf_dB:.3}dB\n"
        )
        for phase in final_phases:
            f.write(f"{phase}\n")
    print(f"Output Phases: {output_phase}")

    if args.no_plot:
        print("\nPlotting disabled.")
        return

    # Plot time and frequency domain
    plt.figure(figsize=(12, 8))

    # Time domain plot
    plt.subplot(2, 1, 1)
    time_axis = np.arange(nSamples) / sample_rate
    plt.plot(time_axis, final_signal)
    plt.title(
        f"{friendly_noise_names.get(args.noise_type, args.noise_type)} Time Domain (CF = {actual_cf_dB:.2f} dB)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Frequency domain plot with magnitude and phase
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.semilogx(freqs[1:], 20 * np.log10(amplitudes[1:]), "b-", label="Magnitude")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True)
    ax1.set_xlim([2, sample_rate / 2])

    ax2.semilogx(freqs[1:], np.unwrap(final_phases[1:]), "r-", alpha=0.6, label="Phase")
    ax2.set_ylabel("Phase (rad)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Frequency Domain")
    plt.tight_layout()

    # Plot histogram of the signal
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(
        final_signal,
        bins=100,
        density=True,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    plt.title(
        f"{friendly_noise_names.get(args.noise_type, args.noise_type)} Amplitude Distribution (CF = {actual_cf_dB:.2f} dB)"
    )
    plt.xlabel("Amplitude")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)

    mean_val = np.mean(final_signal)
    std_val = np.std(final_signal)
    plt.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.4f}",
    )
    plt.axvline(
        mean_val + std_val,
        color="orange",
        linestyle="--",
        linewidth=1,
        label=f"+1σ: {mean_val + std_val:.4f}",
    )
    plt.axvline(
        mean_val - std_val,
        color="orange",
        linestyle="--",
        linewidth=1,
        label=f"-1σ: {mean_val - std_val:.4f}",
    )

    mu, sigma = norm.fit(final_signal)
    x = np.linspace(final_signal.min(), final_signal.max(), 1000)
    gaussian_fit = norm.pdf(x, mu, sigma)
    plt.plot(
        x,
        gaussian_fit,
        "r-",
        linewidth=2,
        label=f"Gaussian Fit (μ={mu:.4f}, σ={sigma:.4f})",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
