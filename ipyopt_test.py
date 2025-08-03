#!/bin/env python3

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
from freq_dependent_crest_factor import get_fractional_octave_center_frequencies, design_fractional_octave_fir_filter
import signal as signal_handling
import scipy.signal as signal
import scipy.interpolate

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


def generate_pink_amplitudes(freqs, normalization_freq=1000.0):
    # Pink noise has 1/f power spectral density, normalize to 1 at 1kHz
    # Therefore we need 1/sqrt(f) amplitude spectral density, because
    # psd = asd^2, so if psd ~ 1/f, then asd ~ 1/sqrt(f)
    # DC is always 0

    # Generate the amplitudes, prevent divison by zero at DC
    ampls = np.concatenate([[0.0], np.sqrt(normalization_freq) / np.sqrt(np.abs(freqs[1:]))])

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

    return ampls

def generate_pink_a_weighted_amplitudes(freqs):
    a_weighting_fun = lambda f: (12194**2 * f**4) / ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) * (f**2 + 12194**2))

    pink_amplitudes = generate_pink_amplitudes(freqs)
    ampls = pink_amplitudes * a_weighting_fun(freqs)

    return ampls

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
    cs = scipy.interpolate.CubicSpline(freqs_source, avg_magnitudes)
    return cs(freqs)


def generate_music_noise_crests(freqs):

    if freqs is None:
        return 18.06 #broadband crest factor

    one_third_crest_factors = np.array([
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
        [24000, 22.3]
    ])

    # Interpolate the crest factors to the frequencies
    cs = scipy.interpolate.CubicSpline(one_third_crest_factors[:, 0], one_third_crest_factors[:, 1])
    interpolated_crests = cs(freqs)

    return interpolated_crests

@jax.jit
def crest_factor(sig):
    peak = jnp.max(jnp.abs(sig))
    rms = jnp.sqrt(jnp.mean(sig**2))

    return jnp.where(rms <= 0.0, jnp.inf, peak / rms)

@jax.jit
def crest_factor_to_dB(cf):
    return 20 * jnp.log10(cf)

def noise_signal_obj(phases, amplitudes, target_crest):

    curr_spectrum = amplitudes * jnp.exp(1j * phases)

    curr_signal = jnp.fft.irfft(curr_spectrum)

    curr_crest_factor_dB = crest_factor_to_dB(crest_factor(curr_signal))

    curr_obj_fun = jnp.abs(curr_crest_factor_dB - target_crest)

    return curr_obj_fun

@jax.jit
def crest_factor_mtx(signal_mtx):
    # Signals are in each row
    peak = jnp.max(jnp.abs(signal_mtx), axis=1)
    rms = jnp.sqrt(jnp.mean(signal_mtx**2, axis=1))

    return jnp.where(rms <= 0.0, jnp.inf, peak / rms)

def noise_signal_obj_filter(phases, num_phases_lf_pad, num_phases_hf_pad, amplitudes, filters, target_crests, target_crest_weightings):
    
    padded_phases = jnp.pad(phases, (num_phases_lf_pad, num_phases_hf_pad), mode='constant', constant_values=0.0)

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

    curr_obj_fun = jnp.mean(jnp.abs(crest_factors_dB - target_crests) * target_crest_weightings)
    #curr_obj_fun = jnp.std(crest_factors_dB)

    return curr_obj_fun

def eval_g(_x, _out):
    return


def eval_jac_g(_x, _out):
    return


# define the nonzero slots in the jacobian
# there are no nonzeros in the constraint jacobian
eval_jac_g_sparsity_indices = (np.array([]), np.array([]))

def main():
    signal_handling.signal(signal_handling.SIGINT, handle_siginit)

    target_crest = 12
    lf_cutoff = 10.0
    hf_cutoff = 22400.0
    sample_rate = 96000
    nSamples = 65536#*2*2*2*2

    freqs = np.fft.rfftfreq(nSamples, 1/sample_rate)
    num_freqs = len(freqs)
    

    #amplitudes = generate_pink_amplitudes(freqs)
    amplitudes = generate_amplitudes_like(freqs, 'Music-Noise_96kHz.wav')
    amplitudes = np.where((freqs < lf_cutoff) | (freqs > hf_cutoff), np.zeros(len(amplitudes)), amplitudes)


    rng = np.random.default_rng(12345)

    if os.path.exists('starting_point.wav'):
        # Load starting point from WAV file
        starting_point_sample_rate, starting_signal = wavfile.read('starting_point.wav')
        if starting_point_sample_rate != 96000:
            raise ValueError(f"Expected sample rate of 96000Hz, but got {starting_point_sample_rate}Hz")
        starting_point_n_samples = len(starting_signal)
        if starting_point_n_samples != nSamples:
            raise ValueError(f"Expected {nSamples} samples, but got {starting_point_n_samples} samples")
        
        starting_signal = starting_signal.astype(np.float64)
        starting_point_spectrum = np.fft.rfft(starting_signal)

        base_phases = np.angle(starting_point_spectrum)

        print(f"Using initial phases from 'starting_point.wav'")
    else:
        base_phases = rng.uniform(-np.pi, np.pi, num_freqs)

        print(f"Using random initial phases in range [-pi, pi]")

    # Adjust the number of phases to optimize (don't optimize where amplitude is zero / very small)
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

    print(f"Removing {num_phases_lf_pad + num_phases_hf_pad} of {num_freqs} phases from optimization, {num_phases_to_optimize} phases remaining")
    print(f"LF cutoff freq: {lf_cutoff}Hz, HF cutoff freq: {hf_cutoff}Hz")
    print(f"Optimization Indices: [{lf_cutoff_index};{hf_cutoff_index-1}] ->  [{freqs[lf_cutoff_index]};{freqs[hf_cutoff_index-1]}]Hz")
    
    base_phases = base_phases[lf_cutoff_index:hf_cutoff_index]
    if len(base_phases) != num_phases_to_optimize:
        raise ValueError(f"Internal error, Expected {num_phases_to_optimize} phases to optimize, but got {len(base_phases)} phases")

    octave_freqs = get_fractional_octave_center_frequencies(1)
    num_octave_freqs = len(octave_freqs)

    third_octave_freqs = get_fractional_octave_center_frequencies(3)
    num_third_octave_freqs = len(third_octave_freqs)

    tf_oct_freqs = get_fractional_octave_center_frequencies(24)
    num_tf_oct_freqs = len(tf_oct_freqs)

    num_filters = 1 + num_octave_freqs + num_third_octave_freqs + num_tf_oct_freqs

    filters = np.zeros((num_filters, num_freqs), dtype=np.complex64)

    print(f"Building {num_filters} filters...", end=' ', flush=True)
    filters[0] = np.ones(num_freqs, dtype=np.complex64)  # Base filter (no filtering)
    curr_filter_index = 1
    for fc in octave_freqs:
        curr_filter_taps = design_fractional_octave_fir_filter(
            f_center=fc,
            fraction=1,
            fs=sample_rate
        )
        curr_filter_response = np.fft.rfft(curr_filter_taps, nSamples)
        filters[curr_filter_index] = curr_filter_response
        curr_filter_index += 1
    
    for fc in third_octave_freqs:
        curr_filter_taps = design_fractional_octave_fir_filter(
            f_center=fc,
            fraction=3,
            fs=sample_rate
        )
        curr_filter_response = np.fft.rfft(curr_filter_taps, nSamples)
        filters[curr_filter_index] = curr_filter_response
        curr_filter_index += 1

    for fc in tf_oct_freqs:
        curr_filter_taps = design_fractional_octave_fir_filter(
            f_center=fc,
            fraction=24,
            fs=sample_rate
        )
        curr_filter_response = np.fft.rfft(curr_filter_taps, nSamples)
        filters[curr_filter_index] = curr_filter_response
        curr_filter_index += 1
    
    print("Done.", flush=True)

    #target_crests = np.full((num_filters,), target_crest, dtype=np.float32)

    # Get crest factor values
    broadband_crest = generate_music_noise_crests(None)
    octave_crests = generate_music_noise_crests(octave_freqs)
    third_octave_crests = generate_music_noise_crests(third_octave_freqs)
    tf_oct_crests = generate_music_noise_crests(tf_oct_freqs)
    target_crests = np.concatenate([[broadband_crest], octave_crests, third_octave_crests, tf_oct_crests]).astype(np.float32)

    target_crest_weightings = np.ones((num_filters,), dtype=np.float32)
    target_crest_weightings[0] = 20.0
    #target_crest_weightings /= np.max(target_crest_weightings)
    
    #opt_fun_jit = jax.jit(partial(noise_signal_obj, amplitudes=amplitudes, target_crest=target_crest))
    opt_fun_jit = jax.jit(partial(
        noise_signal_obj_filter,
        amplitudes=amplitudes,
        num_phases_lf_pad=num_phases_lf_pad,
        num_phases_hf_pad=num_phases_hf_pad,
        filters=filters,
        target_crests=target_crests,
        target_crest_weightings=target_crest_weightings
    ))
    opt_grad_fun_jit = jax.jit(jax.grad(opt_fun_jit))

    best_solution = base_phases.copy()
    best_objective = math.inf

    def opt_fun(phases):
        obj = opt_fun_jit(phases)

        nonlocal best_solution, best_objective
        if obj < best_objective:
            best_objective = obj
            best_solution = phases.copy()
            #print(f"DEBUG New best solution found and saved, obj is {obj:.6f}dB")

        return obj

    def opt_grad_fun(phases):
        grad = opt_grad_fun_jit(phases)

        #out[()] = grad
        return grad

    # print(res)

    # define the parameters and their box constraints
    nvar = num_phases_to_optimize
    x_l = np.array([-np.pi] * nvar, dtype=float)
    x_u = np.array([np.pi] * nvar, dtype=float)

    # define the inequality constraints
    ncon = 0
    g_l = np.array([], dtype=float)
    g_u = np.array([], dtype=float)
    
    num_iters = 0

    def intermediate_callback(
        intermediate_result: scipy.optimize.OptimizeResult
    ):
        nonlocal num_iters
        num_iters += 1

        obj_value = intermediate_result.fun

        print(f"Iteration {num_iters}, Objective: {obj_value:.6f}dB, Best: {best_objective:.6f}dB", flush=True)

        # global
        if abort_calculation:
            print(f"Aborting optimization at iteration {num_iters} due to user request.")
            raise StopIteration

        # Terminate, 0.0001dB is good enough.
        if obj_value < 1e-03:
            raise StopIteration

    print("Starting optimization:")
    print(f"Sample Rate: {sample_rate}Hz")
    print(f"Signal length {nSamples / sample_rate}s (Number of Samples: {nSamples})")
    print(f"Num Frequencies: {num_freqs}")
    print(f"Num Filters: {num_filters}")
    sys.stdout.flush()

    # define the initial guess
    x0 = base_phases

    # compute the results using ipopt
    global optimization_running
    optimization_running = True

    #_x, obj, status = nlp.solve(x0)
    #CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP
    #CG, BFGS, Newton-CG: No Bounds
    # -> L-BFGS-B, TNC, SLSQP
    # ignore res, we track the best solution ourselves
    scipy.optimize.minimize(
        fun = opt_fun,
        x0 = x0,
        method='L-BFGS-B',
        jac = opt_grad_fun,
        bounds = [(-np.pi, np.pi),] * num_phases_to_optimize,
        options={
            #'maxiter': 5
        },
        callback=intermediate_callback
    )

    optimization_running = False

    final_obj = best_objective

    print(f"Optimization finished after {num_iters} iterations, best objective: {final_obj:.6f}dB")

    final_phases = np.pad(best_solution, (num_phases_lf_pad, num_phases_hf_pad), mode='constant', constant_values=0.0)

    final_signal = np.fft.irfft(amplitudes * np.exp(1j * final_phases))
    final_signal = final_signal / np.max(np.abs(final_signal))

    actual_cf = crest_factor(final_signal)
    actual_cf_dB = crest_factor_to_dB(actual_cf)

    print("\nSUMMARY")
    print(f"=========================================")
    print(f"Optimized {num_freqs} frequencies, reduced to {num_phases_to_optimize} phases")
    print(f"Achieved error of {final_obj:.6f}dB after {num_iters} iterations")
    print(f"Signal statistics ({sample_rate/1000:.1f}kHz):")
    print(f"Duration: {len(final_signal) / sample_rate:.3f} s ({len(final_signal)} samples)")
    print(f"Broadband Crest factor: {actual_cf_dB:.3f}dB ({actual_cf:.3}x)")
    print(f"Peak value: {np.max(np.abs(final_signal)):.3f}")
    print(f"RMS value: {np.sqrt(np.mean(final_signal**2)):.3f}")
    print(f"Mean: {np.mean(final_signal):.6f}")
    print(f"Std dev: {np.std(final_signal):.3f}")

    # Save upsampled signal as WAV file
    wavfile.write(f'generated_{"test"}_noise_{sample_rate/1000.0:.1f}kHz.wav', sample_rate, final_signal.astype(np.float32))

    # Save amplitudes to text file with comment
    with open(f'generated_{"test"}_noise_{sample_rate/1000.0:.1f}kHz_amplitudes.txt', 'w') as f:
        f.write(f"# Amplitude Spectrum for {sample_rate/1000.0:.1f}kHz {"test"} noise with crest factor {actual_cf_dB:.3}dB\n")
        for amp in amplitudes:
            f.write(f"{amp}\n")

    # Save phases to text file with comment
    with open(f'generated_{"test"}_noise_{sample_rate/1000.0:.1f}kHz_phases.txt', 'w') as f:
        f.write(f"# Phases (radians) for {sample_rate/1000.0:.1f}kHz {"test"} noise with crest factor {actual_cf_dB:.3}dB\n")
        for phase in final_phases:
            f.write(f"{phase}\n")

    # Plot time and frequency domain
    plt.figure(figsize=(12, 8))

    # Time domain plot
    plt.subplot(2, 1, 1)
    time_axis = np.arange(nSamples) / sample_rate
    plt.plot(time_axis, final_signal)
    plt.title(f'test Time Domain (CF = {actual_cf_dB:.2f} dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Frequency domain plot with magnitude and phase
    plt.subplot(2, 1, 2)
    # We'll use twinx to have two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot magnitude on left axis
    ax1.semilogx(freqs[1:], 20 * np.log10(amplitudes[1:]), 'b-', label='Magnitude')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.set_xlim([2, sample_rate/2])

    # Plot phase on right axis
    ax2.semilogx(freqs[1:], np.unwrap(final_phases[1:]), 'r-', alpha=0.6, label='Phase')
    ax2.set_ylabel('Phase (rad)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Frequency Domain')
    plt.tight_layout()

    # Plot histogram of the upsampled signal
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(final_signal, bins=100, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'test Amplitude Distribution (CF = {actual_cf_dB:.2f} dB)')
    plt.xlabel('Amplitude')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    mean_val = np.mean(final_signal)
    std_val = np.std(final_signal)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'+1σ: {mean_val + std_val:.4f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1, label=f'-1σ: {mean_val - std_val:.4f}')
    
    # Fit and overlay Gaussian distribution
    mu, sigma = norm.fit(final_signal)
    x = np.linspace(final_signal.min(), final_signal.max(), 1000)
    gaussian_fit = norm.pdf(x, mu, sigma)
    plt.plot(x, gaussian_fit, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu:.4f}, σ={sigma:.4f})')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

