#!/bin/env python3

"""Example for optimizing scipy.optimize.rosen."""

import numpy as np
import jax
import jax.numpy as jnp
import ipyopt
from jax.test_util import check_grads
import scipy.optimize
from functools import partial
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_pink_amplitudes(freqs, lf_cutoff = 10, hf_cutoff = 22400, normalization_freq=1000.0):
    # Pink noise has 1/f power spectral density, normalize to 1 at 1kHz
    # Therefore we need 1/sqrt(f) amplitude spectral density, because
    # psd = asd^2, so if psd ~ 1/f, then asd ~ 1/sqrt(f)
    # DC is always 0

    # Generate the amplitudes, prevent divison by zero at DC
    ampls = np.concatenate([[0.0], np.sqrt(normalization_freq) / np.sqrt(np.abs(freqs[1:]))])

    # Set values below lf_cutoff to 0
    return np.where((freqs < lf_cutoff) | (freqs > hf_cutoff), np.zeros(len(ampls)), ampls)


@jax.jit
def crest_factor(signal):
    peak = jnp.max(jnp.abs(signal))
    rms = jnp.sqrt(jnp.mean(signal**2))

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


def eval_g(_x, _out):
    return


def eval_jac_g(_x, _out):
    return


# define the nonzero slots in the jacobian
# there are no nonzeros in the constraint jacobian
eval_jac_g_sparsity_indices = (np.array([]), np.array([]))

def main():
    """Entry point."""

    sample_rate = 96000
    nSamples = 65536*2*2*2*2

    print(f"{nSamples / sample_rate} seconds")

    target_crest = 12

    freqs = np.fft.rfftfreq(nSamples, 1/sample_rate)
    num_freqs = len(freqs)
    #print(freqs)

    print(f"delta f: {freqs[1] - freqs[0]}Hz")
    

    amplitudes = generate_pink_amplitudes(freqs)
    #print(amplitudes)

    rng = np.random.default_rng(12345)
    base_phases = rng.uniform(-np.pi, np.pi, num_freqs)

    opt_fun_jit = jax.jit(partial(noise_signal_obj, amplitudes=amplitudes, target_crest=target_crest))
    opt_grad_fun_jit = jax.grad(opt_fun_jit)

    opt_fun = lambda phases: opt_fun_jit(phases)
    opt_grad_fun = lambda phases: opt_grad_fun_jit(phases)

    #CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP
    #CG, BFGS, Newton-CG: No Bounds
    # -> L-BFGS-B, TNC, SLSQP
    # res = scipy.optimize.minimize(
    #     fun = opt_fun,
    #     x0 = base_phases,
    #     method='SLSQP',
    #     jac = opt_grad_fun,
    #     bounds = [(-np.pi, np.pi),] * num_freqs,
    #     options={'disp': True,},
    #     callback=lambda xk: print(f"Callback"),
    # )

    # print(res)

    # define the parameters and their box constraints
    nvar = num_freqs
    x_l = np.array([-np.pi] * nvar, dtype=float)
    x_u = np.array([np.pi] * nvar, dtype=float)

    # define the inequality constraints
    ncon = 0
    g_l = np.array([], dtype=float)
    g_u = np.array([], dtype=float)

    del opt_grad_fun

    def opt_grad_fun(x, out):
        out[()] = opt_grad_fun_jit(x)
    
    def intermediate_callback(
        mode: int, iter: int, obj_value: float, inf_pr: float, inf_du: float, mu: float, d_norm: float, regularization_size: float, alpha_du: float, alpha_pr: float, ls_trials: int

    ):
        #print(f"Intermediate callback: iter={iter}, mode={mode}, obj_value={obj_value}, inf_pr={inf_pr}, inf_du={inf_du}, mu={mu}, d_norm={d_norm}, regularization_size={regularization_size}, alpha_du={alpha_du}, alpha_pr={alpha_pr}, ls_trials={ls_trials}")

        # Terminate, 0.0001dB is good enough.
        if obj_value < 0.0001:
            return False

        return True

    # create the nonlinear programming model
    nlp = ipyopt.Problem(
        n = nvar, # number of variables
        x_l = x_l, # lower bounds?
        x_u = x_u, # upper bounds?
        m = ncon, # number of constraints?
        g_l = g_l, # constraint lower bounds
        g_u = g_u, # constraint upper bounds
        sparsity_indices_jac_g = eval_jac_g_sparsity_indices, # sparsity pattern of the jacobian
        sparsity_indices_h = None, # sparsity pattern of the hessian
        eval_f = opt_fun, # objective function
        eval_grad_f = opt_grad_fun, # gradient of the objective function
        eval_g = eval_g, # constraint functions
        eval_jac_g = eval_jac_g, # jacobian of the constraint functions,
        ipopt_options = {
            'print_level': 5,
            #'max_iter': 50,
        },
        intermediate_callback=intermediate_callback
    )
    #nlp.set()

    # define the initial guess
    x0 = base_phases

    # compute the results using ipopt
    _x, obj, status = nlp.solve(x0)

    # report the results
    print(f"obj: {obj}dB")

    final_phases = _x

    final_signal = np.fft.irfft(amplitudes * np.exp(1j * final_phases))
    final_signal = final_signal / np.max(np.abs(final_signal))

    actual_cf = crest_factor(final_signal)
    actual_cf_dB = crest_factor_to_dB(actual_cf)

    print(f"Optimized {num_freqs} frequencies")
    print(f"Signal statistics ({sample_rate/1000:.1f}kHz):")
    print(f"Duration: {len(final_signal) / sample_rate:.3f} s ({len(final_signal)} samples)")
    print(f"Crest factor: {actual_cf_dB:.3f}dB ({actual_cf:.3}x)")
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
