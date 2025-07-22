#!/usr/bin/env python3
"""
Frequency-dependent crest factor calculator for WAV files.
Calculates crest factor using fractional octave filters.
"""

import numpy as np
import argparse
from scipy.io import wavfile
from scipy import signal
import sys
import matplotlib.pyplot as plt


def read_wav_file(filepath):
    """
    Read and preprocess a WAV file.
    
    Args:
        filepath: Path to the WAV file
    
    Returns:
        fs: Sample rate (Hz)
        data: Audio data as normalized float array (mono)
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(filepath)
    
    # Handle stereo/multi-channel audio by converting to mono
    if len(data.shape) > 1:
        print(f"Multi-channel audio detected ({data.shape[1]} channels)")
        print("Converting to mono for statistics calculation...")
        data = np.mean(data, axis=1)
    
    # Determine sample format
    dtype_str = str(data.dtype)
    if 'int16' in dtype_str:
        sample_format = "16-bit integer"
        max_possible_value = 32767.0
    elif 'int32' in dtype_str:
        sample_format = "32-bit integer"
        max_possible_value = 2147483647.0
    elif 'float32' in dtype_str:
        sample_format = "32-bit float"
        max_possible_value = 1.0
    elif 'float64' in dtype_str:
        sample_format = "64-bit float"
        max_possible_value = 1.0
    else:
        sample_format = dtype_str
        max_possible_value = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
    
    # Convert to float for calculations
    data_float = data.astype(np.float64)

    return sample_rate, data_float


def get_fractional_octave_center_frequencies(fraction):
    """
    Generate center frequencies for fractional octave bands.
    Based on ISO 266:1997 and ANSI S1.11-2004.
    
    Args:
        fraction: Fractional octave (1, 3, 6, 12, or 24)
    
    Returns:
        List of center frequencies in Hz
    """

    TFOctCenterFrequencies = [
        20.0,    20.6,    21.2,    21.8,    22.4,    23.0,    23.6,    24.3,    25.0,    25.8,
        26.5,    27.2,    28.0,    29.0,    30.0,    30.7,    31.5,    32.5,    33.5,    34.5,
        35.5,    36.5,    37.5,    38.7,    40.0,    41.2,    42.5,    43.7,    45.0,    46.2,
        47.5,    48.7,    50.0,    51.5,    53.0,    54.5,    56.0,    58.0,    60.0,    61.5,
        63.0,    65.0,    67.0,    69.0,    71.0,    73.0,    75.0,    77.5,    80.0,    82.5,
        85.0,    87.5,    90.0,    92.5,    95.0,    97.5,

        100.0,   103.0,   106.0,   109.0,   112.0,   115.0,   118.0,   122.0,   125.0,   128.0,
        132.0,   136.0,   140.0,   145.0,   150.0,   155.0,   160.0,   165.0,   170.0,   175.0,
        180.0,   185.0,   190.0,   195.0,   200.0,   206.0,   212.0,   218.0,   224.0,   230.0,
        236.0,   243.0,   250.0,   258.0,   265.0,   272.0,   280.0,   290.0,   300.0,   307.0,
        315.0,   325.0,   335.0,   345.0,   355.0,   365.0,   375.0,   387.0,   400.0,   412.0,
        425.0,   437.0,   450.0,   462.0,   475.0,   487.0,   500.0,   515.0,   530.0,   545.0,
        560.0,   580.0,   600.0,   615.0,   630.0,   650.0,   670.0,   690.0,   710.0,   730.0,
        750.0,   775.0,   800.0,   825.0,   850.0,   875.0,   900.0,   925.0,   950.0,   975.0,

        1000.0,  1030.0,  1060.0,  1090.0,  1120.0,  1150.0,  1180.0,  1220.0,  1250.0,  1280.0,
        1320.0,  1360.0,  1400.0,  1450.0,  1500.0,  1550.0,  1600.0,  1650.0,  1700.0,  1750.0,
        1800.0,  1850.0,  1900.0,  1950.0,  2000.0,  2060.0,  2120.0,  2180.0,  2240.0,  2300.0,
        2360.0,  2430.0,  2500.0,  2580.0,  2650.0,  2720.0,  2800.0,  2900.0,  3000.0,  3070.0,
        3150.0,  3250.0,  3350.0,  3450.0,  3550.0,  3650.0,  3750.0,  3870.0,  4000.0,  4120.0,
        4250.0,  4370.0,  4500.0,  4620.0,  4750.0,  4870.0,  5000.0,  5150.0,  5300.0,  5450.0,
        5600.0,  5800.0,  6000.0,  6150.0,  6300.0,  6500.0,  6700.0,  6900.0,  7100.0,  7300.0,
        7500.0,  7750.0,  8000.0,  8250.0,  8500.0,  8750.0,  9000.0,  9250.0,  9500.0,  9750.0,

        10000.0, 10300.0, 10600.0, 10900.0, 11200.0, 11500.0, 11800.0, 12200.0, 12500.0, 12800.0,
        13200.0, 13600.0, 14000.0, 14500.0, 15000.0, 15500.0, 16000.0, 16500.0, 17000.0, 17500.0,
        18000.0, 18500.0, 19000.0, 19500.0, 20000.0, 20600.0, 21200.0, 21800.0, 22400.0, 23000.0
    ]
    
    TwelthOctCenterFrequencies = range(2, len(TFOctCenterFrequencies), 2)

    SixthOctIndices = range(4, len(TFOctCenterFrequencies), 4)

    ThirdOctIndices = range(8, len(TFOctCenterFrequencies), 8)
        
    OctIndices = range(16, len(TFOctCenterFrequencies), 24)

    if fraction == 24:
        return TFOctCenterFrequencies

    if fraction == 12:
        return [TFOctCenterFrequencies[index] for index in TwelthOctCenterFrequencies]

    if fraction == 6:
        return [TFOctCenterFrequencies[index] for index in SixthOctIndices]

    if fraction == 3:
        return [TFOctCenterFrequencies[index] for index in ThirdOctIndices]
    
    if fraction == 1:
        return [TFOctCenterFrequencies[index] for index in OctIndices]

    raise ValueError(f"nsupported fractional octave: 1/{fraction}")

def get_fractional_octave_edges(fc, fraction):
    """
    Calculate the edge frequencies for a fractional octave band.
    
    Args:
        fc: Center frequency (Hz)
        fraction: Fraction of octave (e.g., 1 for octave, 3 for third-octave)
    
    Returns:
        fc_low, fc_high: Lower and upper edge frequencies
    """
    factor = 2 ** (1 / (2 * fraction))
    fc_low = fc / factor
    fc_high = fc * factor
    return fc_low, fc_high

def design_fractional_octave_butterworth_filter(f_center, fraction, fs):

    low, high = get_fractional_octave_edges(f_center, fraction)

    sos = signal.butter(6, [low, high], btype='band', fs=fs, output='sos')
    return sos


def design_fractional_octave_fir_filter(f_center, fraction, fs):
    low, high = get_fractional_octave_edges(f_center, fraction)
    
    h = signal.firwin(512, [low, high], pass_zero=False, fs=fs)
    return h



def filter_signal(signal_data, f_center, fraction, fs, filter_type):
    
    # Prepend signal to handle filter transients
    prepended_signal = np.concatenate([signal_data, signal_data])
    
    if filter_type == 'iir':
        sos_filter = design_fractional_octave_butterworth_filter(f_center, fraction, fs)
        filtered = signal.sosfilt(sos_filter, prepended_signal)
    else:  # fir
        coeffs = design_fractional_octave_fir_filter(f_center, fraction, fs)
        filtered = signal.lfilter(coeffs, 1, prepended_signal)

    # Return only the second half (original signal length)
    return filtered[len(signal_data):]


def calculate_crest_factor(signal_data):
    """
    Calculate the crest factor of a signal.
    Crest factor = peak amplitude / RMS value
    
    Args:
        signal_data: Input signal
    
    Returns:
        Crest factor in dB
    """
    if len(signal_data) == 0:
        return float('nan')
    
    peak = np.max(np.abs(signal_data))
    rms = np.sqrt(np.mean(signal_data ** 2))
    
    if rms == 0:
        return float('inf')
    
    crest_factor_linear = peak / rms
    crest_factor_db = 20 * np.log10(crest_factor_linear)
    
    return crest_factor_db


def plot_crest_factors(center_frequencies, crest_factors, broadband_cf, input_file, fraction, filter_type):
    """
    Create a bar plot of the frequency-dependent crest factors.
    
    Args:
        center_frequencies: List of center frequencies
        crest_factors: List of corresponding crest factors in dB
        broadband_cf: Broadband crest factor in dB
        input_file: Input file name for the plot title
        fraction: Fractional octave width
    """
    plt.figure(figsize=(14, 8))
    
    # Create bar plot with log-spaced x positions
    x_positions = np.arange(len(center_frequencies))
    bars = plt.bar(x_positions, crest_factors, alpha=0.7, color='steelblue', 
                   edgecolor='darkblue', linewidth=0.5)
    
    # Set x-axis labels to center frequencies
    # Show fewer labels for readability
    step = max(1, len(center_frequencies) // 30)  # Show about 15 labels max
    plt.xticks(x_positions[::step], 
               [f"{fc:.0f}" if fc >= 100 else f"{fc:.1f}" for fc in center_frequencies[::step]], 
               rotation=45, ha='right')
    
    # Add horizontal line for broadband crest factor
    plt.axhline(y=broadband_cf, color='red', linestyle='--', linewidth=2, 
                label=f'Broadband CF: {broadband_cf:.2f} dB')
    
    # Add value labels on top of bars (for every nth bar to avoid clutter)
    label_step = 1 #max(1, len(center_frequencies) // 10)
    for i in range(0, len(bars), label_step):
        if not np.isnan(crest_factors[i]) and not np.isinf(crest_factors[i]):
            plt.text(i, crest_factors[i] + 0.1, f'{crest_factors[i]:.1f}', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Formatting
    plt.xlabel('Center Frequency (Hz)')
    plt.ylabel('Crest Factor (dB)')
    plt.title(f'Frequency-Dependent Crest Factor\n1/{fraction} octave bands, {'IIR' if filter_type == 'iir' else 'FIR'} Filters\n{input_file}', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc='upper right')
    
    # Set y-axis limits with some padding
    y_min = 0 # min(min(crest_factors), broadband_cf) - 1
    y_max = max(max(crest_factors), broadband_cf) + 2
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate frequency-dependent crest factor of WAV files'
    )
    parser.add_argument('input_file', help='Input WAV file path')
    parser.add_argument(
        '--fraction', '-f', type=int, choices=[1, 3, 6, 12, 24], default=3,
        help='Fractional octave width (1, 3, 6, 12, or 24; default: 3)'
    )
    parser.add_argument(
        '--filter-type', '-t', choices=['iir', 'fir'], default='iir',
        help='Filter type (default: iir)'
    )
    parser.add_argument(
        '--plot', '-p', action='store_true',
        help='Display bar plot of crest factors'
    )
    
    args = parser.parse_args()
    
    # Read WAV file
    try:
        fs, signal_data = read_wav_file(args.input_file)
    except Exception as e:
        print(f"Error reading WAV file: {e}", file=sys.stderr)
        return 1
    
    # Get center frequencies for the specified fractional octave
    center_frequencies = get_fractional_octave_center_frequencies(args.fraction)
    
    print(f"\nFrequency-Dependent Crest Factor Analysis")
    print(f"=========================================")
    print(f"File: {args.input_file}")
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(signal_data)/fs:.2f} seconds")
    print(f"Filter type: {args.filter_type}")
    print(f"Fractional octave: 1/{args.fraction}")
    print(f"\nResults:")
    print(f"{'Center Freq (Hz)':>15} | {'Crest Factor (dB)':>17}")
    print("-" * 70)
    
    # Store results for plotting
    crest_factors = []
    
    # Process each frequency band
    for fc in center_frequencies:
        
        # Apply filter
        filtered_signal = filter_signal(signal_data, fc, args.fraction, fs, args.filter_type)
        
        # Calculate crest factor
        crest_factor_db = calculate_crest_factor(filtered_signal)
        crest_factors.append(crest_factor_db)
        
        print(f"{fc:>15.1f} | {crest_factor_db:>17.2f}dB")
            

    # Calculate broadband crest factor
    print("-" * 70)
    broadband_cf = calculate_crest_factor(signal_data)
    print(f"{'Broadband':>15} | {broadband_cf:>17.2f}dB")
    
    # Generate plot if requested
    if args.plot:
        plot_crest_factors(center_frequencies, crest_factors, broadband_cf, 
                          args.input_file, args.fraction, args.filter_type)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
