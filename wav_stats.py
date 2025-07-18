#!/usr/bin/env python3
"""
WAV File Signal Statistics Analyzer
Reads a WAV file and outputs various signal statistics.
"""

import sys
import os
import numpy as np
from scipy.io import wavfile


def calculate_wav_statistics(filepath):
    """
    Calculate and display statistics for a WAV file.
    
    Parameters:
    filepath (str): Path to the WAV file
    """
    try:
        # Read the WAV file
        sample_rate, data = wavfile.read(filepath)
        
        # Get file info
        file_size = os.path.getsize(filepath)
        
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
        
        # Calculate statistics
        duration = len(data) / sample_rate
        peak_max = np.max(data_float)
        peak_min = np.min(data_float)
        peak_absolute = max(abs(peak_max), abs(peak_min))
        
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(data_float**2))
        
        # Mean and standard deviation
        mean = np.mean(data_float)
        stdev = np.std(data_float)
        
        # Crest factor
        if rms > 0:
            crest_factor = peak_absolute / rms
            crest_factor_db = 20 * np.log10(crest_factor)
        else:
            crest_factor = float('inf')
            crest_factor_db = float('inf')
        
        # Display results
        print("\n" + "="*50)
        print(f"WAV FILE STATISTICS: {os.path.basename(filepath)}")
        print("="*50)
        print(f"File Size:          {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"Sample Format:      {sample_format}")
        print(f"Sample Rate:        {sample_rate:,} Hz")
        print(f"Duration:           {duration:.3f} seconds")
        print(f"Total Samples:      {len(data):,}")
        print("\nSignal Statistics:")
        print(f"Peak Max:           {peak_max:.6f}")
        print(f"Peak Min:           {peak_min:.6f}")
        print(f"Peak Absolute:      {peak_absolute:.6f}")
        print(f"RMS Value:          {rms:.6f}")
        print(f"Mean:               {mean:.6f}")
        print(f"Std Deviation:      {stdev:.6f}")
        print(f"Crest Factor:       {crest_factor:.3f}")
        print(f"Crest Factor (dB):  {crest_factor_db:.2f} dB")
        
        # Normalized values (if applicable)
        if np.issubdtype(data.dtype, np.integer):
            print(f"\nNormalized Values (relative to {max_possible_value}):")
            print(f"Peak Max (norm):    {peak_max/max_possible_value:.6f}")
            print(f"Peak Min (norm):    {peak_min/max_possible_value:.6f}")
            print(f"RMS (norm):         {rms/max_possible_value:.6f}")
        
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading WAV file: {str(e)}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python wav_stats.py <wav_file_path>")
        print("Example: python wav_stats.py audio.wav")
        sys.exit(1)
    
    filepath = sys.argv[1]
    calculate_wav_statistics(filepath)


if __name__ == "__main__":
    main()
