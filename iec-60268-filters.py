import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Create frequency vector in Hz
f = np.logspace(np.log10(20), np.log10(20000), 1000)  # 0.1 to 1000 Hz
w = 2 * np.pi * f  # Convert to rad/s for scipy.signal.freqs

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
w1, h1 = signal.freqs(num1, den1, w)
w2, h2 = signal.freqs(num2, den2, w)
w3, h3 = signal.freqs(num3, den3, w)
w4, h4 = signal.freqs(num4, den4, w)

# Pink Noise amplitudes, normalized to 1 at 1khz
h5 = np.concatenate([[0.0], np.sqrt(200.0) / np.sqrt(np.abs(f[1:]))])

# Combine by multiplying transfer functions
h_combined = h1 * h2 * h3 * h4 * h5

print(h_combined)

# Convert back to Hz for plotting
f_plot = w1 / (2*np.pi)

# Create the plot
plt.figure(figsize=(12, 8))

# Magnitude plot
plt.subplot(2, 1, 1)
plt.semilogx(f_plot, 20*np.log10(np.abs(h1)), 'b-', label=f'Second Order High-pass (fh={fh} Hz)', alpha=0.7)
plt.semilogx(f_plot, 20*np.log10(np.abs(h2)), 'r-', label=f'Biquadratic Peaking Filter (fc={fc} Hz, Q={Q2}, G={gain2}dB)', alpha=0.7)
plt.semilogx(f_plot, 20*np.log10(np.abs(h3)), 'g-', label=f'First Order Low Pass (fl={fl} Hz)', alpha=0.7)
plt.semilogx(f_plot, 20*np.log10(np.abs(h4)), 'c-', label=f'Gain (G={gain4}dB)', alpha=0.7)
plt.semilogx(f_plot, 20*np.log10(np.abs(h5)), 'm-', label=f'Pink Noise', alpha=0.7)
plt.semilogx(f_plot, 20*np.log10(np.abs(h_combined)), 'k--', linewidth=3, label='Combined Response')
plt.ylabel('Magnitude (dB)')
plt.title('Combined Analog Filter Response')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([20, 20000])

# Phase plot
plt.subplot(2, 1, 2)
plt.semilogx(f_plot, np.angle(h1)*180/np.pi, 'b-', alpha=0.7)
plt.semilogx(f_plot, np.angle(h2)*180/np.pi, 'r-', alpha=0.7)
plt.semilogx(f_plot, np.angle(h3)*180/np.pi, 'g-', alpha=0.7)
plt.semilogx(f_plot, np.angle(h4)*180/np.pi, 'g-', alpha=0.7)
plt.semilogx(f_plot, np.angle(h_combined)*180/np.pi, 'k--', linewidth=3)
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (Hz)')
plt.grid(True, alpha=0.3)
plt.xlim([20, 20000])

plt.tight_layout()
plt.show()

# Print some key information
print("Filter Specifications:")
print(f"Second Order High-pass (fh={fh} Hz)")
print(f"Biquadratic Peaking Filter (fc={fc} Hz), Q={Q2}, G={gain2}dB)")
print(f"First Order Low Pass (fl={fl} Hz)")
print(f"Gain (G={gain4}dB)")
