# Noise Generator with Crest Factor Control

This project uses a simple Python script to generate various types of noise with a specified crest factor. The script supports white, pink, brown noise, speech noise (IEC-60268-16 A.6.1 Sprachförmiges Geräusch), and A-weighted pink noise. It can also process an external WAV file.

The generated noise is Periodic Noise, i.e. noise that is synthesized from an inverse FFT and not pseudorandom noise. This has some advantages when you perform measurements (i.e. spectrum is correct without lots of averaging).

Some utilities are included:
- `iec-60268-filters.py`: Plots the filter response of the IEC-60268-16 A.6.1 Speech Noise filter, hoping that I understood the standard correctly.
- `wav_stats.py`: Prints some statistics about a WAV file, such as sample rate, number of channels, duration, and crest factor.
- `pn_pink_noise_test.py`, `pn_white_noise_test.py`, `white_noise_test.py`: Test files to generate noise and check their crest factor

Some generate noise signals are included in the `generated` folder.

## Usage
**There is no guarantee that the script will be able to achieve the desired crest factor.** If it doesn't, play with the parameters.

Generate pink noise with a crest factor of 12dB:
```bash
python generate_noise.py pink 12
```

Generate pink noise with a crest factor of 18dB:
```bash
python generate_noise.py pink 18
```

Generate a noise signal with the same spectrum as M-Noise / Music Noise and also 18dB crest factor:
```bash
python3 ./generate_noise.py external 18 --external-wav Music-Noise_96kHz.wav
```
(`Music-Noise_96kHz.wav` is not included for copyright reasons, but you can download it from https://www.aes.org/standards/AES75/)

## Working principle
The script uses the numeric steam hammer, aka gradient-free optimization. First, the desired amplitude spectrum is generated, then the gradient-free optimization changes the phase spectrum to achieve the desired crest factor.

Because COBYLA is by far the best performing algorithm I tested, but limited to 32bit (~35k parameters), the script has a separate internal sample rate and target sample rate.

The internal sample rate is the space over which the optimization is performed. Sadly, when resampling a signal to a higher sample rate, the crest factor will change.

Therefore, during the optimization process, the following steps are performed:
1. Generate the desired amplitude spectrum for internal sample rate
2. Generate an initial phase spectrum
3. Synthesize the noise signal with the current phase spectrum
4. Resample the noise signal to the target sample rate
5. Calculate the crest factor of the resampled signal
6. If the crest factor is not within the desired range, adjust the phase spectrum and repeat

