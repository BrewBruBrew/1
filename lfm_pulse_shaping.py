
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, windows
from scipy.fft import fft, fftfreq

# Parameters
lfm_params = {
    "center_freq": 10e6,        # Hz
    "bandwidth": 5e6,           # Hz
    "pulse_width": 20e-6,       # seconds
    "prf": 1000,                # Hz
    "num_pulses": 8,
    "sample_rate": 100e6,       # Hz
    "ramp_in": 0.1,             # Fraction of pulse width
    "ramp_out": 0.1,            # Fraction of pulse width
    "compression_point_dB": 10, # dB above target signal
    "target_level_dB": 0        # dBFS
}

# Derived parameters
T = lfm_params["pulse_width"]
Fs = lfm_params["sample_rate"]
N = int(T * Fs)
t = np.linspace(0, T, N, endpoint=False)
ramp_in_samples = int(N * lfm_params["ramp_in"])
ramp_out_samples = int(N * lfm_params["ramp_out"])
flat_samples = N - ramp_in_samples - ramp_out_samples

# Base LFM signal
def generate_lfm_pulse():
    return chirp(t, f0=lfm_params["center_freq"] - lfm_params["bandwidth"] / 2,
                 f1=lfm_params["center_freq"] + lfm_params["bandwidth"] / 2,
                 t1=T, method='linear')

# Envelope functions
def linear_envelope():
    ramp_in = np.linspace(0, 1, ramp_in_samples, endpoint=False)
    flat = np.ones(flat_samples)
    ramp_out = np.linspace(1, 0, ramp_out_samples, endpoint=False)
    return np.concatenate([ramp_in, flat, ramp_out])

def gaussian_envelope():
    sigma = ramp_in_samples / 2
    gauss = np.exp(-0.5 * ((np.arange(N) - N / 2) / sigma) ** 2)
    norm = np.max(gauss)
    return gauss / norm

def hamming_envelope():
    return windows.hamming(N)

def blackman_envelope():
    return windows.blackman(N)

def taylor_envelope(nbar=4, sll=30):
    return windows.taylor(N, nbar=nbar, sll=sll)

# Apply compression (simulate soft clipping)
def apply_compression(signal, compression_point_dB, target_level_dB):
    rms = np.sqrt(np.mean(np.square(signal)))
    signal_dB = 20 * np.log10(rms)
    headroom = compression_point_dB - (signal_dB - target_level_dB)
    limit = 10 ** ((target_level_dB + headroom) / 20)
    compressed = np.clip(signal, -limit, limit)
    return compressed

# Generate base pulse
base_pulse = generate_lfm_pulse()

# Generate shaped pulses
shaped_signals = {
    "Rectangular": base_pulse,
    "Linear": base_pulse * linear_envelope(),
    "Gaussian": base_pulse * gaussian_envelope(),
    "Hamming": base_pulse * hamming_envelope(),
    "Blackman": base_pulse * blackman_envelope(),
    "Taylor": base_pulse * taylor_envelope()
}

# Compressed signals
compressed_signals = {
    name: apply_compression(sig, lfm_params["compression_point_dB"], lfm_params["target_level_dB"])
    for name, sig in shaped_signals.items()
}

# Frequency axis
freq = fftfreq(N, 1 / Fs)
mask = freq > 0

# Plotting
fig, axes = plt.subplots(len(shaped_signals), 4, figsize=(20, 3 * len(shaped_signals)))
for idx, (name, sig) in enumerate(shaped_signals.items()):
    comp_sig = compressed_signals[name]
    
    # Time-domain original
    axes[idx, 0].plot(t * 1e6, np.real(sig))
    axes[idx, 0].set_title(f"{name} (Time Domain)")
    axes[idx, 0].set_xlabel("Time (µs)")
    axes[idx, 0].set_ylabel("Amplitude")
    
    # Spectrum original
    axes[idx, 1].plot(freq[mask] / 1e6, np.abs(fft(sig))[mask])
    axes[idx, 1].set_title(f"{name} (Spectrum)")
    axes[idx, 1].set_xlabel("Frequency (MHz)")
    axes[idx, 1].set_ylabel("Magnitude")
    
    # Time-domain compressed
    axes[idx, 2].plot(t * 1e6, np.real(comp_sig))
    axes[idx, 2].set_title(f"{name} Compressed (Time Domain)")
    axes[idx, 2].set_xlabel("Time (µs)")
    axes[idx, 2].set_ylabel("Amplitude")
    
    # Spectrum compressed
    axes[idx, 3].plot(freq[mask] / 1e6, np.abs(fft(comp_sig))[mask])
    axes[idx, 3].set_title(f"{name} Compressed (Spectrum)")
    axes[idx, 3].set_xlabel("Frequency (MHz)")
    axes[idx, 3].set_ylabel("Magnitude")

for ax in axes.flat:
    ax.grid(True)

fig.tight_layout()
plt.show()
