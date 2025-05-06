
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
    "num_pulses": 4,
    "sample_rate": 100e6,       # Hz
    "compression_point_dB": 10, # dB above target signal
    "target_level_dB": 0        # dBFS
}

# Derived parameters
T = lfm_params["pulse_width"]
Fs = lfm_params["sample_rate"]
N = int(T * Fs)
t = np.linspace(0, T, N, endpoint=False)

# Base LFM signal
def generate_lfm_pulse():
    return chirp(t, f0=lfm_params["center_freq"] - lfm_params["bandwidth"] / 2,
                 f1=lfm_params["center_freq"] + lfm_params["bandwidth"] / 2,
                 t1=T, method='linear')

# Envelope functions (full duration shaping)
def rectangular_envelope():
    return np.ones(N)

def linear_envelope():
    return np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])

def gaussian_envelope():
    sigma = N / 6
    gauss = np.exp(-0.5 * ((np.arange(N) - N / 2) / sigma) ** 2)
    return gauss / np.max(gauss)

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
    return np.clip(signal, -limit, limit)

# Generate shaped pulses
base_pulse = generate_lfm_pulse()
envelopes = {
    "Rectangular": rectangular_envelope(),
    "Linear": linear_envelope(),
    "Gaussian": gaussian_envelope(),
    "Hamming": hamming_envelope(),
    "Blackman": blackman_envelope(),
    "Taylor": taylor_envelope()
}

shaped_signals = {name: base_pulse * env for name, env in envelopes.items()}
compressed_signals = {
    name: apply_compression(sig, lfm_params["compression_point_dB"], lfm_params["target_level_dB"])
    for name, sig in shaped_signals.items()
}

# Generate pulse trains
PRI = 1 / lfm_params["prf"]
total_time = lfm_params["num_pulses"] * PRI
train_samples = int(total_time * Fs)
time_train = np.linspace(0, total_time, train_samples, endpoint=False)

def build_train(pulse, num_pulses):
    train = np.zeros(train_samples)
    step = int(PRI * Fs)
    for i in range(num_pulses):
        start = i * step
        if start + N <= train_samples:
            train[start:start+N] += pulse
    return train

pulse_trains = {name: build_train(sig, lfm_params["num_pulses"]) for name, sig in shaped_signals.items()}
compressed_trains = {name: build_train(sig, lfm_params["num_pulses"]) for name, sig in compressed_signals.items()}

# Frequency axis
freq_train = fftfreq(train_samples, 1 / Fs)
mask_train = freq_train > 0

# Plot full comparison
fig, axes = plt.subplots(len(shaped_signals), 4, figsize=(20, 3 * len(shaped_signals)))
for idx, name in enumerate(shaped_signals.keys()):
    axes[idx, 0].plot(t * 1e6, shaped_signals[name])
    axes[idx, 0].set_title(f"{name} Shaped Pulse (Time)")
    
    axes[idx, 1].plot(freq_train[mask_train] / 1e6, np.abs(fft(shaped_signals[name], train_samples))[mask_train])
    axes[idx, 1].set_title(f"{name} Spectrum")
    
    axes[idx, 2].plot(time_train * 1e6, compressed_trains[name])
    axes[idx, 2].set_title(f"{name} Compressed Train (Time)")
    
    axes[idx, 3].plot(freq_train[mask_train] / 1e6, np.abs(fft(compressed_trains[name]))[mask_train])
    axes[idx, 3].set_title(f"{name} Compressed Spectrum")

    for j in range(4):
        axes[idx, j].grid(True)

fig.tight_layout()

# Superimposed spectrum plot
plt.figure(figsize=(10, 6))
for name, sig in shaped_signals.items():
    spectrum = np.abs(fft(sig, train_samples))[mask_train]
    plt.plot(freq_train[mask_train] / 1e6, spectrum, label=name)

plt.title("Superimposed Frequency Spectra of Shaped Pulses")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
