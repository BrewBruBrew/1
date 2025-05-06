
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, windows
from scipy.fft import fft, fftfreq

# Parameters
lfm_params = {
    "center_freq": 10e6,
    "bandwidth": 5e6,
    "pulse_width": 20e-6,
    "prf": 1000,
    "num_pulses": 4,
    "sample_rate": 100e6,
    "compression_point_dB": 10,
    "target_level_dB": 0
}

T = lfm_params["pulse_width"]
Fs = lfm_params["sample_rate"]
N = int(T * Fs)
t = np.linspace(0, T, N, endpoint=False)
PRI = 1 / lfm_params["prf"]
train_samples = int(PRI * Fs * lfm_params["num_pulses"])
time_train = np.linspace(0, lfm_params["num_pulses"] * PRI, train_samples, endpoint=False)

def generate_lfm_pulse():
    return chirp(t, f0=lfm_params["center_freq"] - lfm_params["bandwidth"] / 2,
                 f1=lfm_params["center_freq"] + lfm_params["bandwidth"] / 2,
                 t1=T, method='linear')

def normalize(envelope):
    return envelope / np.max(np.abs(envelope))

def rectangular_envelope():
    return np.ones(N)

def linear_envelope():
    return np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])

def gaussian_envelope():
    sigma = N / 6
    gauss = np.exp(-0.5 * ((np.arange(N) - N / 2) / sigma) ** 2)
    return normalize(gauss)

def hamming_envelope():
    return normalize(windows.hamming(N))

def hann_envelope():
    return normalize(windows.hann(N))

def blackman_envelope():
    return normalize(windows.blackman(N))

def taylor_envelope(nbar=4, sll=30):
    extended = windows.taylor(N * 3, nbar=nbar, sll=sll)
    center = extended[N:2*N]
    return normalize(center)

def apply_compression(signal, compression_point_dB, target_level_dB):
    rms = np.sqrt(np.mean(np.square(signal)))
    signal_dB = 20 * np.log10(rms)
    headroom = compression_point_dB - (signal_dB - target_level_dB)
    limit = 10 ** ((target_level_dB + headroom) / 20)
    return np.clip(signal, -limit, limit)

base_pulse = generate_lfm_pulse()
envelopes = {
    "Rectangular": rectangular_envelope(),
    "Linear": linear_envelope(),
    "Gaussian": gaussian_envelope(),
    "Hamming": hamming_envelope(),
    "Hann": hann_envelope(),
    "Blackman": blackman_envelope(),
    "Taylor": taylor_envelope()
}

shaped_signals = {name: base_pulse * env for name, env in envelopes.items()}
compressed_signals = {
    name: apply_compression(sig, lfm_params["compression_point_dB"], lfm_params["target_level_dB"])
    for name, sig in shaped_signals.items()
}

def build_train(pulse, num_pulses, train_samples):
    train = np.zeros(train_samples)
    step = int(PRI * Fs)
    for i in range(num_pulses):
        start = i * step
        if start + N <= train_samples:
            train[start:start+N] += pulse
    return train

pulse_trains = {name: build_train(sig, lfm_params["num_pulses"], train_samples) for name, sig in shaped_signals.items()}
compressed_trains = {name: build_train(sig, lfm_params["num_pulses"], train_samples) for name, sig in compressed_signals.items()}

freq = fftfreq(train_samples, 1 / Fs)
mask = freq > 0

fig, axes = plt.subplots(len(shaped_signals), 4, figsize=(20, 3 * len(shaped_signals)))
for idx, name in enumerate(shaped_signals.keys()):
    axes[idx, 0].plot(time_train * 1e6, pulse_trains[name])
    axes[idx, 0].set_title(f"{name} Pulse Train (Time)")
    axes[idx, 0].set_xlabel("Time (µs)")
    axes[idx, 0].set_ylabel("Amplitude")

    axes[idx, 1].plot(freq[mask] / 1e6, np.abs(fft(pulse_trains[name]))[mask])
    axes[idx, 1].set_title(f"{name} Spectrum")
    axes[idx, 1].set_xlabel("Frequency (MHz)")
    axes[idx, 1].set_ylabel("Magnitude")

    axes[idx, 2].plot(time_train * 1e6, compressed_trains[name])
    axes[idx, 2].set_title(f"{name} Compressed Train (Time)")
    axes[idx, 2].set_xlabel("Time (µs)")
    axes[idx, 2].set_ylabel("Amplitude")

    axes[idx, 3].plot(freq[mask] / 1e6, np.abs(fft(compressed_trains[name]))[mask])
    axes[idx, 3].set_title(f"{name} Compressed Spectrum")
    axes[idx, 3].set_xlabel("Frequency (MHz)")
    axes[idx, 3].set_ylabel("Magnitude")

    for j in range(4):
        axes[idx, j].grid(True)

fig.tight_layout()

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

for name in shaped_signals:
    ax1.plot(freq[mask] / 1e6, np.abs(fft(pulse_trains[name]))[mask], label=name)
    ax2.plot(freq[mask] / 1e6, np.abs(fft(compressed_trains[name]))[mask], label=name)

ax1.set_title("Superimposed Uncompressed Spectra")
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Magnitude")
ax1.grid(True)
ax1.legend()

ax2.set_title("Superimposed Compressed Spectra")
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("Magnitude")
ax2.grid(True)
ax2.legend()

fig2.tight_layout()
plt.show()
