
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, windows, firwin, lfilter
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
    "target_level_dB": 0,
    "lpf_taps": 128,
    "lpf_window": "hamming"
}

# Derived parameters
T = lfm_params["pulse_width"]
Fs = lfm_params["sample_rate"]
N = int(T * Fs)
t = np.linspace(0, T, N, endpoint=False)
PRI = 1 / lfm_params["prf"]
train_samples = int(PRI * Fs * lfm_params["num_pulses"])
time_train = np.linspace(0, lfm_params["num_pulses"] * PRI, train_samples, endpoint=False)

# Pulse generation
def generate_lfm_pulse():
    return chirp(t, f0=lfm_params["center_freq"] - lfm_params["bandwidth"] / 2,
                 f1=lfm_params["center_freq"] + lfm_params["bandwidth"] / 2,
                 t1=T, method='linear')

# Amplitude shaping envelopes
def rectangular(): return np.ones(N)
def linear(): return np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
def gaussian(): return np.exp(-0.5 * ((np.arange(N) - N / 2) / (N / 6)) ** 2)
def hamming(): return windows.hamming(N)
def blackman(): return windows.blackman(N)
def taylor(): return windows.taylor(N, nbar=4, sll=30)

# Frequency domain LPF shaping
def apply_spectral_shaping(signal):
    nyq = Fs / 2
    cutoff = lfm_params["bandwidth"] / 2
    taps = firwin(
        lfm_params["lpf_taps"],
        cutoff=cutoff / nyq,
        window=lfm_params["lpf_window"]
    )
    return lfilter(taps, 1.0, signal)

# Compression
def apply_compression(signal, compression_point_dB, target_level_dB):
    rms = np.sqrt(np.mean(signal**2))
    signal_dB = 20 * np.log10(rms)
    headroom = compression_point_dB - (signal_dB - target_level_dB)
    limit = 10 ** ((target_level_dB + headroom) / 20)
    return np.clip(signal, -limit, limit)

# Pulse train builder
def build_train(pulse, num_pulses, total_samples):
    train = np.zeros(total_samples)
    step = int(PRI * Fs)
    for i in range(num_pulses):
        idx = i * step
        if idx + N <= total_samples:
            train[idx:idx+N] += pulse
    return train

# Define envelope dictionary
envelopes = {
    "Rectangular": rectangular(),
    "Linear": linear(),
    "Gaussian": gaussian(),
    "Hamming": hamming(),
    "Blackman": blackman(),
    "Taylor": taylor()
}

# Generate base LFM
lfm = generate_lfm_pulse()
shaped_signals = {k: lfm * v for k, v in envelopes.items()}
shaped_signals["LPF_Shaped"] = apply_spectral_shaping(lfm)

# Apply compression
compressed_signals = {
    k: apply_compression(v, lfm_params["compression_point_dB"], lfm_params["target_level_dB"])
    for k, v in shaped_signals.items()
}

# Build pulse trains
pulse_trains = {k: build_train(v, lfm_params["num_pulses"], train_samples) for k, v in shaped_signals.items()}
compressed_trains = {k: build_train(v, lfm_params["num_pulses"], train_samples) for k, v in compressed_signals.items()}

# Frequency setup
freq = fftfreq(train_samples, 1 / Fs)
mask = freq > 0
def to_db(x): return 20 * np.log10(np.abs(x) + 1e-12)

# Plot full comparison
fig, axes = plt.subplots(len(pulse_trains), 4, figsize=(20, 3 * len(pulse_trains)))
for idx, name in enumerate(pulse_trains.keys()):
    axes[idx, 0].plot(time_train * 1e6, pulse_trains[name])
    axes[idx, 0].set_title(f"{name} Pulse Train (Time)")
    axes[idx, 0].set_xlabel("Time (µs)")
    axes[idx, 0].set_ylabel("Amplitude")

    axes[idx, 1].plot(freq[mask] / 1e6, to_db(fft(pulse_trains[name])[mask]))
    axes[idx, 1].set_title(f"{name} Spectrum (dB)")
    axes[idx, 1].set_xlabel("Frequency (MHz)")
    axes[idx, 1].set_ylabel("Magnitude (dB)")

    axes[idx, 2].plot(time_train * 1e6, compressed_trains[name])
    axes[idx, 2].set_title(f"{name} Compressed Train (Time)")
    axes[idx, 2].set_xlabel("Time (µs)")
    axes[idx, 2].set_ylabel("Amplitude")

    axes[idx, 3].plot(freq[mask] / 1e6, to_db(fft(compressed_trains[name])[mask]))
    axes[idx, 3].set_title(f"{name} Compressed Spectrum (dB)")
    axes[idx, 3].set_xlabel("Frequency (MHz)")
    axes[idx, 3].set_ylabel("Magnitude (dB)")

    for ax in axes[idx]: ax.grid(True)

fig.tight_layout()

# Superimposed Spectra Plot
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
for name in pulse_trains:
    ax1.plot(freq[mask] / 1e6, to_db(fft(pulse_trains[name])[mask]), label=name)
    ax2.plot(freq[mask] / 1e6, to_db(fft(compressed_trains[name])[mask]), label=name)

ax1.set_title("Superimposed Uncompressed Spectra (dB)")
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.grid(True)
ax1.legend()

ax2.set_title("Superimposed Compressed Spectra (dB)")
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("Magnitude (dB)")
ax2.grid(True)
ax2.legend()

fig2.tight_layout()
plt.show()
