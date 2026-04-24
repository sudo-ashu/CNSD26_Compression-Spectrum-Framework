import numpy as np
import matplotlib.pyplot as plt


def plot_all(comp_ratio, scale, Ent):
    # Compression spectrum
    plt.figure(figsize=(8,4))
    plt.stem(range(1, len(comp_ratio)+1), comp_ratio)
    plt.xlabel("Scale")
    plt.ylabel("Log Compression Ratio")
    plt.title("Compression Spectrum")
    plt.grid()

    # Scale evolution
    plt.figure(figsize=(8,4))
    plt.step(range(len(scale)), scale)
    plt.xlabel("Iteration")
    plt.ylabel("Scale")
    plt.title("Scale Evolution")
    plt.grid()

    # Entropy of scale formation
    plt.figure(figsize=(8,4))
    plt.plot(Ent)
    plt.xlabel("Scale")
    plt.ylabel("Entropy")
    plt.title("Entropy of Scale Formation")
    plt.grid()

    plt.show()

def plot_compression_vs_scale(comp_ratio):
    scales = np.arange(1, len(comp_ratio) + 1)
    
    plt.figure(figsize=(8,4))
    plt.stem(scales, comp_ratio)
    plt.xlabel("Scale")
    plt.ylabel("Log Compression Ratio")
    plt.title("Compression Spectrum (Log Compression Ratio vs Scale)")
    plt.grid()

def plot_frequency_spectrum(signal, fs):
    from scipy.fft import fft, fftshift
    L = len(signal)
    
    # FFT
    Y = np.fft.fft(signal)
    
    # Normalizing
    Y = Y / L
    
    # Shift zero frequency to center
    Y_shifted = fftshift(Y)
    
    # Frequency axis
    f = np.linspace(-fs/2, fs/2, L)
    
    # Amplitude
    amplitude = np.abs(Y_shifted)
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(f, amplitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Double-Sided Amplitude Spectrum")
    plt.grid()

def plot_entropy_vs_scale(Ent):
    scales = np.arange(1, len(Ent) + 1)
    
    plt.figure(figsize=(8,4))
    plt.plot(scales, Ent, marker='o')
    plt.xlabel("Scale")
    plt.ylabel("Entropy")
    plt.title("Entropy of Scale Formation vs Scale")
    plt.grid()