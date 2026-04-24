import numpy as np
import matplotlib.pyplot as plt
import compression_spec_scale_info as comp_spec

Fs = 1000
T = 1 / Fs
# signal length
L = 10000 
# time vector
t = np.arange(0, L) * T 

# required signal - sinusoid or random
# S = np.sin(2*np.pi*50*t)
# X = S
X = np.random.rand(len(t))

plt.plot(1000*t[:50], X[:50])
plt.title("Signal")
plt.xlabel("t (milliseconds)")
plt.ylabel("X(t)")

Y = np.fft.fft(X)
P2 = np.abs(Y / L)
P1 = P2[:L//2 + 1]
P1[1:-1] = 2 * P1[1:-1]
f = Fs * np.arange(0, L//2 + 1) / L

plt.figure()
plt.plot(f, P1)
plt.title("Single-Sided Amplitude Spectrum of X(t)")
plt.xlabel("f (Hz)")
plt.ylabel("|P1(f)|")

# ETC spectrum evaluation
NumBins_array = 8
thresh = 2000
LEN_to_take = L

comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(X[:LEN_to_take], NumBins_array)

samp_time = 1 / Fs
samp = np.arange(1, L//2 + 1)

f_ETC = (samp * samp_time) ** (-1)

scale1 = np.arange(1, len(comp_ratio)+1)
plt.figure()
plt.stem(scale1, comp_ratio)
plt.ylabel('Log of Comp. ratio')
plt.xlabel('Scale')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# For plot with frequency on x axis
n = min(len(f_ETC), len(comp_ratio))
plt.figure()
plt.stem(np.log(f_ETC[:n]), comp_ratio[:n])
plt.ylabel('Log of Comp. ratio')
plt.xlabel('log(Frequency)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

non_zero_vals = np.count_nonzero(comp_ratio)

ETC_val = N

plt.figure()
plt.step(np.arange(len(scale[NumBins_array:])), scale[NumBins_array:], where='mid', linewidth=1.2)
plt.ylabel('Scale')
plt.xlabel('Iteration no.')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.figure()
plt.plot(Ent, linewidth=1.5)
plt.xlabel('Scale')
plt.ylabel('Entropy of scale formation')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()