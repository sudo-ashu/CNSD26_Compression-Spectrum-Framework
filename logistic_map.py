import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import compression_spec_scale_info as comp_spec
import trajectory as traj

# ==========================
# Parameters
# ==========================

L = 14000
a = 3.6                     # Logistic map parameter
num_symbols = 8             # NumBins
LEN_TO_TAKE = 10000

# ==========================
# Generating Logistic Trajectory
# ==========================

ini_val = np.random.rand()
X, LE = traj.get_trajectory_logistic( initial_value=ini_val,  length=L, a=a)

# ==========================
# Compression Spectrum
# ========================== 
comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(X[:LEN_TO_TAKE], num_symbols)


# ==========================
# Peaks
# ==========================

peaks, _ = find_peaks(comp_ratio)
num_peaks = len(peaks)
num_nonzero = np.count_nonzero(comp_ratio)

print(f"Lyapunov Exponent : {LE:.4f}")
print(f"Number of Peaks   : {num_peaks}")
print(f"Bandwidth         : {num_nonzero}")

print("Length of comp_ratio:", len(comp_ratio))
print("Length of scale_obt:", len(scale))
print("N:", N)
# ==========================
# Compression Spectrum
# ==========================

plt.figure(figsize=(6,4))
scale = np.arange(2, 21)
plt.stem( scale, comp_ratio[1:20], basefmt="r-")
plt.xlabel("Scale")
plt.ylabel("Log Compression Ratio")
plt.title("Compression Spectrum")
plt.grid(True)
plt.tight_layout()

# ==========================
# Scale vs Iteration
# ==========================

plt.figure(figsize=(6,4))
plt.step( np.arange(len(scale[num_symbols:])), scale[num_symbols:], where="post")
plt.xlabel("Iteration")
plt.ylabel("Scale")
plt.title("Scale vs Iteration")
plt.grid(True)
plt.tight_layout()

# ==========================
# Entropy vs Scale
# ==========================

plt.figure(figsize=(6,4))
plt.plot( Ent, linewidth=2)
plt.xlabel("Scale")
plt.ylabel("Entropy of Scale Formation")
plt.title("Entropy vs Scale")
plt.grid(True)
plt.tight_layout()

# ==========================
# Return Map (optional)
# ==========================

# plt.figure(figsize=(5,5))
# plt.scatter(
#     X[:-1],
#     X[1:],
#     s=5
# )
#
# plt.xlabel(r"$x_n$")
# plt.ylabel(r"$x_{n+1}$")
# plt.title("Logistic Map Return Plot")
# plt.grid(True)

plt.show()