import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import compression_spec_scale_info as comp_spec

# load 1/f noise
data = loadmat('gen_pink_noise.mat')
invfn = data['invfn'].flatten()

NumBins_array = 8
LEN_to_take = 10000
X = invfn[:LEN_to_take]

comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(X, NumBins_array)

scale1 = np.arange(2, 10)
plt.figure()
plt.stem(scale1, comp_ratio[1:9])
plt.ylabel('Log of Comp. ratio')
plt.xlabel('Scale')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

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