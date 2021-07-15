import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from ..bit_flip_protocols import asym_bf_1D

L=50

N=np.linspace(50, 100_000, L).astype(int)
KE_array = np.zeros(L)
PE_array = np.zeros(L)


for i, item in enumerate(N):
    eq_state = asym_bf_1D.eq_state(item)

    KE_array[i] = asym_bf_1D.get_kinetic_energy(eq_state).mean()

    PE_array[i] = asym_bf_1D.get_potential(eq_state, 0).mean()


plt.plot(N, KE_array)
plt.plot(N, PE_array)
plt.legend(['KE','PE'])
plt.show()