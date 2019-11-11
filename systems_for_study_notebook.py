# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### This notebook is to showcase the finished systems that are ready for simulation and study
# #### *finished systems already include a protocol, potential, force, etc... (see tutorial for details)

# +
import numpy as np 

import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from IPython.display import HTML

#dummy coordinates for later
N=8
coords_2D=2*np.random.random_sample((N,2,2))-1
coords_1D=2*np.random.random_sample((N,1,2))-1
coords_5D=2*np.random.random_sample((N,5,2))-1

# +
#premade systems for the szilard protocol #
import szilard_protocols as zp

#Alec version one with linear coupling
Duff2D_szilard= zp.D2D_szilard
#higher couplings, of the form x^2 y and y^2 x
BLW_szilard= zp.BLW_szilard
#exponential potential allowing for precise localization
Exp_szilard=zp.EW2_szilard



# +

Duff2D_szilard.show_potential(.2)
Duff2D_szilard.scale_potential(10)

BLW_szilard.show_potential(.2)

Exp_szilard.show_potential(.2)
# -

ani = BLW_szilard.animate_protocol()
HTML(ani.to_jshtml(fps=5))

ani = Duff2D_szilard.animate_protocol(n_contours=100)
HTML(ani.to_jshtml(fps=5))

ani = Exp_szilard.animate_protocol(frames=50, surface=False)
HTML(ani.to_jshtml(fps=5))

Exp_szilard.Protocol.show_params()




