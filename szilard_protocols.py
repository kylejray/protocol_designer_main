from protocol_designer.protocol import Protocol, Compound_Protocol, sequential_protocol
from protocol_designer.potentials import Potential, Duffing_2D, BLW, Exp_wells_2D
from protocol_designer.system import System
import numpy as np
import matplotlib.pyplot as plt


# ALECS 12 STEP SZILARD, using linearly coupled 2D duffing:
##########################################################
##########################################################

# we prepare a list of the which parameters will change, and also their values at each substage time
which = (3, 4, 6, 7)

p3 = (-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1)
p4 = (-1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1)
p6 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0)
p7 = (0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0)
non_triv_param = (p3, p4, p6, p7)
# there are 12 steps
NS = 12
# and 7 parameters
NP = 7
# then we create the Compound Protocol (note that default_params is optional, and defaults will be 0 without it)
D2D_szilard_prot = sequential_protocol(NS, NP, which, non_triv_param, initial_params=Duffing_2D.default_params)
D2D_szilard = System(D2D_szilard_prot, Duffing_2D)

##########################################################
##########################################################

# THE BLW version of szilard protocol, 6 steps currently:

##########################################################
##########################################################

R0R1 = (1, 0, 0, 1, 1, 1, 1)
L0L1 = (1, 0, 0, 1, 1, 1, 1)
L1R1 = (1, 1, 1, 1, 0, 0, 1)
L0R0 = (1, 1, 1, 1, 0, 0, 1)
L0 = (0, 0, -1, -1, -1, 0, 0)
# L1 is constant 0
# R0 is constant 0
R1 = (0, 0, -1, -1, -1, 0, 0)

which_p = (1, 2, 3, 4, 5, 8)

ntp = (R0R1, L0L1, L1R1, L0R0, L0, R1)
BLW_szilard_prot = sequential_protocol(6, 12, which_p, ntp, initial_params=BLW.default_params)
BLW_szilard = System(BLW_szilard_prot, BLW)

##########################################################
##########################################################

# an exponential version of szilard protocol

##########################################################
##########################################################

L0L1 = (1, 0, 0, 1, 1, 1, 1)
R0R1 = (1, 0, 0, 1, 1, 1, 1)
L0R0 = (1, 1, 1, 1, 0, 0, 1)
L1R1 = (1, 1, 1, 1, 0, 0, 1)

L0 = np.multiply(-1, (0, 0, -1, -1, -1, 0, 0))+1
# L1 is constant 0
# R0 is constant 0
R1 = np.multiply(-1, (0, 0, -1, -1, -1, 0, 0))+1

which_p = (1, 2, 3, 4, 5, 8)

ntp = (L0L1, R0R1, L0R0, L1R1, L0, R1)
EW2_szilard_prot = sequential_protocol(6, 16, which_p, ntp, initial_params=Exp_wells_2D.default_params)
EW2_szilard = System(EW2_szilard_prot, Exp_wells_2D)

#BLW_szilard.animate_protocol(fps=5, surface=True)
#D2D_szilard.animate_protocol(fps=5, surface=False)
#EW2_szilard.animate_protocol(fps=5, surface=True)
