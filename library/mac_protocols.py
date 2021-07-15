from ..protocol_designer import Protocol, Compound_Protocol, Potential, System 
from ..protocol_designer.protocol import sequential_protocol
from .potentials import duffing_2D, blw, exp_wells_2D
import numpy as np


# MULTIPLY

# 1,2,3,4:                                  barrier heights for L0:L1,R0:R1,L0:R0,L1:R1      (0,1)
# 5,6,7,8:                                  well depths for L0,L1,R0,R1,                     (absolute)
# (9,10),(11,12),(13,14),(15,16):           (x,y) coordiantes of the L0,L1,R0,R1 wells       (absolute)

# # MULTIPLY # #

L0L1 = (1, 0, 0, 1, 1)
# R0R1 = (1, 1, 1, 1, 1)
L0R0 = (1, 0, 0, 1, 1)
# L1R1 = (1, 1, 1, 1, 1)
L0 = (1, 1, 2, 2, 1)
# L1,R0,R1 are constant 1

which_p = (1, 3, 5)

ntp = (L0L1, L0R0, L0)

exp_mult_prot = sequential_protocol(4, 16, which_p, ntp, initial_params=exp_wells_2D.default_params)
exp_mult = System(exp_mult_prot, exp_wells_2D)

# ADD
# 1,2,3,4:                                  barrier heights for L0:L1,R0:R1,L0:R0,L1:R1      (0,1)
# 5,6,7,8:                                  well depths for L0,L1,R0,R1,                     (absolute)
# (9,10),(11,12),(13,14),(15,16):           (x,y) coordiantes of the L0,L1,R0,R1 wells       (absolute)

# Flip #

# L0L1 = (1, 1, 1, 1, 1)
# R0R1 = (1, 1, 0, 1, 1)
# L0R0
# L1R1
# L0
# L1
# R0 = (1, 1, 2, 2, 1)
# R1
# L0 x,y
L1_x = (-1, 0, 1)
L1_y = (1, .5, 1)
# R0 x,y
R1_x = (1, 0, -1)
R1_y = (1, 2, 1)
which_p = (11, 12, 15, 16)
ntp = (L1_x, L1_y, R1_x, R1_y)

exp_flip_prot = sequential_protocol(2, 16, which_p, ntp, initial_params=exp_wells_2D.default_params)

# Erase #

# L0L1 = (1, 1, 1, 1, 1)
R0R1 = (1, 0, 0, 1, 1)
# L0R0
# L1R1
# L0
# L1
R0 = (1, 1, 2, 2, 1)
# R1


which_p = (2, 7)

ntp = (R0R1, R0)

exp_R_erase_prot = sequential_protocol(4, 16, which_p, ntp, initial_params=exp_wells_2D.default_params)
exp_R_erase_prot.time_shift(1)

p_rots = []
for item in exp_flip_prot.protocols:
    p_rots.append(item)

for item in exp_R_erase_prot.protocols:
    p_rots.append(item)


exp_add_prot = Compound_Protocol(p_rots)

exp_add = System(exp_add_prot, exp_wells_2D)
