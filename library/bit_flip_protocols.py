from ..protocol_designer import Protocol, Compound_Protocol, Potential, System
from ..protocol_designer.protocol import sequential_protocol

from .potentials import even_1DW, exp_wells_2D, asym_1DW, fredkin_pot
import numpy as np



# 1D bit flip using qudratic/quartic only
p1 = (2, 2, 0, 0, 2, 2)
p2 = (-16, -16, 1, 1, -16, -16)
which = (1, 2)
nontrivial = (p1, p2)
times = (0, .99, 1, 1 + np.pi, 1.01 + np.pi, 2 + np.pi)

bit_flip_1D = sequential_protocol(5, 2, which, nontrivial, times, initial_params=even_1DW.default_params)

bf_1D = System(bit_flip_1D, even_1DW)

trivial_sys = System(even_1DW.trivial_protocol(), even_1DW)

# 1D  asymetric "bit flip" using qudratic/quartic only
p1 = (2, 2, 0, 0, 2, 2)
p2 = (2, 2, 0, 0, 2, 2)
p3 = (-16, -16, 1, 1, -16, -16)
p4 = (-16, -16, .25, .25, -16, -16)
which = (1, 2, 3, 4)
nontrivial = (p1, p2, p3, p4)
times = (0, .99, 1, 1 + np.pi, 1.01 + np.pi, 2 + np.pi)

asym_flip_1D = sequential_protocol(5, 4, which, nontrivial, times, initial_params=even_1DW.default_params)

asym_bf_1D = System(asym_flip_1D, asym_1DW)

# fredkin_gate_flip using 4th order confinement for inf storage
p3 = (0, 0, 1, 1, 0, 0)
p4 = (0, 0, 1, 1, 0, 0)
which = (3, 4)
nontrivial = (p3, p4)
times = (0, .99, 1, 1 + np.pi, 1.01 + np.pi, 2 + np.pi)

fred_gate_prot = sequential_protocol(5, 4, which, nontrivial, times, initial_params=even_1DW.default_params)

fredkin_3D = System(fred_gate_prot, fredkin_pot)


# 2D Exponential translational Flip #
# 1,2,3,4:                                  barrier heights for L0:L1,R0:R1,L0:R0,L1:R1      (0,1)
# 5,6,7,8:                                  well depths for L0,L1,R0,R1,                     (absolute)
# (9,10),(11,12),(13,14),(15,16):           (x,y) coordiantes of the L0,L1,R0,R1 wells       (absolute)

# L0L1 = (1, 1, 1, 1, 1)
# R0R1 = (1, 1, 0, 1, 1)
# L0R0
# L1R1
# L0
# L1
# R0 = (1, 1, 2, 2, 1)
# R1
# L0 x,y
L0_x = (-1, 0, 1)
L0_y = (0, -1, 0)
# R0 x,y
R0_x = (1, 0, -1)
R0_y = (0, 1, 0)
which_p = (9, 10, 13, 14)
ntp = (L0_x, L0_y, R0_x, R0_y)

default_vals = (1, 1, 1, 1, 1, 0, 1, 0, -1, 0, -1, 1, 1, 0, 1, 1)


exp_flip_prot = sequential_protocol(2, 16, which_p, ntp, initial_params=default_vals)

exp_flip_sys = System(exp_flip_prot, exp_wells_2D)
exp_flip_sys.potential.domain = np.array([[-5,-5],[5,5]])
