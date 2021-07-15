from ..protocol_designer import Protocol, Compound_Protocol, Potential
from .potentials import duffing_2D
import numpy as np
import matplotlib.pyplot as plt


# Protocols take an input for time t=(t_i,t_f) and an input list of all parameters initial and final values:
# params = ((p1_i,p1_f),(p2_i,p2_f),(p3_i,p3_f),...)

# define the one protocol
t_lower = (0, .25)
p_lower = ((1, 0), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (-1, -1), (1, 1), (-1, -1), (1, 1))

lower = Protocol(t_lower, p_lower)

#then we can copy it, and change just some params
temp = lower.copy()
temp.time_shift(.25)
temp.change_params((1, 7), ((0, 0), (0, -1)))
tilt = temp.copy()

temp = lower.copy()
temp.time_shift(.5)
temp.reverse()
temp.change_params(7, (-1, -1))
unlower = temp.copy()

temp = tilt.copy()
temp.time_shift(.5)
temp.reverse()
temp.change_params(1, (1, 1))
untilt = temp.copy()

#then combine the substages into a single CompoundProtocol


CP = Compound_Protocol((tilt, lower, untilt, unlower))
