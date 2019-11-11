from protocol_designer.protocol import Protocol, Compound_Protocol
from protocol_designer.potentials import Potential, Duffing_2D
import numpy as np
import matplotlib.pyplot as plt


t_lower = (0, .25)
p_lower = ((1, 0), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (-1, -1), (1, 1), (-1, -1), (1, 1))

lower = Protocol(t_lower, p_lower)

temp = lower.copy()
temp.time_shift(.25)
temp.change_param((1, 7), ((0, 0), (0, -1)))
tilt = temp.copy()

temp = lower.copy()
temp.time_shift(.5)
temp.reverse()
temp.change_param(7, (-1, -1))
unlower = temp.copy()

temp = tilt.copy()
temp.time_shift(.5)
temp.reverse()
temp.change_param(1, (1, 1))
untilt = temp.copy()

# ####end test values
# define the four substage Protocols. Protocols take an input for time t=(t_i,t_f)
# and an input list of all parameters initial and final values: params =(p1_i,p1_f),(p2_i,p2_f),(p3_i,p3_f),...
CP = Compound_Protocol((tilt, lower, untilt, unlower))
# ind = (1, 7, 10)
# CP.show_params()

# plt.show()
