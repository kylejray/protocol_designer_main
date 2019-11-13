import numpy as np
from protocol_designer.protocol import Protocol


class Potential:

    def __init__(self, potential, external_force, N_params, N_dim, default_params=None):
        self.potential = potential
        self.external_force = external_force
        self.N_params = N_params
        self.N_dim = N_dim
        self.default_params = default_params

    def trivial_protocol(self, t_i=0, t_f=1):
        t = (t_i, t_f)
        if self.default_params is not None:
            assert len(default_params) == self.N_params, "number of default parameters doesnt match potential"
            params = []
            for i in range(self.N_params):
                params.append((default_params[i], default_params[i]))
        if self.default_params is None:
            params = np.ones((self.N_params, 2))

        return Protocol(t, params)

    def info(self):
        print("This potential has {} parameters and {} dimensions".format(self.N_params, self.N_dim))

# A simple 1D potential, for testing one dimensional systems
# its just an absolute value. parameters are:
# 1: the slope
# 2: zero point


def one_D_V(x, params):
    slope, x_0 = params
    return(slope*abs(x-x_0))


def one_D_V_force(x, params):
    slope, x_0 = params
    return(slope*np.sign(x-x_0))


odv = Potential(one_D_V, one_D_V_force, 2, 1)

# the coupled duffing potential:
# defautls are set so that it is 4 equal wells


def coupled_duffing_2D(x, y, params):
    a_x, a_y, b_x, b_y, c_x, c_y, d = params
    return(a_x*x**4 + a_y*y**4 + b_x*x**2 + b_y*y**2 + c_x*x + c_y*y + d*x*y)


def coupled_duffing_2D_force(x, y, params):
    a_x, a_y, b_x, b_y, c_x, c_y, d = params
    dx = 4*a_x*x**3+2*b_x*x+c_x+d*y
    dy = 4*a_y*y**3+2*b_y*y+c_y+d*x
    return(-dx, -dy)


duffing_2D = Potential(coupled_duffing_2D, coupled_duffing_2D_force, 7, 2, default_params=(1., 1., -1., -1., 0, 0, 0))

# Next we have a more complicated potential, that uses higher the next order coupling xy^2 and yx^2:
# BLW stands for barriers lifts wells,

# note that these potentials involve a more complex conversion betweem parameters and the values that plug into the function,
# this is manifest in the option scaled_params=True. When absolute params is set to true the function takes the protocol
# parameters as 1 to 1 inputs. When it is set to false, the parameters try to act in a more functional way:
# for example, setting parameter #1 (a) to 0 should mean lowering the R0:R1 barrier to its minimum,
# setting it to 1 will raise the barrier to its "maximum". The ranges in the parenthesis are the ranges scaled_params=True
# are meant to deal with. (absolute) means the parameter is taken in unscaled and unshifted.

# parameters are:

# 1,2,3,4:  barrier b/w R0:R1, L0:L1, L1:R1, L0:R0      (0,1)
# 5,6,7,8:  lifts and lowers L0,L1,R0,R1                (-1,1)
# 9,10:     x coord for L,R wells                       (absolute)
# 11,12:    y coord for 0,1 wells                       (absolute)


def blw_potential(x, y, params, scaled_params=True):
    L = 2
    WD = 1
    B = .5
    if scaled_params:
        B_scale = .7
        L_scale = .5
        B_shift = -.2
        L_shift = 0
        B = B_scale+B_shift
        scale_vector = (B_scale, B_scale, B_scale, B_scale, L_scale, L_scale, L_scale, L_scale, 1, 1, 1, 1)
        shift_vector = (B_shift, B_shift, B_shift, B_shift, L_shift, L_shift, L_shift, L_shift, 0, 0, 0, 0)
        params = np.multiply(scale_vector, params) + shift_vector

    a, b, c, d, L0, L1, R0, R1, x1, x2, y1, y2 = params

    barriers = a*(x+L)*(y+L)*(L-y)+L*(B-a)*x+b*(L-x)*(y+L)*(L-y)-L*(B-b)*x+c*(x+L)*(L-x)*(y+L)+L*(B-c)*y+d*(x+L)*(L-x)*(L-y)-L*(B-d)*y
    lifts = L0*(L-x)*(L-y)+L1*(L-x)*(y+L)+R0*(x+L)*(L-y)+R1*(x+L)*(y+L)
    wells = (x-x1)**2*(x-x2)**2 + (y-y1)**2*(y-y2)**2
    return (barriers+lifts+WD*wells)


def blw_potential_force(x, y, params, scaled_params=True):
    WD = 1
    L = 2
    B = 2
    if scaled_params:
        B_scale = .7
        L_scale = .5
        B_shift = -.2
        L_shift = 0
        B = B_scale+B_shift
        scale_vector = (B_scale, B_scale, B_scale, B_scale, L_scale, L_scale, L_scale, L_scale, 1, 1, 1, 1)
        shift_vector = (B_shift, B_shift, B_shift, B_shift, L_shift, L_shift, L_shift, L_shift, 0, 0, 0, 0)
        params = np.multiply(scale_vector, params) + shift_vector

    a, b, c, d, L0, L1, R0, R1, x1, x2, y1, y2 = params

    dx_barriers = a*(y+L)*(L-y)-b*(y+L)*(L-y)-c*2*x*(y+L)-d*2*x*(L-y)+L*(B-a)-L*(B-b)
    dx_lifts = -L0*(L-y)-L1*(y+L)+R0*(L-y)+R1*(y+L)
    dx_wells = 2*(x-x1)*(x-x2)*(2*x-x1-x2)
    f_x = -(dx_barriers+dx_lifts+WD*dx_wells)

    dy_barriers = -a*(x+L)*2*y-b*(L-x)*2*y+c*(x+L)*(L-x)-d*(x+L)*(L-x)+L*(B-c)-L*(B-d)
    dy_lifts = -L0*(L-x)+L1*(L-x)-R0*(x+L)+R1*(x+L)
    dy_wells = 2*(y-y1)*(y-y2)*(2*y-y1-y2)
    f_y = -(dy_barriers+dy_lifts+WD*dy_wells)

    return(f_x, f_y)


blw = Potential(blw_potential, blw_potential_force, 12, 2, default_params=(1, 1, 1, 1, 0, 0, 0, 0, -1, 1, -1, 1))


# Next we have exponential wells, in order to test really well localized wells.
# They are really easy to program, but perhaps not as physical?


# parameters are:

# 1,2,3,4:                                  barrier heights for L0:L1,R0:R1,L0:R0,L1:R1      (0,1)
# 5,6,7,8:                                  well depths for L0,L1,R0,R1,                     (absolute)
# (9,10),(11,12),(13,14),(15,16):           (x,y) coordiantes of the L0,L1,R0,R1 wells       (absolute)

# first we define some helper functions:
def exp_well(x, y, Depth, x_loc, y_loc, x0, y0):
    return -Depth*np.exp(-(x_loc*(x-x0)**2+y_loc*(y-y0)**2))


def exp_well_derivs(x, y, Depth, x_loc, y_loc, x0, y0):
    dx = -2*x_loc*(x-x0)*exp_well(x, y, Depth, x_loc, y_loc, x0, y0)
    dy = -2*y_loc*(y-y0)*exp_well(x, y, Depth, x_loc, y_loc, x0, y0)
    return(dx, dy)


def exp_potential(x, y, params, scaled_params=True):

    if scaled_params:
        B_scale = 5.5
        L_scale = 10
        B_shift = -.5
        L_shift = 0
        B = B_scale+B_shift

        scale_vector = (B_scale, B_scale, B_scale, B_scale, L_scale, L_scale, L_scale, L_scale, 1, 1, 1, 1, 1, 1, 1, 1)
        shift_vector = (B_shift, B_shift, B_shift, B_shift, L_shift, L_shift, L_shift, L_shift, 0, 0, 0, 0, 0, 0, 0, 0)
        params = np.multiply(scale_vector, params) + shift_vector

    L0L1, R0R1, L0R0, L1R1, L0, L1, R0, R1, xL0, yL0, xL1, yL1, xR0, yR0, xR1, yR1 = params

    WL0 = exp_well(x, y, L0, 1+L0R0, 1+L0L1, xL0, yL0)
    WL1 = exp_well(x, y, L1, 1+L1R1, 1+L0L1, xL1, yL1)
    WR0 = exp_well(x, y, R0, 1+L0R0, 1+R0R1, xR0, yR0)
    WR1 = exp_well(x, y, R1, 1+L1R1, 1+R0R1, xR1, yR1)
    s = .1
    stability = s*(x**2+y**2)
    return (WL0+WL1+WR0+WR1+stability)


def exp_potential_force(x, y, params, scaled_params=True):

    if scaled_params:
        B_scale = 5.5
        L_scale = 10
        B_shift = -.5
        L_shift = 0
        B = B_scale+B_shift

        scale_vector = (B_scale, B_scale, B_scale, B_scale, L_scale, L_scale, L_scale, L_scale, 1, 1, 1, 1, 1, 1, 1, 1)
        shift_vector = (B_shift, B_shift, B_shift, B_shift, L_shift, L_shift, L_shift, L_shift, 0, 0, 0, 0, 0, 0, 0, 0)
        params = np.multiply(scale_vector, params) + shift_vector

    L0L1, R0R1, L0R0, L1R1, L0, L1, R0, R1, xL0, yL0, xL1, yL1, xR0, yR0, xR1, yR1 = params

    WL0_dx, WL0_dy = exp_well_derivs(x, y, L0, 1+L0R0, 1+L0L1, xL0, yL0)
    WL1_dx, WL1_dy = exp_well_derivs(x, y, L1, 1+L1R1, 1+L0L1, xL1, yL1)
    WR0_dx, WR0_dy = exp_well_derivs(x, y, R0, 1+L0R0, 1+R0R1, xR0, yR0)
    WR1_dx, WR1_dy = exp_well_derivs(x, y, R1, 1+L1R1, 1+R0R1, xR1, yR1)
    s = .1
    xs = 2*s*x
    ys = 2*s*y
    fx, fy = -(WL0_dx+WL1_dx+WR0_dx+WR1_dx+xs), -(WL0_dy+WL1_dy+WR0_dy+WR1_dy+ys)

    return (fx, fy)


exp_defaults = (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1)
exp_wells_2D = Potential(exp_potential, exp_potential_force, 16, 2, default_params=exp_defaults)
