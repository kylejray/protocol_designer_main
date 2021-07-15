import numpy as np

from ..protocol_designer import Potential


def one_D_V(x, params):
    """
    A simple 1D potential, for testing one dimensional systems
    its just an absolute value.

    Parameters
    ----------
    x: the coordinates
    params: (1,2)
        1: the slope
        2: zero point

    Returns
    -------
    the value of the potential at location x with the given params
    """

    slope, x_0 = params
    return slope * abs(x - x_0)


def one_D_V_force(x, params):
    """
    See one_D_V function, it has the same input format.
    """
    slope, x_0 = params
    return slope * np.sign(x - x_0)


# define the potential 2 parameters, 1 dimension, no default values
odv = Potential(one_D_V, one_D_V_force, 2, 1)


def e_well_3D(x, y, z, params):
    """
    A simple 1D potential, for testing one dimensional systems
    its just an absolute value.

    Parameters
    ----------
    x, y, z: the coordinates
    params: (1,2)
        1: the slope
        2: zero point

    Returns
    -------
    the value of the potential at location x with the given params
    """

    a, b, x_0, y_0, z_0 = params
    return -a * np.exp(-b * ((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2))


def e_well_3D_force(x, y, z, params):
    """
    See exp_well_3D function, it has the same input format.
    """
    a, b, x_0, y_0, z_0 = params
    dx = 2 * a * b * (x-x_0) * np.exp(-b * ((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2))
    dy = 2 * a * b * (y-y_0) * np.exp(-b * ((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2))
    dz = 2 * a * b * (z-z_0) * np.exp(-b * ((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2))
    return (-dx, -dy, -dz)


ew_3D = Potential(e_well_3D, e_well_3D_force, 5, 3)


def five_d_test(x1, x2, x3, x4, x5, params):
    slope = params
    return slope * (np.abs(x1) + np.abs(x2) + np.abs(x3) + np.abs(x4) + np.abs(x5))


def five_d_test_force(x1, x2, x3, x4, x5, params):
    slope = params
    dx1 = slope * (np.abs(x2) + np.abs(x3) + np.abs(x4) + np.abs(x5))
    dx2 = slope * (np.abs(x1) + np.abs(x3) + np.abs(x4) + np.abs(x5))
    dx3 = slope * (np.abs(x1) + np.abs(x2) + np.abs(x4) + np.abs(x5))
    dx4 = slope * (np.abs(x1) + np.abs(x2) + np.abs(x3) + np.abs(x5))
    dx5 = slope * (np.abs(x1) + np.abs(x2) + np.abs(x3) + np.abs(x4))
    return (-dx1, -dx2, -dx3, -dx4, -dx5)


fdt = Potential(five_d_test, five_d_test_force, 1, 5)


def coupled_duffing_2D(x, y, params):
    """
    the coupled 2D duffing potential:
    defautls are set so that it is 4 equal wells

    Parameters
    ----------
    x: ndarray of dimension [N,]
        the x coordinates for N positions
    y: ndarray of dimension [N,]
        the y coordinates for N positions
    params: list/tuple (1, 2, 3, 4, 5, 6, 7)
        1, 2 : coefficients of the x^4 and y^4 terms, respectively
        3, 4 : coefficients of the x^2 and y^2 terms, respectively
        5, 6 : coefficients of the x^1 and y^1 terms, respectively
        7: coefficient of the coupling term, x*y

    Returns
    -------
    the value of the potential at locations x,y with the given params
    """

    a_x, a_y, b_x, b_y, c_x, c_y, d = params
    return (
        a_x * x ** 4
        + a_y * y ** 4
        + b_x * x ** 2
        + b_y * y ** 2
        + c_x * x
        + c_y * y
        + d * x * y
    )


def coupled_duffing_2D_force(x, y, params):
    """
    See coupled_duffing_2D function, it has the same input format.
    """
    a_x, a_y, b_x, b_y, c_x, c_y, d = params
    dx = 4 * a_x * x ** 3 + 2 * b_x * x + c_x + d * y
    dy = 4 * a_y * y ** 3 + 2 * b_y * y + c_y + d * x
    return (-dx, -dy)


duffing_2D = Potential(
    coupled_duffing_2D,
    coupled_duffing_2D_force,
    7,
    2,
    default_params=(1.0, 1.0, -1.0, -1.0, 0, 0, 0),
)


# Next we have a more complicated potential, that uses higher the next order coupling xy^2 and yx^2:
# BLW stands for barriers lifts wells,

# note that these potentials involve a more complex conversion betweem parameters and the values that plug into the function,
# this is manifest in the option scaled_params=True. When absolute params is set to true the function takes the protocol
# parameters as 1 to 1 inputs. When it is set to false, the parameters try to act in a more functional way:
# for example, setting parameter #1 (a) to 0 should mean lowering the R0:R1 barrier to its minimum,
# setting it to 1 will raise the barrier to its "maximum". The ranges in the parenthesis are the ranges scaled_params=True
# are meant to deal with. (absolute) means the parameter is taken in unscaled and unshifted.


def blw_potential(x, y, params, scaled_params=True):
    """
    4 wells in 2D, using higher order terms than the duffing: xy^2 and yx^2

    Parameters
    ----------
    x: ndarray of dimension [N,]
        the x coordinates for N positions
    y: ndarray of dimension [N,]
        the y coordinates for N positions
    params: list/tuple (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    1,2,3,4:  barrier b/w R0:R1, L0:L1, L1:R1, L0:R0            (0,1)
    5,6,7,8:  lifts/lowers the L0,L1,R0,R1 wells                (-1,1)
    9,10:     x coord for L,R wells                             (absolute)
    11,12:    y coord for 0,1 wells                             (absolute)

    scaled_parameters: True or False
        if True will scale the parameters to be functionally meaningful
        i.e. setting parameter 1 to 0 means that the barrier is dropped to its lowest value
        and setting it to 1 means that it is at its highers value
        if False, parameters are taken to be the actual inputs without the extra layer or functionalizing them
    Returns
    -------
    the value of the potential at locations x,y with the given params
    """
    L = 2
    WD = 1
    B = 0.5
    if scaled_params:
        B_scale = 0.7
        L_scale = 0.5
        B_shift = -0.2
        L_shift = 0
        B = B_scale + B_shift
        scale_vector = (B_scale, B_scale, B_scale, B_scale, L_scale, L_scale, L_scale, L_scale, 1, 1, 1, 1,)
        shift_vector = (B_shift, B_shift, B_shift, B_shift, L_shift, L_shift, L_shift, L_shift, 0, 0, 0, 0,)
        params = np.multiply(scale_vector, params) + shift_vector

    a, b, c, d, L0, L1, R0, R1, x1, x2, y1, y2 = params

    barriers = (
        a * (x + L) * (y + L) * (L - y)
        + L * (B - a) * x
        + b * (L - x) * (y + L) * (L - y)
        - L * (B - b) * x
        + c * (x + L) * (L - x) * (y + L)
        + L * (B - c) * y
        + d * (x + L) * (L - x) * (L - y)
        - L * (B - d) * y
    )
    lifts = (
        L0 * (L - x) * (L - y)
        + L1 * (L - x) * (y + L)
        + R0 * (x + L) * (L - y)
        + R1 * (x + L) * (y + L)
    )
    wells = (x - x1) ** 2 * (x - x2) ** 2 + (y - y1) ** 2 * (y - y2) ** 2
    return barriers + lifts + WD * wells


def blw_potential_force(x, y, params, scaled_params=True):
    """
    See blw_potential documentation
    """
    WD = 1
    L = 2
    B = 2
    if scaled_params:
        B_scale = 0.7
        L_scale = 0.5
        B_shift = -0.2
        L_shift = 0
        B = B_scale + B_shift
        scale_vector = (B_scale, B_scale, B_scale, B_scale, L_scale, L_scale, L_scale, L_scale, 1, 1, 1, 1,)
        shift_vector = (B_shift, B_shift, B_shift, B_shift, L_shift, L_shift, L_shift, L_shift, 0, 0, 0, 0,)
        params = np.multiply(scale_vector, params) + shift_vector

    a, b, c, d, L0, L1, R0, R1, x1, x2, y1, y2 = params

    dx_barriers = (
        a * (y + L) * (L - y)
        - b * (y + L) * (L - y)
        - c * 2 * x * (y + L)
        - d * 2 * x * (L - y)
        + L * (B - a)
        - L * (B - b)
    )
    dx_lifts = -L0 * (L - y) - L1 * (y + L) + R0 * (L - y) + R1 * (y + L)
    dx_wells = 2 * (x - x1) * (x - x2) * (2 * x - x1 - x2)
    f_x = -(dx_barriers + dx_lifts + WD * dx_wells)

    dy_barriers = (
        -a * (x + L) * 2 * y
        - b * (L - x) * 2 * y
        + c * (x + L) * (L - x)
        - d * (x + L) * (L - x)
        + L * (B - c)
        - L * (B - d)
    )
    dy_lifts = -L0 * (L - x) + L1 * (L - x) - R0 * (x + L) + R1 * (x + L)
    dy_wells = 2 * (y - y1) * (y - y2) * (2 * y - y1 - y2)
    f_y = -(dy_barriers + dy_lifts + WD * dy_wells)

    return (f_x, f_y)


blw = Potential(
    blw_potential,
    blw_potential_force,
    12,
    2,
    default_params=(1, 1, 1, 1, 0, 0, 0, 0, -1, 1, -1, 1),
)


# Next we have exponential wells, in order to test really well localized wells.
# They are really easy to program, but perhaps not as physical?


# parameters are:

# 1,2,3,4:                                  barrier heights for L0:L1,R0:R1,L0:R0,L1:R1      (0,1)
# 5,6,7,8:                                  well depths for L0,L1,R0,R1,                     (absolute)
# (9,10),(11,12),(13,14),(15,16):           (x,y) coordiantes of the L0,L1,R0,R1 wells       (absolute)

# first we define some helper functions:

def exp_well(x, y, Depth, x_loc, y_loc, x0, y0):
    '''
    helper function used in other potentials
    '''
    return -Depth * np.exp(-(x_loc * (x - x0) ** 2 + y_loc * (y - y0) ** 2))


def exp_well_derivs(x, y, Depth, x_loc, y_loc, x0, y0):
    '''
    helper function used in other potentials
    '''
    dx = -2 * x_loc * (x - x0) * exp_well(x, y, Depth, x_loc, y_loc, x0, y0)
    dy = -2 * y_loc * (y - y0) * exp_well(x, y, Depth, x_loc, y_loc, x0, y0)
    return (dx, dy)

def find_localization_scales(well_locations):
    '''
    helper function used in exp_wells potential
    '''
    x_loc = well_locations[::2]
    y_loc = well_locations[1::2]
    B = y_loc[1] - y_loc[0], y_loc[3] - y_loc[2], x_loc[2] - x_loc[0], x_loc[3] - x_loc[1]
    B = 2 / np.array(B)**2
    B_scale = 11*B
    B_shift = B
    return B_scale, B_shift


def exp_potential(x, y, params, scaled_params=True):
    """
    4 wells in 2D, using a exponentially localized wells

    Parameters
    ----------
    x: ndarray of dimension [N,]
        the x coordinates for N positions
    y: ndarray of dimension [N,]
        the y coordinates for N positions
    params: list/tuple (1, 2, 3, ..., 16)

    1,2,3,4: barrier heights b/w L0:L1, R0:R1, L0:R0, L1:R1                         (0,1)
    5,6,7,8: well depths L0,L1,R0,R1 wells                                          (absolute)
    (9,10),(11,12),(13,14),(15,16): (x,y) coordiantes of the L0,L1,R0,R1 wells      (absolute)

    scaled_parameters: True or False
        if True will scale the parameters to be functionally meaningful
        i.e. setting parameter 1 to 0 means that the barrier is dropped to its lowest value
        and setting it to 1 means that it is at its highers value
        if False, parameters are taken to be the actual inputs without the extra layer or functionalizing them
    Returns
    -------
    the value of the potential at locations x,y with the given params
    """
    if scaled_params:
        B_scale, B_shift = find_localization_scales(np.array(params)[8:])
        assert len(B_scale) == 4, 'error in setting the localization scale'
        assert len(B_shift) == 4, 'error in setting the localization scale'
        no_scale = np.ones(12)
        no_shift = np.zeros(12)

        scale_vector = (*B_scale, *no_scale)
        shift_vector = (*B_shift, *no_shift)
        params = np.multiply(scale_vector, params) + shift_vector

    (L0L1, R0R1, L0R0, L1R1, L0, L1, R0, R1, xL0, yL0, xL1, yL1, xR0, yR0, xR1, yR1,) = params

    WL0 = exp_well(x, y, L0, L0R0, L0L1, xL0, yL0)
    WL1 = exp_well(x, y, L1, L1R1, L0L1, xL1, yL1)
    WR0 = exp_well(x, y, R0, L0R0, R0R1, xR0, yR0)
    WR1 = exp_well(x, y, R1, L1R1, R0R1, xR1, yR1)
    
    s = 0.03
    stability = s * (x ** 4 + y ** 4)
    return WL0 + WL1 + WR0 + WR1 + stability


def exp_potential_force(x, y, params, scaled_params=True):
    """
    see exp_potential function docstring
    """

    if scaled_params:
        B_scale, B_shift = find_localization_scales(np.array(params)[8:])
        assert len(B_scale) == 4, 'error in setting the localization scale'
        assert len(B_shift) == 4, 'error in setting the localization scale'
        no_scale = np.ones(12)
        no_shift = np.zeros(12)

        scale_vector = (*B_scale, *no_scale)
        shift_vector = (*B_shift, *no_shift)
        params = np.multiply(scale_vector, params) + shift_vector

    (L0L1, R0R1, L0R0, L1R1, L0, L1, R0, R1, xL0, yL0, xL1, yL1, xR0, yR0, xR1, yR1,) = params

    WL0_dx, WL0_dy = exp_well_derivs(x, y, L0, L0R0, L0L1, xL0, yL0)
    WL1_dx, WL1_dy = exp_well_derivs(x, y, L1, L1R1, L0L1, xL1, yL1)
    WR0_dx, WR0_dy = exp_well_derivs(x, y, R0, L0R0, R0R1, xR0, yR0)
    WR1_dx, WR1_dy = exp_well_derivs(x, y, R1, L1R1, R0R1, xR1, yR1)
    s = 0.03
    s_dx = 4 * s * x ** 3
    s_dy = 4 * s * y ** 3
    fx, fy = (
        -(WL0_dx + WL1_dx + WR0_dx + WR1_dx + s_dx),
        -(WL0_dy + WL1_dy + WR0_dy + WR1_dy + s_dy),
    )

    return (fx, fy)


exp_defaults = (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1)
exp_domain = (-2,-2),(2,2)
exp_wells_2D = Potential(exp_potential, exp_potential_force, 16, 2, default_params=exp_defaults, relevant_domain=exp_domain)


def even_parity_1D_well(x, params):
    """
    1D double well potential using just even parity terms x^2 and x^4:
        a*x^4 + (b/2)*x^2

    Parameters
    ----------
    x: ndarray of dimension [N,]
        the x coordinates for N positions
    params: list/tuple (1, 2)
        1, 2 : coefficients of the x^4 and x^2 terms, respectively

    Returns
    -------
    the value of the potential at locations x,y with the given params
    """

    a, b = params
    return a * x ** 4 + (b/2) * x ** 2


def even_parity_1D_well_force(x, params):
    """
    see even_parity_1D_well function documentation
    """

    a, b = params
    return -4 * a * x ** 3 - 2 * (b/2) * x


even_1DW_defaults = (1, -8)
even_1DW = Potential(even_parity_1D_well, even_parity_1D_well_force, 2, 1, default_params=even_1DW_defaults)


def asym_1D_well(x, params):
    """
    1D asymetric double well potential using just even parity terms x^2 and x^4. coefficients for x<0 and x>0 can vary.

    Parameters
    ----------
    x: ndarray of dimension [N,]
        the x coordinates for N positions
    params: list/tuple (1, 2, 3, 4)
        1, 2, 3, 4 : 1,2 are coefficients of the x^4 terms and  3,4  of the x^2 terms, respectively

    Returns
    -------
    the value of the potential at locations x,y with the given params
    """

    a1, a2, k1, k2 = params

    positive_well = a1 * x ** 4 + (k1/2) * x ** 2
    negative_well = a2 * x ** 4 + (k2/2) * x ** 2


    max_depth = np.max( [k1**2/(16*a1), k2**2/(16*a2)] )

    return max_depth + np.heaviside(x, 0) * positive_well + np.heaviside(-x, 0) * negative_well


def asym_1D_well_force(x, params):
    """
    see even_parity_1D_well function documentation
    """

    a1, a2, k1, k2 = params
    return np.heaviside(x, 0) * (-4 * a1 * x ** 3 - k1 * x) + np.heaviside(-x, 0) * (-4 * a2 * x ** 3 - k2 * x)


asym_1DW_defaults = (1, 1, -16, -16)

asym_1DW = Potential(asym_1D_well, asym_1D_well_force, 4, 1, default_params=asym_1DW_defaults)


def exp_well_3D(x, y, z, depth, localization, x0, y0, z0):
    '''
    helper function used in other potentials
    '''
    U = - depth * np.exp(-localization*((x-x0)**2 + (y-y0)**2 + (z-z0)**2))
    fx = 2 * localization * depth * (x-x0) * U
    fy = 2 * localization * depth * (y-y0) * U
    fz = 2 * localization * depth * (z-z0) * U
    return U, (fx, fy, fz)


def stability_3D(x, y, z, s=.2):
    '''
    helper function used in other potentials
    '''
    U = s*(x**4 + y**4 + z**4)
    fx = - 4*s*x**3
    fy = - 4*s*x**3
    fz = - 4*s*x**3
    return U, (fx, fy, fz)


def symmetric_exp_wells_3D_pot(x, y, z, params):
    """
    3D 8-well potential. Used to implement a fredkin gate

    Parameters
    ----------
    x,y,z : ndarrays of dimension [N,]
        the x,y,z coordinates for N positions
    params: list/tuple (1, 2, 3)
        1, 2, 3:     1 is well depths
                     2 is well separation
                     3 is well localization

    Returns
    -------
    the value of the potential at locations x,y,z with the given params
    """
    D, L, loc = params
    well_positions = [(-L, -L, -L),
                      (-L, -L, L),
                      (-L, L, -L),
                      (-L, L, L),
                      (L, -L, -L),
                      (L, -L, L),
                      (L, L, -L),
                      (L, L, L),
                      ]
    wells = []
    for item in well_positions:
        wells.append(exp_well_3D(x, y, z, D, loc, *item)[0])

    wells.append(stability_3D(x, y, z, s=.2)[0])

    U = 0
    for item in wells:
        U = np.add(U, item)
    return U


def symmetric_exp_wells_3D_force(x, y, z, params):
    '''
    see docs for symmetric_exp_wells_3D_pot
    '''
    D, L, loc = params
    well_positions = [(-L, -L, -L),
                      (-L, -L, L),
                      (-L, L, -L),
                      (-L, L, L),
                      (L, -L, -L),
                      (L, -L, L),
                      (L, L, -L),
                      (L, L, L),
                      ]
    wells = []
    for item in well_positions:
        wells.append(exp_well_3D(x, y, z, D, loc, *item)[1])

    wells.append(stability_3D(x, y, z, s=.2)[1])

    f = np.zeros((3, *np.shape(x)))
    for item in wells:
        f = np.add(f, item)
    return f


symm_3D_wells = Potential(symmetric_exp_wells_3D_pot, symmetric_exp_wells_3D_force, 3, 3, default_params=(10, 1, 8))


def fredkin_flip_pot(x, y, z, params):
    """
    3D 8-well potential. Used to implement a fredkin gate

    Parameters
    ----------
    x: ndarray of dimension [N,]
        the x coordinates for N positions
    params: list/tuple (1, 2, 3, 4)
        1, 2, 3, 4 : 1,2 are the coefficients of the 4th and 2nd order terms in the storage potential
                     3 turns off the y-z subspace storage potential fir x>0. Should genrally be 0(storage on) or 1(storage off)
                     4 is the k-value of the computational potential that implements the swap (like spring contant k)

    Returns
    -------
    the value of the potential at locations x,y with the given params
    """
    a, b, s, k = params
    r2 = np.sqrt(2)
    yp = (y-z)/r2
    zp = (y+z)/r2

    U0 = a*(x**4 + y**4 + z**4) + b*(x**2 + y**2 + z**2)
    U1 = U0 + s*(-a * (y**4 + z**4) - b * (y**2 + z**2) + k*(yp**2/2 + 2*zp**2))

    return np.heaviside(-x, 0) * U0 + np.heaviside(x, 0) * U1


def fredkin_flip_force(x, y, z, params):
    '''
    see docs for fredkin_flip_pot
    '''
    a, b, s, k = params
    r2 = np.sqrt(2)
    yp = (y-z)/r2
    zp = (y+z)/r2

    U0_dx = 4*a*(x**3) + 2*b*x
    U0_dy = 4*a*(y**3) + 2*b*y
    U0_dz = 4*a*(z**3) + 2*b*z
    U1_dx = U0_dx
    U1_dy = U0_dy + s * (-4 * a * y**3 - 2 * b * y + k * (yp/r2 + 4*zp/r2))
    U1_dz = U0_dz + s * (-4 * a * z**3 - 2 * b * z + k * (-yp/r2 + 4*zp/r2))

    fx = -np.heaviside(-x, 0) * U0_dx - np.heaviside(x, 0) * U1_dx
    fy = -np.heaviside(-x, 0) * U0_dy - np.heaviside(x, 0) * U1_dy
    fz = -np.heaviside(-x, 0) * U0_dz - np.heaviside(x, 0) * U1_dz

    return (fx, fy, fz)


fp_def_param = (2, -16, 0, 0)
fp_domain = ((-3, -3, -3), (3, 3, 3))
fredkin_pot = Potential(fredkin_flip_pot, fredkin_flip_force, 4, 3, default_params=fp_def_param, relevant_domain=fp_domain)
