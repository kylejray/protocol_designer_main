from ..protocol_designer import Potential, Protocol, Compound_Protocol
import numpy as np



def ew_1D(x, depth, x_0, localization):
    return -depth * np.exp( -localization * (x-x_0)**2 )
def ew_1D_deriv(x, depth, x_0, localization):
    return 2 * depth * (x-x_0) * localization * np.exp( -localization * (x-x_0)**2 )
def symm_quartic_1D(x, depth, x_0):
    a, b = depth/x_0**4, 2*depth/x_0**2
    return a*x**4 - b*x**2
def symm_quartic_deriv(x, depth, x_0):
    a, b = depth/x_0**4, 2*depth/x_0**2
    return 4*a*x**3 - 2 * b * x
'''
 WIP
def quart_gauss_potential(x, params):
    params = depth, x_0, quart_weight
    local = 20

    quartic = symm_quart(x, depth, x_0)

    gauss = ew_1D(x, depth, x_0, local) + ew_1D(ew_1D(x, depth, -x_0, local)
'''
    
def odw_gaussian_pot(x, params):
    '''
    x_0, x_1, d_0, d_1, local_0, local_1 = params
    '''
    x_0, x_1, d_0, d_1, local_0, local_1 = params

    z_gauss = ew_1D(x, d_0, x_0, local_0)
    o_gauss = ew_1D(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    slope_0, slope_1 = -d_0 * np.sqrt(2*local_0/ np.e), d_1 * np.sqrt(2*local_1/ np.e)
    y_0, y_1 = -d_0/np.sqrt(np.e), -d_1/np.sqrt(np.e)

    s_values = [[slope_0, x_m, y_0], [slope_1, x_p, y_1]]
    
    linear_stab = [ item[0]*(x-item[1]) + item[2] for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*linear_stab[0] + np.heaviside(x-x_p, 0)*linear_stab[1]


    U = np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return U

def odw_gaussian_force(x, params):
    x_0, x_1, d_0, d_1, local_0, local_1 = params

    z_gauss = ew_1D_deriv(x, d_0, x_0, local_0)
    o_gauss = ew_1D_deriv(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    slope_0, slope_1 = -d_0 * np.sqrt(2*local_0/ np.e), d_1 * np.sqrt(2*local_1/ np.e)
    y_0, y_1 = -d_0/np.sqrt(np.e), -d_1/np.sqrt(np.e)

    s_values = [[slope_0, x_m, y_0], [slope_1, x_p, y_1]]
    
    linear_stab_deriv = [ item[0] for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*linear_stab_deriv[0] + np.heaviside(x-x_p, 0)*linear_stab_deriv[1]

    d_U = np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return -d_U

odwg_params = (-1, 1, 1, 1, 5, 5)
odwg_domain = [[-5],[5]]

odw_gaussian = Potential(odw_gaussian_pot, odw_gaussian_force, 6, 1, default_params=odwg_params, relevant_domain=odwg_domain)



def symm_quart_exp_potential(x, params):
    '''
    potential that interpolates between quartic and exponential 1d confinement
    '''
    local = 20
    a, b, quart_weight= params

    
    depth = b**2 / (4*a)
    location = np.sqrt(b/(2*a))

    quart = a*x**4 - b*x**2

    stability = a*x**4
    s_depth= a*location**4
    exponential = ew_1D(x, depth+.05*s_depth, location, local) + ew_1D(x, depth+.05*s_depth, -location, local)+ .05*stability
    U = quart_weight * quart + (1-quart_weight) * exponential + depth
    return U

def symm_quart_exp_force(x,params):
    '''
    potential that interpolates between quartic and exponential 1d confinement
    '''
    a, b, quart_weight= params
    local=20

    d_quart = 4*a*x**3 - 2*b*x
    depth = b**2 / (4*a)
    location = np.sqrt(b/(2*a))
    d_stability = 4*a*x**3
    s_depth= a*location**4
    d_exponential = ew_1D_deriv(x, depth+.05*s_depth, location, local) + ew_1D_deriv(x, depth+.05*s_depth, -location, local) + .05*d_stability
    d = quart_weight * d_quart + (1-quart_weight) * d_exponential
    return -d

default = (2, 4, 1)
dom = [[-3],[3]]

symm_quart_exp_pot = Potential(symm_quart_exp_potential, symm_quart_exp_force, 3, 1, default_params=default, relevant_domain=dom)


prm = np.zeros((3,2))
prm[:, 0] = symm_quart_exp_pot.default_params
prm[:, 1] = symm_quart_exp_pot.default_params
prm[2,1] = 0


t=(0,1)

q_to_g_prot = Protocol(t, prm)





