from ..protocol_designer import Potential, Protocol, Compound_Protocol
import numpy as np

###### SINGLE FLUX QUBIT######

#first we defined the potential

def flux_qubit_pot(p,pdc, params):
    px, pxdc, gamma, beta, dbeta= params
    U = .5 * (p-px)**2 + .5*gamma*(pdc-pxdc)**2 + beta*np.cos(.5*pdc)*np.cos(p) - dbeta*np.sin(.5*pdc)*np.sin(p)
    return U

def flux_qubit_force(p,pdc, params):
    px, pxdc, gamma, beta, dbeta= params
    
    dp = (p-px) - beta*np.cos(.5*pdc)*np.sin(p) - dbeta*np.sin(.5*pdc)*np.cos(p)
    
    dpdc = gamma*(pdc-pxdc) - .5*beta*np.sin(.5*pdc)*np.cos(p) - .5*dbeta*np.cos(.5*pdc)*np.sin(p)
    
    return (-dp, -dpdc)
#realistic:
default_real = (.084, -2.5, 12, 6.2, .2)
#symmetric approximation:
default_symm = (0, -2.3, 12, 6.2, 0)
#symmetric ideal:
default_isymm = (0, -2.8, 27.5, 19.2, 0)
#domain
dom = ((-4,-4),(4,-1))


fq_pot = Potential(flux_qubit_pot, flux_qubit_force, 5, 2, default_params=default_real, relevant_domain=dom)
#once the potential is done, we make some simple one-time-step protocols, that will serves as buulding blocks

#the protocol below will start with the default parameters and then go to the 'flip parameters' at t=1
prm = np.zeros((5,2))
prm[:, 0] = fq_pot.default_params
prm[:, 1] = fq_pot.default_params
prm[1, 1] = -2*3.1416

t=(0,1)

flip_on = Protocol(t, prm)
flip_on.interpolation = 'step'

#the protocol below will start with the 'flip parameters' and go to the defaults at t=1
flip_off = flip_on.copy()
flip_off.reverse()

#same as above but shifted to run from t=1 to t=2 instead
flip_off_shift = flip_off.copy()
flip_off_shift.time_shift(1)


#combinig protocol steps into a Compound_Protocol allows for a full computational cycle
flip_prot = Compound_Protocol([flip_on,flip_off_shift])

#new default parameters to represent a more ideal device, given this kind of freedom is possible.

###### 2 COUPLED RF/RF QUBITS######

## RF RF COUPLING ##
#assumes derivative with respect to dimensionless arguments for the force, in other words the fluxes are scaled by 2pi/Phi_0 #

#helper functions to make the potential and force functions more palatable
def RF_RF_helper(beta, dbeta, arg1, arg2):
    return  beta * np.cos(arg1)* np.cos(arg2/2) - dbeta * np.sin(arg1)* np.sin(arg2/2)

def RF_RF_helper_deriv(beta, dbeta, arg1, arg2, which):
    if which == 1:
        return  ( -beta * np.sin(arg1)* np.cos(arg2/2) - dbeta * np.cos(arg1)* np.sin(arg2/2))
    if which == 2: 
        return   .5*(-beta * np.cos(arg1)* np.sin(arg2/2) - dbeta * np.sin(arg1)* np.cos(arg2/2))

#define the potential energy
def RF_RF_potential(zeta, zetap, pdc, pdcp, params):
    zetax, zetaxp, pxdc, pxdcp, mu, gamma, beta, dbeta = params

    z_quad = 1/(2*(1-mu)) * (zeta-zetax)**2 + 1/(2*(1+mu)) * (zetap-zetaxp)**2
    dc_quad = (gamma/2) * ( (pdc-pxdc)**2 + (pdcp-pxdcp)**2 )

    phi = (-zeta+zetap)/np.sqrt(2)
    phip = (zeta+zetap)/np.sqrt(2)

    coupling = RF_RF_helper(beta, dbeta, phi, pdc) + RF_RF_helper(beta, dbeta, phip, pdcp)

    return z_quad + dc_quad + coupling

#define the force from it
def RF_RF_force(zeta, zetap, pdc, pdcp, params):
    zetax, zetaxp, pxdc, pxdcp, mu, gamma, beta, dbeta = params
    phi = (-zeta+zetap)/np.sqrt(2)
    phip = (zeta+zetap)/np.sqrt(2)

    d_zeta = 1/(1-mu) * (zeta-zetax) + - RF_RF_helper_deriv(beta, dbeta, phi, pdc, 1)/np.sqrt(2) + RF_RF_helper_deriv(beta, dbeta, phip, pdcp, 1)/np.sqrt(2)
    d_zetap = 1/(1+mu) * (zetap-zetaxp) + RF_RF_helper_deriv(beta, dbeta, phi, pdc, 1)/np.sqrt(2)+ RF_RF_helper_deriv(beta, dbeta, phip, pdcp, 1)/np.sqrt(2)
    d_pdc = gamma * (pdc-pxdc) + RF_RF_helper_deriv(beta, dbeta, phi, pdc, 2)
    d_pdcp = gamma * (pdcp-pxdcp) + RF_RF_helper_deriv(beta, dbeta, phip, pdcp, 2)

    return (-d_zeta, -d_zetap, -d_pdc, -d_pdcp)


RFRF_default = (0, 0, 0, 0, .5, 12, 6, 0,)
RFRF_dom = ((-7, -7, -7, -7),(7, 7, 7, 7))

RF_RF_pot = Potential(RF_RF_potential, RF_RF_force, 8, 4, default_params=RFRF_default, relevant_domain=RFRF_dom)

#no systems/protocols prebuilt
