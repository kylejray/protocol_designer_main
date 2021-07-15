import numpy as np

from .protocol import Protocol


class Potential:
    """
    This class is relatively simple in function. It bundles a force function and a potential energy function
    together with methods to pull out the forces and energies when given coordinates and parameters. There are
    also some other useful pieces of information stored, as well as utility methods

    Attributes
    ----------
    scale: float
        a multiplicative scale for the whole potential
    pot: func
        the potential energy function
    force: func
        the force function
    N_params: int
        the number of parameters that the force/potential energy need to give well defined answers
    N_dim: int
        number of dimensions the potential is over
    default_params = None or list
        if None, will set each default to 1
        if list (length N_params), list becomes the default values for each parameter
    domain: None or ndarray of dimension [2, N_dim]
        stores the relevant working domain of the potential, where we expect interesting dynamics to happen
        if None, uses -2,2 for all dimensions
        if ndarray, take the array to be [ [x1_min, x2_min,....], [x1_max, x2_max,...]]
    """

    def __init__(
        self,
        potential,
        external_force,
        N_params,
        N_dim,
        default_params=None,
        relevant_domain=None,
    ):
        """
        potential: func
            the potential energy function
        external_force: func
            the force function
        N_params: int
            the number of parameters that the force/potential energy need to give well defined answers
        N_dim: int
            number of dimensions the potential is over
        default_params = None or list
            if None, will set each default to 1
            if list (length N_params), list becomes the default values for each parameter
        relevant_domain: None or ndarray of dimension [2, N_dim]
            stores the relevant working domain of the potential, where we expect interesting dynamics to happen
            if None, uses -2,2 for all dimensions
            if ndarray, take the array to be [ [x1_min, x2_min,....], [x1_max, x2_max,...]]
        """

        self.scale = 1
        self.pot = potential
        self.force = external_force
        self.N_params = N_params
        self.N_dim = N_dim
        self.default_params = default_params
        if relevant_domain is None:
            self.domain = np.asarray(
                (-2 * np.ones(self.N_dim), 2 * np.ones(self.N_dim))
            )
        else:
            self.domain = np.asarray(relevant_domain)

    def potential(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args: the arguments to be fed into the potential function

        Returns
        -------
        a scaled version of the potential function

        """
        return np.multiply(self.scale, self.pot(*args, **kwargs))

    def external_force(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args: the arguments to be fed into the force function

        Returns
        -------
        a scaled version of the force function

        """
        return np.multiply(self.scale, self.force(*args, **kwargs))

    def trivial_protocol(self, t_i=0, t_f=1):
        """
        makes a trivial (all parameters held fixed) protocol that will work with this potential

        Parameters
        ----------
        t_i,t_f : floats
            the initial and final times of the protocol

        Returns
        -------
        Protocol: instance of Protocol class
            this will be a simple one step protocol, where all parameters are held
            fixed at their default values, potential.default_params.
        """
        t = (t_i, t_f)
        if self.default_params is not None:
            assert (
                len(self.default_params) == self.N_params
            ), "number of default parameters doesnt match potential"
            params = []
            for i in range(self.N_params):
                params.append((self.default_params[i], self.default_params[i]))
        if self.default_params is None:
            params = np.ones((self.N_params, 2))

        return Protocol(t, params)

    def info(self, verbose=False):
        """
        prints basic info about the potential
        """
        if verbose:
            print(self.pot.__doc__)
        else:
            print("This potential has {} parameters and {} dimensions".format(self.N_params, self.N_dim))
            print("The current scale is {}".format(self.scale))
            print('To see details about the specific potential set verbose=True')


# A simple 1D potential, for testing one dimensional systems
# its just an absolute value. parameters are:
# 1: the slope
# 2: zero point


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
