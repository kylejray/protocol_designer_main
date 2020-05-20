import numpy as np
import matplotlib.pyplot as plt
import copy


class Protocol:
    """
    The Protocol class can be thought of as a time dependent signal that is sent to an
    instance of the Potential class, which gives the Potential the value of its parameters
    as a function of time. The simple Protocol class can only hold one initial and final value
    for each parameter.

    Attributes
    ----------
    params: ndarray of dimension [N_params, 2]
        the initial and final values of each parameter

    protocols: None
        this is used for the inherited class, Compound_Protocol

    t_i, t_f : float
        the initial and final times of the protocol, t_i < t_f

    N_params: int
        number of parameters in the protocol

    """

    def __init__(self, t, params):
        self.params = np.asarray(params)
        self.protocols = None
        self.t_i = float(t[0])
        self.t_f = float(t[1])
        self.N_params = len(self.params[:, 0])

    def get_params(self, t):
        """
        returns a list of length N_params, that gives the value of each parameter at time t
        currently uses linear interpolation to determine the parameter

        Parameters
        ----------
        t: float
            the time at which you want the parameter values, t_i <= t <= t_f

        Returns
        -------
        parameter_vals: ndarray of dimension [N_params]
            gives the value of each parameter at the input time

        """

        if t < self.t_i:

            return self.get_linear(self.params[:, 0], self.params[:, 1], self.t_i)

        if self.t_f < t:

            return self.get_linear(self.params[:, 0], self.params[:, 1], self.t_f)

        if self.t_i <= t and t <= self.t_f:

            return self.get_linear(self.params[:, 0], self.params[:, 1], t)

    def time_shift(self, dt):
        """
        shifts the  protocol.t_i and protocol.t_f attributes
        by an amount dt, there are no returns

        Parameters
        ----------
        dt: float
            the amount we want to shift time by
        """
        self.t_i = self.t_i + dt
        self.t_f = self.t_f + dt

    def time_stretch(self, t_mult):
        """
        stretches the protocol to be longer by a factor of t_mult,
        changes the attributes protocol.t_i and protocol.t_f, there are no returns

        Parameters
        ----------
        t_mult: float
            the amount we want to dilate the timescale by
        """
        self.t_f = self.t_i + t_mult * (self.t_f - self.t_i)

    def normalize(self):
        """
        normalizes the protocol timescale so it begins at t_i=0
        and ends at t_f=1, no inputs and no outputs
        """
        t_i = self.t_i
        self.time_shift(-t_i)
        t_f = self.t_f
        self.time_stretch(1 / t_f)

    def reverse(self):
        """
        inverts protocol.params, so the initial parameters become the final ones,
        no inputs no outputs
        """
        self.params = np.flip(self.params, axis=1)

    def change_param(self, which_params, new_params):
        """
        Manually changes some or all of the parameter values in the protocol.
        There are no returns

        Parameters
        ----------
        which_params: list
            list of which paramters you want to change. i.e. which_params=(1,3) means you want to change the values for p1 and p3

        new_params: list of tuples of the form (p_i, p_f)
            list with the new initial and final values of every parameter you want to change
        """

        index = np.asarray(which_params) - 1
        self.params[index, :] = new_params

    def copy(self):
        """
        Returns
        -------
        A copy of your current protocol
        """

        return copy.deepcopy(Protocol((self.t_i, self.t_f), self.params))

    def show_params(self, which=None):
        """
        Shows plots of the chosen parameters over times, no returns

        Parameters
        ----------
        which: None, all,  or list
            if None, shows only nontrivial parameters that change over time
            if all, shows all parameters no amtter what
            if list, shows only the parameters in the list. i.e. which=3 will only show parameter numbe 3
        """
        N_t = 50
        t = np.linspace(self.t_i, self.t_f, N_t)

        if which is "all" or which is None:
            indices = np.asarray(range(self.N_params))

        if which is not None and which is not "all":
            indices = np.asarray(which) - 1

        p_array = np.zeros((N_t, len(indices)))

        for i, item in enumerate(t):
            p_array[i, :] = self.get_params(item)[indices]

        if which is None:
            idx = []
            p_test = p_array - p_array[0, :]
            p_t_sum = np.sum(p_test, axis=0)
            for i, item in enumerate(p_t_sum):
                if item != 0:
                    idx.append(i)
            indices = np.asarray(idx)
            assert (
                len(indices) > 0
            ), "protocol is completely trivial, use which = 'all' "
            p_array = p_array[:, indices]

        img_size = 5
        fig, ax = plt.subplots(
            len(indices), 1, figsize=(img_size, img_size * len(indices) / 5)
        )
        fig.subplots_adjust(hspace=0.5)

        for i, item in enumerate(ax[:]):
            y_range = max(np.abs(np.max(p_array[:, i])), np.abs(np.min(p_array[:, i])))
            if y_range == 0:
                y_range = 1
            item.set_xlim(self.t_i, self.t_f)
            item.set_ylim(-1.5 * y_range, 1.5 * y_range)
            item.yaxis.tick_right()
            item.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            # x_lines=np.flatten(self.times)

            item.text(
                0,
                0,
                "p{}".format(indices[i] + 1),
                horizontalalignment="right",
                verticalalignment="center",
            )
            if i > 0:
                item.set_xticks([])

            item.plot(t, p_array[:, i])

    def get_linear(self, init, final, t):
        """
        basic linear interpolation function, used internally by other methods
        """

        return init + (t - self.t_i) * (final - init) / (self.t_f - self.t_i)

    def get_logistic(self, init, final, t, ramp=5, offset=0):
        """
        basic logistic interpolation function, used internally by other methods
        """
        delta_y = final - init
        t_scaled = t - (self.t_i + self.t_f) / 2
        return init + delta_y / (1 + np.exp(-ramp * (t_scaled - offset)))


class Compound_Protocol(Protocol):
    """
    Stitches a list of protocols into a single object, provided that the protocol times do not overlap with eachother.

    Attributes
    ----------
    See attributes of the Protocol class, additionally we also have:

    protocols: list of Protocols
        list of the Protocol objects that make up the compound protocol, reffered to in the documentation as 'substages'

    times: ndarray of dimensions [N_prot, 2]
        array that stores the start/end times of each substage
    """

    def __init__(self, protocols):
        N = len(protocols)
        protocols = list(protocols)

        def sorting_t_i(prot):
            return prot.t_i

        def sorting_t_f(prot):
            return prot.t_f

        protocols.sort(key=sorting_t_i)
        sort_check = sorted(protocols, key=sorting_t_f)
        assert protocols == sort_check, "sorting error: check protocol times"

        times = np.zeros((N, 2))
        N_params = len(protocols[0].params[:, 0])

        for idx, item in enumerate(protocols):
            assert N_params == len(
                item.params[:, 0]
            ), "all substages must have the same number of parameters"
            times[idx, 0], times[idx, 1] = item.t_i, item.t_f

            if idx < N - 1:
                assert times[idx, 1] <= protocols[idx + 1].t_i, "protocol times overlap"

        self.times = times
        self.protocols = protocols
        self.t_i = float(np.min(times))
        self.t_f = float(np.max(times))
        self.N_params = N_params
        self.params = np.asarray(
            tuple(zip(self.get_params(self.t_i), self.get_params(self.t_f)))
        )

    def get_params(self, t):
        """
        Same as the parent class function, but requires slightly different code to implement.
            No Parameters or returns
        """
        counter = sum(self.times[:, 0] <= t)
        if counter > 0:
            counter = counter - 1

        return self.protocols[counter].get_params(t)

    def show_substage_times(self):
        """
        Prints the substage times for each piece of Compound_Protocol.
            No parameters or Returns
        """
        i = 1
        for item in self.times:
            print("stage {} times:".format(i), item)
            i += 1

    def time_stretch(self, scale, which_stages=None):
        """
        This extension of the parent class time_shift protocol requires an additional input
            that tells us which substages we want to stretch in time

        Parameters
        ----------
        scale : float
            how much we are going to dilate time

        which_stages: None or list of ints
            which substages are going to get stretched
            if None, stretches all stages
            if list, stretches only the selected substages
                (other times will translate automatically to keep the protocol going in forward time always)
        """
        if which_stages is None:
            new_times = scale * (self.times - np.min(self.times)) + np.min(self.times)

        if which_stages is not None:
            new_times = np.copy(self.times)

            if np.size(which_stages) == 1:
                which_stages = np.array([which_stages])
                index = which_stages - 1
            if np.size(which_stages) > 1:
                index = np.asarray(which_stages) - 1

            for idx in index:
                t0 = new_times[idx, 0]
                t1 = new_times[idx, 1]
                new_times[idx, 1] = scale * (t1 - t0) + t0
                delta_t = new_times[idx, 1] - t1

                for i in range(idx + 1, len(self.protocols)):
                    new_times[i, :] = new_times[i, :] + delta_t

        self.times = new_times
        self.refresh_substage_times()

    def time_shift(self, delta_t, which_stages=None):
        """
        This extension of the parent class time_shift protocol requires an additional input
            that tells us which substages we want to shift in time

        Parameters
        ----------
        delta_t : float
            how much we are going to shift the time

        which_stages:None or list of ints
            which substages are going to get shifted
            if None, shifts all stages
            if list shifts only the selected substages
                (other times will adjust to keep the protocol going in forward time always)
        """
        if which_stages is None:
            self.times = self.times + delta_t
            self.refresh_substage_times()

        if which_stages is not None:
            new_times = np.copy(self.times)

            if np.size(which_stages) == 1:
                which_stages = np.array([which_stages])
                index = which_stages - 1
            if np.size(which_stages) > 1:
                index = np.asarray(which_stages) - 1

            for idx in index:
                if delta_t > 0:
                    for i in range(idx, len(self.protocols)):
                        new_times[i, :] = new_times[i, :] + delta_t
                if delta_t < 0:
                    for i in range(0, idx + 1):
                        j = idx - i
                        new_times[j, :] = new_times[j, :] + delta_t

            self.times = new_times
            self.refresh_substage_times()

    def refresh_substage_times(self):
        """
        This is a helper function, used internally by other methods.
        It makes sure that the individual substage protocol t_i and t_f
        and the Compound_Protocol.times array match by generating a new
        Compound_Protocol.times array from the substage t_i's and t_f's
        """
        self.t_f = float(np.min(self.times))
        self.t_f = float(np.max(self.times))
        for idx, item in enumerate(self.protocols):
            item.t_i = float(self.times[idx, 0])
            item.t_f = float(self.times[idx, 1])

    def copy(self):
        """
        Returns a copy of the Protocol
        """
        return copy.deepcopy(Compound_Protocol(self.protocols))


def sequential_protocol(
    N_steps, N_params, which_params, nontrivial_params, times=None, initial_params=None
):
    """
    This function is to faciliate the creation of a common type of Compound Protocol,
        often, a protocol has only a few nontrivial (actually changing) parameters
        in the signal. This function is especially useful in these cases.

    Parameters
    ----------
    N_steps: int
        number of stages in the protocol
    N_params: int
        number of parameters in the signal
    which_params: list of ints
        lists which parameters we are going to be changing in the full protocol
    nontrivial_params: list of lists
        the list is of length len(which_param)), each element corresponds to a different parameter
        each element should have a length of N_steps+1, and contain the parameter values at each substage start and end time
    times: None or list of floats
        if None will make equally spaced substages b/w t=0 and t=1
        if list (length = N_steps+1)
    initial_params: None or list of lenth N_params
        if None, the trivial parameters will be set to 0
        if list, usues the corresponding list element for the trivial parameter values

    Returns
    -------
    Compound Protocol: instance of Compount_Protocol class

    Examples
    --------
    Return a 5 parameter protocol, with 2 equal length substages, where the 3rd and 4th parameters are the only nontrivial ones.
    The rest are held fixed at 1. Total time will be set to the default: t_f =1

    >>> p3, p4 = (-1,1,0), (0,0,2)
    >>> which_p = (3,4)
    >>> nontrivial_p = (p3, p4)
    >>> init_vals = np.ones(5)
    >>> seq_prot = sequential_protocol(2, 5, which_p, nontrivial_p, initial_params=init_vals )
    >>> seq_prot.show_substage_times()
    """
    if times is None:
        times = np.linspace(0, 1, N_steps + 1)

    indices = np.asarray(which_params) - 1

    t = np.zeros((N_steps, 2))

    p = np.zeros((N_steps, N_params, 2))

    ntp = np.asarray(nontrivial_params)

    if initial_params is not None:
        for idx, item in enumerate(initial_params):
            p[:, idx, :] = item

    for i in range(N_steps):
        new_params = []
        for j in range(len(indices)):
            new_params.append((ntp[j, i], ntp[j, i + 1]))

        t[i, :] = times[i], times[i + 1]
        p[i, indices, :] = new_params

    prots = []

    for i in range(N_steps):
        current_prot = Protocol(t[i, :], p[i, :, :])
        prots.append(current_prot)

    return Compound_Protocol(prots)
