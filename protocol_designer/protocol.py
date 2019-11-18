import numpy as np
import matplotlib.pyplot as plt
import copy


class Protocol:

    def __init__(self, t, params):
        self.params = np.asarray(params)
        self.protocols = None
        self.t_i = t[0]
        self.t_f = t[1]
        self.N_params = len(self.params[:, 0])

    def get_params(self, t):

        if t < self.t_i:

            return self.get_linear(self.params[:, 0], self.params[:, 1], self.t_i)

        if self.t_f < t:

            return self.get_linear(self.params[:, 0], self.params[:, 1], self.t_f)

        if self.t_i <= t and t <= self.t_f:

            return self.get_linear(self.params[:, 0], self.params[:, 1], t)

    def get_linear(self, init, final, t):

        return init+(t-self.t_i)*(final-init)/(self.t_f-self.t_i)

    def get_logistic(self, init, final, t, ramp=5, offset=0):

        delta_y = final-init
        t_scaled = t-(self.t_i+self.t_f)/2
        return(init+delta_y/(1+np.exp(-ramp*(t_scaled-offset))))

    def time_shift(self, dt):
        self.t_i = self.t_i+dt
        self.t_f = self.t_f+dt

    def time_stretch(self, t_mult):
        self.t_f = self.t_i+t_mult*(self.t_f-self.t_i)

    def reverse(self):
        self.params = np.flip(self.params, axis=1)

    def change_param(self, which_params, new_params):
        index = np.asarray(which_params)-1
        self.params[index, :] = new_params

    def copy(self):
        return copy.deepcopy(Protocol((self.t_i, self.t_f), self.params))

    def show_params(self, which=None):
        N_t = 50
        t = np.linspace(self.t_i, self.t_f, N_t)

        if which is 'all' or which is None:
            indices = np.asarray(range(self.N_params))

        if which is not None and which is not 'all':
            indices = np.asarray(which)-1

        p_array = np.zeros((N_t, len(indices)))

        for i, item in enumerate(t):
            p_array[i, :] = self.get_params(item)[indices]

        if which is None:
            idx = []
            p_test = p_array-p_array[0, :]
            p_t_sum = np.sum(p_test, axis=0)
            for i, item in enumerate(p_t_sum):
                if item != 0:
                    idx.append(i)
            indices = np.asarray(idx)
            assert len(indices) > 0, "protocol is completely trivial, use which = 'all' "
            p_array = p_array[:, indices]

        img_size = 5
        fig, ax = plt.subplots(len(indices), 1, figsize=(img_size, img_size*len(indices)/5))
        fig.subplots_adjust(hspace=.5)

        for i, item in enumerate(ax[:]):
            y_range = max(np.abs(np.max(p_array[:, i])), np.abs(np.min(p_array[:, i])))
            if y_range == 0:
                y_range = 1
            item.set_xlim(self.t_i, self.t_f)
            item.set_ylim(-1.5*y_range, 1.5*y_range)
            item.yaxis.tick_right()
            item.axhline(y=0, color='k', linestyle='--', alpha=.5)
            # x_lines=np.flatten(self.times)

            item.text(0, 0, "p{}".format(indices[i]+1), horizontalalignment='right', verticalalignment='center')
            if i > 0:
                item.set_xticks([])

            item.plot(t, p_array[:, i])


class Compound_Protocol(Protocol):

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
            assert N_params == len(item.params[:, 0]), "all substages must have the same number of parameters"
            times[idx, 0], times[idx, 1] = item.t_i, item.t_f

            if idx < N-1:
                assert times[idx, 1] <= protocols[idx+1].t_i, "protocol times overlap"

        self.times = times
        self.protocols = protocols
        self.t_i = float(np.min(times))
        self.t_f = float(np.max(times))
        self.N_params = N_params
        self.params = np.asarray(tuple(zip(self.get_params(self.t_i), self.get_params(self.t_f))))

    def get_params(self, t):
        counter = sum(self.times[:, 0] <= t)
        if counter > 0:
            counter = counter-1

        return self.protocols[counter].get_params(t)

    def show_substage_times(self):
        i = 1
        for item in self.times:
            print('stage {} times:'.format(i), item)
            i += 1

    def time_stretch(self, scale, which_stages=None):
        if which_stages is None:
            new_times = scale * (self.times-np.min(self.times)) + np.min(self.times)

        if which_stages is not None:
            new_times = np.copy(self.times)

            if np.size(which_stages) == 1:
                which_stages = np.array([which_stages])
                index = which_stages-1
            if np.size(which_stages) > 1:
                index = np.asarray(which_stages) - 1

            for idx in index:
                t0 = new_times[idx, 0]
                t1 = new_times[idx, 1]
                new_times[idx, 1] = scale*(t1-t0)+t0
                delta_t = new_times[idx, 1]-t1

                for i in range(idx+1, len(self.protocols)):
                    new_times[i, :] = new_times[i, :] + delta_t 

        self.times = new_times
        self.t_i = np.min(self.times)
        self.t_f = np.max(self.times)
        self.refresh_substage_times()

    def time_shift(self, delta_t, which_stages=None):
        if which_stages is None:
            self.t_i = self.t_i + delta_t
            self.t_f = self.t_f + delta_t
            self.times = self.times + delta_t
            self.refresh_substage_times()

        if which_stages is not None:
            new_times = np.copy(self.times)

            if np.size(which_stages) == 1:
                which_stages = np.array([which_stages])
                index = which_stages-1
            if np.size(which_stages) > 1:
                index = np.asarray(which_stages) - 1

            for idx in index:
                if delta_t > 0:
                    for i in range(idx, len(self.protocols)):
                        new_times[i, :] = new_times[i, :] + delta_t
                if delta_t < 0:
                    for i in range(0, idx+1):
                        j = idx - i
                        new_times[j, :] = new_times[j, :] + delta_t

            self.times = new_times
            self.t_i = np.min(self.times)
            self.t_f = np.max(self.times)
            self.refresh_substage_times()

    def refresh_substage_times(self):
        for idx, item in enumerate(self.protocols):
            item.t_i = self.times[idx, 0]
            item.t_f = self.times[idx, 1]

    def copy(self):
        return copy.deepcopy(Compound_Protocol(self.protocols))


def sequential_protocol(N_steps, N_params, which_params, nontrivial_params, times=None, initial_params=None):

    if times is None:
        times = np.linspace(0, 1, N_steps+1)

    indices = np.asarray(which_params)-1

    t = np.zeros((N_steps, 2))

    p = np.zeros((N_steps, N_params, 2))

    ntp = np.asarray(nontrivial_params)

    if initial_params is not None:
        for idx, item in enumerate(initial_params):
            p[:, idx, :] = item

    for i in range(N_steps):
        new_params = []
        for j in range(len(indices)):
            new_params.append((ntp[j, i], ntp[j, i+1]))

        t[i, :] = times[i], times[i+1]
        p[i, indices, :] = (new_params)

    prots = []

    for i in range(N_steps):
        current_prot = Protocol(t[i, :], p[i, :, :])
        prots.append(current_prot)

    return Compound_Protocol(prots)
