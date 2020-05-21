import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from .protocol import Protocol
from .potentials import Potential
import copy

# expected form for data for coordinates is (N,D,2)
# N is number of trials, D is dimension, 2 is for position(0) vs velocity(1)
# examples: coords[204,1,1] would be v_y of the 204th trial.


class System:
    """
    This class bridges the gap between protocol designer
    and the info engine sims package. It take a protocol
    and a potential and packages it into a system that can
    be simulated.

    Attributes
    ----------
    protocol: instance of the Protocol class
        this is the signal that controls the potential parameters

    potential: instance of the Potential class
        this is the potential energy landscape we will apply the
        protocol too
    """

    def __init__(self, protocol, potential):
        self.protocol = protocol
        self.potential = potential
        msg = "the number of protocol parameters must match the potential"
        assert (
            len(self.protocol.get_params(self.protocol.t_i)) == self.potential.N_params
        ), msg

    def copy(self):
        """
        Generate a copy of the system

        Returns
        -------
        copy: instance of the System class

        """
        return copy.deepcopy(System(self.protocol, self.potential))

    def get_energy(self, coords, t):
        """
        Calculate the energy of a particle at location coords at time t

        Parameters
        ----------

        coords: ndarray of dimensions [N_c, N_d, 2]
            array of N_c sets of coordinates in N_d dimensions

        t: float or int
            time at which you want to evaluate the energy

        Returns
        -------

        U+T : ndarray of dimension [N_c,]

        """

        v = coords[:, :, 1]
        U = self.get_potential(coords, t)
        T = np.sum(0.5 * np.square(v), axis=1)
        return U + T

    def get_potential(self, coords, t):
        """
        Calculate the potential energy of a particle at location coords at time t

        Parameters
        ----------

        coords: ndarray of dimensions [N_c, N_d, 2]
            array of N_c sets of coordinates in N_d dimensions

        t: float or int
            time at which you want to evaluate the energy

        Returns
        -------

        U : ndarray of dimension [N_c,]

        """
        params = self.protocol.get_params(t)
        positions = np.transpose(coords[:, :, 0])

        return self.potential.potential(*positions, params)

    def get_external_force(self, coords, t):
        """
        Calculate the forces on a particle due to the potential energy
        at location coords at time t

        Parameters
        ----------

        coords: ndarray of dimensions [N_c, N_d, 2]
            array of N_c sets of coordinates in N_d dimensions

        t: float or int
            time at which you want to evaluate the energy

        Returns
        -------

        U : ndarray of dimension [N_c, N_d]

        """
        params = self.protocol.get_params(t)
        positions = np.transpose(coords[:, :, 0])
        F = np.zeros((len(coords[:, 0, 0]), self.potential.N_dim))
        force = self.potential.external_force(*positions, params)
        for i in range(0, self.potential.N_dim):
            F[:, i] = force[i]

        return F

    def eq_state(self, Nsample,  t=None, resolution=1000, damped=None, manual_domain=None, axis1=1, axis2=2, slice_vals=None):
        '''
        function still in development, docstring will come later.
        generates Nsample coordinates from an equilibrium distribution at
        time t.

        '''
        NT = Nsample
        state = np.zeros((max(100, int(2*NT)), self.potential.N_dim, 2))

        def get_prob(self, state):
            E_curr = self.get_energy(state, t)
            Delta_U = E_curr-U0
            return np.exp(-Delta_U)

        x_min, x_max, y_min, y_max = self.get_domain(axis1, axis2, domain=manual_domain)
        mins = (x_min, y_min)
        maxes = (x_max, y_max)

        if t is None:
            t = self.protocol.t_i

        U = self.lattice(t, resolution, x_min, x_max, y_min, y_max, axis1, axis2, slice_values=slice_vals)[0]

        U0 = np.min(U)
        i = 0

        while i < Nsample:
            n_coords = 2
            if self.potential.N_dim == 1:
                n_coords = 1
            test_coords = np.zeros((NT, n_coords, 2))
            test_state = np.copy(test_coords)
            if slice_vals is not None:
                test_state[:, :, 0] = slice_vals

            if n_coords == 1:
                test_coords[:, :, 0] = np.random.uniform(x_min, x_max, (NT, n_coords))
            if n_coords == 2:
                test_coords[:, :, 0] = np.random.uniform(mins, maxes, (NT, n_coords))
            if damped is None:
                test_coords[:, :, 1] = np.random.normal(0, 1, (NT, n_coords))
            if n_coords == 2:
                test_state[:, axis1-1, :] = test_coords[:, 0, :]
                test_state[:, axis2-1, :] = test_coords[:, 1, :]
            if n_coords == 1:
                test_state = test_coords

            p = get_prob(self, test_state)
            decide = np.random.uniform(0, 1, NT)
            n_sucesses = np.sum(p > decide)
            if i == 0:
                ratio = max(n_sucesses/NT, .1)
            state[i:i+n_sucesses, :, :] = test_state[p > decide, :, :]
            i = i + n_sucesses
            NT = max(int((Nsample-i)/ratio), 100)
        state = state[0:Nsample, :, :]
        return(state)

    def show_potential(
        self,
        t,
        resolution=100,
        surface=False,
        manual_domain=None,
        contours=50,
        axis1=1,
        axis2=2,
        slice_values=None,
    ):
        """
        Shows a 1 or 2D plot of the potential at a time t

        Parameters
        ----------

        t: float
            the time you want to plot the potential at

        resolution: int
            the number of sample points to plot along each axis

        surface: True/False
            if True plots a wireframe surface in 3D
            if False plots a contour plot in 2D

        manual_domain: None or ndarray of dimension (2, N_d)
            if None, we pull the domain from the default potential.domain
            if ndarray, a manual domain of the form [ (xmin,ymin,...), (xmax, ymax,...) ]

        contours: int or list
            sets number of contours to plot, or list of manually set contours

        axis1, axis2: int
            which coordinate we will consider to be 'x' and 'y' for the plot

        slice_values: ndarray of dimension [N_d,]
            these are the values we keep the other coordinates fixed at while sweeping through axis1 and axis2

        Returns
        -------
        no returns, just plots a figure
        """

        x_min, x_max, y_min, y_max = self.get_domain(axis1, axis2, domain=manual_domain)

        if self.potential.N_dim >= 2:
            U, X, Y = self.lattice(
                t, resolution, x_min, x_max, y_min, y_max, axis1, axis2, slice_values
            )

            if surface is False:
                fig, ax = plt.subplots()
                CS = ax.contour(X, Y, U, contours)
                # ax.clabel(CS, inline=1, fontsize=10)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.text(
                    x_min - 0.1 * (x_max - x_min),
                    y_min - 0.1 * (y_max - y_min),
                    "t={:.2f}".format(t),
                    horizontalalignment="right",
                    verticalalignment="top",
                    fontsize=12,
                    color="k",
                )

                plt.show()

            if surface is True:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_wireframe(X, Y, U)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.text(
                    -0.3,
                    -0.3,
                    0,
                    "t={:.2f}".format(t),
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=12,
                    color="k",
                )

                plt.show()
        if self.potential.N_dim == 1:
            U, X = self.lattice(t, resolution, x_min, x_max, y_min, y_max)
            fig, ax = plt.subplots()
            ax.plot(X, U)
            ax.set_xlabel("x")
            ax.text(
                x_min - 0.1 * (x_max - x_min),
                y_min - 0.1 * (y_max - y_min),
                "t={:.2f}".format(t),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=12,
                color="k",
            )

    def animate_protocol(
        self,
        mesh=40,
        fps=10,
        frames=50,
        surface=False,
        save=False,
        manual_domain=None,
        n_contours=50,
        axis1=1,
        axis2=2,
        slice_values=None,
    ):
        """
        Shows an animation of how the potential changes over the duration of your protocol, can be a little slow

        Parameters
        ----------

        mesh: int
            the number of sample points to plot along each axis

        fps: int
            frames per second in the animation

        frame: int
            number of frames to render

        surface: True/False
            if True plots a wireframe surface in 3D
            if False plots a contour plot in 2D

        manual_domain: None or ndarray of dimension (2, N_d)
            if None, we pull the domain from the default potential.domain
            if ndarray, a manual domain of the form [ (xmin,ymin,...), (xmax, ymax,...) ]

        n_contours: int or list
            sets number of contours to plot, or list of manually set contours

        axis1, axis2: int
            which coordinate we will consider to be 'x' and 'y' for the plot

        slice_values: ndarray of dimension [N_d,]
            these are the values we keep the other coordinates fixed at while sweeping through axis1 and axis2

        Returns
        -------
        anim: animation.FuncAnimate object

        """
        x_min, x_max, y_min, y_max = self.get_domain(axis1, axis2, domain=manual_domain)
        t_i = self.protocol.t_i
        t_f = self.protocol.t_f
        t = np.linspace(t_i, t_f, frames)

        if self.potential.N_dim == 1:

            U_array = np.zeros((mesh, frames))
            U_array[:, 0], X = self.lattice(t_i, mesh, x_min, x_max, y_min, y_max)

            for idx, item in enumerate(t):
                U_array[:, idx] = self.lattice(item, mesh, x_min, x_max, y_min, y_max)[
                    0
                ]

            fig, ax = plt.subplots()
            (line,) = ax.plot([], [])
            ax.set_xlabel("x")
            ax.set_ylabel("U")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(U_array.min(), U_array.max())

            # txt=ax.text(-.1, -.3, '', horizontalalignment='center', verticalalignment='top', fontsize=12,color='k')
            def init():
                line.set_data([], [])
                # txt.set_text('')

                return (line,)  # txt

            def update_plot(frame_number):
                U_current = U_array[:, frame_number]
                line.set_data(X, U_current)
                # txt=ax.text(-.1, -.3, 't={:.2f}'.format(t[frame_number]),
                # horizontalalignment='center', verticalalignment='top', fontsize=12,color='k')
                return (line,)  # txt

            anim = animation.FuncAnimation(
                fig,
                update_plot,
                init_func=init,
                frames=frames,
                interval=1000 / fps,
                blit=True,
            )

            if save:
                anim.save("animation.gif", fps)
                print("finished")

            return anim

        if self.potential.N_dim >= 2:
            U_array = np.zeros((mesh, mesh, frames))
            # T_array=np.zeros((mesh,mesh,frames))
            #
            # for idx,item in enumerate(t):
            #    T_array[:,:,idx]=item

            U_array[:, :, 0], X, Y = self.lattice(t_i, mesh, x_min, x_max, y_min, y_max)
            for idx, item in enumerate(t):
                U_array[:, :, idx] = self.lattice(
                    item, mesh, x_min, x_max, y_min, y_max
                )[0]

            if surface:

                def update_plot_2D(frame_number, U_array, plot):
                    plot[0].remove()
                    plot[0] = ax.plot_wireframe(
                        X, Y, U_array[:, :, frame_number], cmap="magma"
                    )
                    txt.set_text("t={:.2f}".format(t[frame_number]))

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                plot = [
                    ax.plot_surface(
                        X, Y, U_array[:, :, 0], color="0.75", rstride=1, cstride=1
                    )
                ]
                txt = ax.text(
                    x_min - 0.2 * (x_max - x_min),
                    y_min - 0.2 * (y_max - y_min),
                    0,
                    "t={:.2f}".format(t[0]),
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=12,
                    color="k",
                )
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlim(U_array.min(), U_array.max())

                anim = animation.FuncAnimation(
                    fig,
                    update_plot_2D,
                    frames,
                    fargs=(U_array, plot),
                    interval=1000 / fps,
                    blit=False,
                )

                if save:
                    anim.save("anim.gif", fps)
                    print("finished")

                plt.show()

                return anim

            else:
                fig, ax = plt.subplots(1, 1)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                # cont = plt.contour(X,Y,U_array[:,:,0],50)
                # ax.xaxis.tick_top()
                # ax.invert_yaxis()
                # ax.set_title('$\delta$={:.1f},$\gamma$={:.1f},L={:.0f},I={:.0f}'.format(d,g,L,I))
                # cb2 = fig.colorbar(cont,ax=ax,fraction=0.00001,pad=0)
                # txt=ax.text(0, 0, '',horizontalalignment='center',verticalalignment='top', fontsize=12,color='k')
                ############################
                # def init():
                #    txt.set_text('')
                #    cont.set_array([],[],[])
                #    return cont,txt
                ############################

                def animate_step(iter):
                    ax.clear()
                    txt = ax.text(
                        x_min - 0.1 * (x_max - x_min),
                        y_min - 0.1 * (y_max - y_min),
                        "t={:.2f}".format(t[iter]),
                        horizontalalignment="center",
                        verticalalignment="top",
                        fontsize=12,
                        color="k",
                    )
                    U_current = U_array[:, :, iter]
                    cont = plt.contour(X, Y, U_current, n_contours)
                    return cont

                anim = animation.FuncAnimation(
                    fig, animate_step, frames, interval=1000 / fps, blit=False
                )
                if save:
                    anim.save("animation.gif", fps)
                    print("finished")

                # plt.show()

                return anim

    def get_domain(self, axis1, axis2, domain=None):
        """
        a helper function used only internally by other methods
        it is for deciding what the relevant domain is
        for visualization purposes.

        Parameters
        ----------

        axis1, axis2 : int
            the coordinates you want to get the domain for, i.e. axis1=2 means
            we are plotting the 'y' coordiante on the first axis

        domain: None, or array of dimension [2, N_d]
            if None, we pull the domain from the potential.domain
            if ndarray, a manual domain of the form [ (xmin,ymin,...), (xmax, ymax,...) ]

        Returns
        -------
        if a 1D potential:

        x_min, x_max, [], []: float
            x_min, x_max is our relevant domain boundaries
            y_min ,y_max are empty placeholder lists

        if a 2D potential

        x_min, x_max, y_min, y_max : floats
            relevant domain boundaries for axis1 and axis2, respectively
        """
        if self.potential.N_dim == 1:
            if domain is None:
                x_min, x_max = self.potential.domain[:, axis1 - 1]
                y_min, y_max = [], []
            else:
                x_min, x_max = domain[0], domain[1]
                y_min, y_max = [], []
        if self.potential.N_dim >= 2:
            if domain is None:
                x_min, x_max = self.potential.domain[:, axis1 - 1]
                y_min, y_max = self.potential.domain[:, axis2 - 1]
            else:
                domain = np.asarray(domain)
                x_min, x_max = domain[:, axis1 - 1]
                y_min, y_max = domain[:, axis2 - 1]
        return x_min, x_max, y_min, y_max

    def lattice(
        self,
        t,
        resolution,
        x_min,
        x_max,
        y_min,
        y_max,
        axis1=1,
        axis2=2,
        slice_values=None,
    ):

        """
        Helper function used internally by the visualization code. Creates a
        1D or 2D lattice of coordiantes and calculates the potential at those coordinates

        Parameters
        ----------

        t: float
            time of interest for the potential energy

        resolution: int
            how many points we want to sample along each axis

        x_min, x_max, y_min, y_max: float
            min/max values of the horizontal and vertical axes, respectively

        axis1, axis2: int
            which coordinate we will consider to be 'x' and 'y' for the plot

        slice_values: ndarray of dimension [N_d,]
            there are the values we keep the other coordinates fixed at while sweeping through axis1 and axis2

        Returns
        -------

        if 1D:
            U: ndarray of dimension [resolution,]
                the potential at our test points, X

            X: ndarray of dimensiion [resolution]
                array of our test points
        if 2D:
            U: ndarray of dimension [resolution, resolution]
                the potential at our test points: X,Y

            X,Y: np arrays of dimension [resolution, resolution]
                X/Y gives the axis1/axis2 coordinates at each lattice point
                they are the results of an np.meshgrid operation
        """
        params = self.protocol.get_params(t)

        if self.potential.N_dim == 2:
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x, y)

            U = self.potential.potential(X, Y, params)

            return (U, X, Y)

        if self.potential.N_dim == 1:
            X = np.linspace(x_min, x_max, resolution)
            U = self.potential.potential(X, params)

            return (U, X)

        if self.potential.N_dim > 2:
            axis1 = axis1 - 1
            axis2 = axis2 - 1

            if slice_values is None:
                slice_values = [0] * self.potential.N_dim

            x1 = np.linspace(x_min, x_max, resolution)
            x2 = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x1, x2)

            slice_list = list(slice_values)
            slice_list[axis1] = X
            slice_list[axis2] = Y
            U = self.potential.potential(*slice_list, params)

            return (U, X, Y)
