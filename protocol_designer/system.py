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
        self.has_velocity = True
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
    
    def get_kinetic_energy(self, coords, mass):
        '''
        gives the kinetic energy of a set of coordinates where [..., 1] are the velocities
        
        Parameters
        ----------
        coords: ndarray of dimension [..., 2], conventionally [N_c, N_d, 2]
            array of N_c sets of coordinates in N_d dimensions
        mass: float or ndarray of length N_c
            the masses of the particles
        
        Returns
        -------

        T: ndarray
            The  N_c kinetic energies of the set of coordinates

        if has_velocity is False, returns a 0
        '''
        if self.has_velocity:
            velocity = coords[..., 1]
            velocity_squared = np.sum(np.square(velocity), axis=np.size(np.shape(velocity))-1)
            T = np.multiply(.5 * mass, velocity_squared)
            return T
        else:
            return 0

    def get_energy(self, coords, t, mass=1):
        """
        Calculate the energy of a particle at location coords at time t

        Parameters
        ----------

        coords: ndarray of dimensions [N_c, N_d, 2]
            array of N_c sets of coordinates in N_d dimensions

        t: float or int
            time at which you want to evaluate the energy
        mass: float
            the mass of the particles, generally it makes sense to scale to the mass is 1

        Returns
        -------

        U+T : ndarray of dimension [N_c,]

        """
        U = self.get_potential(coords, t)
        if self.has_velocity:
            T = self.get_kinetic_energy(coords, mass)
            return T+U
        else:
            return U

    def get_positions(self, coords):
        '''
        simple helper function that returns only the positions from a set of coordinates
        '''
        if self.has_velocity:
            return np.transpose(coords[..., 0])
        else:
            return np.transpose(coords)

    def get_potential(self, coords, t):
        """
        Calculate the potential energy of a particle at location coords at time t

        Parameters
        ----------

        coords: ndarray of dimensions [N_c, N_d, 2]
            array of N_c sets of coordinates in N_d dimensions

        t: float or int
            time at which you want to evaluate the potential energy

        Returns
        -------

        U : ndarray of dimension [N_c,]

        """
        params = self.protocol.get_params(t)
        positions = self.get_positions(coords)

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
        positions = self.get_positions(coords)

        if self.potential.N_dim == 1:
            force = self.potential.external_force(positions, params)
        else:
            force = self.potential.external_force(*positions, params)

        return np.transpose(force)

    def eq_state(self, Nsample,  t=None, resolution=500, beta=1, M=1, manual_domain=None, axes=None, slice_vals=None):
        '''
        function still in development, docstring will come later.
        generates Nsample coordinates from an equilibrium distribution at
        time t.

        '''
        if self.potential.N_dim >= 3 and (axes is None or len(axes) > 3):
            if resolution > 100:
                print('using a lower resolution for searching a space in >2 dimensions')
                if self.potential.N_dim == 3:
                    resolution = 100
                if self.potential.N_dim >= 4:
                    resolution = 20
                print('new resolution is {}'.format(resolution))

        NT = Nsample
        state = np.zeros((max(100, int(2*NT)), self.potential.N_dim, 2))

        def get_prob(self, state):
            if self.has_velocity is False:
                state = state[:, :, 0]
            E_curr = self.get_energy(state, t)
            Delta_U = E_curr-U0
            return np.exp(-Delta_U)

        if t is None:
            t = self.protocol.t_i

        U, X = self.lattice(t, resolution, axes=axes, slice_values=slice_vals, manual_domain=manual_domain)
        mins = []
        maxes = []
        for item in X:
            mins.append(np.min(item))
            maxes.append(np.max(item))

        U0 = np.min(U)
        i = 0

        if axes is None:
            axes = [_ for _ in range(1, self.potential.N_dim + 1)]
        axes = np.array(axes) - 1

        while i < Nsample:
            test_coords = np.zeros((NT, self.potential.N_dim, 2))
            if slice_vals is not None:
                test_state[:, :, 0] = slice_vals

            test_coords[:, axes, 0] = np.random.uniform(mins, maxes, (NT, len(axes)))

            p = get_prob(self, test_coords)

            decide = np.random.uniform(0, 1, NT)
            n_sucesses = np.sum(p > decide)
            if i == 0:
                ratio = max(n_sucesses/NT, .05)

            state[i:i+n_sucesses, :, :] = test_coords[p > decide, :, :]
            i = i + n_sucesses
            print("found {} samples out of {}".format(i, Nsample), end="\r")

            NT = max(int((Nsample-i)/ratio), 100)

        state = state[0:Nsample, :, :]

        if self.has_velocity:
            state[:, :, 1] = np.random.normal(0, np.sqrt(1/(M*beta)), (Nsample, self.potential.N_dim))
        else:
            return state[:, :, 0]

        return state

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
        if self.potential.N_dim >= 2 and axis2 is not None:
            U, X_mesh = self.lattice(t, resolution, axes=(axis1, axis2), slice_values=slice_values, manual_domain=manual_domain)
            X = X_mesh[0]
            Y = X_mesh[1]

            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)
            if surface is False:
                fig, ax = plt.subplots()
                CS = ax.contour(X, Y, U, contours)
                # ax.clabel(CS, inline=1, fontsize=10)
                ax.set_xlabel('x{}'.format(axis1))
                ax.set_ylabel('x{}'.format(axis2))
                ax.set_title("t={:.2f}".format(t))

                plt.show()

            if surface is True:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_wireframe(X, Y, U)
                ax.set_xlabel('x{}'.format(axis1))
                ax.set_ylabel('x{}'.format(axis2))
                ax.set_title("t={:.2f}".format(t))

                plt.show()

        if self.potential.N_dim == 1 or axis2 is None:
            U, X_mesh = self.lattice(t, resolution, axes=[axis1], slice_values=slice_values, manual_domain=manual_domain)
            X = X_mesh[0]
            Y = U[0]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)
            fig, ax = plt.subplots()
            ax.plot(X, Y)
            ax.set_xlabel("x")
            ax.set_xlabel("U")
            ax.set_title("t={:.2f}".format(t))

        return fig, ax

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
        t_i = self.protocol.t_i
        t_f = self.protocol.t_f
        t = np.linspace(t_i, t_f, frames)

        if self.potential.N_dim == 1:

            U_array = np.zeros((mesh, frames))
            U_array[:, 0], X = self.lattice(t_i, mesh, axes=[axis1], slice_values=slice_values, manual_domain=manual_domain)

            for idx, item in enumerate(t):
                U_array[:, idx] = self.lattice(item, mesh, axes=[axis1], slice_values=slice_values, manual_domain=manual_domain)[0]

            fig, ax = plt.subplots()
            (line,) = ax.plot([], [])
            ax.set_xlabel("x")
            ax.set_ylabel("U")
            x_min, x_max = np.min(X), np.max(X)
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
            U_array[:, :, 0], X_mesh = self.lattice(t_i, mesh, axes=(axis1, axis2), slice_values=slice_values, manual_domain=manual_domain)
            X = X_mesh[0]
            Y = X_mesh[1]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)
            for idx, item in enumerate(t):
                U_array[:, :, idx] = self.lattice(item, mesh, axes=(axis1, axis2), slice_values=slice_values, manual_domain=manual_domain)[0]

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
                ax.set_xlabel("x{}".format(axis1))
                ax.set_ylabel("x{}".format(axis2))
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

    def lattice(self, t, resolution, axes=None, slice_values=None, manual_domain=None):

        """
        Helper function used internally by other pieces of code. Creates a
        lattice of coordiantes and calculates the potential at those coordinates
        at the given time

        Parameters
        ----------

            t: float
                time of interest for the potential energy

            resolution: int
                how many points we want to sample along each axis

            axes: list of dimension <N_dim or None
                which coordinates we want to sweep through for the potential
                if None, will do all coordiantes


            slice_values: ndarray of dimension [N_d,] or None
                these are the values we keep the other coordinates fixed at while sweeping through the selected axes
                if None, sets all non-swept coordiantes to 0

            manual_domain: ndarray of dimension [2, N_d]
                lists a manual domain if you want to sweep a different domain than the potentials default values

        Returns
        -------

        U: ndarray of dimension [resolution, resolution, ...]
            the potential at our test points, X_mesh has as many dimensions as len(axes) or N_dim if axes is None

        X: ndarray of dimension [len(axes) or N_dim, shape(U)]
            array of our test points
        """
        params = self.protocol.get_params(t)

        if self.potential.N_dim == 1:
            lims = self.potential.domain
            if manual_domain is not None:
                lims = manual_domain
            x_vec = np.transpose(np.linspace(*lims, resolution))
            U = self.potential.potential(x_vec, params)
            return U, x_vec

        else:
            if manual_domain is None:
                lims = self.potential.domain

                if axes is None:
                    axes = [_ for _ in range(1, self.potential.N_dim+1)]

                axes = np.array(axes) - 1
                lims = lims[:, axes]

            else:
                assert axes is not None, "when using a manual domain, must include the 'which_axes' keyword"
                manual_domain = np.array(manual_domain)
                axes = np.array(axes) - 1
                lims = manual_domain[:, axes]

        x_vec = np.transpose(np.linspace(*lims, resolution))
        X_mesh = np.meshgrid(*x_vec)

        if slice_values is None:
            slice_values = [0] * self.potential.N_dim

        slice_list = list(slice_values)

        for i, item in enumerate(axes):
            slice_list[item] = X_mesh[i]

        U = self.potential.potential(*slice_list, params)

        return (U, X_mesh)
