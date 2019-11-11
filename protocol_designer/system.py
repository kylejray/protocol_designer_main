import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from .protocol import Protocol
from .potentials import Potential

# expected form for data for coordinates is (N,D,2)
# N is number of trials, D is dimension, 2 is for position(0) vs velocity(1)
# examples: coords[204,1,1] would be v_y of the 204th trial.


class System:

    def __init__(self, Protocol, Potential):
        self.Protocol = Protocol
        self.Potential = Potential
        self.scale = 1
        msg = "the number of protocol parameters must match the potential"
        assert len(Protocol.get_params(Protocol.t_i)) == Potential.N_params, msg

    def scale_potential(self, scaling):
        self.scale = scaling

    def get_energy(self, coords, t):
        v = coords[:, :, 1]
        U = self.get_potential(coords, t)
        T = np.sum(np.square(v), axis=1)
        return(U+T)

    def get_potential(self, coords, t):
        params = self.Protocol.get_params(t)
        positions = np.transpose(coords[:, :, 0])

        return(self.scale*self.Potential.potential(*positions, params))

    def get_external_force(self, coords, t):
        params = self.Protocol.get_params(t)
        positions = np.transpose(coords[:, :, 0])
        F = np.zeros((len(coords[:, 0, 0]), self.Potential.N_dim))
        force = self.Potential.external_force(*positions, params)
        for i in range(0, self.Potential.N_dim):
            F[:, i] = force[i]

        return(self.scale*F)

    def lattice(self, t, resolution, x_min, x_max, y_min, y_max, axis1=1, axis2=2, slice_values=None):
        params = self.Protocol.get_params(t)

        if self.Potential.N_dim == 2:
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x, y)

            U = self.scale*self.Potential.potential(X, Y, params)

            return(U, X, Y)

        if self.Potential.N_dim == 1:
            X = np.linspace(x_min, x_max, resolution)
            U = self.scale*self.Potential.potential(X, params)

            return(U, X)

        if self.Potential.N_dim > 2:
            axis1 = axis1-1
            axis2 = axis2-1

            if slice_values is None:
                slice_values = [0] * self.Potential.N_dim

            x1 = np.linspace(x_min, x_max, resolution)
            x2 = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x1, x2)

            slice_list = list(slice_values)
            slice_list[axis1] = X
            slice_list[axis2] = Y
            U = self.scale*self.Potential.potential(*slice_list, params)

            return(U, X, Y)

    def show_potential(self, t, resolution=100, surface=False, x_min=-2, x_max=2, y_min=-2, y_max=2, contours=50, axis1=1, axis2=2, slice_values=None):
        if self.Potential.N_dim >= 2:
            U, X, Y = self.lattice(t, resolution, x_min, x_max, y_min, y_max, axis1, axis2, slice_values)

            if surface is False:
                fig, ax = plt.subplots()
                CS = ax.contour(X, Y, U, contours)
                # ax.clabel(CS, inline=1, fontsize=10)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.text(x_min-.1*(x_max-x_min), y_min-.1*(y_max-y_min), 't={:.2f}'.format(t), horizontalalignment='right', verticalalignment='top', fontsize=12, color='k')

                plt.show()

            if surface is True:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(X, Y, U)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.text(-.3, -.3, 0, 't={:.2f}'.format(t), horizontalalignment='center', verticalalignment='top', fontsize=12, color='k')

                plt.show()
        if self.Potential.N_dim == 1:
            U, X = self.lattice(t, resolution, x_min, x_max, y_min, y_max)
            fig, ax = plt.subplots()
            ax.plot(X, U)
            ax.set_xlabel('x')
            ax.text(x_min-.1*(x_max-x_min), y_min-.1*(y_max-y_min), 't={:.2f}'.format(t), horizontalalignment='right', verticalalignment='top', fontsize=12, color='k')

    def animate_protocol(self, mesh=40, fps=10, frames=50, surface=False, save=False, x_min=-2, x_max=2, y_min=-2, y_max=2, axis1=0, axis2=1, slice_values=None, n_contours=50):
        t_i = self.Protocol.t_i
        t_f = self.Protocol.t_f
        t = np.linspace(t_i, t_f, frames)

        if self.Potential.N_dim == 1:
            U_array = np.zeros((mesh, frames))
            U_array[:, 0], X = self.lattice(t_i, mesh, x_min, x_max, y_min, y_max)

            for idx, item in enumerate(t):
                U_array[:, idx] = self.lattice(item, mesh, x_min, x_max, y_min, y_max)[0]

            fig, ax = plt.subplots()
            line, = ax.plot([], [])
            ax.set_xlabel('x')
            ax.set_ylabel('U')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(U_array.min(), U_array.max())

            # txt=ax.text(-.1, -.3, '', horizontalalignment='center', verticalalignment='top', fontsize=12,color='k')
            def init():
                line.set_data([], [])
                # txt.set_text('')

                return line, #txt

            def update_plot(frame_number):
                U_current = U_array[:, frame_number]
                line.set_data(X, U_current)
                # txt=ax.text(-.1, -.3, 't={:.2f}'.format(t[frame_number]), horizontalalignment='center', verticalalignment='top', fontsize=12,color='k')
                return line,#txt

            anim = animation.FuncAnimation(fig, update_plot, init_func=init, frames=frames, interval=1000/fps, blit=True)

            if save:
                anim.save('animation.gif', fps)
                print('finished')

            return(anim)

        if self.Potential.N_dim >= 2:
            U_array = np.zeros((mesh, mesh, frames))
            # T_array=np.zeros((mesh,mesh,frames))
            #
            # for idx,item in enumerate(t):
            #    T_array[:,:,idx]=item

            U_array[:, :, 0], X, Y = self.lattice(t_i, mesh, x_min, x_max, y_min, y_max)
            for idx, item in enumerate(t):
                U_array[:, :, idx] = self.lattice(item, mesh, x_min, x_max, y_min, y_max)[0]

            if surface:
                def update_plot_2D(frame_number, U_array, plot):
                    plot[0].remove()
                    plot[0] = ax.plot_wireframe(X, Y, U_array[:, :, frame_number], cmap="magma")

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                plot = [ax.plot_surface(X, Y, U_array[:, :, 0], color='0.75', rstride=1, cstride=1)]
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlim(U_array.min(), U_array.max())

                anim = animation.FuncAnimation(fig, update_plot_2D, frames, fargs=(U_array, plot), interval=1000/fps, blit=False)

                if save:
                    anim.save('anim.gif', fps)
                    print('finished')

                plt.show()

                return(anim)

            else:
                fig, ax = plt.subplots(1, 1)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
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
                    txt = ax.text(x_min-.1*(x_max-x_min), y_min-.1*(y_max-y_min), 't={:.2f}'.format(t[iter]), horizontalalignment='center', verticalalignment='top', fontsize=12, color='k')
                    U_current = U_array[:, :, iter]
                    cont = plt.contour(X, Y, U_current, n_contours)
                    return cont

                anim = animation.FuncAnimation(fig, animate_step, frames, interval=1000/fps, blit=False)
                if save:
                    anim.save('animation.gif', fps)
                    print('finished')
                
                # plt.show()

                return(anim)
