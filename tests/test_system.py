import sys
import os
source_path = os.path.expanduser('~/source/')
sys.path.append(source_path)

from system_under_study.bit_flip_protocols import fredkin_3D, asym_bf_1D

import numpy as np


class Test_3D_sim_calls:
    '''
    Tests the basic methods for an example system in 3D
    '''

    def set_inputs(self):
        ''' this function sets up inputs, it is called inside other test functions '''
        self.system = fredkin_3D
        self.N_c = np.random.randint(10)+1
        self.N_d = self.system.potential.N_dim
        domain = self.system.potential.domain
        state = np.zeros((self.N_c, self.N_d, 2))
        state[..., 1] = np.random.uniform(0, 1, (self.N_c, self.N_d))
        state[..., 0] = np.random.uniform(domain[0], domain[1], (self.N_c, self.N_d))
        self.coords = state
        self.x = np.random.rand(self.N_d, 2)
        self.time = np.random.uniform(self.system.protocol.t_i, self.system.protocol.t_f)

    def test_force(self):
        self.set_inputs()
        assert np.shape(self.system.get_external_force(self.coords, self.time)) == (self.N_c, self.N_d)
        assert np.shape(self.system.get_external_force(self.x, self.time)) == (self.N_d,)

    def test_potential(self):
        self.set_inputs()
        assert np.shape(self.system.get_potential(self.coords, self.time)) == (self.N_c,)
        assert np.shape(self.system.get_potential(self.x, self.time)) == ()

    def test_energy(self):
        self.set_inputs()
        assert np.shape(self.system.get_kinetic_energy(self.coords, self.time)) == (self.N_c,)
        assert np.shape(self.system.get_energy(self.coords, self.time)) == (self.N_c,)
        assert np.shape(self.system.get_kinetic_energy(self.x, self.time)) == ()
        assert np.shape(self.system.get_energy(self.x, self.time)) == ()

    def test_eq_state(self):
        self.system = fredkin_3D
        sample_mass = np.random.uniform(.1, 10)
        sample_beta = np.random.uniform(.1, 10)
        N = np.random.randint(1000, 10000)
        eq_state = self.system.eq_state(N, beta=sample_beta, M=sample_mass)
        v_square = np.square(eq_state[..., 1])
        sample_sigma = np.std(v_square)/np.sqrt(N)
        assert np.isclose(np.mean(v_square), 1/(sample_mass*sample_beta), atol=3*sample_sigma)

