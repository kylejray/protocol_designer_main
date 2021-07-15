import sys
import os
source_path = os.path.expanduser('~/source/')
sys.path.append(source_path)

from system_under_study.protocol_designer.potentials import Potential, odv, duffing_2D
import numpy as np


class TestOneDim:
    '''
    Tests the basic methods for an example one dimensional potenial
    '''

    def set_inputs(self):
        ''' this function sets up inputs, it is called inside other test functions '''
        triv_prot = odv.trivial_protocol()
        self.params = triv_prot.params[0]
        self.N = np.random.randint(10, size=2) + 1
        self.x_vec = np.linspace(-5, 5, self.N[0])
        self.x_array = np.random.rand(*self.N)
        self.x = np.random.rand()

    def test_force(self):
        ''' tests that the external_force method returns the appropriate sized arrays for single, vector, matrix inputs '''
        self.set_inputs()
        assert np.shape(odv.external_force(self.x, self.params)) == ()
        assert np.shape(odv.external_force(self.x_vec, self.params)) == (self.N[0],)
        assert all(np.shape(odv.external_force(self.x_array, self.params)) == self.N)

    def test_potential(self):
        ''' tests that the potential method returns the appropriate sized arrays for single, vector, matrix inputs '''
        self.set_inputs()
        assert np.shape(odv.potential(self.x, self.params)) == ()
        assert np.shape(odv.potential(self.x_vec, self.params)) == (self.N[0],)
        assert all(np.shape(odv.potential(self.x_array, self.params)) == self.N)

    def test_scaling(self):
        ''' tests that the scale attribute propoerly scales the external_force and potential'''
        self.set_inputs()
        args = self.x, self.params
        odv.scale = self.N[0]
        potential_value = odv.potential(*args)
        force_value = odv.external_force(*args)
        odv.scale = 1
        new_potential_value = odv.potential(*args)
        new_force_value = odv.external_force(*args)
        assert potential_value == self.N[0] * new_potential_value
        assert force_value == self.N[0] * new_force_value


class TestTwoDim:
    '''
    tests the basic methods for an example one dimensional potenial, same tests as in the test class TestOneDim
    '''

    def set_inputs(self):
        triv_prot = duffing_2D.trivial_protocol()
        self.params = triv_prot.params[..., 0]
        self.N = np.random.randint(10, size=2) + 1
        self.x_vec = np.transpose(np.linspace((-2, -2), (2, 2), self.N[0]))
        self.x_array = np.random.rand(duffing_2D.N_dim, *self.N)
        self.x = np.random.rand(duffing_2D.N_dim)

    def test_force(self):
        self.set_inputs()
        assert np.shape(duffing_2D.external_force(*self.x, self.params)) == (duffing_2D.N_dim,)
        assert np.shape(duffing_2D.external_force(*self.x_vec, self.params)) == (duffing_2D.N_dim, self.N[0])
        assert np.shape(duffing_2D.external_force(*self.x_array, self.params)) == np.shape(self.x_array)

    def test_potential(self):
        self.set_inputs()
        assert np.shape(duffing_2D.potential(*self.x, self.params)) == ()
        assert np.shape(duffing_2D.potential(*self.x_vec, self.params)) == (self.N[0],)
        assert np.shape(duffing_2D.potential(*self.x_array, self.params)) == np.shape(self.x_array[0])