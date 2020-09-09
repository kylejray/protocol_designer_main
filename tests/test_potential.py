import sys
sys.path.append("..")
from protocol_designer.potentials import Potential, odv, duffing_2D
import numpy as np


class TestOneDim:

    def set_inputs(self):
        self.params = (1, 0)
        self.N = np.random.randint(10, size=2)
        self.x_vec = np.linspace(-5, 5, self.N[0])
        self.x_array = np.ones(self.N)
        self.x = 1

    def test_scaling(self):
        self.set_inputs()
        args = self.x, self.params
        odv.scale = 2
        value = odv.potential(*args)
        odv.scale = 1
        new_value = odv.potential(*args)
        assert value == 2*new_value

    def test_force(self):
        self.set_inputs()
        assert np.shape(odv.force(self.x, self.params)) == ()
        assert np.shape(odv.force(self.x_vec, self.params)) == (self.N[0],)
        assert all(np.shape(odv.force(self.x_array, self.params)) == self.N)
    
    def test_potential(self):
        self.set_inputs()
        


