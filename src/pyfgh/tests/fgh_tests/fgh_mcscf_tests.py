import unittest
import numpy as np
from pyfgh.fgh import fgh_mcscf

hartree2kcal = 627.509
kcal2wavenumber = 349.755

class TestOneDimensional(unittest.TestCase):
    def test_fgh_mcscf_1d(self):

        print("\nStart One Dimensional test\n")

        self.potential = np.array([0.5*(i/64-8)**2 for i in range(1024)])
        self.nx = np.array([1024])
        self.dx = np.array([1/64])
        self.mass = 1
        self.scf_iter = 2
        self.basis_size = 5

        self.energies, self.wavefunctions = fgh_mcscf(self.potential, self.nx, self.dx, self.mass, self.scf_iter, self.basis_size)
        bench = [0.5, 1.5, 2.5, 3.5, 4.5]

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-bench[i])<0.01)

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

class TestTwoDimensional(unittest.TestCase):
    def test_fgh_mcscf_2d(self):

        print("\nStart Two Dimensional test\n")

        self.potential = np.array([0.5*(i/16-8)**2 + 0.5*(j/16-8)**2 for j in range(256) for i in range(256)])
        self.nx = np.array([256,256])
        self.dx = np.array([1/16,1/16])
        self.mass = 1
        self.scf_iter = 2
        self.basis_size = 5

        self.energies, self.wavefunctions = fgh_mcscf(self.potential, self.nx, self.dx, self.mass, self.scf_iter, self.basis_size)
        bench = [1.0, 2.0, 2.0, 3.0, 3.0]

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-bench[i])<0.01)

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

class TestThreeDimensional(unittest.TestCase):
    def test_fgh_mcscf_3d(self):

        print("\nStart Three Dimensional Test\n")
    
        self.potential = np.array([0.5*(i/8-8)**2 + 0.5*(j/8-8)**2 + 0.5*(k/8-8)**2 for k in range(128) for j in range(128) for i in range(128)])
        self.nx = np.array([128,128,128])
        self.dx = np.array([1/8,1/8,1/8])
        self.mass = 1
        self.scf_iter = 2
        self.basis_size = 2

        self.energies, self.wavefunctions = fgh_mcscf(self.potential, self.nx, self.dx, self.mass, self.scf_iter, self.basis_size)
        bench = [1.5, 2.5, 2.5, 2.5, 3.5]

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-bench[i])<0.03)

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

if __name__ == '__main__':
    unittest.main()
