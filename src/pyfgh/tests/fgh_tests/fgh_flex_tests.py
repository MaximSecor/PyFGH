import unittest
import numpy as np
from pyfgh.fgh import fgh_flex

hartree2kcal = 627.509
kcal2wavenumber = 349.755

class TestOneDimensional(unittest.TestCase):
    def test_fgh_flex_1d(self):

        print("\nStart One Dimensional Test\n")

        print("FGH flex HO test")
        self.potential = np.array([0.5*(i/2-16)**2 for i in range(64)])
        self.nx = np.array([64])
        self.dx = np.array([0.5])
        self.mass = 1
        self.energies, self.wavefunctions = fgh_flex(self.potential, self.nx, self.dx, self.mass)

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-(0.5+i))<(1/(hartree2kcal*kcal2wavenumber)))

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

class TestTwoDimensional(unittest.TestCase):
    def test_fgh_flex_2d(self):

        print("\nStart Two Dimensional Test\n")

        print("FGH flex HO test")
        self.potential = np.array([0.5*(i/2-8)**2 + 0.5*(j/2-8)**2 for j in range(32) for i in range(32)])
        self.nx = np.array([32, 32])
        self.dx = np.array([0.5, 0.5])
        self.mass = 1
        self.energies, self.wavefunctions = fgh_flex(self.potential, self.nx, self.dx, self.mass)

        bench = [1.0, 2.0, 2.0, 3.0, 3.0]
        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-bench[i])<(1/(hartree2kcal*kcal2wavenumber)))

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)


class TestThreeDimensional(unittest.TestCase):
    def test_fgh_flex_3d(self):

        print("\nStart Three Dimensional Test\n")

        print("FGH flex HO test")
        self.potential = np.array([0.5*(i-8)**2 + 0.5*(j-8)**2 + 0.5*(k-8)**2 for k in range(16) for j in range(16) for i in range(16)])
        self.nx = np.array([16, 16, 16])
        self.dx = np.array([1, 1, 1])
        self.mass = 1

        self.energies, self.wavefunctions = fgh_flex(self.potential, self.nx, self.dx, self.mass)
        bench = [1.5, 2.5, 2.5, 2.5, 3.5]

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-bench[i])<0.01)

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

if __name__ == '__main__':
    unittest.main()
