import unittest
import numpy as np
from pyfgh.fgh import fgh_hardcode_1d, fgh_hardcode_2d, fgh_hardcode_3d

hartree2kcal = 627.509
kcal2wavenumber = 349.755

class TestOneDimensional(unittest.TestCase):
    def test_fgh_hardcode_1d(self):

        print("\nStart one_dimensional test\n")

        print("FGH hardcode HO test")
        self.potential = np.array([0.5*(i/2-16)**2 for i in range(64)])
        self.nx = 64
        self.dx = 0.5
        self.mass = 1
        self.energies, self.wavefunctions = fgh_hardcode_1d(self.potential, self.nx, self.dx, self.mass)

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-(0.5+i))<(1/(hartree2kcal*kcal2wavenumber)))

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

        print("\nFGH hardcode PIB test")
        self.potential = np.array([0 for _ in range(512)])
        self.nx = 64
        self.dx = 32/self.nx
        self.mass = 1
        self.energies = fgh_hardcode_1d(self.potential, self.nx, self.dx, self.mass)[0][:5]
        for i, en in enumerate(self.energies):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-(((np.pi*(i+1))**2)/(2*32**2)))<(1000/(hartree2kcal*kcal2wavenumber)))

class TestTwoDimensional(unittest.TestCase):
    def test_fgh_hardcode_2d(self):

        print("\nStart two_dimensional test\n")
    
        print("FGH hardcode HO test")
        self.potential = np.array([0.5*(i/4-8)**2 + 0.5*(j/4-8)**2 for j in range(64) for i in range(64)])
        self.nx = np.array([64,64])
        self.dx = np.array([1/4,1/4])
        self.mass = 1

        self.energies, self.wavefunctions = fgh_hardcode_2d(self.potential.reshape(self.nx[0],self.nx[1]), self.nx[0], self.dx[0], self.mass)
        bench = [1.0, 2.0, 2.0, 3.0, 3.0]

        for i, en in enumerate(self.energies[:5]):
            print(f"energy level {i} has energy {en} hartree")
            self.assertTrue(np.abs(en-bench[i])<(1/(hartree2kcal*kcal2wavenumber)))

        for i, wvfn in enumerate(self.wavefunctions[:5]):
            norm = np.sum(wvfn**2)
            print(f"{i}th state wavefunction normalization check: {norm}")
            self.assertTrue(np.abs(norm-1)<10**-12)

class TestThreeDimensional(unittest.TestCase):
    def test_fgh_hardcode_3d(self):

        print("\nStart Three Dimensional Test\n")
    
        print("FGH hardcode HO test")
        self.potential = np.array([0.5*(i-8)**2 + 0.5*(j-8)**2 + 0.5*(k-8)**2 for k in range(16) for j in range(16) for i in range(16)])
        self.nx = np.array([16,16,16])
        self.dx = np.array([1,1,1])
        self.mass = 1

        self.energies, self.wavefunctions = fgh_hardcode_3d(self.potential.reshape(self.nx[0],self.nx[1],self.nx[2]), self.nx[0], self.dx[0], self.mass)
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
