#%%
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft, cc
from pyscf.geomopt.berny_solver import optimize
from pyfgh.fgh import fgh_object

hartree2kcal = 627.509
kcal2wavenumber = 349.755
hartree2wavenumber = hartree2kcal*kcal2wavenumber

class qm_proton(gto.Mole):
    
    def spe(self):
        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)
        self.en = self.HF().run().e_tot
        
    def spe_mp2(self):
        self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
        self.mp2_en = self.HF().run().MP2().run().e_tot
        
    def spe_ccsd(self):
        self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
        self.ccsd_en = cc.CCSD(self.HF().run()).run().e_tot

    def geom_opt(self):
        self.sys = optimize(self.HF(), maxsteps=100)._atom
        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)    
        self.en = self.HF().run().e_tot
        self.eqen = self.en 
        self.eqcoords = self.atom_coords(unit='Bohr')
        
    def grid_gen(self, h_atom: int, nx: np.ndarray, dx: np.ndarray):

        self.h_atom = h_atom        
        self.nx = nx
        self.dx = dx

        self.sys = self._atom
        self.coords = self.atom_coords(unit='Bohr')
        self.coords[h_atom] -= (self.nx//2)*self.dx
        
        self.grid_ener = []
        self.grid_coord = []
        
        for i in range(self.nx[0]):
            print(i)
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):

                    self.temp = self.coords[self.h_atom] + np.array([i*self.dx[0],j*self.dx[1],k*self.dx[2]])                    
                    for l,dist in enumerate(self.temp): 
                        self.sys[h_atom][1][l] = dist
                    self.spe()
                    
                    self.grid_ener.append(self.en)
                    self.grid_coord.append(self.sys)
                    
        self.grid_ener = np.array(self.grid_ener) - np.min(self.grid_ener)
        self.grid_coord = np.array(self.grid_coord)
        
    def grid_refine(self, method='mp2', enlim=5*10**3):
        
        self.method = method
        self.enlim = enlim
        self.q = 0
    
        for ener,coord in zip(self.grid_ener,self.grid_coord):
            if ener*hartree2wavenumber<self.enlim:
                self.sys = coord
                if self.method == 'mp2':
                    print(self.q)
                    self.spe_mp2()
                    self.grid_ener[self.q] = self.mp2_en
                if self.method == 'ccsd':
                    self.spe_ccsd()
                    self.grid_ener[self.q] = self.ccsd_en
            self.q += 1

        self.grid_ener -= np.min(self.grid_ener)

    def grid_solve(self):
        self.fgh_solver = fgh_object(self.grid_ener, self.nx, self.dx, mass=1836)
        self.fgh_solver.fgh_fci_solve()
        self.solutions = self.fgh_solver.get_solutions()
                    
if __name__ == "__main__":

    mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"
    
    mol = qm_proton()
    mol.build(unit = 'B', atom = mol_geom, basis = '321g', verbose = 0)    
    mol.geom_opt()
    mol.grid_gen(2, np.array([8,8,8]), np.array([0.2,0.2,0.2]))
    # mol.grid_solve()
    # hf_soln = mol.solutions
    mol.grid_refine()
    mol.grid_solve()
    mp2_soln = mol.solutions
    print(mp2_soln[0][0]*hartree2kcal,(mp2_soln[0][:10]-mp2_soln[0][0])*hartree2wavenumber)
