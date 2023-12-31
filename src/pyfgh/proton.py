"""
===================================
===================================
              ______  _____  _   _  
              |  ___||  __ \| | | | 
 _ __   _   _ | |_   | |  \/| |_| | 
| '_ \ | | | ||  _|  | | __ |  _  | 
| |_) || |_| || |    | |_\ \| | | | 
| .__/  \__, |\_|     \____/\_| |_/ 
| |      __/ |                      
|_|     |___/                       
===================================
===================================

This module quantizes a proton on a molecule and utilizes the main FGH functionality

-----::::Constants::::-----
KCAL   : Hartree to kcal/mol   : 627.509        
WVNMBR : Hartree to wavenumber : 349.755*627.509

-----::::Classes included::::-----
qm_proton : This class is a subclass of the gto.Mole class of the PySCF library and facilitates the quantization of proton by solving the nuclear Time-Independent Schrodinger Equation (TDSE) using FGH. 

"""

import numpy as np
from pyscf import gto, cc
from pyscf.geomopt.berny_solver import optimize
from pyfgh.fgh import fgh_object

KCAL = 627.509
WVNMBR = 349.755*627.509

class qm_proton(gto.Mole):
        
    """
    fgh_object : This class provides a brief description of the purpose and functionality of the class. See pyscf.gto.Mole class to find the definition of inherited methods.

    ---Methods---
        spe         : sto-3g, HF single point energy (SPE) calculation
        spe_mp2     : ccpvtz, MP2 SPE calculation  
        spe_ccsd    : ccpvtz, CCSD SPE calculation
        geom_opt    : Geomerty optimizes the molecule
        grid_gen    : Generates the PES of the proton using sto-3g, HF SPEs
        grid_refine : Refines the PES of the proton using ccpvtz, MP2 OR ccpvtz, CCSD SPEs
        grid_solve  : Solves the TDSE of the proton on the PES using FGH
    
    ---Usage---
        Instantiate the class         : `instance = qm_proton()`
        Define the system             : `instance.build(unit = 'B', atom = mol_geom, basis = '321g', verbose = 0)
        Perform geometry optimization : `instance.geom_opt()`
        Generate the PES              : `instance.grid_gen(2, np.array([8,8,8]), np.array([0.2,0.2,0.2]))`
        Refine the PES                : `instance.grid_refine()`
        Solve the TDSE                : `instance.grid_solve()`
    """
    
    def spe(self):

        """
        spe method : This method performs a sto-3g, HF SPE calculation

        ---Params---
            en : float : The total energy using sto-3g, HF
        """

        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)
        self.en = self.HF().run().e_tot
        
    def spe_mp2(self):

        """
        spe_mp2 method : This method performs a ccpvtz, MP2 SPE calculation

        ---Params---
            mp2_en : float : The total energy using ccpvtz, MP2
        """

        self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
        self.mp2_en = self.HF().run().MP2().run().e_tot
        
    def spe_ccsd(self):

        """
        spe_ccsd method : This method performs a ccpvtz, CCSD SPE calculation

        ---Params---
            ccsd_en : float : The total energy using ccpvtz, CCSD
        """

        self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
        self.ccsd_en = cc.CCSD(self.HF().run()).run().e_tot

    def geom_opt(self):

        """
        geom_opt method : This method performs a sto-3g, HF geometry optimization

        ---Params---
            sys      : list    : List of atoms and coordinates describing the molecule
            en       : float   : The total energy using sto-3g, HF
            eqen     : float   : The optimized total energy using sto-3g, HF
            eqcoords : ndarray : The optimized coordinates
        """

        self.sys = optimize(self.HF(), maxsteps=100)._atom
        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)    
        self.en = self.HF().run().e_tot
        self.eqen = self.en 
        self.eqcoords = self.atom_coords(unit='Bohr')
        
    def grid_gen(self, h_atom: int, nx: np.ndarray, dx: np.ndarray):

        """
        grid_gen method : This method generates the PES of the proton using sto-3g, HF SPEs

        ---Args---
            h_atom : int        : Index of proton to be quantized
            nx     : 1D ndarray : Interger numbers of points on the grid in each dimension
            dx     : 1D ndarray : Distance between grid points in each dimension

        ---Params---
            sys        : list       : List of atoms and coordinates describing the molecule
            coords     : ndarray    : XYZ coordinates of each atom
            grid_ener  : 1D ndarray : The sto-3g, HF total energy at each grid point
            grid_coord : 2D ndarray : XYZ coordinates of each atom at each grid point
            temp       : 1D ndarray : Temporary array holding the XYZ coordiantes of the proton at each grid point
        """

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

        """
        grid_refine method : This method refines the PES of the proton using MP2 or CCSD SPEs with a ccpvtz basis set

        ---Args---
            method : str   : The post-Hartree-Fock method to be used to refine the grid: 'mp2' OR 'ccsd'
            enlim  : float : The energy cutoff for refining the grid

        ---Params---
            q          : list       : A grid point counter
            coords     : ndarray    : XYZ coordinates of each atom
        """
        
        self.method = method
        self.enlim = enlim
        self.q = 0
    
        for ener,coord in zip(self.grid_ener,self.grid_coord):
            if ener*WVNMBR<self.enlim:
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

        """
        grid_solve method : This method solves the TDSE of the proton on the PES using FGH

        ---Params---
            protsoln : tuple : The proton vibrational energies and wavefunctions calcualted useing FGH
        """

        self.fgh_solver = fgh_object(self.grid_ener, self.nx, self.dx, mass=1836)
        self.fgh_solver.fgh_fci_solve()
        self.protsoln = self.fgh_solver.solutions
                    
if __name__ == "__main__":

    mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"
    
    mol = qm_proton()
    mol.build(unit = 'B', atom = mol_geom, basis = '321g', verbose = 0)    
    mol.geom_opt()
    mol.grid_gen(2, np.array([8,8,8]), np.array([0.2,0.2,0.2]))
    # mol.grid_solve()
    # hf_soln = mol.protsoln
    mol.grid_refine()
    mol.grid_solve()
    mp2_soln = mol.protsoln
    print(mp2_soln[0][0]*KCAL,(mp2_soln[0][:10]-mp2_soln[0][0])*WVNMBR)
