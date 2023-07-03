#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:33:42 2023

@author: maximsecor
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft, cc
from pyscf.geomopt.berny_solver import optimize
from pyscf.hessian import thermo
# from pyfgh.fgh import fgh_object

hartree2kcal = 627.509
kcal2wavenumber = 349.755
hartree2wavenumber = hartree2kcal*kcal2wavenumber

#%%

class normal_modes(gto.Mole):
    
    def spe(self):
        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)
        self.en = self.HF().run().e_tot
        
    def geom_opt(self):
        self.sys = optimize(self.HF(), maxsteps=100)._atom
        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)    
        self.en = self.HF().run().e_tot
        self.eqen = self.en 
        self.eqcoords = self.atom_coords(unit='Bohr')
    
    def hessian(self):
        self.build(atom = self.sys, basis = 'sto3g', verbose = 0)   
        self.hessian = self.HF().run().Hessian().kernel()
        self.freq_info = thermo.harmonic_analysis(self.build(atom = self.sys, basis = 'sto3g', verbose = 0), self.hessian)

    def grid_gen(self, nm_idx):
        self.nm_idx = nm_idx
        self.nm_disp = self.freq_info['norm_mode'][self.nm_idx]
        
if __name__ == "__main__":

    mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"
    
    mol = normal_modes()
    mol.build(unit = 'B', atom = mol_geom, basis = '321g', verbose = 0)    
    mol.geom_opt()
    mol.hessian()
    print(mol.freq_info['freq_wavenumber'])
    mol.grid_gen(2)
    print(np.sum((mol.nm_disp*)))

#%%

mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"

mol = gto.Mole()
mol.build(unit = 'B', atom = mol_geom, basis = '321g', verbose = 0)    

mf = mol.RHF().run()
hessian = mf.Hessian().kernel()
freq_info = thermo.harmonic_analysis(mol, hessian)

print(freq_info)

#%%

print(mf.mol)
