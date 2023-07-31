#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:33:42 2023

@author: maximsecor
"""
import numpy as np
from pyscf import gto, cc
from pyscf.geomopt.berny_solver import optimize
from pyscf.hessian import thermo
from pyfgh.fgh import fgh_object
import matplotlib.pyplot as plt

KCAL = 627.51
WVNMBR = 627.51*349.76

class normal_modes(gto.Mole):
    
    def spe(self):
    
        # self.build(atom = self.sys, basis = 'augccpv5z', verbose = 0)
        self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)
        self.en = self.HF().run().e_tot

    def spe_ccsd(self):

        self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)
        self.ccsd_en = cc.CCSD(self.HF().run()).run().e_tot

        # et = mycc.ccsd_t()
        # print('CCSD(T) total energy', mycc.e_tot + et)
        
    def geom_opt(self):
        
        self.sys = optimize(self.HF(), maxsteps=100)._atom
        self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)    
        self.en = self.HF().run().e_tot
        self.eqen = self.en 
        self.eqcoords = self.atom_coords(unit='Bohr')

        # self.sys = optimize(cc.CCSD(self.HF().run()), maxsteps=100)._atom
        # self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)    
        # self.en = self.HF().run().e_tot
        # self.eqen = self.en 
        # self.eqcoords = self.atom_coords(unit='Bohr')
    
    def hessian(self):
        
        self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)   
        self.hessian = self.HF().run().Hessian().kernel()
        self.freq_info = thermo.harmonic_analysis(self.build(atom = self.sys, basis = 'sto-3g', verbose = 0), self.hessian)

        # self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)   
        # self.hessian = self.HF().run().Hessian().kernel()
        # self.freq_info = thermo.harmonic_analysis(self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0), self.hessian)

    def calculate_moments(self, method = 'hf'):

        self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)   
        nao = self.nao

        dip = self.intor('int1e_r').reshape(3,nao,nao)
        quad = self.intor('int1e_rr').reshape(3,3,nao,nao)
        octa = self.intor('int1e_rrr').reshape(3,3,3,nao,nao)
        hexa = self.intor('int1e_rrrr').reshape(3,3,3,3,nao,nao)
        
        if method == 'hf':
            self.dm = self.HF().run().make_rdm1()

        if method == 'ccsd':
            mf = self.HF().run()
            mycc = cc.CCSD(mf).run()
            dm1 = mycc.make_rdm1()
            self.dm = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1, mf.mo_coeff.conj())
        
        charges = self.atom_charges()
        coords  = self.atom_coords()
        
        self.moments_info = {
            'el_dip'    : -np.einsum('xij,ji->x', dip, self.dm).real,
            'nucl_dip'  : np.einsum('i,ix->x', charges, coords),
            'el_quad'   : -np.einsum('xyij,ji->xy', quad, self.dm).real,
            'nucl_quad' : np.einsum('xi,xj,x->ij', coords, coords, charges),
            'el_octa'   : -np.einsum('xyzij,ji->xyz', octa, self.dm).real,
            'nucl_octa' : np.einsum('xi,xj,xk,x->ijk', coords, coords, coords, charges),
            'el_hexa'   : -np.einsum('xyzaij,ji->xyza', hexa, self.dm).real,
            'nucl_hexa' : np.einsum('xi,xj,xk,xl,x->ijkl', coords, coords, coords, coords, charges)
            }
        
        self.moments_info['mol_dip'] = self.moments_info['el_dip'] + self.moments_info['nucl_dip']
        self.moments_info['mol_quad'] = self.moments_info['el_quad'] + self.moments_info['nucl_quad']
        self.moments_info['mol_octa'] = self.moments_info['el_octa'] + self.moments_info['nucl_octa']
        self.moments_info['mol_hexa'] = self.moments_info['el_hexa'] + self.moments_info['nucl_hexa']
        
    def apply_field(self, E_val: list):
        
        self.build(atom = self.sys, basis = 'sto-3g', verbose = 0)  
        self.E_val = np.array(E_val)
        
        self.set_common_orig([0, 0, 0])
        h =(self.intor('cint1e_kin_sph') + self.intor('cint1e_nuc_sph')
          + np.einsum('x,xij->ij', self.E_val, self.intor('cint1e_r_sph', comp=3)))

        mf = self.HF()
        mf.get_hcore = lambda *args: h
        self.E_en = mf.kernel()

    def grid_gen(self, mode, nx_grid, pot_lim = 120*10**3 / WVNMBR):
        
        self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
        
        self.mode = mode
        self.nx_grid = nx_grid
        self.pot_lim = pot_lim
        factor = np.sqrt(self.freq_info['reduced_mass'][self.mode])

        self.potential = []
        dipole_data = []

        self.x = np.sqrt((2*self.pot_lim)/self.freq_info['force_const_au'][self.mode]) / factor 
        self.dx_grid = 2*self.x / self.nx_grid

        disp = -self.x * self.freq_info['norm_mode'][self.mode] * factor
        for i,atom in enumerate(self._atom):
            for j,coord in enumerate(atom[1]):
                self._atom[i][1][j] = coord + disp[i][j]
        self.sys = self._atom
        self.spe()

        disp = 2*self.x / nx_grid * self.freq_info['norm_mode'][self.mode] * factor
        for q in range(nx_grid):
            print(q)
            for i,atom in enumerate(self._atom):
                for j,coord in enumerate(atom[1]):
                    self._atom[i][1][j] = coord + disp[i][j]
            self.sys = self._atom
            self.spe()
            self.calculate_moments()
            self.potential.append(self.en)
            dipole_data.append(self.moments_info['mol_dip'])

        self.fgh_solver = fgh_object(np.array(self.potential), np.array([self.nx_grid]), np.array([self.dx_grid]), self.freq_info['reduced_mass'][self.mode]*1836)
        self.fgh_solver.fgh_fci_solve()
        self.nm_soln = self.fgh_solver.solutions

        dipole_data = np.array*(dipole_data)
        self.transition_dipole_mat = np.einsum('ix,ji,ki->jkx',dipole_data,self.nm_soln[1],self.nm_soln[1])

            # for j in range(nx_grid):
            #     print(dipole_data[i]['mol_dip']*np.sum(self.nm_soln[1][i]**2)-dipole_data[i]['mol_dip']*np.sum(self.nm_soln[1][j]**2))

        # np.sum(dipole_operator[:,1,0]**2)*(FGH_freq/(627.5096*349.7550))*(2/3)*mass*1836.1527*42.2561
        
if __name__ == "__main__":

    # mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"
    mol_geom = f"H 0 0 0; H 2 0 0;"

    mol = normal_modes()
    mol.build(unit = 'B', atom = mol_geom, basis = 'sto-3g', verbose = 0)    
    mol.geom_opt()
    mol.hessian()
    print(mol.freq_info['freq_wavenumber'])
    mol.calculate_moments()
    mol.grid_gen(0, 32)
    print((mol.nm_soln[0][:10]-mol.nm_soln[0][0])*WVNMBR)
    # print(dipole_data)

    # plt.plot(mol.nm_soln[1][0])
    # plt.show()