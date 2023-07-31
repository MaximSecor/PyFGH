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
"""


import numpy as np
from pyscf import gto, cc
from pyscf.geomopt.berny_solver import optimize
from pyscf.hessian import thermo
from pyfgh.fgh import fgh_object

KCAL = 627.51
WVNMBR = 627.51*349.76

class normal_modes(gto.Mole):
    
    def spe(self):
        
        self.build(atom = self.sys, basis = '631g', verbose = 0)
        self.en = self.HF().run().e_tot
        
    def spe_mp2(self):

        self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
        self.mp2_en = self.HF().run().MP2().run().e_tot

    def spe_ccsd(self, method = 'ccsd'):

        if method == 'ccsd':
            self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)
            self.ccsd_en = cc.CCSD(self.HF().run()).run().e_tot
            
    def geom_opt(self,method = 'hf'):
        
        if method == 'hf':
            self.sys = optimize(self.HF(), maxsteps=100)._atom
            self.build(atom = self.sys, basis = '631g', verbose = 0)    
            self.en = self.HF().run().e_tot
        
        if method == 'mp2':
            self.sys = optimize(self.HF().run().MP2().run(), maxsteps=100)._atom
            self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)    
            self.en = self.HF().run().MP2().run().e_tot
            
        if method == 'ccsd':
            self.sys = optimize(cc.CCSD(self.HF().run()), maxsteps=100)._atom
            self.build(atom = self.sys, basis = 'ccpvtz', verbose = 0)    
            self.en = cc.CCSD(self.HF().run()).run().e_tot
            
        self.eqen = self.en 
        self.eqcoords = self.atom_coords(unit='Bohr')
        
    def hessian(self,method = 'hf'):
        
        if method == 'hf':
            self.build(atom = self.sys, basis = '631g', verbose = 0)   
            self.hessian = self.HF().run().Hessian().kernel()
            self.freq_info = thermo.harmonic_analysis(self.build(atom = self.sys, basis = '631g', verbose = 0), self.hessian)
        
    def efield_calc(self, E_val, method = 'hf'):
        
        self.build(atom = self.sys, basis = '631g', verbose = 0)   
        nao = self.nao

        dip = self.intor('int1e_r').reshape(3,nao,nao)
        quad = self.intor('int1e_rr').reshape(3,3,nao,nao)
        octa = self.intor('int1e_rrr').reshape(3,3,3,nao,nao)
        hexa = self.intor('int1e_rrrr').reshape(3,3,3,3,nao,nao)
        
        self.E_val = E_val
        self.set_common_orig([0, 0, 0])
        h = (self.intor('cint1e_kin_sph') + self.intor('cint1e_nuc_sph') + np.einsum('x,xij->ij', self.E_val, self.intor('cint1e_r_sph', comp=3)))
        
        if method == 'hf':            
            
            mf = self.HF()
            mf.get_hcore = lambda *args: h
            self.E_en = mf.kernel()
            self.dm = mf.make_rdm1()

        if method == 'ccsd':

            mf = self.HF()
            mf.get_hcore = lambda *args: h
            self.E_en = mf.kernel()
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

    def grid_gen(self, mode, nx_grid, pot_lim = 120*10**3 / WVNMBR):
        
        self.build(atom = self.sys, basis = '631g', verbose = 0)
        
        self.mode = mode
        self.nx_grid = nx_grid
        self.pot_lim = pot_lim
        factor = np.sqrt(self.freq_info['reduced_mass'][self.mode])

        self.potential = []
        dip_data = []
        quad_data = []
        octa_data = []
        hexa_data = []
        polar_data = []

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
            for i,atom in enumerate(self._atom):
                for j,coord in enumerate(atom[1]):
                    self._atom[i][1][j] = coord + disp[i][j]
                    
            self.sys = self._atom
            self.spe()
            self.potential.append(self.en)
            
            self.efield_calc([0,0,0])
            dip_data.append(self.moments_info['mol_dip'])
            quad_data.append(self.moments_info['mol_quad'])
            octa_data.append(self.moments_info['mol_octa'])
            hexa_data.append(self.moments_info['mol_hexa'])
            
            self.dip_000 = self.moments_info['mol_dip']  
            self.efield_calc([0.001,0,0])
            self.dip_100 = self.moments_info['mol_dip']
            self.efield_calc([0,0.001,0])
            self.dip_010 = self.moments_info['mol_dip']
            self.efield_calc([0,0,0.001])
            self.dip_001 = self.moments_info['mol_dip']
            self.polar = np.array([self.dip_100-self.dip_000,self.dip_010-self.dip_000,self.dip_001-self.dip_000])/0.001
            polar_data.append(self.polar)
            
        self.test = dip_data
            
        self.fgh_solver = fgh_object(np.array(self.potential), np.array([self.nx_grid]), np.array([self.dx_grid]), self.freq_info['reduced_mass'][self.mode]*1836)
        self.fgh_solver.fgh_fci_solve()
        self.nm_soln = self.fgh_solver.solutions

        self.transition_dip   = np.einsum('ix,ji,ki->jkx',dip_data,self.nm_soln[1],self.nm_soln[1])
        self.transition_quad  = np.einsum('ixy,ji,ki->jkxy',quad_data,self.nm_soln[1],self.nm_soln[1])
        self.transition_octa  = np.einsum('ixyz,ji,ki->jkxyz',octa_data,self.nm_soln[1],self.nm_soln[1])
        self.transition_hexa  = np.einsum('ixyzq,ji,ki->jkxyzq',hexa_data,self.nm_soln[1],self.nm_soln[1])
        self.transition_polar = np.einsum('ixy,ji,ki->jkxy',polar_data,self.nm_soln[1],self.nm_soln[1])

        self.stark_linear     = self.transition_dip[1][1] - self.transition_dip[0][0]
        self.stark_quadratic  = self.transition_polar[1][1] - self.transition_polar[0][0]
        self.irs              = np.sum(self.transition_dip[0][1]**2)*(self.nm_soln[0][1]-self.nm_soln[0][0])*(2/3)*self.freq_info['reduced_mass'][self.mode]*1836.1527*42.2561
        self.irs_polar        = self.transition_polar[1][0]
        
#%%

if __name__ == "__main__":

    mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"

    mol = normal_modes()
    mol.build(unit = 'B', atom = mol_geom, basis = '631g', verbose = 0)
    mol.geom_opt(method = 'hf')
    mol.hessian(method = 'hf')
    
    n_modes = len(mol.freq_info['freq_wavenumber'])
    print(mol.freq_info['freq_wavenumber'])
    
    for i in range(n_modes):
        
        mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"
        mol = normal_modes()
        mol.build(unit = 'B', atom = mol_geom, basis = '631g', verbose = 0)
        mol.geom_opt(method = 'hf')
        mol.hessian(method = 'hf')
        mol.grid_gen(i, 32)
        
        print('\n================================\n')
        print((mol.nm_soln[0][:10]-mol.nm_soln[0][0])*WVNMBR)
        print(mol.stark_linear)
        print(mol.stark_quadratic)
        print('irs: ',mol.irs)
        print(mol.irs_polar)
        
    