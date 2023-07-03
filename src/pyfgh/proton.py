import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft, cc
from numba import jit
import scipy.interpolate as interpolate
from fgh import fgh_object

dx = 0.025
dist_x_0 = 1.75
n_grid = 32
basis_set = ['sto3g','6-31g','ccpvdz'][2]

scan_energy = []
scan_dist = np.array([dist_x_0+i*dx for i in range(n_grid)])

for i in range(n_grid):
    
    mol_geom = f"N 0 0 0; C 1 0 0; H {scan_dist[i]} 0 0;"
    
    mol = gto.Mole()
    mol.build(
        atom = mol_geom,
        charge = 0,
        basis = basis_set,
        verbose = 0
    )
    
    #mf = dft.RKS(mol)
    #mf.xc = 'b3lyp'
    #mf.kernel()
    #en = mf.e_tot

    mf = mol.HF().run()
    en = mf.e_tot

    #mycc = cc.CCSD(mf).run()
    #en = mycc.e_tot

    scan_energy.append(en)

scan_dist_au = scan_dist * 1.889

scan_energy = np.array(scan_energy)
scan_energy = scan_energy - np.min(scan_energy)

domain = np.linspace(np.min(scan_dist_au),np.max(scan_dist_au),1024)
inter_pot_fnc = interpolate.interp1d(scan_dist_au,scan_energy,kind='cubic')

mass = 1836
pot = inter_pot_fnc(domain)

FHG_solutions =  fgh_1D(domain,pot,mass)
wavefunctions = FHG_solutions[1].T*(10**2.75)
energies = FHG_solutions[0]
vibrational = energies[:5]*627*350
print(vibrational[1]-vibrational[0])

plt.ylim(-1000,30000)
plt.xlim(scan_dist[0],scan_dist[-1])
plt.plot(domain/1.889,pot*627*350,'k')
plt.plot(domain/1.889,wavefunctions[0]**2+vibrational[0],'r')
plt.plot(domain/1.889,wavefunctions[1]**2+vibrational[1],'b')
plt.hlines(vibrational[0],scan_dist[0],scan_dist[-1],linestyle='dashed')
plt.hlines(vibrational[1],scan_dist[0],scan_dist[-1],linestyle='dashed')
plt.xlabel('Position (Angstrom)')
plt.ylabel('Energy (kcal/mol)') 
plt.title('Vibrational States')
plt.show()
