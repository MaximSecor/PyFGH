#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:53:47 2022

@author: maximsecor
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft, qmmm
from numba import jit
import scipy.interpolate as interpolate

#%%

@jit(nopython=True)
def fgh_1D(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    vmat = np.zeros((nx,nx))
    tmat = np.zeros((nx,nx))
    hmat = np.zeros((nx,nx))
    
    for i in range(nx):
        for j in range(nx):
            if i == j:
                vmat[i,j] = potential[j]
                tmat[i,j] = (k**2)/3
            else:
                dji = j-i
                vmat[i,j] = 0
                tmat[i,j] = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
            hmat[i,j] = (1/(2*mass))*tmat[i,j] + vmat[i,j]
    
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

#%%

scan_dist = []
scan_energy = []
scan_dipole = []

for i in range(50):
    
    mol_geom =[['C',(0.0,0.0,0.0)],
             ['N',(0.0,0.0,0.8+i*0.02)]]
    
    mol = gto.Mole()
    mol.build(
        atom = mol_geom,
        charge = -1,
        basis = 'sto3g',
    )
    
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    
    coords = [(10.0,0.0,0.0)]
    charges = [1]
    qmmm.mm_charge(scf.RHF(mol), coords, charges)
    
    mf.kernel()
    
    scan_dist.append(0.8+i*0.02)
    scan_energy.append(mf.e_tot)
    scan_dipole.append(mf.dip_moment())

#%%

scan_dist = np.array(scan_dist)
scan_energy = np.array(scan_energy)
scan_dipole = np.array(scan_dipole)

scan_dist_au = scan_dist * 1.889
scan_energy = scan_energy - np.min(scan_energy)

# plt.plot(scan_dist_au,scan_energy*627)

domain = np.linspace(np.min(scan_dist_au),np.max(scan_dist_au),1024)
test = interpolate.interp1d(scan_dist_au,scan_energy,kind='cubic')
# plt.plot(domain,test(domain))

mass = ((12*14)/(12+14))*1836
pot = test(domain)

FHG_solutions =  fgh_1D(domain,pot,mass)
wavefunctions = FHG_solutions[1].T
energies = FHG_solutions[0]
vibrational = (energies[:5]*627*350)
print(vibrational[1]-vibrational[0])

plt.ylim(-1,35)
plt.xlim(1.8/1.889,3.0/1.889)
plt.plot(domain/1.889,pot*627,'k')
plt.plot(domain/1.889,250*wavefunctions[0]**2+vibrational[0]/350,'r')
plt.plot(domain/1.889,250*wavefunctions[1]**2+vibrational[1]/350,'b')
plt.hlines(vibrational[0]/350,1.8,3.0,linestyle='dashed')
plt.hlines(vibrational[1]/350,1.8,3.0,linestyle='dashed')
plt.xlabel('Position (Angstrom)')
plt.ylabel('Energy (kcal/mol)') 
plt.title('Vibrational States')
plt.show()

dx = domain[1]-domain[0]
minx = (np.argmin(pot))
d1 = (pot[minx] - pot[minx-1]) / dx
d2 = (pot[minx+1] - pot[minx]) / dx
k = (d2-d1) / dx

print(np.sqrt(k/mass)*627*350)

test_dipole = interpolate.interp1d(scan_dist_au,scan_dipole[:,2],kind='cubic')
# plt.plot(domain,test_dipole(domain))

# dipole_ref_debye = (test_dipole(domain) - test_dipole(domain)[minx])*(1/2.541)
dipole_ref_debye = (test_dipole(domain))*(1/2.541)

dipole_tran_matrx = np.zeros((5,5))

for i in range(5):
    for j in range(5):
        dipole_tran_matrx[i,j] = np.sum(wavefunctions[i]*wavefunctions[j]*dipole_ref_debye)

print(dipole_tran_matrx)
print(dipole_tran_matrx[1,1]-dipole_tran_matrx[0,0])

#%%

mol = gto.Mole() # Benzene
mol.atom = '''
     C    0.000000000000     1.398696930758     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
     H    0.000000000000     2.491406946734     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
  '''
mol.basis = '6-31g'
mol.build()

#
# Pass 1, generate all HOMOs with external field
#
N = 50 # 50 samples in one period of the oscillated field
mo_id = 20  # HOMO
dm_init_guess = [None]

def apply_field(E):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    dm_init_guess[0] = mf.make_rdm1()
    mo = mf.mo_coeff[:,mo_id]
    if mo[23] < -1e-5:  # To ensure that all MOs have same phase
        mo *= -1
    return mo

fields = np.sin((2*np.pi)/N * np.arange(N))*.2
mos = [apply_field((i+1e-5,0,0)) for i in fields]

#%%

scan_dist = []
scan_energy = []
scan_dipole = []

E = (0.0,0.1,0.0)

for i in range(50):
    
    mol_geom =[['C',(0.0,0.0,0.0)],
             ['N',(0.0,0.0,0.8+i*0.02)]]
    
    mol = gto.Mole()
    mol.build(
        atom = mol_geom,
        charge = -1,
        basis = 'sto3g',
    )
    
    dm_init_guess = [None]
    
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    
    scan_dist.append(0.8+i*0.02)
    scan_energy.append(mf.e_tot)
    scan_dipole.append(mf.dip_moment())

#%%

scan_dist = np.array(scan_dist)
scan_energy = np.array(scan_energy)
scan_dipole = np.array(scan_dipole)

scan_dist_au = scan_dist * 1.889
scan_energy = scan_energy - np.min(scan_energy)

# plt.plot(scan_dist_au,scan_energy*627)

domain = np.linspace(np.min(scan_dist_au),np.max(scan_dist_au),1024)
test = interpolate.interp1d(scan_dist_au,scan_energy,kind='cubic')
# plt.plot(domain,test(domain))

mass = ((12*14)/(12+14))*1836
pot = test(domain)

FHG_solutions =  fgh_1D(domain,pot,mass)
wavefunctions = FHG_solutions[1].T
energies = FHG_solutions[0]
vibrational = (energies[:5]*627*350)
print(vibrational[1]-vibrational[0])

plt.ylim(-1,35)
plt.xlim(1.8/1.889,3.0/1.889)
plt.plot(domain/1.889,pot*627,'k')
plt.plot(domain/1.889,250*wavefunctions[0]**2+vibrational[0]/350,'r')
plt.plot(domain/1.889,250*wavefunctions[1]**2+vibrational[1]/350,'b')
plt.hlines(vibrational[0]/350,1.8,3.0,linestyle='dashed')
plt.hlines(vibrational[1]/350,1.8,3.0,linestyle='dashed')
plt.xlabel('Position (Angstrom)')
plt.ylabel('Energy (kcal/mol)') 
plt.title('Vibrational States')
plt.show()

dx = domain[1]-domain[0]
minx = (np.argmin(pot))
d1 = (pot[minx] - pot[minx-1]) / dx
d2 = (pot[minx+1] - pot[minx]) / dx
k = (d2-d1) / dx

print(np.sqrt(k/mass)*627*350)

test_dipole = interpolate.interp1d(scan_dist_au,scan_dipole[:,2],kind='cubic')
# plt.plot(domain,test_dipole(domain))

# dipole_ref_debye = (test_dipole(domain) - test_dipole(domain)[minx])*(1/2.541)
dipole_ref_debye = (test_dipole(domain))*(1/2.541)

dipole_tran_matrx = np.zeros((5,5))

for i in range(5):
    for j in range(5):
        dipole_tran_matrx[i,j] = np.sum(wavefunctions[i]*wavefunctions[j]*dipole_ref_debye)

print(dipole_tran_matrx)
print(dipole_tran_matrx[1,1]-dipole_tran_matrx[0,0])


#%%

scan_dist = []
scan_energy = []
scan_dipole = []

for i in range(50):
    
    mol_geom =[['C',(10.0,0.0,0.0)],
             ['N',(10.0,(0.8+i*0.02)*(1/np.sqrt(2)),(0.8+i*0.02)*(1/np.sqrt(2)))]]
    
    mol = gto.Mole()
    mol.build(
        atom = mol_geom,
        charge = -1,
        basis = '631g',
    )
    
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    
    scan_dist.append(0.8+i*0.02)
    scan_energy.append(mf.e_tot)
    scan_dipole.append(mf.dip_moment())

#%%

scan_dist = np.array(scan_dist)
scan_energy = np.array(scan_energy)
scan_dipole = np.array(scan_dipole)

scan_dist_au = scan_dist * 1.889
scan_energy = scan_energy - np.min(scan_energy)

# plt.plot(scan_dist_au,scan_energy*627)

domain = np.linspace(np.min(scan_dist_au),np.max(scan_dist_au),1024)
test = interpolate.interp1d(scan_dist_au,scan_energy,kind='cubic')
# plt.plot(domain,test(domain))

mass = ((12*14)/(12+14))*1836
pot = test(domain)

FHG_solutions =  fgh_1D(domain,pot,mass)
wavefunctions = FHG_solutions[1].T
energies = FHG_solutions[0]
vibrational = (energies[:5]*627*350)
print(vibrational[1]-vibrational[0])

plt.ylim(-1,35)
plt.xlim(1.8/1.889,3.0/1.889)
plt.plot(domain/1.889,pot*627,'k')
plt.plot(domain/1.889,250*wavefunctions[0]**2+vibrational[0]/350,'r')
plt.plot(domain/1.889,250*wavefunctions[1]**2+vibrational[1]/350,'b')
plt.hlines(vibrational[0]/350,1.8,3.0,linestyle='dashed')
plt.hlines(vibrational[1]/350,1.8,3.0,linestyle='dashed')
plt.xlabel('Position (Angstrom)')
plt.ylabel('Energy (kcal/mol)') 
plt.title('Vibrational States')
plt.show()

dx = domain[1]-domain[0]
minx = (np.argmin(pot))
d1 = (pot[minx] - pot[minx-1]) / dx
d2 = (pot[minx+1] - pot[minx]) / dx
k = (d2-d1) / dx

print(np.sqrt(k/mass)*627*350)

test_dipole_x = interpolate.interp1d(scan_dist_au,scan_dipole[:,0],kind='cubic')
test_dipole_y = interpolate.interp1d(scan_dist_au,scan_dipole[:,1],kind='cubic')
test_dipole_z = interpolate.interp1d(scan_dist_au,scan_dipole[:,2],kind='cubic')
plt.plot(domain,test_dipole_x(domain))
plt.plot(domain,test_dipole_y(domain))
plt.plot(domain,test_dipole_z(domain))
plt.show()

#%%

# dipole_ref_debye = (test_dipole(domain) - test_dipole(domain)[minx])*(1/2.541)
dipole_ref_debye = (test_dipole(domain))*(1/2.541)

dipole_tran_matrx = np.zeros((5,5,3))

for i in range(5):
    for j in range(5):
        dipole_tran_matrx[i,j,0] = np.sum(wavefunctions[i]*wavefunctions[j]*test_dipole_x(domain)*(1/2.541))
        dipole_tran_matrx[i,j,1] = np.sum(wavefunctions[i]*wavefunctions[j]*test_dipole_y(domain)*(1/2.541))
        dipole_tran_matrx[i,j,2] = np.sum(wavefunctions[i]*wavefunctions[j]*test_dipole_z(domain)*(1/2.541))

# print(dipole_tran_matrx)
print(np.sqrt(np.sum((dipole_tran_matrx[1,1]-dipole_tran_matrx[0,0])**2)))

#%% 50 test


mass = ((12*14)/(12+14))*1836

FHG_solutions =  fgh_1D(scan_dist_au,scan_energy,mass)
wavefunctions = FHG_solutions[1].T
energies = FHG_solutions[0]
vibrational = (energies[:5]*627*350)
print(vibrational[1]-vibrational[0])

plt.ylim(-1,35)
plt.xlim(1.8/1.889,3.0/1.889)
plt.plot(scan_dist_au/1.889,scan_energy*627,'k')
plt.plot(scan_dist_au/1.889,25*wavefunctions[0]**2+vibrational[0]/350,'r')
plt.plot(scan_dist_au/1.889,25*wavefunctions[1]**2+vibrational[1]/350,'b')
plt.hlines(vibrational[0]/350,1.8,3.0,linestyle='dashed')
plt.hlines(vibrational[1]/350,1.8,3.0,linestyle='dashed')
plt.xlabel('Position (Angstrom)')
plt.ylabel('Energy (kcal/mol)') 
plt.title('Vibrational States')
plt.show()

#%%


dx = scan_dist_au[1]-scan_dist_au[0]

minx = (np.argmin(scan_energy))

d1 = (scan_energy[minx] - scan_energy[minx-1]) / dx
d2 = (scan_energy[minx+1] - scan_energy[minx]) / dx

k = (d2-d1) / dx

print(np.sqrt(k/mass)*627*350)

#%%

test_dipole = interpolate.interp1d(scan_dist_au,scan_dipole[:,2],kind='cubic')
plt.plot(domain,test_dipole(domain))

#%%

dipole_ref_debye = (scan_dipole[:,2] - scan_dipole[:,2][minx])*(1/2.541)

dipole_tran_matrx = np.zeros((5,5))

for i in range(5):
    for j in range(5):
        dipole_tran_matrx[i,j] = np.sum(wavefunctions[i]*wavefunctions[j]*dipole_ref_debye)

print(dipole_tran_matrx)

#%%

print(dipole_tran_matrx[0,4])

#%%

def apply_field(E):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    en = mf.kernel()
    return en

#%%

scan_dist = []
scan_energy = []

for i in range(16):
    
    mol_geom =[['C',(0.0,0.0,0.0)],
             ['N',(0.0,0.0,0.7+i*0.1)]]
    
    mol = gto.Mole()
    mol.build(
        atom = mol_geom,
        charge = -1,
        basis = '631g',
    )
    
    E = np.array([0,0,0.001])
    mol.set_common_orig([0, 0, 0])
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    
    en = mf.kernel()
    
    scan_dist.append(0.7+i*0.1)
    scan_energy.append(en)

#%%

scan_dist = np.array(scan_dist)
scan_energy = np.array(scan_energy)
scan_dipole = np.array(scan_dipole)

scan_dist = scan_dist * 1.889
scan_energy = scan_energy - np.min(scan_energy)

plt.plot(scan_dist,scan_energy*627)

#%%

domain = np.linspace(np.min(scan_dist),np.max(scan_dist),1024)
test = interpolate.interp1d(scan_dist,scan_energy,kind='cubic')
plt.plot(domain,test(domain))

#%%

mass = ((12*14)/(12+14))*1836
pot = test(domain)

FHG_solutions =  fgh_1D(domain,pot,mass)
wavefunctions = FHG_solutions[1].T
energies = FHG_solutions[0]
vibrational = (energies[:5]*627*350)
print(vibrational[1]-vibrational[0])

plt.ylim(-1,35)
plt.xlim(1.8,3.0)
plt.plot(domain,pot*627,'k')
plt.plot(domain,250*wavefunctions[0]**2+vibrational[0]/350,'r')
plt.plot(domain,250*wavefunctions[1]**2+vibrational[1]/350,'b')
plt.hlines(vibrational[0]/350,1.8,3.0,linestyle='dashed')
plt.hlines(vibrational[1]/350,1.8,3.0,linestyle='dashed')
plt.xlabel('Position (Angstrom)')
plt.ylabel('Energy (kcal/mol)') 
plt.title('Vibrational States')
plt.show()


#%%

from pyscf.geomopt.berny_solver import optimize
mol_eq = optimize(mf, maxsteps=100)
print(mol_eq.atom_coords())

#%%

C_coord = mol_eq.atom_coords()[0] / 1.889
N_coord = mol_eq.atom_coords()[1] / 1.889

#%%

stencil = np.array([-0.0001,0,0.0001])
scan_en = []

for i in range(len(stencil)):
    
    N_coord_i = N_coord + np.array([0,0,stencil[i]])
    
    mol_geom = []
    mol_geom.append(['C',tuple(C_coord)])
    mol_geom.append(['N',tuple(N_coord_i)])
    mol_geom = np.array(mol_geom)

    mol = gto.Mole()
    mol.build(atom = mol_geom,
        charge = -1,
        basis = '631g',)
    
    mf = scf.RHF(mol)
    scan_en.append(mf.kernel())
    mf.dip_moment()

scan_en = np.array(scan_en)

print(scan_en)

#%%

dr2 = (scan_en[0] - 2*scan_en[1] + scan_en[2]) / (4*(0.00005*1.889)**2)
print(np.sqrt(dr2/mass)*627*350)

#%%

mol_geom = []
mol_geom.append(['C',tuple(C_coord)])
mol_geom.append(['N',tuple(N_coord)])
mol_geom = np.array(mol_geom)

mol = gto.Mole()
mol.build(atom = mol_geom,
    charge = -1,
    basis = '631g',)

mf = mol.RHF().run()
h = mf.Hessian().kernel()
print(h.shape)

#%%

print(np.sqrt(np.linalg.eigh(h.reshape(6,6))[0]/mass)*627*350)

#%%

from pyscf.hessian import thermo

mf = mol.RHF().run()
hessian = mf.Hessian().kernel()
freq_info = thermo.harmonic_analysis(mf.mol, hessian)

print(freq_info)

#%%

mol_geom = []
mol_geom.append(['C',tuple(C_coord)])
mol_geom.append(['N',tuple(N_coord)])
mol_geom = np.array(mol_geom)

#%%

EField_dir = np.asarray([1, 0, 0])

mol = gto.Mole()
mol.build(atom = mol_geom,
    charge = -1,
    basis = '631g',)


mf.E = 1*EField_dir
mf = mol.RHF().run()

#%%

import numpy
from pyscf import gto, scf

mol = gto.M(
    verbose = 0,
    atom = 'H 0 0 0; H 0 0 1.5; H 0 1 1; H 1.1 0.2 0',
    basis = 'ccpvdz'
)

nao = mol.nao

# Dipole integral
dip = mol.intor('int1e_r').reshape(3,nao,nao)

# Quadrupole
quad = mol.intor('int1e_rr').reshape(3,3,nao,nao)

# Octupole
octa = mol.intor('int1e_rrr').reshape(3,3,3,nao,nao)

# hexadecapole
hexa = mol.intor('int1e_rrrr').reshape(3,3,3,3,nao,nao)

print(nao)

#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:26:26 2022

@author: maximsecor
"""


import numpy
from pyscf import gto, scf, cc

mol = gto.M(
    verbose = 0,
    atom = 'O 0 0 0; H 0 0 1; H 0 1 0;',
    basis = 'ccpvdz'
)

nao = mol.nao

# Multipole integrals for atomic orbitals
dip = mol.intor('int1e_r').reshape(3,nao,nao)
quad = mol.intor('int1e_rr').reshape(3,3,nao,nao)
octa = mol.intor('int1e_rrr').reshape(3,3,3,nao,nao)
hexa = mol.intor('int1e_rrrr').reshape(3,3,3,3,nao,nao)

mf = mol.HF().run()
mycc = cc.CCSD(mf).run()

charges = mol.atom_charges()
coords  = mol.atom_coords()
# dm = (mf.make_rdm1())
dm1 = mycc.make_rdm1()
dm = numpy.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1, mf.mo_coeff.conj())

el_dip = -numpy.einsum('xij,ji->x', dip, dm).real
nucl_dip = np.einsum('i,ix->x', charges, coords)
mol_dip = nucl_dip + el_dip

el_quad = -numpy.einsum('xyij,ji->xy', quad, dm).real
nucl_quad = np.einsum('xi,xj,x->ij', coords, coords, charges)
mol_quad = nucl_quad + el_quad

el_octa = -numpy.einsum('xyzij,ji->xyz', octa, dm).real
nucl_octa = np.einsum('xi,xj,xk,x->ijk', coords, coords, coords, charges)
mol_octa = nucl_octa + el_octa

el_hexa = -numpy.einsum('xyzaij,ji->xyza', hexa, dm).real
nucl_hexa = np.einsum('xi,xj,xk,xl,x->ijkl', coords, coords, coords, coords, charges)
mol_hexa = nucl_hexa + el_hexa

print(mol_dip,'\n \n',mol_quad)

#%%

test = np.zeros((3,3))
for i in range(len(coords[0])):
    for j in range(len(coords[0])):
        for l in range(len(coords)):
            test[i,j] = coords[l,i]*coords[l,j]*charges[l]
            
print(test)

#%%

coords_quad = np.einsum('i,j->ij', coords, coords)
print(coords_quad)


#%%

a = np.array([[1,1,1],[1,1,1]])
b = np.array([1,1])

a = coords
b = charges

c = np.einsum('xi,xj,x->ij', a, a, b)
# c = np.einsum('ij,ij->ij', a, a)

print(c)

#%%

def dip_moment(mol, dm):

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = numpy.einsum('xij,ji->x', ao_dip, dm).real

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = numpy.einsum('i,ix->x', charges, coords)
    mol_dip = nucl_dip - el_dip

    return mol_dip

#%%

dm = (mf.make_rdm1())
print(dip_moment(mol, dm)*2.541)


