#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:25:23 2023

@author: maximsecor
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from pyscf import gto, scf, dft, qmmm
from numba import jit
import itertools
import time

@jit(nopython=True)
def fgh_hardcode_1d(potential,nx,dx,mass):
    
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

@jit(nopython=True)
def fgh_hardcode_2d(potential,nx,dx,mass):

    k = np.pi/dx
    
    vmat = 0
    tmat = 0
    hmat = np.zeros((nx**2,nx**2))

    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    if xi == xj and yi == yj:
                        vmat = potential[xj,yj]
                        tmat = (k**2)*(2/3)
                    elif xi != xj and yi == yj:
                        dji = xj-xi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    elif xi == xj and yi != yj:
                        dji = yj-yi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    else:
                        vmat = 0
                        tmat = 0
                    hmat[xi+nx*yi,xj+nx*yj] = (1/(2*mass))*tmat + vmat

    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln 

@jit(nopython=True)
def fgh_hardcode_3d(potential,nx,dx,mass):
    
    k = np.pi/dx
    
    vmat = 0
    tmat = 0
    
    hmat = np.zeros((nx**3,nx**3))
    
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    for zi in range(nx):
                        for zj in range(nx):
                            if xi == xj and yi == yj and zi == zj:
                                vmat = potential[xj,yj,zj]
                                tmat = (k**2)
                            elif xi != xj and yi == yj and zi == zj:
                                dji = xj-xi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi != yj and zi == zj:
                                dji = yj-yi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi == yj and zi != zj:
                                dji = zj-zi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            else:
                                vmat = 0
                                tmat = 0
                            hmat[xi+nx*yi+nx*nx*zi,xj+nx*yj+nx*nx*zj] = (1/(2*mass))*tmat + vmat
    
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

def get_ID(q,nx):
    q_idx = []
    
    t = 1
    for i in nx:
        q_idx.append((q//t)%i)
        t *= i

    return q_idx

def fgh_flex(potential,nx,dx,mass):

    k = np.pi/dx
    dh = np.prod(nx)
    hmat = np.zeros((dh,dh))

    for i in range(dh):
        for j in range(dh):
            i_id = get_ID(i,nx)
            j_id = get_ID(j,nx)
            id_match = [k1 == k2 for k1,k2 in zip(i_id,j_id)]
            
            if sum(id_match) == len(j_id):
                pot = potential[j]
                kin = 0.5*mass*np.sum([k1**2 for k1 in k])/3
                hmat[i,j] = kin + pot
                
            if sum(id_match) == len(j_id)-1:
                for k1,k2 in enumerate(id_match):
                    if k2 == False:
                        dji = float(j_id[k1]-i_id[k1])
                        kin = 0.5*mass*((2*k[k1]**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))
                pot = 0
                hmat[i,j] = kin + pot

    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

def tensor_product(arrays):
    result = arrays[0]
    for i in range(1, len(arrays)):
        result = np.tensordot(result, arrays[i], axes=0)
    return result

def average_except_one(tensor, axis):
    dimensions = len(tensor.shape)
    axes = tuple(i for i in range(dimensions) if i != axis)
    average = np.sum(tensor, axis=axes)
    return average

def mean_field_potential(potential, wavefnc, axis):
    
    wavefnc_temp = []
    for k,wave_k in enumerate(wavefnc):
        if k != axis:
            wavefnc_temp.append(wave_k*wave_k)
        else:
            wavefnc_temp.append(np.array([1 for j in range(len(wave_k))]))

    return average_except_one(potential*tensor_product(wavefnc_temp), axis)

def fgh_mcscf(potential,nx,dx,mass,SCF_iter = 2,basis_size = 5):

    potential = potential.reshape(nx)
    wavefnc = [np.array([np.sqrt(1/nx[i]) for j in range(nx[i])]) for i in range(len(nx))]

    for k in range(SCF_iter):
        mean_field_pots = [mean_field_potential(potential, wavefnc, axis) for axis in range(len(wavefnc))]
        wavefnc = [fgh_hardcode_1d(pot,nx[i],dx[i],mass)[1].T[0] for i, pot in enumerate(mean_field_pots)]
                                
    dimensional_basis_sets = [fgh_hardcode_1d(pot,nx[i],dx[i],mass)[1].T[:basis_size] for i, pot in enumerate(mean_field_pots)]

    CI_idx = [[j for j in range(basis_size)] for i in range(len(nx))]
    combinations = list(itertools.product(*CI_idx))
    full_basis = np.array([tensor_product([dim[idx] for dim, idx in zip(dimensional_basis_sets, combination)]) for combination in combinations])

    full_basis_reshaped = full_basis.reshape(full_basis.shape[0], -1)
    vmat = np.dot(full_basis_reshaped * potential.ravel(), full_basis_reshaped.T)

    laplacian = [np.gradient(np.gradient(full_basis, dx[i], axis=i+1), dx[i], axis=i+1) for i in range(len(nx))]
    tmat = np.zeros((len(full_basis),len(full_basis)))
    for lap_k in laplacian:
        lap_k_reshaped = lap_k.reshape(lap_k.shape[0], -1)
        tmat += np.dot(lap_k_reshaped, full_basis_reshaped.T)
            
    hmat = -0.5*mass*tmat + vmat
    hmat_soln = np.linalg.eigh(hmat)

    return hmat_soln

class FGH_object:

    def __init__(self, potential: list[float], nx: list[int], dx: list[int], mass: int):
        self.potential = potential
        self.nx = nx
        self.dx = dx
        self.mass = mass

    def __str__(self):
        return f"The potential is {len(self.nx)} dimensional.\nThe length of the dimensions are {np.array(nx)*np.array(dx)}"
    
    def fgh_fci_solve(self):

        q = sum([1 if i != self.nx[0] else 0 for i in self.nx])
        q += sum([1 if i != self.dx[0] else 0 for i in self.dx])

        if q != 0:
            self.solutions = fgh_flex(self.potential,self.nx,self.dx,self.mass)
        
        if q == 0:
            self.potential_reshape = self.potential.reshape(self.nx)
            if len(self.nx) == 1: self.solutions = fgh_hardcode_1d(self.potential_reshape,self.nx[0],self.dx[0],self.mass)
            if len(self.nx) == 2: self.solutions = fgh_hardcode_2d(self.potential_reshape,self.nx[0],self.dx[0],self.mass)
            if len(self.nx) == 3: self.solutions = fgh_hardcode_3d(self.potential_reshape,self.nx[0],self.dx[0],self.mass)
        
    def fgh_mcscf_solve(self):
        self.solutions = fgh_mcscf(self.potential,self.nx,self.dx,self.mass)
        
    def get_solutions(self):
        return self.solutions

if __name__ == "__main__":

    potential = []
    for i in range(16):
        for j in range(32):
            potential.append(0.5*(i-8)**2 + 0.5*((j/2)-8)**2)
    potential = np.array(potential)

    nx = np.array([32,16])
    dx = np.array([0.5,1])
    mass = 1

    test_2 = FGH_object(potential, nx, dx, mass)
    test_2.fgh_fci_solve()
    temp_2 = test_2.get_solutions()
    print(temp_2[0][:10])

    potential = []
    for i in range(64):
        for j in range(64):
            potential.append(0.5*(i/4-8)**2 + 0.5*((j/4)-8)**2)
    potential = np.array(potential)

    nx = np.array([64,64])
    dx = np.array([0.25,0.25])
    mass = 1

    test_2 = FGH_object(potential, nx, dx, mass)
    test_2.fgh_mcscf_solve()
    temp_2 = test_2.get_solutions()
    print(temp_2[0][:10])
