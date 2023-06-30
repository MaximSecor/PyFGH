"""
This module implements the main functionality of pyFGH.

::::Functions included::::
fgh_hardcode_1d: Hard-coded FCI, jitted 1D FGH
fgh_hardcode_2d: Hard-coded FCI, jitted 2D FGH, same number of grid points and same distance between grid points in each dimension
fgh_hardcode_3d: Hard-coded FCI, jitted 3D FGH, same number of grid points and same distance between grid points in each dimension
fgh_flex: FCI FGH flexible accept any number of dimensions with any number of points and any distance seperation along each dimension
fgh_mcscf: MCSCF FGH flexible accept any number of dimensions with any number of points and any distance seperation along each dimension

::::Classes included::::
fgh_object:

"""

__author__ = "Maxim Secor"
__email__ = "maxim.secor@yale.edu"

import numpy as np
from numba import jit
import itertools

@jit(nopython=True)
def fgh_hardcode_1d(potential: np.ndarray, nx: int, dx: float, mass: float) -> 'tuple[np.ndarray]':
    
    '''Hard-coded, jitted 1D FGH

    Args:
        potential : 1D ndarray
            Potential energy in Hartree along 1 dimension
        nx : int
            number of points along the grid
        dx : float
            distance between grid points
        mass : float
            Mass of particle
    Returns:
        Energy eigenvalues and eigenfunctions, tuple of a 1D ndarray and a 2D ndarray
    '''
        
    k = np.pi/dx
    m = (1/(2*mass))
    hmat = []

    for i in range(nx):
        for j in range(nx):
            dji = j-i
            if i == j: 
                hmat.append(potential[j] + (m*k**2)/3)
            else: 
                hmat.append((m*2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))

    hmat_soln = np.linalg.eigh(np.array(hmat).reshape(nx,nx))
    energies = hmat_soln[0]
    wavefunctions = hmat_soln[1].T

    return energies, wavefunctions

@jit(nopython=True)
def fgh_hardcode_2d(potential: np.ndarray, nx: int, dx: float, mass: float) -> 'tuple[np.ndarray]':

    '''Hard-coded, jitted 2D FGH, same number of grid points and same distance between grid points in each dimension

    Args:
        potential : 2D ndarray
            2D potential energy surface in Hartree
        nx : int
            Number of points in each dimension
        dx : float
            Distance between grid points in each dimension
        mass : float
            Mass of particle
    Returns:
        Energy eigenvalues and eigenfunctions, tuple of a 1D ndarray and a 2D ndarray
    '''

    k = np.pi/dx
    m = (1/(2*mass))
    hmat = []

    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    if xi == xj and yi == yj:
                        hmat.append(potential[xj,yj] + m*(k**2)*(2/3))
                    elif xi != xj and yi == yj:
                        dji = xj-xi
                        hmat.append(m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))
                    elif xi == xj and yi != yj:
                        dji = yj-yi
                        hmat.append(m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))
                    else:
                        hmat.append(0)

    hmat_soln = np.linalg.eigh(np.array(hmat).reshape(nx**2,nx**2))
    energies = hmat_soln[0]
    wavefunctions = hmat_soln[1].T

    return energies, wavefunctions

@jit(nopython=True)
def fgh_hardcode_3d(potential: np.ndarray, nx: int, dx: float, mass: float) -> 'tuple[np.ndarray]':

    '''Hard-coded, jitted 3D FGH, same number of grid points and same distance between grid points in each dimension

    Args:
        potential : 3D ndarray
            3D potential energy surface in Hartree
        nx : int
            Number of points in each dimension
        dx : float
            Distance between grid points in each dimension
        mass : float
            Mass of particle
    Returns:
        Energy eigenvalues and eigenfunctions, tuple of a 1D ndarray and a 2D ndarray
    '''

    k = np.pi/dx
    m = 1/(2*mass)
    hmat = []
    
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    for zi in range(nx):
                        for zj in range(nx):
                            if xi == xj and yi == yj and zi == zj:
                                hmat.append(potential[xj,yj,zj] + m*k**2)
                            elif xi != xj and yi == yj and zi == zj:
                                dji = xj-xi
                                hmat.append(m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))
                            elif xi == xj and yi != yj and zi == zj:
                                dji = yj-yi
                                hmat.append(m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))
                            elif xi == xj and yi == yj and zi != zj:
                                dji = zj-zi
                                hmat.append(m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))
                            else:
                                hmat.append(0)
    
    hmat_soln = np.linalg.eigh(np.array(hmat).reshape(nx**3,nx**3))
    energies = hmat_soln[0]
    wavefunctions = hmat_soln[1].T

    return energies, wavefunctions

def get_ID(q,nx):

    '''Hard-coded, jitted 3D FGH, same number of grid points and same distance between grid points in each dimension

    Args: Distance between grid points in each dimension
    Returns: Energy eigenvalues and eigenfunctions, tuple of a 1D ndarray and a 2D ndarray
    '''

    q_idx = []
    t = 1
    for i in nx:
        q_idx.append((q//t)%i)
        t *= i

    return q_idx

def fgh_flex(potential,nx,dx,mass):

    '''Hard-coded, jitted 3D FGH, same number of grid points and same distance between grid points in each dimension

    Args:
        potential : 1D ndarray
            N-dimensional potential energy surface in Hartree flattened into a 1D array
        nx : list[float]
            Number of points in each dimension
        dx : list[int]
            Distance between grid points in each dimension
        mass : float
            Mass of particle
    Returns:
        Energy eigenvalues and eigenfunctions, tuple of a 1D ndarray and a 2D ndarray
    '''

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

    def __init__(self, potential: 'list[float]', nx: 'list[int]', dx: 'list[int]', mass: int):
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
