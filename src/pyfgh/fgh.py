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

This module implements the main functionality of pyFGH.

-----::::Functions included::::-----
fgh_hardcode_1d : This function is a hard-coded, jitted 1D FGH calculation
fgh_hardcode_2d : Hard-coded FCI, jitted 2D FGH, same number of grid points and same distance between grid points in each dimension
fgh_hardcode_3d : Hard-coded FCI, jitted 3D FGH, same number of grid points and same distance between grid points in each dimension
fgh_flex : FCI FGH flexible accept any number of dimensions with any number of points and any distance seperation along each dimension
fgh_mcscf : MCSCF FGH flexible accept any number of dimensions with any number of points and any distance seperation along each dimension

-----::::Classes included::::-----
fgh_object : This class provides a brief description of the purpose and functionality of the class.

"""

__author__ = "Maxim Secor"
__email__ = "maxim.secor@yale.edu"

import numpy as np
from numba import jit
import itertools
import matplotlib.pyplot as plt

@jit(nopython=True)
def fgh_hardcode_1d(potential: np.ndarray, nx: int, dx: float, mass: float) -> tuple:
    
    """
    fgh_hardcode_1d : This function is a hard-coded, jitted 1D FGH calculation

    ---Args---
        potential : 1D ndarray : potential energy in Hartree along 1 dimension
        nx        : int        : number of points along the grid
        dx        : float      : distance between grid points
        mass      : float      : mass of the particle

    ---Returns---
        energies      : 1D ndarray : Energy eigenvalues in Hartree
        wavefunctions : 2D ndarray : Energy eigenfunctions
    """

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
def fgh_hardcode_2d(potential: np.ndarray, nx: int, dx: float, mass: float) -> tuple:

    """
    fgh_hardcode_2d : This function is a hard-coded, jitted 2D FGH calculation. The number of grid points and distance between grid points is the same for ALL dimension

    ---Args---
        potential : 2D ndarray : 2D potential energy surface in Hartree
        nx        : int        : number of points along the grid
        dx        : float      : distance between grid points
        mass      : float      : mass of the particle

    ---Returns---
        energies      : 1D ndarray : Energy eigenvalues in Hartree
        wavefunctions : 2D ndarray : Energy eigenfunctions
    """

    k = np.pi/dx
    m = (1/(2*mass))

    hmat = np.zeros((nx**2,nx**2))
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    if xi == xj and yi == yj:
                        hmat[xi+nx*yi,xj+nx*yj] = potential[xj,yj] + m*(k**2)*(2/3)
                    elif xi != xj and yi == yj:
                        dji = xj-xi
                        hmat[xi+nx*yi,xj+nx*yj] = m*((2*k**2)/(np.pi**2))*(((-1)**dji)/(dji**2))
                    elif xi == xj and yi != yj:
                        dji = yj-yi
                        hmat[xi+nx*yi,xj+nx*yj] = m*((2*k**2)/(np.pi**2))*(((-1)**dji)/(dji**2))
                    else:
                        hmat[xi+nx*yi,xj+nx*yj] = 0

    hmat_soln = np.linalg.eigh(hmat)

    energies = hmat_soln[0]
    wavefunctions = hmat_soln[1].T

    return energies, wavefunctions

@jit(nopython=True)
def fgh_hardcode_3d(potential: np.ndarray, nx: int, dx: float, mass: float) -> tuple:

    """
    fgh_hardcode_3d : This function is a hard-coded, jitted 3D FGH calculation. The number of grid points and distance between grid points is the same for ALL dimension

    ---Args---
        potential : 3D ndarray : 3D potential energy surface in Hartree
        nx        : int        : number of points along the grid
        dx        : float      : distance between grid points
        mass      : float      : mass of the particle

    ---Returns---
        energies      : 1D ndarray : Energy eigenvalues in Hartree
        wavefunctions : 2D ndarray : Energy eigenfunctions
    """

    k = np.pi/dx
    m = (1/(2*mass))
    hmat = np.zeros((nx**3,nx**3))
    
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    for zi in range(nx):
                        for zj in range(nx):
                            if xi == xj and yi == yj and zi == zj:
                                hmat[xi+nx*yi+nx*nx*zi,xj+nx*yj+nx*nx*zj] = m*(k**2) + potential[xj,yj,zj]
                            elif xi != xj and yi == yj and zi == zj:
                                dji = xj-xi
                                hmat[xi+nx*yi+nx*nx*zi,xj+nx*yj+nx*nx*zj] = m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi != yj and zi == zj:
                                dji = yj-yi
                                hmat[xi+nx*yi+nx*nx*zi,xj+nx*yj+nx*nx*zj] = m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi == yj and zi != zj:
                                dji = zj-zi
                                hmat[xi+nx*yi+nx*nx*zi,xj+nx*yj+nx*nx*zj] = m*(2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
    
    hmat_soln = np.linalg.eigh(hmat)
    energies = hmat_soln[0]
    wavefunctions = hmat_soln[1].T

    return energies, wavefunctions

def get_ID(q: int, nx: int) -> list:

    """
    get_ID : This function allows the fgh_flex function to index discrete variable representation (DVR) Hartree product basis functions along each dimension

    ---Args---
        q  : int : DVR hartree product basis function index
        nx : int : number of points along the grid

    ---Returns---
        q_idx : list[int] : list of indices of length N providing an N-dimensional index of the qth DVR basis function
    """

    q_idx = []
    t = 1
    for i in nx:
        q_idx.append((q//t)%i)
        t *= i

    return q_idx

def fgh_flex(potential: np.ndarray, nx: np.ndarray, dx: np.ndarray, mass: int) -> tuple:

    """
    fgh_flex : This function performs FGH calculation in any dimension with any number of grid points spaced consistently across individual dimensions.

    ---Args---
        potential : 1D ndarray : Flattened N-dimensional potential energy surface in Hartree 
        nx        : 1D ndarray : Interger numbers of points on the grid in each dimension
        dx        : 1D ndarray : Distance between grid points in each dimension
        mass      : float      : Mass of the particle

    ---Returns---
        energies      : 1D ndarray : Energy eigenvalues in Hartree
        wavefunctions : 2D ndarray : Energy eigenfunctions
    """

    k = np.pi/dx
    dh = np.prod(nx)
    m = (1/(2*mass))
    hmat = np.zeros((dh,dh))

    for i in range(dh):
        for j in range(dh):
            i_id = get_ID(i,nx)
            j_id = get_ID(j,nx)
            id_match = [k1 == k2 for k1,k2 in zip(i_id,j_id)]
            
            if sum(id_match) == len(j_id):
                hmat[i,j] = m*np.sum([k1**2 for k1 in k])/3 + potential[j]
                
            if sum(id_match) == len(j_id)-1:
                for k1,k2 in enumerate(id_match):
                    if k2 == False:
                        dji = float(j_id[k1]-i_id[k1])
                        hmat[i,j] = m*((2*k[k1]**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))

    hmat_soln = np.linalg.eigh(hmat)
    energies = hmat_soln[0]
    wavefunctions = hmat_soln[1].T

    return energies, wavefunctions

def tensor_product(arrays: list):

    """
    tensor_product : This function is used by fgh_mcasf to iteratively takes outer products N of a list of 1D ndarrays

    ---Args---
        arrays  : list : list of ndarrays containing basis functions produced from the dimensional SCF procedure

    ---Returns---
        result : ND ndarray : list of indices of the basis function in each dimension
    """

    result = arrays[0]
    for i in range(1, len(arrays)):
        result = np.tensordot(result, arrays[i], axes=0)
    return result

def average_except_one(tensor: np.ndarray, axis: int):

    """
    tensor_product : This function is used by fgh_mcasf to take the average along all dimensions except one

    ---Args---
        tensor  : ND np.ndarray : The ...
        axis    : int           : The average along which the mean field potential is being calculated

    ---Returns---
        average : 1D ndarray : list of indices of the basis function in each dimension
    """
        
    dimensions = len(tensor.shape)
    axes = tuple(i for i in range(dimensions) if i != axis)
    average = np.sum(tensor, axis=axes)
    return average

def mean_field_potential(potential, wavefnc, axis):

    """
    mean_field_potential : This function calculates the meanfield potential along a given dimension

    ---Args---
        potential : ND np.ndarray : The total potential
        wavefnc   : ND np.ndarray : The total wavefunction
        axis      : int           : The axis associated with dimension for which the mean field potential is calculated

    ---Returns---
        result : 1D ndarray : The mean field potential along a given dimension 
    """
    
    wavefnc_temp = []
    for k,wave_k in enumerate(wavefnc):
        if k != axis:
            wavefnc_temp.append(wave_k*wave_k)
        else:
            wavefnc_temp.append(np.array([1 for j in range(len(wave_k))]))

    return average_except_one(potential*tensor_product(wavefnc_temp), axis)

def potential_term_calc(potential, wavefnc_1, wavefnc_2):

    """
    mean_field_potential : This function calculates the meanfield potential along a given dimension

    ---Args---
        potential : ND np.ndarray : The total potential
        wavefnc   : ND np.ndarray : The total wavefunction
        axis      : int           : The axis associated with dimension for which the mean field potential is calculated

    ---Returns---
        result : 1D ndarray : The mean field potential along a given dimension 
    """
    
    wavefnc_temp = []
    for k,wave_k in enumerate(wavefnc):
        if k != axis:
            wavefnc_temp.append(wave_k*wave_k)
        else:
            wavefnc_temp.append(np.array([1 for j in range(len(wave_k))]))

    return average_except_one(potential*tensor_product(wavefnc_temp), axis)

@jit(nopython=True)
def fgh_get_kin(potential: np.ndarray, nx: int, dx: float, mass: float) -> tuple:
    
    """
    fgh_hardcode_1d : This function is a hard-coded, jitted 1D FGH calculation

    ---Args---
        potential : 1D ndarray : potential energy in Hartree along 1 dimension
        nx        : int        : number of points along the grid
        dx        : float      : distance between grid points
        mass      : float      : mass of the particle

    ---Returns---
        energies      : 1D ndarray : Energy eigenvalues in Hartree
        wavefunctions : 2D ndarray : Energy eigenfunctions
    """

    k = np.pi/dx
    m = (1/(2*mass))
    tmat = []

    for i in range(nx):
        for j in range(nx):
            dji = j-i
            if i == j: 
                tmat.append((m*k**2)/3)
            else: 
                tmat.append((m*2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2)))

    return np.array(tmat).reshape(nx,nx)

def fgh_mcscf(potential: np.ndarray, nx: np.ndarray, dx: np.ndarray, mass: float, scf_iter = 2, basis_size = 5):

    """
    fgh_mcscf : This function performs FGH calculation in any dimension with any number of grid points spaced consistently across individual dimensions.

    ---Args---
        potential  : 1D ndarray : N-dimensional potential energy surface in Hartree
        nx         : list       : Number of points along the grid
        dx         : list       : Distance between grid points
        mass       : float      : Mass of the particle
        SCF_iter   : int        : Number of SCF cycles (default = 2)
        basis_size : int        : Basis set size in each dimension (default = 5)

    ---Returns---
        energies      : 1D ndarray : Energy eigenvalues in Hartree
        wavefunctions : 2D ndarray : Energy eigenfunctions
    """

    scf_iter = 10
    basis_size = 5

    nx = nx[::-1]
    dx = dx[::-1]
    potential = potential.reshape(nx)
    wavefnc = [np.array([np.sqrt(1/nx[i]) for j in range(nx[i])]) for i in range(len(nx))]

    for k in range(scf_iter):
        mean_field_pots = [mean_field_potential(potential, wavefnc, axis) for axis in range(len(wavefnc))]
        wavefnc = [fgh_hardcode_1d(pot,nx[i],dx[i],mass)[1][0] for i, pot in enumerate(mean_field_pots)]

    dimensional_basis_sets = [fgh_hardcode_1d(pot,nx[i],dx[i],mass)[1][:basis_size] for i, pot in enumerate(mean_field_pots)]
    CI_idx = [[j for j in range(basis_size)] for i in range(len(nx))]
    combinations = list(itertools.product(*CI_idx))
    full_basis_size = len(combinations)

    hmat = np.zeros((full_basis_size,full_basis_size))
    for i in range(0,full_basis_size,1):
        for j in range(i,full_basis_size,1):
            
            V, T = 0, 0
            wave_similarity = [a==b for a,b in zip(combinations[i],combinations[j])]
            basis_function_1 = [dimensional_basis_sets[k][lvl] for k,lvl in enumerate(combinations[i])]
            basis_function_2 = [dimensional_basis_sets[k][lvl] for k,lvl in enumerate(combinations[j])]
    
            V = np.sum(tensor_product(basis_function_1)*potential*tensor_product(basis_function_2))
            
            if sum(wave_similarity) == len(combinations[i]):
                T = np.sum([np.dot(np.dot(fgh_get_kin(mean_field_pots[i], nx[i], dx[i], mass),basis),basis) for i,basis in enumerate(basis_function_1)])
                
            if sum(wave_similarity) == len(combinations[i])-1:
                idx_f = np.where(np.array(wave_similarity)==False)[0][0]
                T = np.dot(np.dot(fgh_get_kin(mean_field_pots[idx_f], nx[idx_f], dx[idx_f], mass),basis_function_2[idx_f]),basis_function_1[idx_f])
                
            hmat[i,j] = T + V
                          
    hmat = hmat+hmat.T-np.diag(np.diag(hmat))
    hmat_soln = np.linalg.eigh(hmat)
    energies = hmat_soln[0]
    
    wavefunctions = []
    for i in range(len(energies)):
        wavefunctions.append(np.sum([coeff*tensor_product([dimensional_basis_sets[k][lvl] for k,lvl in enumerate(combinations[j])]).T for j,coeff in enumerate(hmat_soln[1].T[i])],axis=0))

    return energies, wavefunctions

class fgh_object:

    """
    fgh_object : This class provides a brief description of the purpose and functionality of the class.

    ---Methods---
        __init__        : Initializes fgh_object with potential, # grid point, distance between points, and particle mass.
        fgh_fci_solve   : Solves the Time Independent Schrodinger Equation (TDSE) using the appropriate FCI FGH function.
        fgh_mcscf_solve : Solves the TDSE using the MCSCF FGH function.
    
    ---Usage---
        Instantiate the class : `instance = fgh_object(potential, nx, dx, mass)`
        Solve the system      : `instance.fgh_fci_solve()` OR `instance.fgh_mcscf_solve()`
    """

    def __init__(self, potential: np.ndarray, nx: np.ndarray, dx: np.ndarray, mass: float):

        """
        __init__ method : This method initializes the class instance

        ---Args---
            potential : 1D ndarray : Flattened N-dimensional potential energy surface in Hartree 
            nx        : 1D ndarray : Interger numbers of points on the grid in each dimension
            dx        : 1D ndarray : Distance between grid points in each dimension
            mass      : float      : Mass of the particle
        """

        self.potential = potential
        self.nx = nx
        self.dx = dx
        self.mass = mass

    def fgh_fci_solve(self):

        """
        fgh_fci_solve(): The method selects which FGH FCI solver will be best and then solves the TISE. If dimensions < 4 and all the dimensions have the same number of grid points and grid point separation then the appropriately dimensioned hardcoded solver will be chosen, otherwise the flex solven will be used.
        """

        q = sum([1 if i != self.nx[0] else 0 for i in self.nx])
        q += sum([1 if i != self.dx[0] else 0 for i in self.dx])

        if len(self.nx) <= 3:
            if q != 0:
                self.solutions = fgh_flex(self.potential,self.nx,self.dx,self.mass)
            else:
                self.potential_reshape = self.potential.reshape(self.nx)
                if len(self.nx) == 1: self.solutions = fgh_hardcode_1d(self.potential_reshape,self.nx[0],self.dx[0],self.mass)
                if len(self.nx) == 2: self.solutions = fgh_hardcode_2d(self.potential_reshape,self.nx[0],self.dx[0],self.mass)
                if len(self.nx) == 3: self.solutions = fgh_hardcode_3d(self.potential_reshape,self.nx[0],self.dx[0],self.mass)
        else:
            self.solutions = fgh_flex(self.potential,self.nx,self.dx,self.mass)
        
    def fgh_mcscf_solve(self):

        """
        fgh_mcscf_solve(): The method calls the FGH MCSCF solver.
        """
        
        self.solutions = fgh_mcscf(self.potential,self.nx,self.dx,self.mass)

if __name__ == "__main__":

    """
    2d - testing
    """

    pts = 32
    spc = 2

    potential = np.array([0.5*(i/spc-8)**2 + 0.5*(j/spc-8)**2 for j in range(pts) for i in range(pts)])
    nx = np.array([pts,pts])
    dx = np.array([1/spc,1/spc])
    mass = 1

    test = fgh_object(potential, nx, dx, mass)
    test.fgh_mcscf_solve()
    temp = test.solutions
    print(temp[0][:10])