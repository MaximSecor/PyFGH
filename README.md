# PyFGH
PyFGH is a Python library for dealing with FGH

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyFGH. (Available Soon)

```bash
pip install pyfgh (Available Soon)
```

## Usage

```python
from pyfgh import fgh
from pyfgh import proton
from pyfgh import irsec
from pyfgh import stark

# returns eigenenergies and eigenfunctions of the simple harmonic oscillator 

potential = np.array([0.5*(i/8-8)**2 for i in range(128)])
nx = np.array([128])
dx = np.array([1/8])
mass = 1

pot_grid = fgh.fgh_object(potential, nx, dx, mass)
pot_grid.fgh_fci_solve()
print(pot_grid.solutions)

# returns eigenenergies of the proton of an HCN molecule

mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"
mol = proton.qm_proton()
mol.build(unit = 'B', atom = mol_geom, basis = '321g', verbose = 0)    

mol.geom_opt()
mol.grid_gen(2, np.array([8,8,8]), np.array([0.2,0.2,0.2]))
mol.grid_refine()

mol.grid_solve()
mp2_soln = mol.protsoln
print(mp2_soln[0][0]*KCAL,(mp2_soln[0][:10]-mp2_soln[0][0])*WVNMBR)

# returns and IRSEC plot

path_neutral = 'YOUR_PATH_NEUTRAL_MOLECULE'
path_other = ['YOUR_PATH_OTHER1_MOLECULE','YOUR_PATH_OTHER2_MOLECULE', ...]
scaling_factor = 0.962

mol = irsec.IRSEC(path_neutral, path_other, scaling_factor)
mol.plotSpectra(0,num=5,xlim=[1650,1500],show=True,save='test.png')

# returns fundamental and overtones of the first normal mode of an HCN molecule

mol_geom = f"N 0 0 0; C 2 0 0; H 4 0 0;"

mol = stark.normal_modes()
mol.build(unit = 'B', atom = mol_geom, basis = 'sto-3g', verbose = 0)    
mol.geom_opt()
mol.hessian()
mol.calculate_moments()
mol.grid_gen(0, 128)
print((mol.nm_soln[0][:10]-mol.nm_soln[0][0])*WVNMBR)

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

