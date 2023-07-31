import numpy as np
import matplotlib.pyplot as plt
from pyfgh.fgh import fgh_object

hartree2kcal = 627.509
kcal2wavenumber = 349.755
proton_mass = 1836

k = proton_mass*((3000/(hartree2kcal*kcal2wavenumber))**2)

potential_1 = np.array([0.75*k*(i/16-5/4)**2 for i in range(64)])
potential_2 = np.array([1.5*k*(i/16-11/4)**2 for i in range(64)])
potential_dw = np.array([np.linalg.eigh([[a,0.2],[0.2,b]])[0][0] for a, b in zip(potential_1,potential_2)])
potential_dw = potential_dw - np.min(potential_dw)

nx = np.array([64])
dx = np.array([1/16])

fgh_instance = fgh_object(potential_dw, nx, dx, proton_mass)
fgh_instance.fgh_fci_solve()
energies = fgh_instance.solutions[0]
wavefunctions = fgh_instance.solutions[1]

print(energies[:10]*hartree2kcal*kcal2wavenumber)

plt.ylim(-1,100)
plt.plot(potential_dw*hartree2kcal)
plt.plot(50*wavefunctions[0]**2 + energies[0]*hartree2kcal)
plt.plot(50*wavefunctions[1]**2 + energies[1]*hartree2kcal)
plt.plot(50*wavefunctions[2]**2 + energies[2]*hartree2kcal)
plt.show()


