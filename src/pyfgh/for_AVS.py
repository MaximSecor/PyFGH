import numpy as np
from pyfgh.fgh import fgh_hardcode_1d
import matplotlib.pyplot as plt

hartree2kcal = 627.509
kcal2wavenumber = 349.755

file = open("src/pyfgh/potential_2.txt")
content = file.readlines()
file.close()

dom = []
pot = []
for i in content:
    temp = i.split(' ')
    temp_2 = []
    for j in temp:
        if j != '':
            temp_2.append(j)
    dom.append(float(temp_2[0]))
    pot.append(float(temp_2[1]))

potential = np.array(pot)/hartree2kcal
nx = len(dom)
dx = (dom[1] - dom[0])*1.889
mass = 1836

print(dom[-1]-dom[0])
print(nx)

energies, wavefunctions = fgh_hardcode_1d(potential, nx, dx, mass)

print((energies[:5]-energies[0])*hartree2kcal)

plt.plot(potential)
plt.plot(wavefunctions[0])
plt.show()

