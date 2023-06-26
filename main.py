from FGH import FGH_object
from utilities import read_input, read_potential
import numpy as np
import datetime

if __name__ == "__main__":

    inputs = read_input("example_input.in")

    ''' FGH Calculation on a provided potential '''
    if inputs['calctype'] == '0':

        ''' Calculation using new format '''
        if inputs['readin'] == '0':

            potential = read_potential("example_pot.in")
            nx = np.array(inputs['nx'].split(','),dtype=int)
            dx = np.array(inputs['dx'].split(','),dtype=float)
            mass = float(inputs['mass'])

            if "method" in inputs:
                if inputs["method"] == "MCSCF":
                    test = FGH_object(potential, nx, dx, mass)
                    test.fgh_mcscf_solve()
                if inputs["method"] == "FCI":
                    test = FGH_object(potential, nx, dx, mass)
                    test.fgh_fci_solve()
            else:
                test = FGH_object(potential, nx, dx, mass)
                test.fgh_fci_solve()

        with open("example.out", "w") as file:

            print_temp = test.get_solutions()
            now = datetime.datetime.now()
            formatted_date = now.strftime("%Y-%m-%d")
            formatted_time = now.strftime("%H:%M:%S")

            L = ["========================================================\n","PySCF: FGH Calculation on a provided potential\n","========================================================\n\n","Here are the eigenvalues\n",f"{print_temp[0]}\n\n","Here are the eigenfunctions\n",f"{print_temp[1]}\n","\n========================================================\n\n",f"Current date: {formatted_date}\nCurrent time: {formatted_time}"]

            file.writelines(L)

    #''' FGH Calculation on a proton'''
    #if inputs['calctype'] == '1':
