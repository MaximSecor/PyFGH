
######################
### IMPORT MODULES ###
######################

import math
import numpy as np
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d
from numba import jit
import matplotlib.pyplot as plt
np.set_printoptions(precision=4,suppress=True)
np.seterr(all="ignore")

##################
### DEFINE FGH ###
##################

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

###################################
### READ NORMAL MODE PARAMETERS ###
###################################

temp_name = "anharm_temp/temp.out"

a_file = open(temp_name, "r")
list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)
a_file.close()
list_of_lists = np.array(list_of_lists)

mass = float(list_of_lists[0])    
freq = float(list_of_lists[1])
field_dir = str(list_of_lists[2])
field_str = int(list_of_lists[3])
dist = float(list_of_lists[4])

#########################
### READ GRID OUTPUTS ###
#########################

scan = []
dipole = []

for k in range(21):
    
    a_file = open("anharm_temp/test_"+str(k)+".log", "r")
    list_of_lists = []
    for line in a_file:
      stripped_line = line.strip()
      line_list = stripped_line.split()
      list_of_lists.append(line_list)
    a_file.close()
    list_of_lists = np.array(list_of_lists,dtype=object)
    
    for i in range(int(len(list_of_lists))):
        for j in range(int(len(list_of_lists[i]))):
            if (list_of_lists[i][j])=="converged.":
                scan.append(float(list_of_lists[i][4]))
                
    for i in range(int(len(list_of_lists))):
        for j in range(int(len(list_of_lists[i]))):
            if (list_of_lists[i][j])=="Dipole":
                dipole.append(np.array([list_of_lists[i+1][1],list_of_lists[i+1][3],list_of_lists[i+1][5]],dtype=float))
            
#################################
### GRID CALCULATION ANALYSIS ###
#################################

dipole = np.array(dipole)
scan = np.array(scan)
scan = scan - np.min(scan)

xlist = np.linspace(-dist,dist,21)*1.8897
xlist_res = np.linspace(-dist,dist,128)*1.8897

plt.close()
ylim = (0.5*mass*1836.1527*((freq/(349.7550*627.5096))**2)*dist**2*627.5096)
plt.ylim(-1,ylim/2)
plt.xlim(-0.2,0.2)
plt.plot(xlist,scan*627)
plt.savefig("scan.tif",dpi=100)

spline_scan = interp1d(xlist, scan, kind='cubic')
scan_res = spline_scan(xlist_res)

FGH_Solutions = fgh_1D(xlist_res,scan_res,mass*1836.1527)
FGH_freq = (FGH_Solutions[0][1]-FGH_Solutions[0][0])*627.5096*349.7550

x_min = np.argmin(scan_res)
dx = xlist_res[1]-xlist_res[0]
Finite_Diff = (scan_res[x_min+1]-2*scan_res[x_min]+scan_res[x_min-1])/(dx**2)

print("\n-----FREQUENCY RESULTS-----")
print("Hessian (cm^-1): ", freq)
print("Finite Difference (cm^-1): ",627.5096*349.7550*np.sqrt(Finite_Diff/(mass*1836.1527)))
print("FGH (cm^-1): ",FGH_freq)
print("Anharmonic Effect: ", 627.5096*349.7550*np.sqrt(Finite_Diff/(mass*1836.1527)) - FGH_freq)

#######################
### DIPOLE ANALYSIS ###
#######################

spline_dipole_x = interp1d(xlist, dipole[:,0], kind='cubic')
spline_dipole_y = interp1d(xlist, dipole[:,1], kind='cubic')
spline_dipole_z = interp1d(xlist, dipole[:,2], kind='cubic')
    
dipole_x = (spline_dipole_x(xlist_res))
dipole_y = (spline_dipole_y(xlist_res))
dipole_z = (spline_dipole_z(xlist_res))

test_dip = (np.concatenate((dipole_x,dipole_y,dipole_z)))
test_dip = test_dip.reshape(3,128)
    
dipole_operator = np.zeros((3,10,10))
for i in range(10):
    for j in range(10):
        dipole_operator[0,i,j] = (np.sum(FGH_Solutions[1][:,i]*FGH_Solutions[1][:,j]*spline_dipole_x(xlist_res)))
for i in range(10):
    for j in range(10):
        dipole_operator[1,i,j] = (np.sum(FGH_Solutions[1][:,i]*FGH_Solutions[1][:,j]*spline_dipole_y(xlist_res)))
for i in range(10):
    for j in range(10):
        dipole_operator[2,i,j] = (np.sum(FGH_Solutions[1][:,i]*FGH_Solutions[1][:,j]*spline_dipole_z(xlist_res)))

print("\n-----DIPOLE RESULTS-----")
print("Dipole at Minima (D): ",dipole[10])

print("Diagonal Vibrational Transition Dipole Matrix Elements (D): ")
for i in range(10):
    print("    ",dipole_operator[:,i,i])

print("Vibrational Transition Dipole (D): ",dipole_operator[:,1,0])

diff_dipole = (dipole_operator[:,0,0]-dipole_operator[:,1,1])
print("Vibrational Dipole Difference (D): ",diff_dipole)

print("\n-----OSCILLATOR STRENGTH RESULTS-----")
dip_der = (((dipole[11,:]-dipole[10,:])/(xlist[1]-xlist[0])+(dipole[10,:]-dipole[9,:])/(xlist[1]-xlist[0]))/2)
print("Dipole Derivative: (D/Angstrom)",dip_der) 
    
harm_test = fgh_1D(xlist_res,0.5*mass*1836.1527*((freq/(349.7550*627.5096))**2)*xlist_res**2,mass*1836.1527)
test_trans = (((dipole[11,:]-dipole[10,:])/(xlist[1]-xlist[0])+(dipole[10,:]-dipole[9,:])/(xlist[1]-xlist[0]))/2)*(np.sum(harm_test[1][:,0]*xlist_res*harm_test[1][:,1]))
harm_osc = np.sum(test_trans*test_trans)*(freq/(627.5096*349.7550))*(2/3)*(mass*1836.1527)*42.2561
print("Harmonic Oscillator Strength (KM/mole): ",harm_osc)
   
anharm_osc = np.sum(dipole_operator[:,1,0]*dipole_operator[:,1,0])*(FGH_freq/(627.5096*349.7550))*(2/3)*(mass*1836.1527)*42.2561
print("Anharmonic Oscillator Strength (KM/mole): ",anharm_osc,"\n")    

#####################
### VISUALIZATION ###
#####################

dipole_mag = np.sqrt(dipole_x**2+dipole_y**2+dipole_z**2)
dipole_mag = dipole_mag - dipole_mag[np.argmin(spline_scan(xlist_res/1.8897))]

plt.close()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

dx = xlist_res[np.argmin(spline_scan(xlist_res/1.8897))]/1.8897
dE = np.min(spline_scan(xlist_res/1.8897))

ax1.plot(xlist_res/1.8897-dx,FGH_Solutions[1][:,0]**2*(ylim*1.25)+FGH_Solutions[0][0]*627.5096,'b')
ax1.plot(xlist_res/1.8897-dx,FGH_Solutions[1][:,1]**2*(ylim*1.25)+FGH_Solutions[0][1]*627.5096,'b')
ax1.plot(xlist_res/1.8897-dx,(spline_scan(xlist_res/1.8897)-dE)*627.5096,'k')
ax1.plot(xlist_res/1.8897,0.5*mass*1836.1527*((freq/(349.7550*627.5096))**2)*(xlist_res/1.8897)**2*627.5096,'r--')

ax2.plot(xlist_res/1.8897, dipole_mag, 'g.')

ax1.set_xlabel('NM Displacement (Angstroms)')
ax1.set_ylabel('Energy (kcal/mol)', color='k')
ax2.set_ylabel('Dipole (D)', color='g')

ylim = (0.5*mass*1836.1527*((freq/(349.7550*627.5096))**2)*dist**2*627.5096)

ax1.set_ylim([-1,ylim/2])
ax1.set_xlim([-0.2,0.2])
#ax2.set_ylim([-np.max(np.abs(dipole_mag)),np.max(np.abs(dipole_mag))])

plt.savefig("FGH_2.tif",dpi=300)
plt.close()

ylim = (0.5*mass*1836.1527*((freq/(349.7550*627.5096))**2)*dist**2*627.5096)
plt.ylim(-1,ylim/2)
plt.xlim(-0.2,0.2)

dx = (xlist_res[np.argmin(spline_scan(xlist_res/1.8897))]/1.8897)
dE = np.min(spline_scan(xlist_res/1.8897))

plt.plot(xlist_res/1.8897-dx,FGH_Solutions[1][:,0]**2*(ylim*1.25)+FGH_Solutions[0][0]*627.5096,'b')
plt.plot(xlist_res/1.8897-dx,FGH_Solutions[1][:,1]**2*(ylim*1.25)+FGH_Solutions[0][1]*627.5096,'b')
plt.plot(xlist_res/1.8897-dx,(spline_scan(xlist_res/1.8897)-dE)*627.5096,'k')
plt.plot(xlist_res/1.8897,0.5*mass*1836.1527*((freq/(349.7550*627.5096))**2)*(xlist_res/1.8897)**2*627.5096,'r--')

plt.savefig("FGH.tif",dpi=100)

dipole_ref = dipole - dipole[10]
plt.close()
plt.xlim(-dist,dist)
#plt.plot(xlist,dipole_ref[:,0],'k',xlist,dipole_ref[:,1],'b',xlist,dipole_ref[:,2],'r')
#plt.scatter(xlist/1.8897,dipole_ref[:,0],marker='x',color='k')
#plt.scatter(xlist/1.8897,dipole_ref[:,1],marker=(5,2),color='r')
#plt.scatter(xlist/1.8897,dipole_ref[:,2],marker='+',color='b')
plt.plot(xlist_res/1.8897,dipole_mag)
plt.savefig("dipole.tif",dpi=100)

#####################
### DATA PRINTOUT ###
#####################

file_scan = 'PES.csv'
os.system('touch ' + file_scan)
df_features_potential = pd.DataFrame(scan_res)
df_features_potential.to_csv(file_scan, index = False, header=True)

file_present = 'dipole.csv'
os.system('touch ' + file_present)
df_features_potential = pd.DataFrame(test_dip)
df_features_potential.to_csv(file_present, index = False, header=True)


