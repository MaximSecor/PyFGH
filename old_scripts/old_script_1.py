######################
### IMPORT MODULES ###
######################

import math
import numpy as np
import os
import sys
np.set_printoptions(precision=2,suppress=True)

#######################
### INPUT VARIABLES ###
#######################

top_card = np.array(["%chk=temp.chk","\n","%nprocshared=2","\n","%mem=5GB","\n","\n","#p CCSD/6-311++g(d,p) nosymm","\n","\n","Title","\n","\n","-1 1","\n"])
logfile_name = "CBA_P_A_MP2_NG.log" 
mode_choice = 35
atom_at_origin = 14
atom_on_xaxis = 15
atom_in_xyplane = 2

####################
### READ LOGFILE ###
####################

a_file = open(logfile_name, "r")
list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)
a_file.close()
list_of_lists = np.array(list_of_lists,dtype=object)

### NUMBER OF ATOM ###

q = 0
for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if q == 0:
            if (list_of_lists[i][j])=="NAtoms=":
                q = 1
                n_atoms = int(list_of_lists[i][1])

### ATOM TYPES ###

atom_type = []
for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if (list_of_lists[i][j])=="Symbolic":
            for k in range(n_atoms):
                atom_type.append(list_of_lists[i+k+2][0])
atom_type = np.array(atom_type)

### ATOM POSITIONS ###

for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if (list_of_lists[i][j])=="Abelian":
            idx_geom = i
            
atom_pos = []
for i in range(n_atoms):
    atom_pos.append(list_of_lists[idx_geom+6+i][3:])
atom_pos = np.array(atom_pos,dtype=float)

### NORMAL MODES COORDINATES ###

harm_temp = []
for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if (list_of_lists[i][j])=="Harmonic":
            harm_temp.append(i)
harm_temp = np.array(harm_temp)
temp_idx = (int(((harm_temp[1]-harm_temp[0])-4)/(n_atoms*3+10)))

norm_masses = []
norm_freqs = []

for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if (list_of_lists[i][j])=="Frequencies":
            temp_freqs = (list_of_lists[i][2:])
            for q in range(len(temp_freqs)):
                norm_freqs.append(temp_freqs[q])

for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if (list_of_lists[i][j])=="Reduced":
            temp_masses = (list_of_lists[i][3:])
            for q in range(len(temp_masses)):
                norm_masses.append(temp_masses[q])

m = 0
for i in range(int(len(list_of_lists))):
    for j in range(int(len(list_of_lists[i]))):
        if (list_of_lists[i][j])=="Coord":
            temp_freq = []
            for k in range(n_atoms*3):
                temp_freq.append(list_of_lists[i+k+1])
            temp_freq = np.array(temp_freq,dtype=float)
            if m==0:
                m = 1
                freq_vec = temp_freq[:,3:]
            else:
                freq_vec = np.concatenate((freq_vec, temp_freq[:,3:]),1)

norm_masses = np.array(norm_masses)
norm_freqs = np.array(norm_freqs)
n_modes = (freq_vec.shape)[1]
mode_choice = mode_choice - 1

############################################
### ROTATION OF MOLECULE AND NORMAL MODE ###
############################################
 
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

### ORIENT MOLECULE ###

atom_at_origin = atom_at_origin - 1
atom_on_xaxis = atom_on_xaxis - 1
atom_in_xyplane = atom_in_xyplane - 1

atom_pos_shifted = atom_pos-atom_pos[atom_at_origin]

v01 = atom_pos_shifted[atom_on_xaxis]
xaxis = np.array([1.0,0.0,0.0])
rot_axis = np.cross(v01,xaxis)/np.linalg.norm(np.cross(v01,xaxis))

axis = rot_axis
theta = np.arccos(np.dot(v01,xaxis)/(np.linalg.norm(v01)*np.linalg.norm(xaxis)))

n = len(atom_pos_shifted)
atom_pos_xaxis = np.zeros((n,3))

for i in range(n):
    atom_pos_xaxis[i] = np.dot(rotation_matrix(axis, theta), atom_pos_shifted[i])

v02 = np.array([0.0,atom_pos_xaxis[atom_in_xyplane,1],atom_pos_xaxis[atom_in_xyplane,2]])
yaxis =  np.array([0.0,1.0,0.0])

axis = atom_pos_xaxis[atom_on_xaxis]
theta = np.arccos(np.dot(v02,yaxis)/(np.linalg.norm(v02)*np.linalg.norm(yaxis)))

atom_pos_plane = np.zeros((n,3))

for i in range(n):
    atom_pos_plane[i] = np.dot(rotation_matrix(axis, theta), atom_pos_xaxis[i])
    turn_signal = 0

atom_pos_plane[np.abs(atom_pos_plane)<0.001] = 0

if atom_pos_plane[atom_in_xyplane,2] != 0:
    for i in range(n):
        atom_pos_plane[i] = np.dot(rotation_matrix(axis, np.pi-theta), atom_pos_xaxis[i])
    turn_signal = 1

print('\nFinal positions: \n', atom_pos_plane,'\n')

### ORIENT NORMAL MODE ###

norm_mode = (freq_vec[:,mode_choice].reshape(n_atoms,3))
norm_mode_shifted = norm_mode

v01 = atom_pos_shifted[atom_on_xaxis]
xaxis = np.array([1.0,0.0,0.0])
rot_axis = np.cross(v01,xaxis)/np.linalg.norm(np.cross(v01,xaxis))
axis = rot_axis
theta = np.arccos(np.dot(v01,xaxis)/(np.linalg.norm(v01)*np.linalg.norm(xaxis)))

norm_mode_xaxis = np.zeros((n,3))
for i in range(n):
    norm_mode_xaxis[i] = np.dot(rotation_matrix(axis, theta), norm_mode_shifted[i])

v02 = np.array([0.0,atom_pos_xaxis[atom_in_xyplane,1],atom_pos_xaxis[atom_in_xyplane,2]])
yaxis =  np.array([0.0,1.0,0.0])
axis = atom_pos_xaxis[atom_on_xaxis]
theta = np.arccos(np.dot(v02,yaxis)/(np.linalg.norm(v02)*np.linalg.norm(yaxis)))

norm_mode_plane = np.zeros((n,3))
if turn_signal == 0:
    for i in range(n):
        norm_mode_plane[i] = np.dot(rotation_matrix(axis, theta), norm_mode_xaxis[i])
else:
    for i in range(n):
        norm_mode_plane[i] = np.dot(rotation_matrix(axis, np.pi-theta), norm_mode_xaxis[i])

print('\nFinal positions: \n', norm_mode_plane,'\n')

###############################
### SET-UP GRID CALCULATION ###
###############################

dist_mass = float(norm_masses[mode_choice])*1836.1527
dist_freq = float(norm_freqs[mode_choice])/(349.7550*627.5096)
dist_ener = 200/627.5096
dist = np.sqrt((2*dist_ener)/(dist_mass*dist_freq**2))/1.8897

if os.path.isdir("anharm_temp") == True:
    os.system("rm -r anharm_temp")
os.system("mkdir anharm_temp")

for q in range(21):
    atom_pos_plane_step = atom_pos_plane + norm_mode_plane*(-dist+((dist*q)/10))
    file_name_rotated = "anharm_temp/test_"+str(q)+".gjf"
    
    if os.path.exists(file_name_rotated) == True:
        os.remove(file_name_rotated)
    
    with open(file_name_rotated, "a") as myfile:
        myfile.write("%chk=temp_"+str(q)+".chk") 
        for i in range(len(top_card)-1):
            myfile.write(top_card[i+1])
        for i in range(n_atoms):
            test = np.insert(atom_pos_plane_step[i].astype(str),0,atom_type[i])
            test1 = test.tolist()
            line = test1[0]+"      "+test1[1]+"   "+test1[2]+"   "+test1[3]+"\n"
            myfile.write(line)
        myfile.write("\n")

file_name_rotated = "anharm_temp/temp.out"

if os.path.exists(file_name_rotated) == True:
    os.remove(file_name_rotated)

field_dir = "X" #dummy
field_str = "0" #dummy
with open(file_name_rotated, "a") as myfile:
    myfile.write(norm_masses[mode_choice])
    myfile.write("\n")
    myfile.write(norm_freqs[mode_choice])
    myfile.write("\n")
    myfile.write(field_dir)
    myfile.write("\n")
    myfile.write(str(field_str))
    myfile.write("\n")
    myfile.write(str(dist))
    myfile.write("\n")

###############################
### SUBMIT GRID CALCULATION ###
###############################

cwd = os.getcwd()
os.chdir(cwd+"/anharm_temp")
for i in range(21):
     os.system("bash gaussian.sh test_"+str(i)+".gjf")
os.chdir(cwd)


