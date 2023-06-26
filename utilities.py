'''Miscellaneous functions needed for operations of main.py'''

import numpy as np

'''Function to read in inputs'''
def read_input(file_name: str):

    with open(file_name, "r") as file:
        inputs = file.readlines()

    input_dict = {}
    input_items = [i.replace(' ','').replace('\n','').split('=') for i in inputs]
    for i in input_items:
        input_dict[i[0]] = i[1] 

    return input_dict

'''Function to read in potential'''
def read_potential(file_name: str):

    with open(file_name, "r") as file:
        inputs = np.array(file.readlines(),dtype=float)

    return inputs