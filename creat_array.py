"""The file implements functions for working with files."""
from bitstring import BitArray
from tqdm import trange
import os
from re import findall
import random


def creat_array(path: str, size: int):
    """The variable 'path' takes as input a string with the name of a file with a binary sequence. The variable
    'size' takes as input the number of bytes to be read from the file. The function outputs a list with a binary
    sequence. The file can be with the extension '.txt' or '.bin.' Also, from this function the standard Python random
    number generator can be called if the variable 'path' is equal to 'gen'."""
    if path == 'gen':
        return python_generator(size)
    else:
        file_type = findall(r'\.\D+', path)[0]  # output the file extension
        if file_type == '.bin':
            return from_bin_file(path, size)
        elif file_type == '.txt':
            return from_txt_file(path, size)
        else:
            print('\nError creat array: no such file type')


def from_bin_file(file: str, size: int):  # creating an array from a binary file
    """The variable 'path' takes as input a string with the name of a file with a binary sequence. The variable
    'size' takes as input the number of bytes to be read from the file. The function outputs a list with a binary
    sequence. Creating an array from a binary file"""
    print("Creat array from bin file")
    size_of_file = os.path.getsize(file)  # file size
    if size_of_file < size:
        print('\nVariable size larger than file size', size, '>', size_of_file)
    elif size == 0:
        array = []  # output array
        with open(file, "rb") as f:
            for i in trange(1, size_of_file + 1):
                byte = BitArray(f.read(1)).bin  # converting hexadecimal to binary
                for j in range(8):
                    array.append(int(byte[j]))
        print('Done')
        return array
    else:
        array = []  # output array

        with open(file, "rb") as f:
            for i in trange(1, size + 1):
                byte = BitArray(f.read(1)).bin  # converting hexadecimal to binary
                for j in range(8):
                    array.append(int(byte[j]))
        print('Array has been created!')
        return array


def from_txt_file(file: str, size: int):
    """The variable 'path' takes as input a string with the name of a file with a binary sequence. The variable
    'size' takes as input the number of bytes to be read from the file. The function outputs a list with a binary
    sequence. Creating an array from a TXT file"""
    print('\nCreat array from TXT file')
    size_of_file = os.path.getsize(file)  # size of the file
    if size_of_file < size * 8:
        print('\nVariable size larger than file size', size * 8, '>', size_of_file)
    else:
        array = []
        with open(file, "r") as f:
            for i in trange(1, (size * 8) + 1):
                bit = f.read(1)
                array.append(int(bit))
        return array


def python_generator(size: int):
    print('\n Creat array by python\'s random')
    array = []
    for i in trange(size*8):
        array.append(random.randint(0, 1))
    return array
