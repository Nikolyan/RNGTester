from math import floor
from copy import deepcopy
from numpy import zeros
from tqdm import trange
import scipy.special as spc

def longest_runs(array):
    bin_data = deepcopy(array)

    if len(bin_data) < 128:
        print("\t", "Not enough data to run test!")
        return -1.0
    elif len(bin_data) < 6272:
        k, m = 3, 8
        v_values = [1, 2, 3, 4]
        pik_values = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif len(bin_data) < 75000:
        k, m = 5, 128
        v_values = [4, 5, 6, 7, 8, 9]
        pik_values = [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
    else:
        k, m = 6, 10000
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pik_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    # Work out the number of blocks, discard the remainder
    # pik = [0.2148, 0.3672, 0.2305, 0.1875]
    num_blocks = floor(len(bin_data) / m)
    frequencies = zeros(k + 1)
    block_start, block_end = 0, m
    for i in trange(num_blocks):
        # Slice the binary string into a block
        block_data = bin_data[block_start:block_end]
        # Keep track of the number of ones
        max_run_count, run_count = 0, 0
        for j in range(0, m):
            if block_data[j] == '1':
                run_count += 1
                max_run_count = max(max_run_count, run_count)
            else:
                max_run_count = max(max_run_count, run_count)
                run_count = 0
        max_run_count = max(max_run_count, run_count)
        if max_run_count < v_values[0]:
            frequencies[0] += 1
        for j in range(k):
            if max_run_count == v_values[j]:
                frequencies[j] += 1
        if max_run_count > v_values[k - 1]:
            frequencies[k] += 1
        block_start += m
        block_end += m
    chi_squared = 0
    for i in range(len(frequencies)):
        chi_squared += (pow(frequencies[i] - (num_blocks * pik_values[i]), 2.0)) / (num_blocks * pik_values[i])
    result = spc.gammaincc(float(k / 2), float(chi_squared / 2))

    if result >= 0.01:
        return f'------------ \nLongest Runs Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nLongest Runs Test \nUnsuccess P-value = {str(result)} \n------------'