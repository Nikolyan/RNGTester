from math import floor
from numpy import zeros
import scipy.special as spc


def longest_runs(bin_data: list, path: str):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of the tests is the longest run of ones within M-bit blocks. The purpose of this tests is to determine
    whether the length of the longest run of ones within the tested sequences is consistent with the length of the
    longest run of ones that would be expected in a random sequence. Note that an irregularity in the expected
    length of the longest run of ones implies that there is also an irregularity ub tge expected length of the long
    est run of zeroes. Therefore, only one test is necessary for this statistical tests of randomness
    :param bin_data: a binary string
    :return: the p-value from the test
    """
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
    for i in range(num_blocks):
        # Slice the binary string into a block
        block_data = bin_data[block_start:block_end]
        # Keep track of the number of ones
        max_run_count, run_count = 0, 0
        for j in range(0, m):
            if block_data[j] == 1:
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
    # print(frequencies)
    chi_squared = 0
    for i in range(len(frequencies)):
        chi_squared += (pow(frequencies[i] - (num_blocks * pik_values[i]), 2.0)) / (num_blocks * pik_values[i])
    p_val = spc.gammaincc(float(k / 2), float(chi_squared / 2))

    if p_val >= 0.01:
        open(path, 'a').write(
            f'------------\nLongest Runs Test\nSuccess P-value = {str(p_val)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nLongest Runs Test\nUnsuccess P-value = {str(p_val)}\n------------\n')

    return 0
