from math import floor
from numpy import zeros
import scipy.special as spc


def longest_runs(bin_data: list, path: str):
    if len(bin_data) < 128:
        print("Error, short length of array")
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
    num_blocks = floor(len(bin_data) / m)
    freq = zeros(k + 1)
    start, end = 0, m
    for i in range(num_blocks):
        block_data = bin_data[start:end]
        max_count, run_count = 0, 0
        for j in range(m):
            if block_data[j] == 1:
                run_count += 1
                max_count = max(max_count, run_count)
            else:
                max_count = max(max_count, run_count)
                run_count = 0
        max_count = max(max_count, run_count)
        if max_count < v_values[0]:
            freq[0] += 1
        for j in range(k):
            if max_count == v_values[j]:
                freq[j] += 1
        if max_count > v_values[k - 1]:
            freq[k] += 1
        start += m
        end += m
    chi2 = 0
    for i in range(len(freq)):
        chi2 += (pow(freq[i] - (num_blocks * pik_values[i]), 2.0)) / (num_blocks * pik_values[i])
    p_val = spc.gammaincc(float(k / 2), float(chi2 / 2))
    if p_val >= 0.01:
        open(path, 'a').write(
            f'------------\nLongest Runs Test\nSuccess P-value = {str(p_val)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nLongest Runs Test\nUnsuccess P-value = {str(p_val)}\n------------\n')
    return 0
