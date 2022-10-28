from math import floor
import scipy.special as spc
from numpy import zeros, array, exp
from tqdm import trange

def get_prob(u, x):
    out = 1.0 * exp(-x)
    if u != 0:
        out = 1.0 * x * exp(2 * -x) * (2 ** -u) * spc.hyp1f1(u + 1, 2, x)
    return out


def overlapping_template(bin_data: list, path: str, pattern_size=9, block_size=1032):
    n = len(bin_data)
    list_patt = []
    for i in range(pattern_size):
        list_patt.append(1)
    num_blocks = floor(n / block_size)
    eta = (float(block_size - pattern_size + 1) / (2**pattern_size)) / 2.0
    piks = [get_prob(i, eta) for i in range(5)]
    diff = float(array(piks).sum())
    piks.append(1.0 - diff)
    pattern_counts = zeros(6)
    for i in trange(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block_data = bin_data[block_start:block_end]
        counter = 0
        j = 0
        while j < block_size:
            sub_block = block_data[j:j + pattern_size]
            if sub_block == list_patt:
                counter += 1
            j += 1
        if counter <= 4:
            pattern_counts[counter] += 1
        else:
            pattern_counts[5] += 1
    chi_squared = 0.0
    for i in trange(len(pattern_counts)):
        chi_squared += pow(pattern_counts[i] - num_blocks * piks[i], 2.0) / (num_blocks * piks[i])
    result = spc.gammaincc(5.0 / 2.0, chi_squared / 2.0)
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nOverlapping Patterns Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------ \nOverlapping Patterns Test\nUnsuccess P-value = {str(result)}\n------------\n')
    return 0
