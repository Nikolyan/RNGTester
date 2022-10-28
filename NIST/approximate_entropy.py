import scipy.special as spc
from numpy import zeros
from tqdm import trange
from math import log

def approximate_entropy(bin_data: str, path: str, pattern_length=10):
    n = len(bin_data)
    bin_data += bin_data[:pattern_length + 1:]

    max_pattern = ''
    for i in range(pattern_length + 2):
        max_pattern += '1'

    vobs1 = zeros(int(max_pattern[0:pattern_length:], 2) + 1)
    vobs2 = zeros(int(max_pattern[0:pattern_length + 1:], 2) + 1)

    for i in trange(n):
        vobs1[int(bin_data[i:i + pattern_length:], 2)] += 1
        vobs2[int(bin_data[i:i + pattern_length + 1:], 2)] += 1

    all_vobs = [vobs1, vobs2]
    sums = zeros(2)
    for i in range(2):
        for j in range(len(all_vobs[i])):
            if all_vobs[i][j] > 0:
                sums[i] += all_vobs[i][j] * log(all_vobs[i][j] / n)
    sums /= n
    ape = sums[0] - sums[1]
    chi_squared = 2.0 * n * (log(2) - ape)
    result = spc.gammaincc((2**(pattern_length - 1)), chi_squared / 2.0)
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nApproximate Entropy Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------ \nApproximate Entropy Test \nUnsuccess P-value = {str(result)} \n------------\n')

    return 0
