from math import log, floor
import scipy.special as spc
from numpy import zeros
from tqdm import trange

def non_overlapping_template_test(bin_data: list, path: str, template=[0, 0, 0, 0, 0, 0, 0, 0, 1], num_blocks=8):
    n = len(bin_data)
    size_of_pattern = len(template)
    size_of_block = floor(n / num_blocks)
    pattern_counts = zeros(num_blocks)
    for i in trange(num_blocks):
        block_start = i * size_of_block
        block_end = block_start + size_of_block
        block_data = bin_data[block_start:block_end]
        j = 0
        while j < size_of_block:
            sub_block = block_data[j:j + size_of_pattern]
            if sub_block == template:
                pattern_counts[i] += 1
                j += size_of_pattern
            else:
                j += 1
    mean = (size_of_block - size_of_pattern + 1) / (2**size_of_pattern)
    var = size_of_block * ((1 / (2**size_of_pattern)) - (((2 * size_of_pattern) - 1) / (pow(2, size_of_pattern * 2))))
    chi = 0
    for i in trange(num_blocks):
        chi += (pattern_counts[i] - mean**2.0) / var
    result = spc.gammaincc(num_blocks / 2, chi / 2)
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nNon Overlapping Template Test \nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nNon Overlapping Template Test\nUnsuccess P-value = {str(result)}\n------------\n')
    return 0

